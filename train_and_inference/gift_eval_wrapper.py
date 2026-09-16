"""GIFT-Eval adapter for the yearly LGT-PFN.

GIFT-Eval hands you `dataset.test_data.input` (GluonTS entries: `start` Period +
`target`, with the label already held out) and expects `QuantileForecast`s back
for `gluonts.model.evaluation.evaluate_forecasts`. The port's predictor is
already a real `PyTorchPredictor`, so unlike the TensorFlow notebook nothing here
re-implements padding or windowing -- the estimator's own transformation chain is
reused. Four things do have to change, and each is a silent-wrong-answer bug if
it is missed:

1. NO HOLD-OUT AT THE SPLIT POINT. `inference.py` feeds whole series and relies
   on `LastWindowSampler(min_future=h)` to hold out the tail. GIFT-Eval already
   removed the label, so the same sampler would drop the last `h` REAL
   observations from the context. `TestSplitSampler` splits at `len(target)`,
   which is the GluonTS convention for prediction-time chains.

2. CALENDAR FEATURES FROM A PERIOD, NOT A TIMESTAMP. Entries carry no dates.
   `Period.to_timestamp()` (what the TF notebook used) overflows on M4 yearly --
   series start at 1750 and run past pandas' 2262-04-11 limit -- and it collapses
   yearly periods to Jan-1, silently moving every timestamp. Reading the fields
   straight off a `PeriodIndex` avoids the overflow entirely and reports Dec-31,
   the last day of the period, which is what a yearly Period actually denotes.
   See `period_time_features`.

3. DTYPE. GIFT-Eval targets are float32; the network is float64 end to end
   (`constants.FLOAT_DTYPE`). Without an explicit cast the first matmul raises.

4. FORECAST KEYS. GIFT-Eval scores 9 deciles and several metrics ask for
   `forecast.mean`; the head emits 11 levels and no mean. `GiftForecastNet` maps
   the model's columns onto `["mean", "0.1", ..., "0.9"]`, taking the median for
   "mean" exactly as the TF notebook did.

The head is 6 steps wide, so only `term="short"` on a yearly dataset is
supported; `build_gift_predictor` raises on anything longer rather than
truncating a horizon it cannot serve.
"""

from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    SimpleTransformation,
    TestSplitSampler,
    Transformation,
)

from constants import (
    FLOAT_DTYPE,
    MAX_HISTORY,
    MEDIAN_INDEX,
    NP_FLOAT_DTYPE,
    QUANTILE_LEVELS,
    TIME_FEAT_DIM,
)
from dataset import FEAT_TIME, build_instance_splitter
from estimator import PREDICTION_FIELDS
from lightning_module import LGTPFNLightningModule

# GIFT-Eval scores 9 deciles; the head trains on 11 (the 0.05/0.95 tails are
# predicted but never scored here). "mean" is served by the median column.
GIFT_QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FORECAST_KEYS: List[str] = ["mean"] + [str(q) for q in GIFT_QUANTILE_LEVELS]

# Targets past this are treated as infinite and clamped, matching the TF wrapper.
INF_CLAMP = 1e10


# --------------------------------------------------------------------------
# calendar features
# --------------------------------------------------------------------------
def period_time_features(start: pd.Period, length: int) -> np.ndarray:
    """[year, month, day, day_of_week + 1, day_of_year, 0, 0] -> (7, T) float64.

    Built from a `PeriodIndex`, never a `DatetimeIndex`. Two reasons:

      * No 1677-2262 bound. M4 yearly starts at 1750 and the longest series run
        past 2262, where `Period.to_timestamp()` raises OutOfBoundsDatetime.
      * For period frequencies coarser than daily, pandas reports the fields of
        the period's LAST day -- Dec-31 for yearly -- which is the date the
        period denotes. `to_timestamp()` would report its FIRST day instead,
        shifting every GIFT-Eval timestamp by a year relative to what the same
        period means everywhere else in this codebase.

    The Dec-31 expectation is asserted by `check_yearly_anchor` rather than
    trusted, because it rests on pandas' end-of-period convention.
    """
    idx = pd.period_range(start=start, periods=length)
    feats = np.zeros((TIME_FEAT_DIM, length), dtype=NP_FLOAT_DTYPE)
    feats[0] = np.asarray(idx.year, dtype=NP_FLOAT_DTYPE)
    feats[1] = np.asarray(idx.month, dtype=NP_FLOAT_DTYPE)
    feats[2] = np.asarray(idx.day, dtype=NP_FLOAT_DTYPE)
    # Index 0 of the dow vocabulary is reserved for padding, hence the +1.
    feats[3] = np.asarray(idx.dayofweek, dtype=NP_FLOAT_DTYPE) + 1.0
    feats[4] = np.asarray(idx.dayofyear, dtype=NP_FLOAT_DTYPE)
    return feats


def check_yearly_anchor(start: pd.Period) -> None:
    """Fail loudly if yearly periods do not resolve to Dec-31.

    `period_time_features` depends on pandas reporting end-of-period fields for
    frequencies coarser than daily. If that convention ever changes, every
    GIFT-Eval timestamp silently shifts to Jan-1 of the same period -- a year
    away from the date the period denotes and from what M4's own files carry.
    The only symptom would be slightly different metrics, which is exactly the
    kind of bug that gets written up as a modelling result.

    Note this is a tripwire on pandas' behaviour, not a training-anchor check:
    the prior can be re-anchored per series (`convert_prior.py --anchor random`),
    so a Jan-1 slip is no longer out-of-distribution for the embeddings -- it is
    still the wrong date.
    """
    feats = period_time_features(start, 1)
    month, day = int(feats[1, 0]), int(feats[2, 0])
    if (month, day) != (12, 31):
        raise RuntimeError(
            f"yearly Period {start} resolved to month={month} day={day}, "
            f"expected (12, 31). pandas' end-of-period convention changed; "
            f"the calendar features would not match the training prior."
        )


class AddPeriodTimeFeatures(SimpleTransformation):
    """Attach `feat_time` derived from the entry's `start` Period."""

    def __init__(self, check_anchor: bool = True):
        self.check_anchor = check_anchor
        self._checked = False

    def transform(self, data: Dict) -> Dict:
        start = data[FieldName.START]
        if self.check_anchor and not self._checked:
            check_yearly_anchor(start)
            self._checked = True
        data[FEAT_TIME] = period_time_features(start, len(data[FieldName.TARGET]))
        return data


class CastTarget(SimpleTransformation):
    """Cast the target to the network dtype and clamp non-finite values.

    GIFT-Eval serves float32. NaNs are left alone -- `AddObservedValuesIndicator`
    runs after this and turns them into an explicit mask, which the model honours.
    Infinities have no such path, so they are clamped the way the TF wrapper did.
    """

    def transform(self, data: Dict) -> Dict:
        target = np.asarray(data[FieldName.TARGET], dtype=NP_FLOAT_DTYPE)
        if target.ndim != 1:
            raise ValueError(
                f"expected a univariate target, got shape {target.shape}; "
                f"construct the Dataset with to_univariate=True"
            )
        if not np.isfinite(target).all():
            target = np.where(np.isposinf(target), INF_CLAMP, target)
            target = np.where(np.isneginf(target), -INF_CLAMP, target)
        data[FieldName.TARGET] = target
        return data


def build_gift_transformation(
    prediction_length: int,
    context_length: int = MAX_HISTORY,
    check_anchor: bool = True,
) -> Transformation:
    """Cast -> observed mask -> calendar features -> split at the series end."""
    return Chain(
        [
            CastTarget(),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
                dtype=NP_FLOAT_DTYPE,
            ),
            AddPeriodTimeFeatures(check_anchor=check_anchor),
        ]
    ) + build_instance_splitter(
        prediction_length=prediction_length,
        context_length=context_length,
        # Splits at len(target). The label is already held out by GIFT-Eval, so
        # anything else would discard real observations -- see module docstring.
        sampler=TestSplitSampler(),
        min_past=0,
    )


# --------------------------------------------------------------------------
# network / predictor
# --------------------------------------------------------------------------
class GiftForecastNet(torch.nn.Module):
    """Reorders the head's quantile columns into GIFT-Eval's forecast keys.

    `QuantileForecastGenerator` unpacks `net(**inputs)` as `(outputs,), loc,
    scale` and transposes each item to (len(forecast_keys), prediction_length),
    so the emitted column order must match `FORECAST_KEYS` exactly.
    """

    def __init__(
        self,
        module: LGTPFNLightningModule,
        prediction_length: int,
        model_levels: Sequence[float] = QUANTILE_LEVELS,
        median_index: int = MEDIAN_INDEX,
    ):
        super().__init__()
        self.module = module
        self.prediction_length = prediction_length
        levels = np.asarray(model_levels, dtype=np.float64)
        if abs(levels[median_index] - 0.5) > 1e-12:
            raise ValueError("median_index must select the 0.5 level")

        # One column per forecast key. "mean" takes the median, as the TF
        # wrapper did: the head is a pinball head and has no mean output.
        cols = []
        for key in FORECAST_KEYS:
            if key == "mean":
                cols.append(median_index)
                continue
            level = float(key)
            j = int(np.argmin(np.abs(levels - level)))
            if abs(levels[j] - level) > 1e-9:
                raise ValueError(
                    f"the head has no {level} quantile; levels are {list(levels)}"
                )
            cols.append(j)
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))

    def forward(self, past_target, past_feat_time, past_is_pad,
                past_observed_values):
        observed = (past_is_pad < 0.5) & (past_observed_values > 0.5)
        preds = self.module.model.predict(
            past_target, past_feat_time, observed, self.prediction_length
        )  # (B, h, Q)
        return (preds[..., self.cols],), None, None


def load_module(
    checkpoint: str, device: Union[str, torch.device] = "cpu"
) -> LGTPFNLightningModule:
    """Restore a trained module in the network's dtype."""
    torch.set_default_dtype(FLOAT_DTYPE)
    module = LGTPFNLightningModule.load_from_checkpoint(checkpoint, map_location=device)
    module.eval()
    return module


def build_gift_predictor(
    module: LGTPFNLightningModule,
    prediction_length: int,
    context_length: int = MAX_HISTORY,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
    check_anchor: bool = True,
) -> PyTorchPredictor:
    """A GIFT-Eval-ready predictor for one (dataset, term) horizon."""
    max_h = module.model.pred_len
    if not 1 <= prediction_length <= max_h:
        raise ValueError(
            f"prediction_length {prediction_length} is outside the head width "
            f"[1, {max_h}]. Only term='short' on a yearly dataset fits this "
            f"checkpoint; medium/long multiply the horizon by 10/15."
        )
    return PyTorchPredictor(
        input_names=PREDICTION_FIELDS,
        prediction_net=GiftForecastNet(module, prediction_length),
        batch_size=batch_size,
        prediction_length=prediction_length,
        input_transform=build_gift_transformation(
            prediction_length, context_length, check_anchor
        ),
        forecast_generator=QuantileForecastGenerator(quantiles=FORECAST_KEYS),
        device=device,
    )
