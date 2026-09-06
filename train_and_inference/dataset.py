"""Prior dataset and the GluonTS transformation chain.

Calendar features come from each series' actual dates, attached as a data field,
rather than from the gluonts `start` Period, which for a yearly frequency always
reports Dec-31. The prior's month-day anchor is stored per series in its
metadata (see `convert_prior.py --anchor`).
"""

from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    SimpleTransformation,
    Transformation,
)

from constants import (
    MAX_HISTORY,
    MIN_CONTEXT,
    NP_FLOAT_DTYPE,
    PAD_VALUE,
    TARGET_LEN,
    TIME_FEAT_DIM,
)

FEAT_TIME = "feat_time"


def dist_rank_world() -> "tuple[int, int]":
    """(rank, world_size), or (0, 1) when not running under DDP."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


def compute_time_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """[year, month, day, day_of_week + 1, day_of_year, 0, 0] -> (7, T).

    `day_of_week` is offset by one because index 0 of each embedding table is
    reserved for padding.
    """
    n = len(dates)
    feats = np.zeros((TIME_FEAT_DIM, n), dtype=NP_FLOAT_DTYPE)
    feats[0] = dates.year.values
    feats[1] = dates.month.values
    feats[2] = dates.day.values
    feats[3] = dates.dayofweek.values + 1
    feats[4] = dates.dayofyear.values
    return feats


def yearly_dates(start_year: int, length: int, month: int = 12,
                 day: int = 31) -> pd.DatetimeIndex:
    """Contiguous yearly grid anchored on a fixed month-day."""
    years = start_year + np.arange(length)
    return pd.DatetimeIndex(
        pd.to_datetime(pd.DataFrame({"year": years, "month": month, "day": day}))
    )


class PriorDataset:
    """GluonTS-style iterable over the converted synthetic prior.

    Series are emitted in file order and never reshuffled. That ordering is part
    of the training regime: the prior is stored as contiguous 250-variant blocks
    per base series, which reshuffling would dissolve.

    The values matrix is memory-mapped and calendar features are cached per
    distinct (start_year, anchor); the cache is bounded and evicts FIFO.
    """

    def __init__(self, y_path: str, meta_path: str, feat_cache_size: int = 2048):
        self.y = np.load(y_path, mmap_mode="r")
        meta = np.load(meta_path, allow_pickle=False)
        self.ids = meta["ids"]
        self.start_year = meta["start_year"].astype(int)
        self.series_len = int(meta["series_len"])

        n = len(self.ids)
        if "anchor_month" in meta:
            self.anchor_month = meta["anchor_month"].astype(int)
            self.anchor_day = meta["anchor_day"].astype(int)
        else:
            # Older prior, written before per-series anchors existed.
            legacy = str(meta["day_anchor"]) if "day_anchor" in meta else "12-31"
            month, day = (int(p) for p in legacy.split("-"))
            self.anchor_month = np.full(n, month, dtype=int)
            self.anchor_day = np.full(n, day, dtype=int)

        assert self.y.shape == (n, self.series_len), (
            f"prior shape {self.y.shape} does not match metadata"
        )
        assert len(self.anchor_month) == n and len(self.anchor_day) == n, (
            "anchor arrays do not cover every series"
        )
        self._feat_cache: Dict["tuple[int, int, int]", np.ndarray] = {}
        self._feat_cache_size = feat_cache_size

    def __len__(self) -> int:
        return len(self.ids)

    def _features(self, start_year: int, month: int, day: int) -> np.ndarray:
        key = (start_year, month, day)
        cached = self._feat_cache.get(key)
        if cached is None:
            dates = yearly_dates(start_year, self.series_len, month, day)
            cached = compute_time_features(dates)
            if len(self._feat_cache) >= self._feat_cache_size:
                self._feat_cache.pop(next(iter(self._feat_cache)))
            self._feat_cache[key] = cached
        return cached

    def _entry(self, i: int) -> Dict:
        return {
            FieldName.ITEM_ID: str(self.ids[i]),
            FieldName.START: pd.Period(int(self.start_year[i]), freq="Y"),
            FieldName.TARGET: np.asarray(self.y[i], dtype=NP_FLOAT_DTYPE),
            FEAT_TIME: self._features(
                int(self.start_year[i]),
                int(self.anchor_month[i]),
                int(self.anchor_day[i]),
            ),
        }

    def __iter__(self) -> Iterator[Dict]:
        """One pass in file order, sharded across DDP ranks.

        The tail is truncated so every rank gets the same number of series, since
        DDP deadlocks if ranks disagree on the step count. At most `world - 1`
        series are dropped (3 of 161,250 on 4 GPUs).
        """
        rank, world = dist_rank_world()
        order = np.arange(len(self.ids))
        if world > 1:
            order = order[: (len(order) // world) * world][rank::world]
        for i in order:
            yield self._entry(int(i))


class AllWindowsSampler(InstanceSampler):
    """Every valid split point, in order."""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if b < a:
            return np.array([], dtype=int)
        return np.arange(a, b + 1)


class LastWindowSampler(InstanceSampler):
    """The one split point that puts the forecast at the end of the series."""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        _, b = self._get_bounds(ts)
        return np.array([b])


class DropUnusedFields(SimpleTransformation):
    """Drop the fields the model does not consume."""

    def __init__(self, keep: List[str]):
        self.keep = set(keep)

    def transform(self, data: Dict) -> Dict:
        return {k: v for k, v in data.items() if k in self.keep}


def build_base_transformation() -> Transformation:
    """The per-series part of the chain, applied before instance splitting."""
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
                dtype=NP_FLOAT_DTYPE,
            )
        ]
    )


def build_instance_splitter(
    prediction_length: int = TARGET_LEN,
    context_length: int = MAX_HISTORY,
    sampler: Optional[InstanceSampler] = None,
    min_past: int = MIN_CONTEXT,
) -> Transformation:
    """Produces `past_target`, `past_feat_time`, `past_is_pad`, `future_target`.

    With `min_past = 12` and `min_future = prediction_length`, a 200-step series
    has a valid split range of [12, 194] -- 183 windows with contexts of 12..194.
    """
    if sampler is None:
        sampler = AllWindowsSampler(min_past=min_past, min_future=prediction_length)

    return Chain(
        [
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=sampler,
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[FEAT_TIME, FieldName.OBSERVED_VALUES],
                # `dummy_value` pads the target and the calendar features alike;
                # index 0 is the padding row in every calendar embedding table.
                dummy_value=PAD_VALUE,
                output_NTC=True,
            ),
            DropUnusedFields(
                [
                    FieldName.ITEM_ID,
                    FieldName.FORECAST_START,
                    f"past_{FieldName.TARGET}",
                    f"past_{FEAT_TIME}",
                    f"past_{FieldName.OBSERVED_VALUES}",
                    f"past_{FieldName.IS_PAD}",
                    f"future_{FieldName.TARGET}",
                ]
            ),
        ]
    )
