"""GluonTS estimator and predictor for the yearly GRAIN.

`prediction_length` is a predictor setting, not a network one: the head is always
built for `max_prediction_length` (6) steps and shorter horizons are served by
slicing. Every (horizon, quantile) pair is an independent output over a shared
trunk, so `result[:, :h]` is what a dedicated h-step head would produce. One
checkpoint therefore serves every horizon in 1..6.
"""

from typing import Any, Dict, Optional, Union

import torch
from gluonts.core.component import validated
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import PseudoShuffled
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import SelectFields, Transformation

from constants import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_HISTORY,
    MIN_CONTEXT,
    QUANTILE_LEVELS,
    SHUFFLE_BUFFER,
    STEPS_PER_EPOCH,
    TARGET_LEN,
)
from dataset import (
    AllWindowsSampler,
    LastWindowSampler,
    build_base_transformation,
    build_instance_splitter,
)
from lightning_module import (
    FUTURE_TARGET,
    GRAINLightningModule,
    PAST_FEAT_TIME,
    PAST_IS_PAD,
    PAST_OBSERVED,
    PAST_TARGET,
)

TRAINING_FIELDS = [
    PAST_TARGET,
    PAST_FEAT_TIME,
    PAST_IS_PAD,
    PAST_OBSERVED,
    FUTURE_TARGET,
]
PREDICTION_FIELDS = [PAST_TARGET, PAST_FEAT_TIME, PAST_IS_PAD, PAST_OBSERVED]


class _InferenceNet(torch.nn.Module):
    """Adapts the model to the shape the gluonts forecast generator expects.

    `QuantileForecastGenerator` unpacks the result as `(outputs,), loc, scale`.
    The model denormalises internally, so loc/scale are None.
    """

    def __init__(self, module: GRAINLightningModule, prediction_length: int):
        super().__init__()
        self.module = module
        self.prediction_length = prediction_length

    def forward(self, past_target, past_feat_time, past_is_pad,
                past_observed_values):
        observed = (past_is_pad < 0.5) & (past_observed_values > 0.5)
        preds = self.module.model.predict(
            past_target, past_feat_time, observed, self.prediction_length
        )  # (B, h, Q)
        return (preds,), None, None


def build_predictor(
    input_transform: Transformation,
    module: GRAINLightningModule,
    prediction_length: int,
    quantile_levels=QUANTILE_LEVELS,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> PyTorchPredictor:
    """Wrap the fixed-width network as a predictor for one horizon."""
    max_h = module.model.pred_len
    if not 1 <= prediction_length <= max_h:
        raise ValueError(
            f"prediction_length must be in [1, {max_h}] (the head width), "
            f"got {prediction_length}"
        )
    return PyTorchPredictor(
        input_names=PREDICTION_FIELDS,
        prediction_net=_InferenceNet(module, prediction_length),
        batch_size=batch_size,
        prediction_length=prediction_length,
        input_transform=input_transform,
        forecast_generator=QuantileForecastGenerator(
            quantiles=[str(q) for q in quantile_levels]
        ),
        device=device,
    )


class GRAINEstimator(PyTorchLightningEstimator):
    """Trains the network at full head width; horizon is chosen at predict time."""

    @validated()
    def __init__(
        self,
        prediction_length: int = TARGET_LEN,
        max_prediction_length: int = TARGET_LEN,
        context_length: int = MAX_HISTORY,
        min_past: int = MIN_CONTEXT,
        lr: float = LEARNING_RATE,
        weight_decay: float = 0.0,
        batch_size: int = BATCH_SIZE,
        num_batches_per_epoch: int = STEPS_PER_EPOCH,
        shuffle_buffer: int = SHUFFLE_BUFFER,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(trainer_kwargs=trainer_kwargs or {})
        if prediction_length > max_prediction_length:
            raise ValueError(
                f"prediction_length {prediction_length} exceeds the head width "
                f"{max_prediction_length}"
            )
        # Every series yields 183 windows, so a full pass is ~28,817 batches --
        # an epoch that would never finish. The epoch length is declared instead.
        if not num_batches_per_epoch or num_batches_per_epoch < 1:
            raise ValueError("num_batches_per_epoch must be a positive integer")
        self.prediction_length = prediction_length
        self.max_prediction_length = max_prediction_length
        self.context_length = context_length
        self.min_past = min_past
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.shuffle_buffer = shuffle_buffer
        self.model_kwargs = dict(model_kwargs or {})
        self.model_kwargs.setdefault("pred_len", max_prediction_length)
        self.model_kwargs.setdefault("seq_len", context_length)

    def create_transformation(self) -> Transformation:
        # Applied to the dataset before instance splitting, so it holds only the
        # part of the chain that does not depend on the split point.
        return build_base_transformation()

    def _splitter(self, sampler, prediction_length: int) -> Transformation:
        return build_instance_splitter(
            prediction_length=prediction_length,
            context_length=self.context_length,
            sampler=sampler,
            min_past=self.min_past,
        )

    def create_lightning_module(self) -> GRAINLightningModule:
        return GRAINLightningModule(
            model_kwargs=self.model_kwargs, lr=self.lr, weight_decay=self.weight_decay
        )

    def create_training_data_loader(self, data, module, **kwargs):
        """Every window of every series, in file order, through a
        `shuffle_buffer`-element buffer, cut into fixed-length epochs."""
        # Train at the full head width so every horizon is supervised.
        transform = self._splitter(
            AllWindowsSampler(
                min_past=self.min_past, min_future=self.max_prediction_length
            ),
            self.max_prediction_length,
        ) + SelectFields(TRAINING_FIELDS, allow_missing=True)

        stream = _Transformed(data, transform, is_train=True, cycle=True)
        if self.shuffle_buffer > 0:
            stream = PseudoShuffled(stream, shuffle_buffer_length=self.shuffle_buffer)
        # Must wrap outside the buffer. See _Continuous.
        stream = _Continuous(stream)

        return as_stacked_batches(
            stream,
            batch_size=self.batch_size,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
            field_names=TRAINING_FIELDS,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: GRAINLightningModule,
        prediction_length: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> PyTorchPredictor:
        h = prediction_length or self.prediction_length
        # The chain must hold out exactly h points, so the splitter is rebuilt
        # for this horizon; the network itself is unchanged.
        infer_transform = transformation + self._splitter(
            LastWindowSampler(min_past=self.min_past, min_future=h), h
        )
        return build_predictor(
            input_transform=infer_transform,
            module=module,
            prediction_length=h,
            batch_size=self.batch_size,
            device=device,
        )


class _Continuous:
    """Iterable that resumes one long-lived iterator on every pass.

    `as_stacked_batches(..., num_batches_per_epoch=N)` ends in `IterableSlice`,
    which takes N batches by calling `islice` on its source. `islice` calls
    `iter()` on that source, so a re-iterable dataset would restart the stream
    every epoch and pin the whole run to the opening N batches -- the same ~923
    series forever, which looks like a smoothly falling train_loss beside
    benchmark metrics that bottom early and then degrade. `IterableSlice` resumes
    only when its source is already an iterator.

    Caveat: the batching generator is dropped at each epoch boundary, so up to
    `batch_size - 1` windows are discarded per epoch (<0.7% of a 160x1024 epoch).
    They are the tail of a contiguous block, so this costs a little data but
    biases nothing.
    """

    def __init__(self, iterable):
        self._iterator = iter(iterable)

    def __iter__(self):
        return self._iterator


class _Transformed:
    """Endless view of `dataset` under `transform`; gluonts loaders want an iterable."""

    def __init__(self, dataset, transform, is_train: bool, cycle: bool = False):
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        self.cycle = cycle

    def _pass(self):
        produced = False
        for entry in self.transform(iter(self.dataset), is_train=self.is_train):
            produced = True
            yield entry
        if not produced:
            raise RuntimeError("transformation produced no training instances")

    def __iter__(self):
        if not self.cycle:
            yield from self._pass()
            return
        while True:
            yield from self._pass()
