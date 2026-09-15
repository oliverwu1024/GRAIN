"""Lightning wrapper"""

from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from gluonts.dataset.field_names import FieldName

from constants import ADAM_EPSILON, LEARNING_RATE
from dataset import FEAT_TIME
from model import GRAINModel

PAST_TARGET = f"past_{FieldName.TARGET}"
PAST_FEAT_TIME = f"past_{FEAT_TIME}"
PAST_IS_PAD = f"past_{FieldName.IS_PAD}"
PAST_OBSERVED = f"past_{FieldName.OBSERVED_VALUES}"
FUTURE_TARGET = f"future_{FieldName.TARGET}"


def observed_from_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:

    is_pad = batch[PAST_IS_PAD]
    observed = batch[PAST_OBSERVED]
    if observed.dim() == 3:
        observed = observed[..., 0]
    return (is_pad < 0.5) & (observed > 0.5)


class GRAINLightningModule(pl.LightningModule):


    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = LEARNING_RATE,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GRAINModel(**(model_kwargs or {}))
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            batch[PAST_TARGET],
            batch[PAST_FEAT_TIME],
            observed_from_batch(batch),
        )

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.model.quantile_loss(out, batch[FUTURE_TARGET])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # eps is set explicitly; torch's default is 1e-8.
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            eps=ADAM_EPSILON,
            weight_decay=self.weight_decay,
        )
