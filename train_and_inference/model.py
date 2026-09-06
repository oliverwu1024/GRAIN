"""The yearly LGT-PFN network and its pinball objective."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import (
    D_FF,
    FLOAT_DTYPE,
    DROPOUT,
    EMBED_DIM,
    LN_EPS,
    LONG_PATCH_SIZE,
    MAX_HISTORY,
    NUM_ENCODER_LAYERS,
    NUM_HEADS,
    NUM_LONG_PATCHES,
    NUM_SHORT_PATCHES,
    POOL_HEADS,
    POOL_QUERIES,
    QUANTILE_LEVELS,
    SCALER_EPSILON,
    SHORT_PATCH_SIZE,
    TARGET_LEN,
    TREND_SMOOTH_LAMBDA,
)
from modules import (
    AttentionPooling,
    DualPatchEmbedding,
    GatedFusion,
    LGTPFNEncoderLayer,
    SeasonalDecomposition,
    TemporalFeatureEmbedding,
    TrendDecomposition,
    init_linear,
)
from scalers import masked_variance_normalization


class LGTPFNModel(nn.Module):
    """Prior-fitted network with a fixed-width multi-horizon quantile head.

    The head always emits `pred_len` steps; shorter horizons are served by slicing
    at prediction time.
    """

    def __init__(
        self,
        seq_len: int = MAX_HISTORY,
        pred_len: int = TARGET_LEN,
        embed_dim: int = EMBED_DIM,
        num_layers: int = NUM_ENCODER_LAYERS,
        num_heads: int = NUM_HEADS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
        quantile_levels: Sequence[float] = QUANTILE_LEVELS,
        horizon_weights: Optional[Sequence[float]] = None,
        scaler_epsilon: float = SCALER_EPSILON,
        trend_smooth_lambda: float = TREND_SMOOTH_LAMBDA,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.scaler_epsilon = scaler_epsilon
        self.trend_smooth_lambda = trend_smooth_lambda
        self.pad_value = pad_value

        self.quantile_levels = tuple(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)
        self.median_index = self.quantile_levels.index(0.5)

        self.register_buffer(
            "_quantiles", torch.tensor(self.quantile_levels), persistent=False
        )
        if horizon_weights is None:
            horizon_weights = [1.0] * pred_len
        if len(horizon_weights) != pred_len:
            raise ValueError(
                f"horizon_weights has length {len(horizon_weights)}, expected {pred_len}"
            )
        self.register_buffer(
            "_horizon_weights", torch.tensor(horizon_weights, dtype=torch.get_default_dtype())
        )

        self.dual_patch_embedding = DualPatchEmbedding(
            LONG_PATCH_SIZE, SHORT_PATCH_SIZE, embed_dim, seq_len
        )
        self.temporal_embedding = TemporalFeatureEmbedding(embed_dim)
        self.seasonal_decomp = SeasonalDecomposition(embed_dim)
        self.trend_decomp = TrendDecomposition(embed_dim)
        self.fusion = GatedFusion(embed_dim)
        self.post_fuse_norm = nn.LayerNorm(embed_dim, eps=LN_EPS)

        self.encoder_layers = nn.ModuleList(
            LGTPFNEncoderLayer(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.pool = AttentionPooling(embed_dim, POOL_QUERIES, POOL_HEADS)

        self.projector1 = nn.Linear(POOL_QUERIES * embed_dim, d_ff)
        self.projector2 = nn.Linear(d_ff, pred_len * self.num_quantiles)
        init_linear(self.projector1)
        init_linear(self.projector2)

        # Cast here so the dtype holds without the caller having to set
        # torch.set_default_dtype() first. `.to()` only touches float tensors, so
        # the integer index buffers below stay int64.
        self.to(FLOAT_DTYPE)

        # Patch-start positions where calendar features are sampled.
        self.register_buffer(
            "_long_idx",
            torch.arange(0, NUM_LONG_PATCHES * LONG_PATCH_SIZE, LONG_PATCH_SIZE),
            persistent=False,
        )
        self.register_buffer(
            "_short_idx",
            torch.arange(0, NUM_SHORT_PATCHES * SHORT_PATCH_SIZE, SHORT_PATCH_SIZE),
            persistent=False,
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _patch_mask(observed: torch.Tensor) -> torch.Tensor:
        """(B, T) point mask -> (B, 120) token mask. A patch is real if ANY point is."""
        b = observed.shape[0]
        long_usable = NUM_LONG_PATCHES * LONG_PATCH_SIZE
        short_usable = NUM_SHORT_PATCHES * SHORT_PATCH_SIZE
        mask_long = observed[:, :long_usable].reshape(
            b, NUM_LONG_PATCHES, LONG_PATCH_SIZE
        ).any(dim=-1)
        mask_short = observed[:, :short_usable].reshape(
            b, NUM_SHORT_PATCHES, SHORT_PATCH_SIZE
        ).any(dim=-1)
        return torch.cat([mask_long, mask_short], dim=1)

    def _encode_temporal(self, time_features: torch.Tensor) -> torch.Tensor:
        long_tf = time_features.index_select(1, self._long_idx)
        short_tf = time_features.index_select(1, self._short_idx)
        # Token order must match DualPatchEmbedding: long first, then short.
        return torch.cat(
            [self.temporal_embedding(long_tf), self.temporal_embedding(short_tf)],
            dim=1,
        )

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        history: torch.Tensor,
        time_features: torch.Tensor,
        observed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            history: (B, T) raw values, padded positions hold `pad_value`.
            time_features: (B, T, 7) calendar features, zero at padded positions.
            observed: (B, T) bool, True where the value is real.
        """
        point_mask = observed.unsqueeze(-1)
        means, stdev, scaled = masked_variance_normalization(
            history.unsqueeze(-1), point_mask, self.scaler_epsilon, self.pad_value
        )

        base = self.dual_patch_embedding(scaled)
        temporal = self._encode_temporal(time_features)
        seasonal = self.seasonal_decomp(base, temporal)
        trend, trend_penalty = self.trend_decomp(base)

        x = self.post_fuse_norm(self.fusion(base, seasonal, trend))

        key_padding_mask = ~self._patch_mask(observed)
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        pooled = self.pool(x, key_padding_mask=key_padding_mask)
        preds = self.projector2(F.gelu(self.projector1(pooled)))
        result = preds.view(-1, self.pred_len, self.num_quantiles)

        # Denormalise, then sort so the reported quantiles do not cross.
        result_scaled = torch.sort(result * stdev + means, dim=-1).values

        return {
            "result": result,  # (B, H, Q) normalised and unsorted; the loss uses this
            "result_scaled": result_scaled,  # (B, H, Q) original scale, sorted
            "scale": stdev.squeeze(-1),  # (B, 1)
            "means": means.squeeze(-1),  # (B, 1)
            "trend_penalty": trend_penalty,
        }

    # -- objective ----------------------------------------------------------
    def quantile_loss(
        self, output: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Pinball loss in normalised space, plus the trend smoothness penalty.

        The target is normalised with the same per-series statistics the model
        derived from the history.
        """
        result = output["result"]
        means = output["means"]
        scale = output["scale"]

        y_norm = ((target.to(result.dtype) - means) / scale).unsqueeze(-1)
        err = y_norm - result  # (B, H, Q)
        q = self._quantiles.to(result.dtype).view(1, 1, -1)
        pinball = torch.maximum(q * err, (q - 1.0) * err)

        w = self._horizon_weights.to(result.dtype).view(1, -1, 1)
        b = result.shape[0]
        # With all-ones weights this is just the mean over B x H x Q.
        loss = (pinball * w).sum() / (b * self.num_quantiles * w.sum())

        return loss + self.trend_smooth_lambda * output["trend_penalty"]

    # -- inference ----------------------------------------------------------
    def predict(
        self,
        history: torch.Tensor,
        time_features: torch.Tensor,
        observed: torch.Tensor,
        prediction_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Quantile forecasts (B, h, Q) in the original scale, sorted ascending."""
        h = self.pred_len if prediction_length is None else prediction_length
        if not 1 <= h <= self.pred_len:
            raise ValueError(
                f"prediction_length must be in [1, {self.pred_len}], got {h}"
            )
        out = self.forward(history, time_features, observed)
        return out["result_scaled"][:, :h, :]
