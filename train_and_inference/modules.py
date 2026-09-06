"""Layers of the yearly LGT-PFN.

Two quirks are kept deliberately rather than fixed, and are marked `QUIRK (kept)`
at their class: the year is discarded by a degenerate LayerNorm, and the fusion
gate is computed from an unmasked token mean.

Initialisers follow the Keras defaults the original implementation used:
Dense/Conv -> glorot_uniform kernel + zero bias, Embedding -> U(-0.05, 0.05).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import (
    DAY_VOCAB,
    DOW_VOCAB,
    IDX_DAY,
    IDX_DOW,
    IDX_MONTH,
    LN_EPS,
    MONTH_VOCAB,
    TREND_DILATION,
    TREND_KERNEL,
    TREND_SMOOTH_KERNEL,
    YEAR_NORM_EPS,
)


# --------------------------------------------------------------------------
# initialisation helpers
# --------------------------------------------------------------------------
def _keras_fans(shape) -> Tuple[int, int]:
    """Keras' fan computation, which differs from PyTorch's for rank > 2."""
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive_field = math.prod(shape[:-2])
    return shape[-2] * receptive_field, shape[-1] * receptive_field


def keras_glorot_uniform_(tensor: torch.Tensor, shape=None) -> None:
    """glorot_uniform over the Keras weight shape, not the torch one."""
    fan_in, fan_out = _keras_fans(tuple(shape) if shape else tuple(tensor.shape))
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


def init_linear(layer: nn.Linear) -> None:
    # torch stores (out, in), Keras (in, out); the fans are the same either way.
    keras_glorot_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_mha(attn: nn.MultiheadAttention, embed_dim: int, num_heads: int) -> None:
    """glorot_uniform over the attention weight shapes Keras uses.

    q/k/v are shaped (embed_dim, num_heads, key_dim) and the output projection
    (num_heads, key_dim, embed_dim). Initialising against the torch shape instead
    makes q/k/v 3.5-4.2x too wide.
    """
    head_dim = embed_dim // num_heads
    keras_glorot_uniform_(attn.in_proj_weight, shape=(embed_dim, num_heads, head_dim))
    nn.init.zeros_(attn.in_proj_bias)
    keras_glorot_uniform_(attn.out_proj.weight, shape=(num_heads, head_dim, embed_dim))
    nn.init.zeros_(attn.out_proj.bias)


def init_conv(layer: nn.Conv1d) -> None:
    # Reconstruct the Keras shape (k, in, out) rather than reading the torch one
    # (out, in/groups, k), which would give fan_in = in*in instead of in*k.
    out_ch, in_ch, k = layer.weight.shape
    keras_glorot_uniform_(layer.weight, shape=(k, in_ch, out_ch))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_embedding(layer: nn.Embedding) -> None:
    nn.init.uniform_(layer.weight, -0.05, 0.05)


# --------------------------------------------------------------------------
# patch embedding
# --------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Non-overlapping patches of raw values -> linear projection + learned position."""

    def __init__(self, patch_size: int, embed_dim: int, max_patches: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)
        self.position_embedding = nn.Embedding(max_patches, embed_dim)
        init_linear(self.projection)
        init_embedding(self.position_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1) -> (B, T // patch_size, embed_dim)
        b, t, _ = x.shape
        n = t // self.patch_size
        patched = x[:, : n * self.patch_size, 0].reshape(b, n, self.patch_size)
        embedded = self.projection(patched)
        positions = torch.arange(n, device=x.device)
        return embedded + self.position_embedding(positions)


class DualPatchEmbedding(nn.Module):
    """Two patch scales concatenated along the token axis: LONG first, then SHORT."""

    def __init__(
        self,
        long_patch_size: int,
        short_patch_size: int,
        embed_dim: int,
        seq_len: int,
    ):
        super().__init__()
        self.long_embed = PatchEmbedding(
            long_patch_size, embed_dim, seq_len // long_patch_size + 1
        )
        self.short_embed = PatchEmbedding(
            short_patch_size, embed_dim, seq_len // short_patch_size + 1
        )
        # Scale-type markers, zero-initialised.
        self.long_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.short_type = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        long_tokens = self.long_embed(x) + self.long_type
        short_tokens = self.short_embed(x) + self.short_type
        return torch.cat([long_tokens, short_tokens], dim=1)


# --------------------------------------------------------------------------
# calendar embedding
# --------------------------------------------------------------------------
class TemporalFeatureEmbedding(nn.Module):
    """Calendar features -> embedding. Reads columns 0-3 of the width-7 input.

    QUIRK (kept): `year_norm` is a LayerNorm over a size-1 axis, so (x - mean) is
    exactly zero and the output is `beta` whatever the year. The year is
    discarded and no gradient flows back to it.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        quarter = embed_dim // 4
        self.month_embedding = nn.Embedding(MONTH_VOCAB, quarter)
        self.day_embedding = nn.Embedding(DAY_VOCAB, quarter)
        self.dow_embedding = nn.Embedding(DOW_VOCAB, quarter)
        self.year_norm = nn.LayerNorm(1, eps=YEAR_NORM_EPS)
        self.year_projection = nn.Linear(1, quarter)
        self.projection = nn.Linear(embed_dim, embed_dim)

        for emb in (self.month_embedding, self.day_embedding, self.dow_embedding):
            init_embedding(emb)
        init_linear(self.year_projection)
        init_linear(self.projection)

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        # time_features: (B, T, 7)
        year = time_features[..., 0:1]
        month = time_features[..., IDX_MONTH].long()
        day = time_features[..., IDX_DAY].long()
        dow = time_features[..., IDX_DOW].long()

        # Degenerate by construction; evaluates to the learned bias. See docstring.
        year_emb = self.year_projection(self.year_norm(year))

        combined = torch.cat(
            [
                self.month_embedding(month),
                self.day_embedding(day),
                self.dow_embedding(dow),
                year_emb,
            ],
            dim=-1,
        )
        return self.projection(combined)


# --------------------------------------------------------------------------
# decomposition branches
# --------------------------------------------------------------------------
class SeasonalDecomposition(nn.Module):
    """Mixes the calendar embedding into the patch embedding, with a residual."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.seasonal_dense = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=LN_EPS)
        init_linear(self.seasonal_dense)
        init_linear(self.combine)

    def forward(self, x: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        seasonal = 0.5 * self.seasonal_dense(temporal)
        combined = self.combine(torch.cat([x, seasonal], dim=-1))
        return self.norm(combined + x)


class TrendDecomposition(nn.Module):
    """Causal separable conv -> causal smoothing conv -> projection -> norm.

    Returns the trend and its second-difference smoothness penalty. The caller
    adds the penalty to the objective.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.kernel = TREND_KERNEL
        self.dilation = TREND_DILATION
        # Separable conv: depthwise then pointwise, with one bias applied after
        # the pointwise stage.
        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=self.kernel,
            groups=embed_dim,
            dilation=self.dilation,
            bias=False,
        )
        self.pointwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.smooth = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=TREND_SMOOTH_KERNEL, bias=True
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=LN_EPS)

        # The depthwise stage starts as a plain moving average.
        with torch.no_grad():
            self.depthwise.weight.fill_(1.0 / self.kernel)
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity="relu")
        nn.init.zeros_(self.pointwise.bias)
        init_conv(self.smooth)
        init_linear(self.proj)

    @staticmethod
    def _smoothness_penalty(x: torch.Tensor) -> torch.Tensor:
        d1 = x[:, 1:, :] - x[:, :-1, :]
        d2 = d1[:, 1:, :] - d1[:, :-1, :]
        return d2.pow(2).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, T, C) -> conv layout (B, C, T)
        h = x.transpose(1, 2)
        # Causal padding: left-pad dilation * (k - 1), nothing on the right.
        h = F.pad(h, (self.dilation * (self.kernel - 1), 0))
        h = self.pointwise(self.depthwise(h))
        h = F.pad(h, (TREND_SMOOTH_KERNEL - 1, 0))
        h = self.smooth(h)
        trend = self.norm(self.proj(h.transpose(1, 2)))
        return trend, self._smoothness_penalty(trend)


class GatedFusion(nn.Module):
    """Softmax gate over the three branches, per series and per channel.

    QUIRK (kept): the gate comes from an unmasked mean over all tokens, so with a
    short context padding dominates it for the whole series.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.fuse_pre = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.gate_gen = nn.Linear(2 * embed_dim, 3 * embed_dim)
        self.fuse_out = nn.Linear(embed_dim, embed_dim)
        for layer in (self.fuse_pre, self.gate_gen, self.fuse_out):
            init_linear(layer)

    def forward(
        self, base: torch.Tensor, seasonal: torch.Tensor, trend: torch.Tensor
    ) -> torch.Tensor:
        b = base.shape[0]
        h = F.gelu(self.fuse_pre(torch.cat([base, seasonal, trend], dim=-1)))
        gates = self.gate_gen(h.mean(dim=1))  # unmasked on purpose, see docstring
        gates = torch.softmax(gates.view(b, 3, self.embed_dim), dim=1)
        fused = (
            gates[:, 0, :].unsqueeze(1) * base
            + gates[:, 1, :].unsqueeze(1) * seasonal
            + gates[:, 2, :].unsqueeze(1) * trend
        )
        return self.fuse_out(fused)


# --------------------------------------------------------------------------
# encoder / pooling
# --------------------------------------------------------------------------
class LGTPFNEncoderLayer(nn.Module):
    """Pre-norm transformer block. Not causal; only key padding is masked."""

    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ff1 = nn.Linear(embed_dim, d_ff)
        self.ff2 = nn.Linear(d_ff, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(embed_dim, eps=LN_EPS)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        init_linear(self.ff1)
        init_linear(self.ff2)
        init_mha(self.attention, embed_dim, num_heads)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        a = self.norm1(x)
        attn, _ = self.attention(
            a, a, a, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout1(attn)
        b = self.norm2(x)
        x = x + self.dropout2(self.ff2(F.gelu(self.ff1(b))))
        return x


class AttentionPooling(nn.Module):
    """Pools the token sequence into `num_queries` learned slots, then flattens."""

    def __init__(self, embed_dim: int, num_queries: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        keras_glorot_uniform_(self.query)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim, eps=LN_EPS)
        init_mha(self.attn, embed_dim, num_heads)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(
            q, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.norm(pooled).reshape(b, -1)
