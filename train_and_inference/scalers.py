"""Per-series normalisation."""

from typing import Tuple

import torch

from constants import SCALER_EPSILON


def masked_variance_normalization(
    values: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = SCALER_EPSILON,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardise each series using only its observed positions.

    values/mask are (B, T, 1); returns means (B, 1, 1), stdev (B, 1, 1) and the
    scaled history, which holds `pad_value` at padded positions.
    """
    zero = torch.zeros((), dtype=values.dtype, device=values.device)

    count = mask.sum(dim=1, keepdim=True).to(values.dtype).clamp(min=1.0)

    masked = torch.where(mask, values, zero)
    means = masked.sum(dim=1, keepdim=True) / count

    diff = torch.where(mask, values - means, zero)
    # Population variance over observed points only (divide by count, not count-1).
    variance = diff.pow(2).sum(dim=1, keepdim=True) / count
    stdev = torch.sqrt(variance + epsilon)

    normalized = (values - means) / stdev
    scaled = torch.where(mask, normalized, torch.full_like(normalized, pad_value))
    return means, stdev, scaled
