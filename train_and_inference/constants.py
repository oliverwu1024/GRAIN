"""Shapes, dtype and architecture constants. These values define the model."""

import numpy as np
import torch

# float64 end to end. The prior reaches ~1e38, and squaring that when computing
# the variance overflows float32 -> inf -> NaN.
# This is the only dtype knob; LGTPFNModel casts itself in __init__.
FLOAT_DTYPE = torch.float64

# The numpy scalar type, not a np.dtype instance: gluonts validates
# AddObservedValuesIndicator's `dtype` argument as a class.
_TORCH_TO_NUMPY = {torch.float64: np.float64, torch.float32: np.float32}
NP_FLOAT_DTYPE = _TORCH_TO_NUMPY[FLOAT_DTYPE]

# --- sequence geometry ------------------------------------------------------
MAX_HISTORY = 200
TARGET_LEN = 6
MIN_CONTEXT = 12
SERIES_LEN = 200

# Padded positions hold 0.0; the mask travels separately as `past_is_pad`.
PAD_VALUE = 0.0

# --- time features ----------------------------------------------------------
# [year, month, day, day_of_week + 1, day_of_year, 0, 0]
# The model reads only columns 0-3. The last two are unused placeholders.
TIME_FEAT_DIM = 7
IDX_YEAR, IDX_MONTH, IDX_DAY, IDX_DOW = 0, 1, 2, 3
MONTH_VOCAB = 13  # 12 months + index 0 for padding
DAY_VOCAB = 32
DOW_VOCAB = 8

# --- quantile head ----------------------------------------------------------
QUANTILE_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
NUM_QUANTILES = len(QUANTILE_LEVELS)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)

# --- network ----------------------------------------------------------------
EMBED_DIM = 128
NUM_ENCODER_LAYERS = 4
NUM_HEADS = 8
D_FF = 512
DROPOUT = 0.1

LONG_PATCH_SIZE = 10
SHORT_PATCH_SIZE = 2
NUM_LONG_PATCHES = MAX_HISTORY // LONG_PATCH_SIZE
NUM_SHORT_PATCHES = MAX_HISTORY // SHORT_PATCH_SIZE
TOTAL_PATCHES = NUM_LONG_PATCHES + NUM_SHORT_PATCHES  # 120

POOL_QUERIES = 4
POOL_HEADS = 4

# The two LayerNorm epsilons differ from each other and from torch's 1e-5
# default, so both are set explicitly.
LN_EPS = 1e-6
YEAR_NORM_EPS = 1e-3

TREND_KERNEL = 15
TREND_DILATION = 2
TREND_SMOOTH_KERNEL = 3
TREND_SMOOTH_LAMBDA = 1e-4

# --- scaler -----------------------------------------------------------------
SCALER_EPSILON = 1e-4

# --- optimisation -----------------------------------------------------------
LEARNING_RATE = 1e-4
ADAM_EPSILON = 1e-7  # torch's default is 1e-8
BATCH_SIZE = 1024
STEPS_PER_EPOCH = 160
SHUFFLE_BUFFER = 5000
SEED = 42
