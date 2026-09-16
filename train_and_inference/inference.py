

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.field_names import FieldName

from constants import (
    FLOAT_DTYPE,
    MAX_HISTORY,
    MEDIAN_INDEX,
    MIN_CONTEXT,
    NP_FLOAT_DTYPE,
    QUANTILE_LEVELS,
    TARGET_LEN,
)
from dataset import (
    FEAT_TIME,
    LastWindowSampler,
    build_base_transformation,
    build_instance_splitter,
    compute_time_features,
)
from estimator import build_predictor
from eval_metrics import SMAPE, naive_MASE, seasonal_MASE
from lightning_module import GRAINLightningModule

# ==========================================================================
# CONFIG
# ==========================================================================
WEIGHTS = "model.ckpt"
SERIES_DIR = "../m3_yearly_validation"
PREDICTION_LENGTH = 6

# 1 == forecast the last PREDICTION_LENGTH points only. N > 1 steps the origin
# back in non-overlapping jumps of PREDICTION_LENGTH, so window N-1 is that final
# window and window 0 is the earliest.
NUM_WINDOWS = 1

SAVE_TXT = False         # per-series .txt reports
SAVE_PLOTS = False       # per-series .png plots

OUTPUT_DIR = "./grain/m3_yearly"
MODEL_NAME = "GRAIN"
SEASONALITY = 1          # 1 for yearly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
# One predict call per series, timed individually, matching how the baselines
# were timed. Off, it predicts in batches and reports the amortised share --
# faster, but not comparable with the baselines.
PER_SERIES_TIMING = True
LIMIT = None             # only the first N series, for a smoke test
# The smallest history that can be scored at all, not a modelling choice: below
# 2, WindowDataset slices from the wrong end (silent history/target
# misalignment, not an error) and naive_MASE divides by an empty diff.
ABSOLUTE_MIN_HISTORY = 2
# Raise to MIN_CONTEXT (12) to keep scoring on-distribution: below that, the
# context is shorter than anything the prior produced.
MIN_HISTORY = ABSOLUTE_MIN_HISTORY
PLOT_CONTEXT = 30        # history points drawn left of the forecast; None = all
VERBOSE = True

C_TRUTH = "#1f77b4"
C_PRED = "#ff7f0e"
C_BAND = "#ffbb78"
C_METRIC = "#1f77b4"


# ==========================================================================
# data
# ==========================================================================
def load_series(
    dir_path: str,
    date_column: str = "date",
    value_column: str = "OT",
    limit: Optional[int] = None,
) -> List[Tuple[str, pd.DatetimeIndex, np.ndarray]]:
    """Read every `T{n}.csv` once, ordered numerically by n."""
    import glob
    import re

    paths = sorted(
        glob.glob(os.path.join(dir_path, "T*.csv")),
        key=lambda p: int(re.search(r"T(\d+)\.csv$", p).group(1)),
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no T*.csv files under {dir_path}")

    out = []
    for path in paths:
        item_id = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, usecols=[date_column, value_column])
        dates = pd.to_datetime(df[date_column])
        order = np.argsort(dates.values)
        dates = pd.DatetimeIndex(dates.values[order])
        values = df[value_column].values[order].astype(NP_FLOAT_DTYPE)
        out.append((item_id, dates, values))
    return out


class WindowDataset:
    """One rolling-origin window, as a GluonTS-style iterable.

    The emitted target is `values[:cut + h]`, so `LastWindowSampler` holds out
    exactly `values[cut:cut + h]`. Dates are the published ones rather than
    derived from the gluonts `start` Period, which for a yearly frequency always
    reports Dec-31 and would re-anchor M3's Jan-1 files.
    """

    def __init__(
        self,
        records: Sequence[Tuple[str, pd.DatetimeIndex, np.ndarray]],
        prediction_length: int,
        drop_last: int,
        min_history: int,
        freq: str = "Y",
    ):
        h = prediction_length
        self.entries: List[Dict] = []
        self.meta: Dict[str, Dict] = {}
        self.skipped: List[Tuple[str, int]] = []

        for item_id, dates, values in records:
            cut = len(values) - drop_last - h
            if cut < min_history:
                self.skipped.append((item_id, cut))
                continue
            window_dates = dates[: cut + h]
            window_values = values[: cut + h]
            history = window_values[:cut]

            self.meta[item_id] = {
                "history": history,
                "history_dates": dates[:cut],
                "target": window_values[cut:],
                "target_dates": dates[cut : cut + h],
            }
            self.entries.append(
                {
                    FieldName.ITEM_ID: item_id,
                    FieldName.START: pd.Period(window_dates[0], freq=freq),
                    FieldName.TARGET: window_values,
                    FEAT_TIME: compute_time_features(window_dates),
                }
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


# ==========================================================================
# model
# ==========================================================================
def load_predictor(
    checkpoint: str,
    prediction_length: int = TARGET_LEN,
    context_length: int = MAX_HISTORY,
    min_past: int = MIN_CONTEXT,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
):
    """Restore a trained module and wrap it as a GluonTS predictor.

    The default dtype is set before the module is built so buffers the checkpoint
    does not cover still land in float64.
    """
    torch.set_default_dtype(FLOAT_DTYPE)
    module = GRAINLightningModule.load_from_checkpoint(checkpoint, map_location=device)
    module.eval()

    transform = build_base_transformation() + build_instance_splitter(
        prediction_length=prediction_length,
        context_length=context_length,
        sampler=LastWindowSampler(min_past=min_past, min_future=prediction_length),
        min_past=min_past,
    )
    return build_predictor(
        input_transform=transform,
        module=module,
        prediction_length=prediction_length,
        batch_size=batch_size,
        device=device,
    )


# ==========================================================================
# reports
# ==========================================================================
def write_report(
    path: str,
    model_name: str,
    season_length: int,
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nmase: float,
    smase: float,
    smape: float,
    predict_t: float,
) -> None:
    lines = [
        "Scaled Metrics:",
        f"  Naive MASE:     {nmase:.4f}",
        f"  Seasonal MASE:  {smase:.4f} (seasonality={season_length})",
        f"  SMAPE:          {smape:.4f}%",
        "",
        "Timing (seconds):",
        "  Train:    0.00s   (zero-shot: no per-series fitting)",
        f"  Predict:  {predict_t:.4f}s",
        f"  Total:    {predict_t:.4f}s",
        "",
        "=" * 70,
        f"DETAILED PREDICTIONS  |  Model: {model_name}",
        "=" * 70,
        "Date                     Actual  Predicted      Error    Error %",
        "-" * 70,
    ]
    for dt, yt, yp in zip(dates, y_true, y_pred):
        err = yp - yt
        err_pct = abs(err) / (abs(yt) + 1e-12) * 100
        lines.append(
            f"{pd.to_datetime(dt).date():<24} "
            f"{yt:10.2f}  {yp:10.2f}  {err:10.2f}    {err_pct:6.1f}%"
        )
    lines.append("=" * 70)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def save_plot(
    path: str,
    title: str,
    hist_dates: pd.DatetimeIndex,
    hist_values: np.ndarray,
    test_dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    mase: float,
    smape: float,
    context: Optional[int] = PLOT_CONTEXT,
) -> None:
    """Ground truth in blue, median in orange, 10-90% band shaded."""
    import sys

    import matplotlib

    # Headless-safe, but leave a backend the caller already chose alone;
    # switching to Agg mid-session would kill inline rendering in a notebook.
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if context is not None and len(hist_dates) > context:
        hist_dates, hist_values = hist_dates[-context:], hist_values[-context:]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    if lower is not None and upper is not None:
        ax.fill_between(test_dates, lower, upper, color=C_BAND, alpha=0.55,
                        linewidth=0, label="Uncertainty (10th-90th quantiles)")
    ax.plot(hist_dates, hist_values, color=C_TRUTH, linewidth=1.6,
            label="Ground Truth")
    # Same colour, no label: one legend entry covers both truth segments.
    ax.plot(test_dates, y_true, color=C_TRUTH, linewidth=1.6)
    ax.plot(test_dates, y_pred, color=C_PRED, linewidth=1.6,
            label="Prediction (50th quantile)")
    cutoff = pd.to_datetime(hist_dates[-1]) + (
        pd.to_datetime(test_dates[0]) - pd.to_datetime(hist_dates[-1])
    ) / 2
    ax.axvline(cutoff, color="0.55", linestyle="--", linewidth=1.0)

    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_title(f"MASE: {mase:.2f}, sMAPE: {smape:.2f}%", loc="right",
                 color=C_METRIC, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def average_block(df: pd.DataFrame, season_length: int, wall_t: float,
                  num_windows: int, per_series_timing: bool = True,
                  batch_size: int = BATCH_SIZE) -> str:
    """Same layout as the baselines' average block, including the timings.

    `Train` is 0 because the model does no per-series fitting, which is the point
    of the comparison: AutoETS/AutoARIMA/AutoTheta pay their fit per series at
    inference time.
    """
    if per_series_timing:
        how = "  (per series, one predict call each -- matches the baselines)"
    else:
        how = (f"  (amortised over batches of {batch_size}; NOT comparable with "
               "the baselines' per-series timings)")
    return "\n".join(
        [
            "=" * 70,
            "AVERAGE SCALED METRICS",
            "=" * 70,
            f"  Naive MASE:     {df['naive_mase'].mean():.4f}",
            f"  Seasonal MASE:  {df['seasonal_mase'].mean():.4f} "
            f"(seasonality={season_length})",
            f"  SMAPE:          {df['smape'].mean():.4f}%",
            "",
            f"  windows:        {num_windows}",
            f"  forecasts:      {len(df)}  "
            f"({df['series_id'].nunique()} series x {num_windows} window(s))",
            "",
            "Average Timing (seconds):",
            "  Train:    0.00s   (zero-shot: no per-series fitting)",
            f"  Predict:  {df['predict_time'].mean():.4f}s",
            f"  Total:    {df['total_time'].mean():.4f}s",
            how,
            f"  (wall clock for all {len(df)} forecasts: {wall_t:.2f}s)",
            "=" * 70,
        ]
    )


# ==========================================================================
# driver
# ==========================================================================
def run(
    weights: str = WEIGHTS,
    series_dir: str = SERIES_DIR,
    prediction_length: int = PREDICTION_LENGTH,
    num_windows: int = NUM_WINDOWS,
    save_txt: bool = SAVE_TXT,
    save_plots: bool = SAVE_PLOTS,
    output_dir: str = OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    seasonality: int = SEASONALITY,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    limit: Optional[int] = LIMIT,
    min_history: int = MIN_HISTORY,
    per_series_timing: bool = PER_SERIES_TIMING,
    quantile_levels: Sequence[float] = QUANTILE_LEVELS,
    median_index: int = MEDIAN_INDEX,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    levels = np.asarray(quantile_levels, dtype=np.float64)
    assert abs(levels[median_index] - 0.5) < 1e-12, "median_index must select 0.5"
    band = (1, 9) if len(levels) == 11 else None  # 10% / 90% columns

    if min_history < ABSOLUTE_MIN_HISTORY:
        raise ValueError(
            f"min_history={min_history} is below ABSOLUTE_MIN_HISTORY="
            f"{ABSOLUTE_MIN_HISTORY}; such windows misalign history against "
            "target rather than failing, so they cannot be accepted"
        )
    if min_history <= seasonality:
        print(f"WARNING: min_history={min_history} <= seasonality={seasonality}; "
              "seasonal_mase is NaN for windows with that little history")

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    records = load_series(series_dir, limit=limit)
    predictor = load_predictor(
        weights,
        prediction_length=prediction_length,
        batch_size=batch_size,
        device=device,
    )
    print(f"{len(records)} series from {series_dir}   h={prediction_length}   "
          f"windows={num_windows}   device={device}   dtype={FLOAT_DTYPE}")

    rows: List[Dict] = []
    pred_rows: List[Dict] = []
    skipped: List[Tuple[str, int, int]] = []
    wall_t = 0.0


    for w in range(num_windows):
        drop_last = (num_windows - 1 - w) * prediction_length
        ds = WindowDataset(records, prediction_length, drop_last, min_history)
        skipped += [(item, w, cut) for item, cut in ds.skipped]
        if len(ds) == 0:
            print(f"window {w}: no series long enough, skipped")
            continue

        if per_series_timing:

            if w == 0:
                list(predictor.predict([next(iter(ds))]))
            forecasts, times = [], []
            for entry in ds:
                t0 = time.perf_counter()
                forecasts += list(predictor.predict([entry]))
                times.append(time.perf_counter() - t0)
            window_t = sum(times)
        else:
            t0 = time.perf_counter()
            forecasts = list(predictor.predict(ds))
            window_t = time.perf_counter() - t0
            times = [window_t / max(len(ds), 1)] * len(ds)
        wall_t += window_t
        predict_t = {str(fc.item_id): t for fc, t in zip(forecasts, times)}
        if len(forecasts) != len(ds):
            raise RuntimeError(
                f"window {w}: predictor returned {len(forecasts)} forecasts "
                f"for {len(ds)} series"
            )

        for fc in forecasts:
            item = str(fc.item_id)
            meta = ds.meta[item]
            y_true, history = meta["target"], meta["history"]
            test_dates = meta["target_dates"]

            pred = np.asarray(fc.forecast_array).T  # (Q, h) -> (h, Q)
            if pred.shape != (len(y_true), len(levels)):
                raise RuntimeError(
                    f"{item} w{w}: forecast {pred.shape}, expected "
                    f"{(len(y_true), len(levels))}"
                )
            y_pred = pred[:, median_index]


            with np.errstate(divide="ignore", invalid="ignore"):
                nmase = float(naive_MASE(y_pred, y_true, history))
                smase = float(seasonal_MASE(y_pred, y_true, history, seasonality))
                smape = float(SMAPE(y_pred, y_true))

            stem = item if num_windows == 1 else f"{item}.w{w}"
            label = item if num_windows == 1 else f"{item}  (window {w})"
            if save_txt:
                write_report(
                    os.path.join(output_dir, f"{stem}.txt"), model_name,
                    seasonality, test_dates, y_true, y_pred,
                    nmase, smase, smape, predict_t[item],
                )
            if save_plots:
                save_plot(
                    os.path.join(plots_dir, f"{stem}.png"),
                    f"{model_name}  {label}",
                    meta["history_dates"], history, test_dates, y_true, y_pred,
                    pred[:, band[0]] if band else None,
                    pred[:, band[1]] if band else None,
                    mase=nmase, smape=smape,
                )

            rows.append({
                "series_id": item, "window": w, "context": len(history),
                "naive_mase": nmase, "seasonal_mase": smase, "smape": smape,
                "train_time": 0.0, "predict_time": predict_t[item],
                "total_time": predict_t[item],
            })
            for step, (dt, yt) in enumerate(zip(test_dates, y_true)):
                rec = {"series_id": item, "window": w, "step": step + 1,
                       "date": pd.Timestamp(dt).date(), "actual": yt}
                rec.update({f"q{q:g}": pred[step, j] for j, q in enumerate(levels)})
                pred_rows.append(rec)

            if verbose:
                print(f"{stem:<12} SMAPE={smape:7.3f}%  MASE={nmase:6.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no forecasts were scored")

    if skipped:
        print(f"\nSKIPPED {len(skipped)} (series, window) pairs with history "
              f"< {min_history} (unscoreable, not a filter):")
        for item, w, cut in skipped[:10]:
            print(f"  {item} window {w}: {cut} points of context")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more (see skipped.csv)")
        pd.DataFrame(skipped, columns=["series_id", "window", "context"]).to_csv(
            os.path.join(output_dir, "skipped.csv"), index=False
        )


    bad = df[["naive_mase", "seasonal_mase", "smape"]].apply(
        lambda c: int((~np.isfinite(c)).sum())
    )
    if bad.sum():
        print(f"WARNING: non-finite metrics -- {bad.to_dict()} "
              f"(these propagate into the mean; not dropped)")

    content = average_block(df, seasonality, wall_t, num_windows,
                            per_series_timing, batch_size)
    if save_txt:
        with open(os.path.join(output_dir, "average_metrics.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(content)
    df.to_csv(os.path.join(output_dir, "per_series_metrics.csv"), index=False)
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(output_dir, "predictions.csv"), index=False
    )

    print("\n" + content + "\n")
    written = ["per_series_metrics.csv", "predictions.csv"]
    if save_txt:
        written.insert(0, "average_metrics.txt")
        written.append("<series>.txt")
    if save_plots:
        written.append("plots/")
    print(f"wrote {output_dir}/  ({', '.join(written)})")
    return df


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Yearly GRAIN scoring, using eval_metrics.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-w", "--weights", default=WEIGHTS,
                   help="checkpoint, e.g. runs/<run>/checkpoints/model.ckpt")
    p.add_argument("-d", "--series-dir", default=SERIES_DIR,
                   help="folder of T{n}.csv files (columns: date,OT)")
    p.add_argument("-p", "--prediction-length", type=int, default=PREDICTION_LENGTH)
    p.add_argument("-n", "--num-windows", type=int, default=NUM_WINDOWS,
                   help="rolling origins; 1 == no sliding window")
    p.add_argument("--save-txt", action=argparse.BooleanOptionalAction,
                   default=SAVE_TXT, help="per-series .txt reports")
    p.add_argument("--save-plots", action=argparse.BooleanOptionalAction,
                   default=SAVE_PLOTS, help="per-series .png plots")
    p.add_argument("-o", "--output-dir", default=OUTPUT_DIR)
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--seasonality", type=int, default=SEASONALITY,
                   help="1 for yearly")
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--limit", type=int, default=LIMIT,
                   help="only the first N series (smoke test)")
    p.add_argument("--min-history", type=int, default=MIN_HISTORY,
                   help="skip a window with less context than this")
    p.add_argument("--per-series-timing", action=argparse.BooleanOptionalAction,
                   default=PER_SERIES_TIMING,
                   help="one predict call per series, matching how the baselines "
                        "were timed; --no-per-series-timing batches instead and "
                        "reports the amortised share (not comparable)")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    run(
        weights=args.weights,
        series_dir=args.series_dir,
        prediction_length=args.prediction_length,
        num_windows=args.num_windows,
        save_txt=args.save_txt,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seasonality=args.seasonality,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        min_history=args.min_history,
        per_series_timing=args.per_series_timing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
