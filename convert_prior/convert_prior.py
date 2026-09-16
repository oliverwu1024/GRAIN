"""One-time conversion of the synthetic prior from the generator's long-format
CSV to a dense float64 .npy matrix plus a small metadata sidecar.

`noise` is identically 1.0 and `ts` is fully determined by (start_year,
anchor_month, anchor_day) on a contiguous yearly grid, so both are dropped; the
asserts below verify that before doing so. The CSV -- not prior_data/*.tfrecords
-- is the source of truth: the tfrecords store `y` as float32.

`--anchor keep` (the default) preserves the CSV's Dec-31 anchor and reproduces
the historical prior exactly. `--anchor random` draws an independent month-day
per series so the calendar embeddings see rows other than 12/31; on a yearly grid
`y` does not depend on the anchor, so no regeneration is needed.

Usage:
    python convert_prior.py --csv "<path>/m3_yearly_200.csv" --out prior/
"""

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc

SERIES_LEN = 200
FREQ = "Y"

# Anchors are drawn from the 365 days of a non-leap year so that every (month,
# day) pair exists in all 200 years a series spans -- Feb-29 does not.
ANCHOR_REF_YEAR = 2001


def draw_anchors(n_series: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Independent uniform month-day anchors, one per series."""
    rng = np.random.default_rng(seed)
    doy = rng.integers(1, 366, size=n_series)
    dates = (np.datetime64(f"{ANCHOR_REF_YEAR}-01-01")
             + (doy - 1).astype("timedelta64[D]"))
    month = (dates.astype("datetime64[M]").astype(int) % 12) + 1
    day = (dates - dates.astype("datetime64[M]")).astype(int) + 1
    return month.astype(np.int8), day.astype(np.int8)


def convert(csv_path: str, out_dir: str, name: str,
            anchor: str = "keep", anchor_seed: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"reading {csv_path} ...")
    table = pc.read_csv(
        csv_path,
        read_options=pc.ReadOptions(block_size=1 << 26),
        convert_options=pc.ConvertOptions(
            include_columns=["id", "ts", "y", "noise"],
            column_types={
                "id": pa.dictionary(pa.int32(), pa.string()),
                "ts": pa.date32(),
                "y": pa.float64(),
                "noise": pa.float64(),
            },
        ),
    )
    n_rows = table.num_rows
    print(f"  {n_rows:,} rows")

    ids_col = table.column("id").combine_chunks()
    y = table.column("y").combine_chunks().to_numpy(zero_copy_only=False)
    noise = table.column("noise").combine_chunks().to_numpy(zero_copy_only=False)
    # date32 is days since epoch; keep it integral to avoid any datetime64[ns] range issues.
    ts_days = table.column("ts").combine_chunks().to_numpy(zero_copy_only=False)
    ts_days = ts_days.astype("datetime64[D]").astype(np.int64)
    del table

    # --- `noise` carries no information -------------------------------------
    assert np.all(noise == 1.0), (
        f"`noise` is not identically 1.0 (min={noise.min()}, max={noise.max()}); "
        "it can no longer be dropped"
    )
    del noise

    # --- series blocks must be contiguous and exactly SERIES_LEN long --------
    codes = ids_col.indices.to_numpy(zero_copy_only=False)
    boundaries = np.flatnonzero(np.diff(codes) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    lengths = np.diff(np.concatenate([starts, [n_rows]]))
    n_series = len(starts)

    assert np.all(lengths == SERIES_LEN), (
        f"not every series has {SERIES_LEN} rows "
        f"(min={lengths.min()}, max={lengths.max()})"
    )
    assert n_series * SERIES_LEN == n_rows, "series blocks do not tile the file"
    assert len(np.unique(codes[starts])) == n_series, (
        "a series id appears in more than one block; the CSV is not grouped by id"
    )

    # --- values -------------------------------------------------------------
    assert np.isfinite(y).all(), "prior contains non-finite values"
    y_mat = y.reshape(n_series, SERIES_LEN)
    del y

    # The dataset pipeline relies on this: with no zeros in the prior, a leading
    # run of exact 0.0 can never be confused with padding, so v15's
    # `unify_pad_leading` transform is unnecessary.
    assert (y_mat > 0).all(), (
        f"prior contains values <= 0 (min={y_mat.min()}); the dataset pipeline "
        "assumes strictly positive values"
    )

    # --- timestamps must be a contiguous yearly grid on a per-series anchor --
    ts_mat = ts_days.reshape(n_series, SERIES_LEN)
    del ts_days
    dates = ts_mat.astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = (dates.astype("datetime64[M]").astype(int) % 12) + 1
    days = (dates - dates.astype("datetime64[M]")).astype(int) + 1

    assert np.all(np.diff(years, axis=1) == 1), (
        "prior timestamps are not a contiguous yearly grid"
    )
    # The anchor must not drift within a series, otherwise (start_year, month,
    # day) cannot reconstruct the dates. A Feb-29 anchor would trip this, since
    # it does not survive the non-leap years in a 200-year span.
    assert np.all(months == months[:, :1]) and np.all(days == days[:, :1]), (
        "a series changes its month-day anchor part-way through; the "
        "(start_year, anchor) encoding is invalid"
    )
    start_year = years[:, 0].astype(np.int16)
    assert np.array_equal(
        years, start_year[:, None].astype(int) + np.arange(SERIES_LEN)[None, :]
    ), "start_year + arange does not reconstruct the original timestamps"

    if anchor == "random":
        anchor_month, anchor_day = draw_anchors(n_series, anchor_seed)
    else:
        anchor_month = months[:, 0].astype(np.int8)
        anchor_day = days[:, 0].astype(np.int8)
    del months, days

    ids = np.asarray(ids_col.dictionary.to_pylist(), dtype=object)[
        codes[starts]
    ].astype(str)

    # --- write --------------------------------------------------------------
    y_path = os.path.join(out_dir, f"{name}.y.npy")
    meta_path = os.path.join(out_dir, f"{name}.meta.npz")
    np.save(y_path, y_mat)

    meta = dict(
        ids=ids,
        start_year=start_year,
        anchor_month=anchor_month,
        anchor_day=anchor_day,
        series_len=np.int64(SERIES_LEN),
        freq=FREQ,
    )
    # The legacy scalar is written only when it is actually true of every series,
    # so a reader that knows nothing about `anchor_month` either gets the right
    # answer or a KeyError -- never a silently wrong anchor.
    uniform = (len(np.unique(anchor_month)) == 1
               and len(np.unique(anchor_day)) == 1)
    if uniform:
        meta["day_anchor"] = f"{int(anchor_month[0]):02d}-{int(anchor_day[0]):02d}"
    np.savez(meta_path, **meta)

    n_anchors = len(np.unique(
        anchor_month.astype(np.int16) * 100 + anchor_day.astype(np.int16)
    ))
    print("\nverification")
    print(f"  series            {n_series:,}")
    print(f"  length each       {SERIES_LEN}")
    print(f"  start year        {start_year.min()} .. {start_year.max()} "
          f"({len(np.unique(start_year))} distinct)")
    print(f"  end year          {(start_year.astype(int) + SERIES_LEN - 1).max()}")
    print(f"  anchor mode       {anchor}"
          + (f" (seed {anchor_seed})" if anchor == "random" else ""))
    print(f"  distinct anchors  {n_anchors} "
          f"({len(np.unique(anchor_month))} months, "
          f"{len(np.unique(anchor_day))} days)")
    print(f"  legacy day_anchor {meta.get('day_anchor', '<omitted: anchors vary>')}")
    print(f"  y range           {y_mat.min():.6g} .. {y_mat.max():.6g}")
    print(f"  y exact zeros     {int((y_mat == 0).sum())}")
    print("\nwrote")
    print(f"  {y_path}   {os.path.getsize(y_path) / 1e6:.1f} MB  "
          f"{y_mat.shape} {y_mat.dtype}")
    print(f"  {meta_path}   {os.path.getsize(meta_path) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="m3_yearly_200.csv", #the location of the csv generated by R
        help="source long-format prior CSV",
    )
    parser.add_argument("--out", default="prior", help="output directory")
    parser.add_argument("--name", default="m3_yearly_200", help="output basename")
    parser.add_argument(
        "--anchor",
        choices=("keep", "random"),
        default="keep",
        help="keep: use the CSV's month-day anchor (Dec-31, the historical "
             "prior). random: draw an independent anchor per series so the "
             "month/day embeddings see more than one row.",
    )
    parser.add_argument(
        "--anchor-seed", type=int, default=0,
        help="seed for --anchor random",
    )
    args = parser.parse_args()
    convert(args.csv, args.out, args.name, args.anchor, args.anchor_seed)


if __name__ == "__main__":
    main()
