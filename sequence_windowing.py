from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.wisdm import ACTIVITY_ORDER, SENSOR_COLUMNS, markdown_table


LABEL_TO_ID = {label: idx for idx, label in enumerate(ACTIVITY_ORDER)}


def count_windows(df: pd.DataFrame, window_size: int, step: int) -> int:
    total = 0
    for _, group in df.groupby(["subject_id", "activity_label"], observed=True, sort=True):
        n = len(group)
        if n >= window_size:
            total += 1 + (n - window_size) // step
    return total


def build_raw_sequence_windows(
    clean_data: Path,
    splits_path: Path,
    out_dir: Path,
    window_seconds: float,
    overlap: float,
    sample_rate_hz: float,
    dataset_name: str = "phone_accel",
    include_labels: list[str] | None = None,
    max_windows: int | None = None,
) -> tuple[Path, Path, Path]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")

    out_dir.mkdir(parents=True, exist_ok=True)
    window_size = int(round(window_seconds * sample_rate_hz))
    step = max(1, int(round(window_size * (1 - overlap))))
    if window_size < 2:
        raise ValueError("window size must contain at least two samples.")

    cols = ["subject_id", "activity_label", "activity_name", "timestamp_ns", *SENSOR_COLUMNS]
    df = pd.read_csv(clean_data, usecols=cols)
    splits = pd.read_csv(splits_path)
    df = df.merge(splits, on="subject_id", how="left")
    label_codes = include_labels if include_labels else ACTIVITY_ORDER
    unknown_labels = sorted(set(label_codes) - set(ACTIVITY_ORDER))
    if unknown_labels:
        raise ValueError(f"Unknown activity labels: {unknown_labels}")
    df = df[df["activity_label"].isin(label_codes)].copy()
    if df.empty:
        raise ValueError("No rows remain after label filtering.")
    label_to_id = {label: idx for idx, label in enumerate(label_codes)}
    df = df.sort_values(["subject_id", "activity_label", "timestamp_ns"])

    n_windows = count_windows(df, window_size, step)
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    x = np.empty((n_windows, window_size, len(SENSOR_COLUMNS)), dtype=np.float32)
    y = np.empty((n_windows,), dtype=np.int64)
    metadata_rows: list[dict[str, object]] = []

    window_id = 0
    group_cols = ["subject_id", "activity_label", "activity_name", "split"]
    for keys, group in df.groupby(group_cols, observed=True, sort=True):
        subject_id, activity_label, activity_name, split = keys
        values = group[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
        timestamps = group["timestamp_ns"].to_numpy(dtype=np.int64)
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1, step):
            if window_id >= n_windows:
                break
            end = start + window_size
            x[window_id] = values[start:end]
            y[window_id] = label_to_id[activity_label]
            metadata_rows.append(
                {
                    "window_id": window_id,
                    "subject_id": int(subject_id),
                    "activity_label": activity_label,
                    "activity_name": activity_name,
                    "split": split,
                    "start_timestamp_ns": int(timestamps[start]),
                    "end_timestamp_ns": int(timestamps[end - 1]),
                    "n_samples": window_size,
                    "window_seconds": float(window_seconds),
                    "overlap": float(overlap),
                }
            )
            window_id += 1
        if window_id >= n_windows:
            break

    metadata = pd.DataFrame(metadata_rows)
    base_slug = f"{dataset_name}_rawseq_{str(window_seconds).replace('.', 'p')}s_{int(overlap * 100)}overlap"
    slug = base_slug if include_labels is None else f"{base_slug}_{'_'.join(include_labels)}"
    npz_path = out_dir / f"{slug}.npz"
    metadata_path = out_dir / f"{slug}_metadata.csv"
    summary_path = out_dir / f"{slug}_summary.md"

    np.savez_compressed(
        npz_path,
        X=x,
        y=y,
        label_codes=np.array(label_codes),
        sensor_columns=np.array(SENSOR_COLUMNS),
        window_seconds=np.array([window_seconds], dtype=np.float32),
        overlap=np.array([overlap], dtype=np.float32),
        sample_rate_hz=np.array([sample_rate_hz], dtype=np.float32),
    )
    metadata.to_csv(metadata_path, index=False)

    split_summary = metadata.groupby("split", observed=True).size().rename("windows").reset_index()
    class_summary = (
        metadata.groupby(["split", "activity_label", "activity_name"], observed=True)
        .size()
        .rename("windows")
        .reset_index()
    )
    class_summary.to_csv(out_dir / f"{slug}_class_distribution.csv", index=False)

    lines = [
        "# Raw Sequence Window Dataset",
        "",
        f"Saved tensor dataset: `{npz_path}`",
        f"Saved window metadata: `{metadata_path}`",
        "",
        "## Tensor schema",
        "",
        f"- `X`: `{x.shape}`, float32, ordered as `(window, time, axis)`.",
        f"- `y`: `{y.shape}`, integer activity IDs aligned to `label_codes`.",
        f"- Labels: `{label_codes}`.",
        f"- Sensor axes: `{SENSOR_COLUMNS}`.",
        f"- Window size: {window_seconds} seconds, {window_size} samples at {sample_rate_hz} Hz.",
        f"- Overlap: {overlap}.",
        "",
        "## Split summary",
        "",
        markdown_table(split_summary),
        "",
        "## Per-split class distribution",
        "",
        markdown_table(class_summary),
        "",
        "Generated by `sequence_windowing.py`.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return npz_path, metadata_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create raw WISDM sequence tensors for CNN/LSTM models.")
    parser.add_argument("--clean-data", type=Path, default=Path("data/processed/phone_accel_clean.csv.gz"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/subject_splits.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sequences"))
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--dataset-name", default="phone_accel", help="Prefix for saved tensor artifacts.")
    parser.add_argument("--include-labels", default=None, help="Comma-separated activity labels to keep, for example A,B,E.")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    include_labels = [label.strip().upper() for label in args.include_labels.split(",")] if args.include_labels else None

    npz_path, metadata_path, summary_path = build_raw_sequence_windows(
        clean_data=args.clean_data,
        splits_path=args.splits,
        out_dir=args.out_dir,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        sample_rate_hz=args.sample_rate_hz,
        dataset_name=args.dataset_name,
        include_labels=include_labels,
        max_windows=args.max_windows,
    )
    print(f"Wrote {npz_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
