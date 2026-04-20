from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import artifact_paths, ensure_dirs, fold_class_counts, log, make_fold_subjects, markdown_table, parse_float_list, timed, write_json


def load_stream(path: Path, prefix: str, max_subjects: int | None) -> pd.DataFrame:
    usecols = ["subject_id", "activity_label", "timestamp_ns", "x", "y", "z"]
    df = pd.read_csv(path, usecols=usecols)
    if max_subjects is not None:
        subjects = sorted(df["subject_id"].dropna().astype(int).unique())[:max_subjects]
        df = df[df["subject_id"].isin(subjects)].copy()
    df = df.sort_values(["subject_id", "timestamp_ns"], kind="mergesort")
    before = len(df)
    df = df.drop_duplicates(["subject_id", "timestamp_ns"], keep="first").reset_index(drop=True)
    if before != len(df):
        log(f"{prefix}: dropped {before - len(df):,} duplicate subject/timestamp rows before alignment")
    return df.rename(
        columns={
            "activity_label": f"{prefix}_label",
            "x": f"{prefix}_x",
            "y": f"{prefix}_y",
            "z": f"{prefix}_z",
        }
    )


def load_aligned_phone_imu(max_subjects: int | None) -> tuple[pd.DataFrame, dict[str, int]]:
    with timed("read phone accelerometer"):
        accel = load_stream(config.PHONE_ACCEL_CLEAN, "accel", max_subjects=max_subjects)
    with timed("read phone gyroscope"):
        gyro = load_stream(config.PHONE_GYRO_CLEAN, "gyro", max_subjects=max_subjects)
    with timed("exact timestamp alignment"):
        merged = accel.merge(gyro, on=["subject_id", "timestamp_ns"], how="inner")
    same_label = merged["accel_label"].eq(merged["gyro_label"])
    known = merged["accel_label"].isin(config.WISDM_LABEL_TO_GROUP)
    aligned = merged.loc[same_label & known].copy()
    aligned["activity_label"] = aligned["accel_label"]
    aligned["activity_name"] = aligned["activity_label"].map(config.WISDM_LABEL_TO_NAME)
    aligned["group_label"] = aligned["activity_label"].map(config.WISDM_LABEL_TO_GROUP)
    aligned["y"] = aligned["group_label"].map(config.GROUP_TO_ID).astype(int)
    aligned = aligned.sort_values(["subject_id", "timestamp_ns"], kind="mergesort").reset_index(drop=True)
    summary = {
        "accel_rows_after_subject_filter": int(len(accel)),
        "gyro_rows_after_subject_filter": int(len(gyro)),
        "exact_aligned_rows": int(len(merged)),
        "label_mismatch_rows_dropped": int((~same_label).sum()),
        "unknown_label_rows_dropped": int((~known).sum()),
        "usable_aligned_rows": int(len(aligned)),
        "subjects": int(aligned["subject_id"].nunique()),
    }
    return aligned, summary


def majority_label(labels: np.ndarray) -> tuple[str | None, float]:
    values, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    winners = values[counts == max_count]
    if len(winners) != 1:
        return None, float(max_count / len(labels))
    return str(winners[0]), float(max_count / len(labels))


def count_candidate_windows(df: pd.DataFrame, window_size: int, step: int) -> int:
    total = 0
    for _, group in df.groupby("subject_id", observed=True, sort=True):
        n = len(group)
        if n >= window_size:
            total += 1 + (n - window_size) // step
    return total


def build_windows(df: pd.DataFrame, window_seconds: float) -> tuple[np.ndarray, pd.DataFrame]:
    window_size = int(round(window_seconds * config.SAMPLE_RATE_HZ))
    step = max(1, int(round(window_size * (1.0 - config.OVERLAP))))
    candidate_count = count_candidate_windows(df, window_size, step)
    x = np.empty((candidate_count, window_size, len(config.CHANNEL_COLUMNS)), dtype=np.float32)
    rows: list[dict[str, object]] = []
    window_id = 0
    dropped_ties = 0

    for subject_id, group in df.groupby("subject_id", observed=True, sort=True):
        group = group.reset_index(drop=True)
        values = group.loc[:, config.CHANNEL_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        labels = group["group_label"].to_numpy()
        timestamps = group["timestamp_ns"].to_numpy(dtype=np.int64, copy=False)
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1, step):
            end = start + window_size
            label, fraction = majority_label(labels[start:end])
            if label is None:
                dropped_ties += 1
                continue
            x[window_id] = values[start:end]
            rows.append(
                {
                    "window_id": window_id,
                    "subject_id": int(subject_id),
                    "start_timestamp_ns": int(timestamps[start]),
                    "end_timestamp_ns": int(timestamps[end - 1]),
                    "n_samples": int(window_size),
                    "window_seconds": float(window_seconds),
                    "overlap": float(config.OVERLAP),
                    "group_label": label,
                    "y": int(config.GROUP_TO_ID[label]),
                    "majority_fraction": fraction,
                }
            )
            window_id += 1

    metadata = pd.DataFrame(rows)
    log(f"{window_seconds:.1f}s: built {len(metadata):,} windows; dropped {dropped_ties:,} tied-majority windows")
    return x[:window_id], metadata


def add_axis_stats(features: dict[str, np.ndarray], values: np.ndarray, prefix: str) -> None:
    q25, q75 = np.percentile(values, [25, 75], axis=1)
    energy = np.mean(values * values, axis=1)
    features[f"{prefix}_mean"] = values.mean(axis=1)
    features[f"{prefix}_std"] = values.std(axis=1, ddof=0)
    features[f"{prefix}_min"] = values.min(axis=1)
    features[f"{prefix}_max"] = values.max(axis=1)
    features[f"{prefix}_median"] = np.median(values, axis=1)
    features[f"{prefix}_iqr"] = q75 - q25
    features[f"{prefix}_range"] = values.max(axis=1) - values.min(axis=1)
    features[f"{prefix}_rms"] = np.sqrt(energy)
    features[f"{prefix}_energy"] = energy


def add_axis_frequency(features: dict[str, np.ndarray], values: np.ndarray, prefix: str) -> None:
    centered = values - values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(values.shape[1], d=1.0 / config.SAMPLE_RATE_HZ)
    total_power = power.sum(axis=1)
    safe = total_power > 1e-12
    non_dc = power.copy()
    if non_dc.shape[1] > 0:
        non_dc[:, 0] = 0.0
    dominant_idx = np.argmax(non_dc, axis=1)
    row_idx = np.arange(len(values))
    prob = np.divide(power, total_power[:, None], out=np.zeros_like(power), where=safe[:, None])
    entropy = -np.sum(prob * np.log2(prob + 1e-12), axis=1) / np.log2(max(power.shape[1], 2))
    low_mask = (freqs > 0.0) & (freqs <= 3.0)
    high_mask = freqs > 3.0
    features[f"{prefix}_freq_total_power"] = total_power
    features[f"{prefix}_freq_dominant_hz"] = np.where(safe, freqs[dominant_idx], 0.0)
    features[f"{prefix}_freq_dominant_power"] = np.divide(power[row_idx, dominant_idx], total_power, out=np.zeros_like(total_power), where=safe)
    features[f"{prefix}_freq_entropy"] = np.where(safe, entropy, 0.0)
    features[f"{prefix}_freq_low_power"] = np.divide(power[:, low_mask].sum(axis=1), total_power, out=np.zeros_like(total_power), where=safe)
    features[f"{prefix}_freq_high_power"] = np.divide(power[:, high_mask].sum(axis=1), total_power, out=np.zeros_like(total_power), where=safe)


def rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denom = np.sqrt(np.sum(left_centered * left_centered, axis=1) * np.sum(right_centered * right_centered, axis=1))
    return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 1e-12)


def extract_feature_table(x: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    feature_values: dict[str, np.ndarray] = {}
    for idx, channel in enumerate(config.CHANNEL_COLUMNS):
        values = x[:, :, idx].astype(np.float64, copy=False)
        add_axis_stats(feature_values, values, channel)
        add_axis_frequency(feature_values, values, channel)

    accel_mag = np.sqrt(np.sum(x[:, :, 0:3].astype(np.float64) ** 2, axis=2))
    gyro_mag = np.sqrt(np.sum(x[:, :, 3:6].astype(np.float64) ** 2, axis=2))
    for name, values in [("accel_magnitude", accel_mag), ("gyro_magnitude", gyro_mag)]:
        add_axis_stats(feature_values, values, name)
        add_axis_frequency(feature_values, values, name)

    for offset, prefix in [(0, "accel"), (3, "gyro")]:
        axes = {
            "x": x[:, :, offset + 0].astype(np.float64, copy=False),
            "y": x[:, :, offset + 1].astype(np.float64, copy=False),
            "z": x[:, :, offset + 2].astype(np.float64, copy=False),
        }
        for left, right in [("x", "y"), ("x", "z"), ("y", "z")]:
            feature_values[f"{prefix}_{left}_{right}_corr"] = rowwise_corr(axes[left], axes[right])

    features = pd.DataFrame(feature_values)
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def write_report(reports_dir: Path, alignment_summary: dict[str, int], window_summaries: list[pd.DataFrame], data_dir: Path) -> None:
    ensure_dirs(reports_dir)
    summary = pd.concat(window_summaries, ignore_index=True) if window_summaries else pd.DataFrame()
    lines = [
        "# Turdalyuly 2026 Window Preparation",
        "",
        "This package prepares WISDM507 phone accelerometer + gyroscope windows for a fair replication of Turdalyuly et al. 2026.",
        "",
        "## Paper Conditions",
        "",
        f"- Paper URL: {config.PAPER_URL}",
        f"- Sensor input: phone accelerometer + phone gyroscope ({len(config.CHANNEL_COLUMNS)} channels).",
        f"- Sample rate: {config.SAMPLE_RATE_HZ:g} Hz.",
        f"- Overlap: {config.OVERLAP:.2f}.",
        f"- Evaluation: {config.N_SPLITS}-fold GroupKFold by subject.",
        "- Window label: majority vote over grouped activity labels.",
        "",
        "## Alignment Summary",
        "",
        markdown_table(pd.DataFrame([alignment_summary])),
        "",
        "## Window Summary",
        "",
        markdown_table(summary),
        "",
        "## Output Directory",
        "",
        f"`{data_dir}`",
        "",
    ]
    (reports_dir / "preparation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paper-style WISDM507 phone accel+gyro windows for Turdalyuly 2026 replication.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--window-sizes", default=None, help="Comma-separated window sizes in seconds. Defaults to 2.0,4.0,6.0.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Use only the first N subjects for smoke testing. Full replication leaves this unset.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    window_sizes = parse_float_list(args.window_sizes, config.WINDOW_SIZES_SECONDS)
    ensure_dirs(args.data_dir, args.reports_dir)

    aligned, alignment_summary = load_aligned_phone_imu(max_subjects=args.max_subjects)
    write_json(args.data_dir / "alignment_summary.json", alignment_summary)

    window_summaries = []
    for window_seconds in window_sizes:
        paths = artifact_paths(args.data_dir, window_seconds)
        if not args.force and paths["tensor"].exists() and paths["metadata"].exists() and paths["features"].exists() and paths["fold_subjects"].exists():
            log(f"{window_seconds:.1f}s artifacts already exist; skipping. Pass --force to rebuild.")
            summary = pd.read_csv(paths["window_summary"]) if paths["window_summary"].exists() else pd.DataFrame()
            if not summary.empty:
                window_summaries.append(summary)
            continue

        with timed(f"build {window_seconds:.1f}s windows"):
            x, metadata = build_windows(aligned, window_seconds)
        if metadata["subject_id"].nunique() < config.N_SPLITS:
            raise ValueError(f"Need at least {config.N_SPLITS} subjects for GroupKFold; got {metadata['subject_id'].nunique()}.")

        with timed(f"extract {window_seconds:.1f}s features"):
            features = extract_feature_table(x, metadata)

        fold_subjects = make_fold_subjects(metadata, config.N_SPLITS)
        class_counts = fold_class_counts(metadata, fold_subjects)
        missing_fold_classes = class_counts[class_counts["eval_windows"].eq(0)]
        if not missing_fold_classes.empty:
            log(f"Warning: some folds are missing grouped classes:\n{missing_fold_classes}")

        np.savez_compressed(
            paths["tensor"],
            X=x,
            y=metadata["y"].to_numpy(dtype=np.int64),
            label_codes=np.array(config.GROUP_LABELS),
            channel_columns=np.array(config.CHANNEL_COLUMNS),
            window_seconds=np.array([window_seconds], dtype=np.float32),
            overlap=np.array([config.OVERLAP], dtype=np.float32),
            sample_rate_hz=np.array([config.SAMPLE_RATE_HZ], dtype=np.float32),
        )
        metadata.to_parquet(paths["metadata"], index=False)
        features.to_parquet(paths["features"], index=False)
        fold_subjects.to_csv(paths["fold_subjects"], index=False)
        class_counts.to_csv(paths["fold_class_counts"], index=False)

        summary = (
            metadata.groupby(["window_seconds", "group_label"], observed=True)
            .size()
            .rename("windows")
            .reset_index()
            .sort_values(["window_seconds", "group_label"])
        )
        summary["subjects"] = metadata["subject_id"].nunique()
        summary["candidate_artifact"] = paths["tensor"].name
        summary.to_csv(paths["window_summary"], index=False)
        window_summaries.append(summary)
        log(f"saved {len(metadata):,} windows for {window_seconds:.1f}s to {args.data_dir}")

    write_report(args.reports_dir, alignment_summary, window_summaries, args.data_dir)
    log(f"wrote {args.reports_dir / 'preparation_report.md'}")


if __name__ == "__main__":
    main()

