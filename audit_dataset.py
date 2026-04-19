from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.wisdm import (
    ACTIVITY_MAP,
    DEFAULT_DATASET_DIR,
    RAW_COLUMNS,
    class_distribution,
    iter_raw_files,
    markdown_table,
    parse_raw_path,
    read_raw_file,
)


def audit_raw_dataset(dataset_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_rows = []
    quality_rows = []
    class_counts: dict[str, int] = {}
    missing_counts = {col: 0 for col in RAW_COLUMNS}
    total_rows = 0
    total_duplicate_rows = 0

    for path in iter_raw_files(dataset_dir):
        info = parse_raw_path(path)
        df = read_raw_file(path)
        rows = len(df)
        total_rows += rows

        for label, count in df["activity_label"].value_counts(dropna=False).items():
            class_counts[str(label)] = class_counts.get(str(label), 0) + int(count)
        for col in RAW_COLUMNS:
            missing_counts[col] += int(df[col].isna().sum())

        exact_subset = ["subject_id", "activity_label", "timestamp_ns", "x", "y", "z", "device", "sensor"]
        duplicate_rows = int(df.duplicated(subset=exact_subset, keep="first").sum())
        total_duplicate_rows += duplicate_rows
        timestamp_duplicate_rows = int(df.duplicated(subset=["activity_label", "timestamp_ns"], keep=False).sum())
        nonmonotonic_rows = int(
            df.groupby("activity_label", observed=True)["timestamp_ns"]
            .diff()
            .lt(0)
            .fillna(False)
            .sum()
        )

        file_rows.append(
            {
                "device": info.device,
                "sensor": info.sensor,
                "subject_id": info.subject_id,
                "file": str(path.relative_to(dataset_dir)),
                "bytes": path.stat().st_size,
                "rows": rows,
            }
        )
        quality_rows.append(
            {
                "device": info.device,
                "sensor": info.sensor,
                "subject_id": info.subject_id,
                "rows": rows,
                "unknown_activity_rows": int((~df["is_known_activity"]).sum()),
                "missing_required_rows": int(df[RAW_COLUMNS].isna().any(axis=1).sum()),
                "nonpositive_timestamp_rows": int(df["timestamp_ns"].le(0).fillna(True).sum()),
                "source_subject_mismatch_rows": int(df["source_subject_mismatch"].fillna(False).sum()),
                "exact_duplicate_rows": duplicate_rows,
                "duplicate_timestamp_rows": timestamp_duplicate_rows,
                "nonmonotonic_timestamp_rows": nonmonotonic_rows,
            }
        )

    inventory = pd.DataFrame(file_rows)
    quality = pd.DataFrame(quality_rows)
    class_df = pd.DataFrame(
        [
            {
                "activity_label": label,
                "activity_name": ACTIVITY_MAP.get(label, "UNKNOWN"),
                "rows": count,
            }
            for label, count in sorted(class_counts.items())
        ]
    )
    class_df["percent"] = (class_df["rows"] / class_df["rows"].sum() * 100).round(2)
    missing_df = pd.DataFrame(
        [{"column": col, "missing_rows": count, "missing_percent": round(count / total_rows * 100, 4)} for col, count in missing_counts.items()]
    )
    duplicate_df = pd.DataFrame(
        [
            {"scope": "within individual raw files", "duplicate_rows": total_duplicate_rows, "duplicate_percent": round(total_duplicate_rows / total_rows * 100, 4)}
        ]
    )

    inventory.to_csv(out_dir / "raw_file_inventory.csv", index=False)
    quality.to_csv(out_dir / "raw_quality_by_file.csv", index=False)
    class_df.to_csv(out_dir / "class_distribution_raw.csv", index=False)
    missing_df.to_csv(out_dir / "missing_values_raw.csv", index=False)
    duplicate_df.to_csv(out_dir / "duplicate_rows_raw.csv", index=False)
    return {
        "inventory": inventory,
        "quality": quality,
        "class_distribution": class_df,
        "missing": missing_df,
        "duplicates": duplicate_df,
    }


def write_report(dataset_dir: Path, results: dict[str, pd.DataFrame], output_path: Path) -> None:
    inventory = results["inventory"]
    quality = results["quality"]
    selected = inventory[(inventory["device"] == "phone") & (inventory["sensor"] == "accel")]
    selected_class_path = dataset_dir
    selected_df = None
    if len(selected) > 0:
        selected_frames = []
        for rel_file in selected["file"]:
            selected_frames.append(read_raw_file(selected_class_path / rel_file))
        selected_df = pd.concat(selected_frames, ignore_index=True)
        selected_class_distribution = class_distribution(selected_df)
    else:
        selected_class_distribution = pd.DataFrame()

    files_by_stream = (
        inventory.groupby(["device", "sensor"], observed=True)
        .agg(files=("file", "count"), subjects=("subject_id", "nunique"), rows=("rows", "sum"), mb=("bytes", lambda s: round(s.sum() / 1_000_000, 2)))
        .reset_index()
    )
    quality_summary = quality.drop(columns=["subject_id"]).groupby(["device", "sensor"], observed=True).sum(numeric_only=True).reset_index()

    data_quality_issues = [
        "No official train/test split is provided; raw files are grouped by subject/device/sensor, so evaluation must create subject-wise splits.",
        "Raw timestamps are nanosecond-like sensor times, not human-readable calendar datetimes; use them for ordering/gaps, not wall-clock interpretation.",
        "The bundle contains multiple devices and sensors. A project should pick one stream first or carefully synchronize streams before sensor fusion.",
        "ARFF files are already windowed feature files, but the recommended project should create its own windows from raw samples to control leakage and preprocessing.",
    ]
    if int(quality["exact_duplicate_rows"].sum()) > 0:
        data_quality_issues.append("Some exact duplicate rows were found within raw files and should be removed during preprocessing.")
    if int(quality["unknown_activity_rows"].sum()) > 0:
        data_quality_issues.append("Unknown activity labels were found and should be dropped or reconciled.")
    if int(quality["missing_required_rows"].sum()) > 0:
        data_quality_issues.append("Rows with missing required fields were found and should be removed or imputed before modeling.")

    lines = [
        "# Dataset Audit",
        "",
        "## Short dataset summary",
        "",
        "Dataset: WISDM activity recognition raw sensor bundle.",
        "",
        "- Participants: 51 subjects, with IDs 1600 through 1650 in the raw filenames.",
        "- Raw streams: phone accelerometer, phone gyroscope, watch accelerometer, watch gyroscope.",
        "- Raw schema: `subject_id`, `activity_label`, `timestamp_ns`, `x`, `y`, `z`; scripts add `device`, `sensor`, `source_file`, and `activity_name` metadata.",
        "- Target label: `activity_label` / `activity_name`, an 18-class human activity label.",
        "- Subject/user ID: `subject_id`.",
        "- Timestamp column: `timestamp_ns`, a high-resolution sensor timestamp used for ordering.",
        "- Sensor columns: `x`, `y`, `z`.",
        "- Train/test split availability: no official split files were found in this bundle.",
        "- Recommended main project task: 18-class activity classification from raw phone accelerometer windows using subject-wise evaluation.",
        "",
        "## Raw files by stream",
        "",
        markdown_table(files_by_stream),
        "",
        "## Overall class distribution",
        "",
        markdown_table(results["class_distribution"]),
        "",
        "## Recommended task class distribution: phone accelerometer",
        "",
        markdown_table(selected_class_distribution),
        "",
        "## Missing-value report",
        "",
        markdown_table(results["missing"]),
        "",
        "## Duplicate-row report",
        "",
        markdown_table(results["duplicates"]),
        "",
        "## Quality checks by stream",
        "",
        markdown_table(quality_summary),
        "",
        "## Data-quality issues list",
        "",
        *[f"- {issue}" for issue in data_quality_issues],
        "",
        "## Project framing recommendation",
        "",
        "Use a subject-wise 18-class activity recognition task on raw phone accelerometer windows. This is clear, reproducible, and avoids the extra synchronization assumptions needed for multi-sensor fusion. Phone accelerometer is also a strong first task because every sample has one label, one subject ID, one timestamp, and three sensor axes.",
        "",
        "Generated by `audit_dataset.py`.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate the WISDM raw dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--report", type=Path, default=Path("dataset_audit.md"))
    args = parser.parse_args()

    results = audit_raw_dataset(args.dataset_dir, args.out_dir)
    write_report(args.dataset_dir, results, args.report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
