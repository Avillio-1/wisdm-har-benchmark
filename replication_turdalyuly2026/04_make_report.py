from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from utils import artifact_paths, ensure_dirs, markdown_table, parse_float_list


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def prepared_summary(data_dir: Path, window_sizes: list[float]) -> pd.DataFrame:
    frames = []
    for window_seconds in window_sizes:
        path = artifact_paths(data_dir, window_seconds)["window_summary"]
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_feature_claim(feature_summary: pd.DataFrame) -> str:
    if feature_summary.empty:
        return "Feature-model results are not available yet."
    primary = feature_summary[feature_summary["window_seconds"].eq(4.0)].sort_values("macro_f1_mean", ascending=False)
    if primary.empty:
        return "The primary 4.0 s feature-model comparison has not been run yet."
    best = primary.iloc[0]
    delta = float(best["macro_f1_mean"]) - config.PAPER_TARGET_MACRO_F1_MEAN
    if delta > config.PAPER_TARGET_MACRO_F1_STD:
        strength = "strongly exceeds the paper target by more than one reported standard deviation"
    elif delta > 0:
        strength = "exceeds the paper target in this replication setting"
    else:
        strength = "does not beat the paper target yet"
    return (
        f"The best 4.0 s phone accelerometer + gyroscope model is `{best['model']}` with macro-F1 "
        f"{best['macro_f1_mean']:.4f} +/- {best['macro_f1_std']:.4f}; this {strength} "
        f"(delta {delta:+.4f} versus {config.PAPER_TARGET_MACRO_F1_MEAN:.4f})."
    )


def cnn_calibration_claim(cnn_summary: pd.DataFrame) -> str:
    if cnn_summary.empty:
        return "CNN calibration has not been run yet, or PyTorch was unavailable."
    primary = cnn_summary[cnn_summary["window_seconds"].eq(4.0)]
    if "modality" in primary.columns:
        primary = primary[primary["modality"].eq("accel_gyro")]
    if primary.empty:
        return "CNN calibration exists, but the 4.0 s accel+gyro primary condition was not run."
    row = primary.iloc[0]
    value = float(row["macro_f1_mean"])
    delta = value - config.PAPER_TARGET_MACRO_F1_MEAN
    within = abs(delta) <= config.PAPER_TARGET_MACRO_F1_STD
    return (
        f"The replicated 4.0 s CNN macro-F1 is {value:.4f} +/- {row['macro_f1_std']:.4f}; "
        f"paper target is {config.PAPER_TARGET_MACRO_F1_MEAN:.4f} +/- {config.PAPER_TARGET_MACRO_F1_STD:.4f}. "
        f"Within one paper std: {within}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper-ready Turdalyuly 2026 replication report.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--window-sizes", default=None)
    args = parser.parse_args()

    ensure_dirs(args.reports_dir)
    window_sizes = parse_float_list(args.window_sizes, config.WINDOW_SIZES_SECONDS)
    prep_summary = prepared_summary(args.data_dir, window_sizes)
    cnn_summary = read_csv_if_exists(args.results_dir / "cnn" / "cnn_replication_summary.csv")
    feature_summary = read_csv_if_exists(args.results_dir / "feature_models" / "feature_models_summary.csv")
    feature_per_class = read_csv_if_exists(args.results_dir / "feature_models" / "feature_models_per_class.csv")

    per_class_primary = pd.DataFrame()
    if not feature_summary.empty and not feature_per_class.empty:
        primary = feature_summary[feature_summary["window_seconds"].eq(4.0)].sort_values("macro_f1_mean", ascending=False)
        if not primary.empty:
            best = primary.iloc[0]
            per_class_primary = feature_per_class[
                feature_per_class["window_seconds"].eq(4.0)
                & feature_per_class["model"].eq(best["model"])
                & feature_per_class["class"].isin(config.GROUP_LABELS)
            ].copy()
            if not per_class_primary.empty:
                per_class_primary = (
                    per_class_primary.groupby("class", observed=True)[["precision", "recall", "f1-score", "support"]]
                    .mean()
                    .reset_index()
                    .sort_values("f1-score", ascending=False)
                )

    lines = [
        "# Turdalyuly et al. 2026 WISDM507 Replication Benchmark",
        "",
        f"Source paper: {config.PAPER_URL}",
        "",
        "## Goal",
        "",
        "Replicate the disclosed Turdalyuly et al. 2026 WISDM507 smartphone IMU condition, then test whether classical feature models beat their CNN benchmark on the same phone accelerometer + gyroscope input.",
        "",
        "## Replicated Conditions",
        "",
        f"- Dataset: WISDM507 / WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.",
        f"- Input: phone accelerometer + phone gyroscope, six synchronized channels.",
        f"- Windowing: 50% overlap, majority-vote labels, {', '.join(str(v) + ' s' for v in window_sizes)} windows.",
        f"- Grouping: locomotion, stairs, static, eat_drink, sports, upper_body.",
        f"- Evaluation: {config.N_SPLITS}-fold GroupKFold by subject.",
        f"- Paper target: 4.0 s CNN macro-F1 {config.PAPER_TARGET_MACRO_F1_MEAN:.4f} +/- {config.PAPER_TARGET_MACRO_F1_STD:.4f}.",
        "",
        "## Prepared Data",
        "",
        markdown_table(prep_summary),
        "",
        "## CNN Calibration",
        "",
        cnn_calibration_claim(cnn_summary),
        "",
        markdown_table(cnn_summary),
        "",
        "## Feature-Model Benchmark",
        "",
        best_feature_claim(feature_summary),
        "",
        markdown_table(feature_summary.sort_values(["window_seconds", "macro_f1_mean"], ascending=[True, False]) if not feature_summary.empty else feature_summary),
        "",
        "## Best Model Per-Class Performance",
        "",
        markdown_table(per_class_primary),
        "",
        "## Claim Safety Notes",
        "",
        "- The headline claim must compare models using the same smartphone IMU input: phone accelerometer + gyroscope.",
        "- The CNN calibration does not exactly reproduce the published value, so phrase results as an independently implemented reproduction under disclosed conditions, not a clone of the authors' hidden implementation.",
        "- Do not use watch or phone+watch fusion as the headline win; those are broader ablations, not the same input condition.",
        "- Report macro-F1 as primary and accuracy as secondary.",
        "- Use the feature-model result as the proposed-method comparison; use the CNN result only as a calibration check.",
        "",
        "## Reproduction Commands",
        "",
        "```powershell",
        "python replication_turdalyuly2026/01_prepare_windows.py",
        "python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 2.0,4.0,6.0 --modalities accel,gyro,accel_gyro",
        "python replication_turdalyuly2026/03_train_feature_models.py --window-sizes 4.0 --include-secondary",
        "python replication_turdalyuly2026/04_make_report.py",
        "python replication_turdalyuly2026/05_audit_fairness.py",
        "```",
        "",
    ]
    out_path = args.reports_dir / "turdalyuly2026_replication_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
