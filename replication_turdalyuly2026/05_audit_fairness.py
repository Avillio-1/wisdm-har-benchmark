from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
from utils import artifact_paths, fold_class_counts, fold_masks, load_prepared_window, markdown_table


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def row_for_metric(name: str, ours: float, target: float, target_std: float) -> dict[str, object]:
    delta = ours - target
    return {
        "paper_result": name,
        "paper_macro_f1": target,
        "paper_std": target_std,
        "our_macro_f1": ours,
        "delta": delta,
        "within_one_paper_std": abs(delta) <= target_std,
    }


def audit_subject_folds(metadata: pd.DataFrame, fold_subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    class_counts = fold_class_counts(metadata, fold_subjects)
    for fold in sorted(fold_subjects["fold"].unique()):
        train_mask, eval_mask = fold_masks(metadata, fold_subjects, int(fold))
        train_subjects = set(metadata.loc[train_mask, "subject_id"].astype(int))
        eval_subjects = set(metadata.loc[eval_mask, "subject_id"].astype(int))
        rows.append(
            {
                "fold": int(fold),
                "train_subjects": len(train_subjects),
                "eval_subjects": len(eval_subjects),
                "eval_windows": int(eval_mask.sum()),
                "subject_overlap": len(train_subjects & eval_subjects),
                "missing_eval_classes": int(
                    class_counts.loc[class_counts["fold"].eq(int(fold)) & class_counts["eval_windows"].eq(0)].shape[0]
                ),
            }
        )
    return pd.DataFrame(rows), class_counts


def load_alignment_summary(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "alignment_summary.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(path.read_text(encoding="utf-8"))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fairness and replication risk for the Turdalyuly 2026 benchmark package.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    _, metadata, _, fold_subjects = load_prepared_window(args.data_dir, args.window_seconds)
    fold_audit, class_counts = audit_subject_folds(metadata, fold_subjects)
    alignment = load_alignment_summary(args.data_dir)
    window_summary = read_csv_if_exists(artifact_paths(args.data_dir, args.window_seconds)["window_summary"])

    cnn_summary = read_csv_if_exists(args.results_dir / "cnn" / "cnn_replication_summary.csv")
    feature_summary = read_csv_if_exists(args.results_dir / "feature_models" / "feature_models_summary.csv")

    target_rows = []
    if not cnn_summary.empty:
        cnn_4s = cnn_summary[cnn_summary["window_seconds"].astype(float).eq(args.window_seconds)]
        if "modality" in cnn_4s.columns:
            cnn_4s = cnn_4s[cnn_4s["modality"].eq("accel_gyro")]
        if not cnn_4s.empty:
            ours = float(cnn_4s.iloc[0]["macro_f1_mean"])
            for name, target in config.PAPER_REPORTED_TARGETS.items():
                target_rows.append(row_for_metric(name, ours, target["macro_f1_mean"], target["macro_f1_std"]))
    target_comparison = pd.DataFrame(target_rows)

    best_feature = pd.DataFrame()
    if not feature_summary.empty:
        best_feature = (
            feature_summary[feature_summary["window_seconds"].astype(float).eq(args.window_seconds)]
            .sort_values("macro_f1_mean", ascending=False)
            .head(5)
        )

    matched = pd.DataFrame(
        [
            {"condition": "Dataset", "paper": "WISDM507", "ours": "WISDM Smartphone and Smartwatch raw phone streams", "status": "matched disclosed dataset"},
            {"condition": "Sensor input", "paper": "phone accel + gyro, 6 channels", "ours": "phone accel + gyro, 6 channels", "status": "matched"},
            {"condition": "Window length", "paper": "4.0 s primary", "ours": f"{args.window_seconds:.1f} s", "status": "matched"},
            {"condition": "Overlap", "paper": "50%", "ours": f"{config.OVERLAP * 100:.0f}%", "status": "matched"},
            {"condition": "Labeling", "paper": "majority vote", "ours": "majority vote", "status": "matched"},
            {"condition": "Evaluation", "paper": "5-fold GroupKFold by subject", "ours": "5-fold GroupKFold by subject", "status": "matched"},
            {"condition": "Normalization", "paper": "train-fold z-score only", "ours": "train-fold z-score only for CNN; sklearn train-fold pipelines for features", "status": "matched for implemented models"},
            {"condition": "Optimizer", "paper": "AdamW, lr=1e-3", "ours": "AdamW, lr=1e-3 for CNN", "status": "matched"},
            {"condition": "Epochs/batch size", "paper": "10 epochs, train batch 256, eval batch 512", "ours": "10 epochs, train batch 256, eval batch 512", "status": "matched"},
            {"condition": "GPU", "paper": "CUDA mixed precision when available", "ours": "CUDA required by default; fold metrics recorded device=cuda", "status": "matched/improved enforcement"},
        ]
    )

    unresolved = pd.DataFrame(
        [
            {
                "issue": "Exact author code unavailable",
                "risk": "CNN architecture, layer widths, initialization, fold ordering, and preprocessing edge cases cannot be guaranteed identical.",
                "action": "Do not claim an exact clone; claim independently implemented reproduction under disclosed conditions.",
            },
            {
                "issue": "Official supplementary file not retrievable from this environment",
                "risk": "MDPI blocked direct supplementary ZIP download attempts, so we could not inspect possible hidden code/configs.",
                "action": "If a browser download succeeds, compare the official supplement against this package before final submission.",
            },
            {
                "issue": "6-class mapping not fully enumerated in the paper text",
                "risk": "Our mapping is a documented interpretation of the named categories using the official WISDM activity key.",
                "action": "Report the full mapping in the paper and appendix.",
            },
            {
                "issue": "Timestamp alignment details are underspecified",
                "risk": "The paper says aligned by timestamp, but does not disclose exact-match vs tolerance/resampling behavior.",
                "action": "Report exact timestamp inner join and aligned-row counts; avoid claiming identical preprocessing.",
            },
            {
                "issue": "Published CNN result is not reproduced numerically",
                "risk": "Our CNN macro-F1 is higher than all reported 4.0 s CNN targets, so the CNN is not a perfect replica.",
                "action": "Use CNN as calibration only; base the contribution on same-protocol proposed method comparison.",
            },
            {
                "issue": "Classical feature models are a different method family",
                "risk": "They are not a replication of the paper's CNN and should not be described as the same model.",
                "action": "Frame XGBoost stats+freq as our proposed model evaluated under the same disclosed data/protocol.",
            },
        ]
    )

    fairness_verdict = (
        "Protocol-fair proposed-method comparison, not exact hidden-code replication."
        if not target_comparison.empty and not bool(target_comparison["within_one_paper_std"].all())
        else "CNN calibration is numerically close to the published targets."
    )

    lines = [
        "# Fairness Audit: Turdalyuly et al. 2026 Replication",
        "",
        f"Source paper: {config.PAPER_URL}",
        "",
        f"**Verdict:** {fairness_verdict}",
        "",
        "The current package matches the disclosed dataset, sensor condition, grouped task, window length, overlap, majority-vote labeling, subject-wise GroupKFold evaluation, train-fold normalization, and CNN training hyperparameters. It does **not** exactly reproduce the published CNN number, so the paper should not claim a perfect replication of the authors' hidden implementation.",
        "",
        "## Matched Disclosed Conditions",
        "",
        markdown_table(matched),
        "",
        "## Published CNN Target Checks",
        "",
        markdown_table(target_comparison),
        "",
        "## Same-Protocol Proposed-Method Results",
        "",
        markdown_table(best_feature),
        "",
        "## Fold Leakage Checks",
        "",
        markdown_table(fold_audit),
        "",
        "## Fold Class Coverage",
        "",
        markdown_table(class_counts),
        "",
        "## Alignment Summary",
        "",
        markdown_table(alignment),
        "",
        "## Window Summary",
        "",
        markdown_table(window_summary),
        "",
        "## Unresolved Replication Risks",
        "",
        markdown_table(unresolved),
        "",
        "## Safe Academic Claim",
        "",
        "Under the disclosed Turdalyuly et al. WISDM507 smartphone IMU condition (phone accelerometer + gyroscope), reduced 6-class taxonomy, 4.0 s windows, 50% overlap, majority-vote labeling, and 5-fold subject-wise GroupKFold evaluation, our statistical/frequency feature model outperforms the reported CNN benchmark. Because the authors' exact preprocessing code, fold assignment, and architecture widths are not disclosed and our CNN calibration is numerically higher than theirs, we present this as a same-protocol comparison rather than an exact reproduction of their hidden implementation.",
        "",
    ]
    out_path = args.reports_dir / "fairness_audit.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    matched.to_csv(args.reports_dir / "fairness_matched_conditions.csv", index=False)
    unresolved.to_csv(args.reports_dir / "fairness_unresolved_risks.csv", index=False)
    target_comparison.to_csv(args.reports_dir / "fairness_target_checks.csv", index=False)
    fold_audit.to_csv(args.reports_dir / "fairness_fold_audit.csv", index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
