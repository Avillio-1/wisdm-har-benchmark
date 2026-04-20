from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from utils import (
    classification_outputs,
    ensure_dirs,
    fold_masks,
    load_prepared_window,
    markdown_table,
    numeric_feature_columns,
    parse_float_list,
    parse_str_list,
    save_metric_bundle,
    summarize_fold_metrics,
    timed,
)


MODEL_ORDER = ("majority", "logistic_regression", "random_forest", "lightgbm", "xgboost")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def make_model(name: str, seed: int, n_jobs: int, n_estimators: int):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=1)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            n_jobs=n_jobs,
        )
    if name == "lightgbm":
        if not module_available("lightgbm"):
            raise RuntimeError("lightgbm is not installed.")
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if name == "xgboost":
        if not module_available("xgboost"):
            raise RuntimeError("xgboost is not installed.")
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


def feature_set_for_model(model_name: str, features: pd.DataFrame) -> tuple[str, list[str]]:
    if model_name in {"lightgbm", "xgboost"}:
        return "stats_freq", numeric_feature_columns(features, include_frequency=True)
    if model_name in {"logistic_regression", "random_forest"}:
        return "stats", numeric_feature_columns(features, include_frequency=False)
    return "none", []


def train_window(window_seconds: float, models: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, metadata, features, fold_subjects = load_prepared_window(args.data_dir, window_seconds)
    y = metadata["y"].to_numpy(dtype=np.int64)
    fold_metrics = []
    per_class_frames = []
    confusion_dir = args.results_dir / "feature_models" / "confusion_matrices"
    ensure_dirs(confusion_dir)

    folds = sorted(fold_subjects["fold"].unique())
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    for model_name in models:
        feature_set, cols = feature_set_for_model(model_name, features)
        if model_name != "majority":
            x_all = features.loc[:, cols].to_numpy(dtype=np.float32, copy=False)
            x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x_all = np.zeros((len(metadata), 1), dtype=np.float32)

        for fold in folds:
            train_mask, eval_mask = fold_masks(metadata, fold_subjects, int(fold))
            model = make_model(model_name, args.seed + int(fold), args.n_jobs, args.n_estimators)
            with timed(f"{model_name} {window_seconds:.1f}s fold {fold}"):
                model.fit(x_all[train_mask], y[train_mask])
                y_pred = model.predict(x_all[eval_mask])
            report, cm, metrics = classification_outputs(y[eval_mask], y_pred)
            cm.to_csv(confusion_dir / f"{model_name}_{window_seconds:.1f}s_fold{fold}.csv")
            report.insert(0, "fold", int(fold))
            report.insert(0, "feature_set", feature_set)
            report.insert(0, "model", model_name)
            report.insert(0, "window_seconds", window_seconds)
            per_class_frames.append(report)
            fold_metrics.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "window_seconds": window_seconds,
                    "fold": int(fold),
                    "eval_subjects": int(metadata.loc[eval_mask, "subject_id"].nunique()),
                    "eval_windows": int(eval_mask.sum()),
                    "n_features": len(cols),
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(fold_metrics)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    summary = summarize_fold_metrics(fold_df, ["model", "feature_set", "window_seconds"])
    return fold_df, per_class, summary


def write_report(summary: pd.DataFrame, reports_dir: Path) -> None:
    ensure_dirs(reports_dir)
    if summary.empty:
        claim = "No feature-model results were found."
    else:
        primary = summary[summary["window_seconds"].eq(4.0)].sort_values("macro_f1_mean", ascending=False)
        if primary.empty:
            claim = "The 4.0 s primary feature-model benchmark was not run."
        else:
            best = primary.iloc[0]
            delta = float(best["macro_f1_mean"]) - config.PAPER_TARGET_MACRO_F1_MEAN
            claim = (
                f"Best 4.0 s phone accelerometer + gyroscope feature model: {best['model']} ({best['feature_set']}), "
                f"macro-F1 mean {best['macro_f1_mean']:.4f}. "
                f"Delta versus Turdalyuly CNN target {config.PAPER_TARGET_MACRO_F1_MEAN:.4f}: {delta:+.4f}."
            )
    lines = [
        "# Turdalyuly 2026 Feature-Model Benchmark",
        "",
        claim,
        "",
        "## Summary",
        "",
        markdown_table(summary.sort_values(["window_seconds", "macro_f1_mean"], ascending=[True, False]) if not summary.empty else summary),
        "",
        "## Claim Safety",
        "",
        "- The headline comparison is valid only for the same smartphone IMU condition: phone accelerometer + gyroscope input.",
        "- Models use the same prepared windows and GroupKFold subject folds as the CNN calibration.",
        "- If the CNN calibration is far from the paper target, report this as an in-repository replication rather than exact paper reproduction.",
        "",
    ]
    (reports_dir / "feature_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical feature models on the Turdalyuly 2026 replication windows.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--window-sizes", default="4.0")
    parser.add_argument("--include-secondary", action="store_true", help="Also run the best 4.0 s model on 2.0 s and 6.0 s windows.")
    parser.add_argument("--models", default=None, help="Comma-separated models. Defaults to majority,logistic_regression,random_forest,lightgbm,xgboost.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-folds", type=int, default=None, help="Smoke-test helper. Full replication leaves this unset.")
    args = parser.parse_args()

    ensure_dirs(args.results_dir / "feature_models", args.reports_dir)
    models = parse_str_list(args.models, MODEL_ORDER)
    all_fold = []
    all_per_class = []
    all_summary = []

    primary_windows = parse_float_list(args.window_sizes, [4.0])
    for window_seconds in primary_windows:
        fold_df, per_class, summary = train_window(window_seconds, models, args)
        all_fold.append(fold_df)
        all_per_class.append(per_class)
        all_summary.append(summary)

    summary_so_far = pd.concat(all_summary, ignore_index=True)
    if args.include_secondary and not summary_so_far.empty:
        primary = summary_so_far[summary_so_far["window_seconds"].eq(4.0)].sort_values("macro_f1_mean", ascending=False)
        if not primary.empty:
            best_model = str(primary.iloc[0]["model"])
            for window_seconds in [2.0, 6.0]:
                if window_seconds not in primary_windows:
                    fold_df, per_class, summary = train_window(window_seconds, [best_model], args)
                    all_fold.append(fold_df)
                    all_per_class.append(per_class)
                    all_summary.append(summary)

    fold_metrics = pd.concat(all_fold, ignore_index=True)
    per_class = pd.concat(all_per_class, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    save_metric_bundle(args.results_dir / "feature_models", "feature_models", fold_metrics, per_class, summary)
    write_report(summary, args.reports_dir)
    print(f"Wrote {args.reports_dir / 'feature_model_report.md'}")


if __name__ == "__main__":
    main()
