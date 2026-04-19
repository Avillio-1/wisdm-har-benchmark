from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.wisdm import ACTIVITY_ORDER, feature_columns, markdown_table


def make_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            objective="multiclass",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    except Exception as exc:
        print(f"LightGBM unavailable, skipping: {exc}")
    return models


def evaluate_baselines(
    windows_path: Path,
    out_dir: Path,
    report_path: Path,
    seed: int,
    label_order: list[str] | None = None,
    refit_train_val: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(windows_path)
    if "split" not in windows:
        raise ValueError("Window table must include a split column. Run splits.py before windowing.py.")

    train = windows[windows["split"] == "train"].copy()
    val = windows[windows["split"] == "val"].copy()
    test = windows[windows["split"] == "test"].copy()
    fit_data = pd.concat([train, val], ignore_index=True) if refit_train_val else train

    features = feature_columns(windows)
    label_encoder = LabelEncoder()
    labels = label_order if label_order else sorted(windows["activity_label"].dropna().unique())
    unknown_labels = sorted(set(labels) - set(ACTIVITY_ORDER))
    if unknown_labels:
        raise ValueError(f"Unknown labels: {unknown_labels}")
    label_encoder.fit(labels)

    x_fit = fit_data[features]
    y_fit = label_encoder.transform(fit_data["activity_label"])

    result_rows = []
    report_sections = []
    for name, model in make_models(seed).items():
        model.fit(x_fit, y_fit)
        split_metrics = {}
        labels_present = list(range(len(label_encoder.classes_)))
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            y_true = label_encoder.transform(split_df["activity_label"])
            pred = model.predict(split_df[features])
            acc = accuracy_score(y_true, pred)
            macro_f1 = f1_score(y_true, pred, average="macro", labels=labels_present)
            split_metrics[split_name] = {"accuracy": acc, "macro_f1": macro_f1}
            cls_report = classification_report(
                y_true,
                pred,
                labels=labels_present,
                target_names=[f"{code}" for code in label_encoder.classes_],
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(cls_report).T.reset_index().rename(columns={"index": "class"})
            report_df.to_csv(out_dir / f"{name}_{split_name}_classification_report.csv", index=False)

            cm = pd.DataFrame(
                confusion_matrix(y_true, pred, labels=labels_present),
                index=label_encoder.classes_,
                columns=label_encoder.classes_,
            )
            cm.to_csv(out_dir / f"{name}_{split_name}_confusion_matrix.csv")

        result_rows.append(
            {
                "model": name,
                "val_accuracy": round(split_metrics["val"]["accuracy"], 4),
                "val_macro_f1": round(split_metrics["val"]["macro_f1"], 4),
                "test_accuracy": round(split_metrics["test"]["accuracy"], 4),
                "test_macro_f1": round(split_metrics["test"]["macro_f1"], 4),
            }
        )

        report_sections.extend(
            [
                f"### {name}",
                "",
                f"- Validation accuracy: {split_metrics['val']['accuracy']:.4f}",
                f"- Validation macro F1: {split_metrics['val']['macro_f1']:.4f}",
                f"- Test accuracy: {split_metrics['test']['accuracy']:.4f}",
                f"- Test macro F1: {split_metrics['test']['macro_f1']:.4f}",
                f"- Test classification report: `{out_dir / f'{name}_test_classification_report.csv'}`",
                f"- Test confusion matrix: `{out_dir / f'{name}_test_confusion_matrix.csv'}`",
                "",
            ]
        )

    results = pd.DataFrame(result_rows).sort_values("test_macro_f1", ascending=False)
    results.to_csv(out_dir / "baseline_metrics.csv", index=False)

    split_counts = windows.groupby("split", observed=True).size().rename("windows").reset_index()
    lines = [
        "# Baseline Model Results",
        "",
        f"Window feature table: `{windows_path}`",
        "",
        "Models use handcrafted statistical window features and a subject-wise split. By default, estimators are fit on train only, validation is reported for model choice, and test remains held out.",
        "",
        "## Split sizes",
        "",
        markdown_table(split_counts),
        "",
        "## Metrics",
        "",
        markdown_table(results),
        "",
        "## Model artifacts",
        "",
        *report_sections,
        "Generated by `baselines.py`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe classical ML baselines on WISDM windows.")
    parser.add_argument("--windows", type=Path, default=Path("data/processed/windows/phone_accel_windows_5p0s_50overlap.csv.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/baselines"))
    parser.add_argument("--report", type=Path, default=Path("baseline_results.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-order", default=None, help="Comma-separated labels in output order, for example A,B,E.")
    parser.add_argument("--refit-train-val", action="store_true", help="Fit final models on train+validation. Leave unset for strict train-only scaling.")
    args = parser.parse_args()
    label_order = [label.strip().upper() for label in args.label_order.split(",")] if args.label_order else None

    evaluate_baselines(args.windows, args.out_dir, args.report, args.seed, label_order=label_order, refit_train_val=args.refit_train_val)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
