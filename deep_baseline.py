from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.wisdm import ACTIVITY_ORDER, feature_columns, markdown_table


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("At least one hidden layer is required.")
    return layers


def save_training_curve(model: MLPClassifier, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("MLP training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_validation_curve(model: MLPClassifier, out_path: Path) -> None:
    scores = getattr(model, "validation_scores_", None)
    if not scores:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(scores) + 1), scores, marker="o", linewidth=1.5, color="#2F6B7C")
    plt.xlabel("Epoch")
    plt.ylabel("Internal validation accuracy")
    plt.title("MLP early-stopping validation curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_split(
    name: str,
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
    out_dir: Path,
) -> dict[str, float | str | int]:
    y_true = label_encoder.transform(y)
    y_pred = model.predict(x)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    labels = list(range(len(label_encoder.classes_)))
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index().rename(columns={"index": "class"})
    report.to_csv(out_dir / f"mlp_{name}_classification_report.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    cm.to_csv(out_dir / f"mlp_{name}_confusion_matrix.csv")

    return {
        "split": name,
        "windows": int(len(x)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }


def run_deep_baseline(
    windows_path: Path,
    out_dir: Path,
    report_path: Path,
    hidden_layers: tuple[int, ...],
    max_iter: int,
    batch_size: int,
    learning_rate: float,
    alpha: float,
    seed: int,
    save_model: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(windows_path)
    if "split" not in windows.columns:
        raise ValueError("Window table must include a split column. Run splits.py and windowing.py first.")

    features = feature_columns(windows)
    train = windows[windows["split"] == "train"].copy()
    val = windows[windows["split"] == "val"].copy()
    test = windows[windows["split"] == "test"].copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(ACTIVITY_ORDER)

    x_train = train[features]
    y_train = label_encoder.transform(train["activity_label"])

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=seed,
        verbose=False,
    )
    model = Pipeline([("scaler", StandardScaler()), ("model", mlp)])
    model.fit(x_train, y_train)
    fitted_mlp: MLPClassifier = model.named_steps["model"]

    metrics = pd.DataFrame(
        [
            evaluate_split("train", model, train[features], train["activity_label"], label_encoder, out_dir),
            evaluate_split("val", model, val[features], val["activity_label"], label_encoder, out_dir),
            evaluate_split("test", model, test[features], test["activity_label"], label_encoder, out_dir),
        ]
    )
    metrics.to_csv(out_dir / "mlp_metrics.csv", index=False)

    save_training_curve(fitted_mlp, out_dir / "mlp_training_loss.png")
    save_validation_curve(fitted_mlp, out_dir / "mlp_internal_validation_accuracy.png")
    if save_model:
        joblib.dump(model, out_dir / "mlp_pipeline.joblib")
        joblib.dump(label_encoder, out_dir / "label_encoder.joblib")

    lines = [
        "# Deep Learning Neural Network Baseline",
        "",
        f"Window feature table: `{windows_path}`",
        "",
        "This is a CPU multi-layer perceptron baseline trained on the existing 5 second statistical window features. PyTorch and TensorFlow were not installed in the current environment, so this baseline uses `sklearn.neural_network.MLPClassifier` with ReLU hidden layers and the Adam optimizer. The split remains subject-wise, matching the classical baselines.",
        "",
        "## Architecture",
        "",
        f"- Input features: {len(features)} handcrafted window features",
        f"- Hidden layers: {hidden_layers}",
        "- Activation: ReLU",
        "- Optimizer: Adam",
        f"- Batch size: {batch_size}",
        f"- Learning rate: {learning_rate}",
        f"- L2 penalty alpha: {alpha}",
        f"- Max epochs: {max_iter}",
        f"- Epochs run: {fitted_mlp.n_iter_}",
        "- Early stopping: enabled with 15% internal split from train subjects only",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Saved outputs",
        "",
        f"- Metrics: `{out_dir / 'mlp_metrics.csv'}`",
        f"- Test classification report: `{out_dir / 'mlp_test_classification_report.csv'}`",
        f"- Test confusion matrix: `{out_dir / 'mlp_test_confusion_matrix.csv'}`",
        f"- Training loss curve: `{out_dir / 'mlp_training_loss.png'}`",
        f"- Internal validation curve: `{out_dir / 'mlp_internal_validation_accuracy.png'}`",
        "",
        "## Interpretation",
        "",
        "This is a neural-network baseline, not a tuned final model. It gives a sanity-check comparison against logistic regression, random forest, and LightGBM while preserving the leakage-safe held-out subject test protocol. A stronger next deep-learning step would train a 1D CNN or LSTM on raw `(time, x, y, z)` windows once a framework such as PyTorch or TensorFlow is available.",
        "",
        "Generated by `deep_baseline.py`.",
        "",
    ]
    if save_model:
        lines.insert(-4, f"- Saved model pipeline: `{out_dir / 'mlp_pipeline.joblib'}`")
        lines.insert(-4, f"- Saved label encoder: `{out_dir / 'label_encoder.joblib'}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural-network baseline on WISDM window features.")
    parser.add_argument("--windows", type=Path, default=Path("data/processed/windows/phone_accel_windows_5p0s_50overlap.csv.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/deep_baseline"))
    parser.add_argument("--report", type=Path, default=Path("deep_baseline_results.md"))
    parser.add_argument("--hidden-layers", default="128,64,32")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    run_deep_baseline(
        windows_path=args.windows,
        out_dir=args.out_dir,
        report_path=args.report,
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        seed=args.seed,
        save_model=args.save_model,
    )
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
