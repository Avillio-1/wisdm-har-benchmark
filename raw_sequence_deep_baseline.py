from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.wisdm import markdown_table


def available_backends() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def backend_status() -> pd.DataFrame:
    rows = []
    for backend, installed in available_backends().items():
        usable = False
        error = ""
        if installed:
            try:
                importlib.import_module("tensorflow" if backend == "tensorflow" else "torch")
                usable = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        rows.append({"backend": backend, "installed": installed, "usable": usable, "error": error})
    return pd.DataFrame(rows)


def choose_backend(requested: str) -> str:
    status = backend_status()
    if requested != "auto":
        row = status[status["backend"] == requested].iloc[0]
        if not bool(row["usable"]):
            raise RuntimeError(
                f"Requested backend {requested!r} is not usable. {row['error']} "
                "Install TensorFlow or PyTorch, then rerun this script."
            )
        return requested
    usable = status[status["usable"]]
    if "tensorflow" in usable["backend"].to_list():
        return "tensorflow"
    if "torch" in usable["backend"].to_list():
        return "torch"
    raise RuntimeError("Neither TensorFlow nor PyTorch is usable in this Python environment. Install or fix one backend to train the CNN/LSTM model.")


def classification_tables(y_true: np.ndarray, y_pred: np.ndarray, label_codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    n_classes = len(label_codes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1

    rows = []
    f1s = []
    weighted_f1_num = 0.0
    total = int(cm.sum())
    for idx, label in enumerate(label_codes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        weighted_f1_num += f1 * support
        rows.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": int(support),
            }
        )
    accuracy = float(np.trace(cm) / total) if total else 0.0
    metrics = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(weighted_f1_num / total) if total else 0.0,
    }
    return pd.DataFrame(rows), pd.DataFrame(cm, index=label_codes, columns=label_codes), metrics


def load_sequence_data(npz_path: Path, metadata_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    payload = np.load(npz_path, allow_pickle=True)
    x = payload["X"].astype(np.float32)
    y = payload["y"].astype(np.int64)
    label_codes = [str(label) for label in payload["label_codes"].tolist()]
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(x):
        raise ValueError("Metadata rows must match X windows.")
    return x, y, metadata, label_codes


def normalize_from_train(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=(0, 1), keepdims=True)
    std = x[train_mask].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.squeeze(), std.squeeze()


def build_tf_model(input_shape: tuple[int, int], n_classes: int, model_type: str, learning_rate: float) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape)
    if model_type == "cnn":
        x = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif model_type == "lstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        raise ValueError("model_type must be 'cnn' or 'lstm'.")

    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_with_tensorflow(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
    out_dir: Path,
    label_codes: list[str],
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    model = build_tf_model(x.shape[1:], len(label_codes), args.model, args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2)),
    ]
    history = model.fit(
        x[masks["train"]],
        y[masks["train"]],
        validation_data=(x[masks["val"]], y[masks["val"]]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    metric_rows = []
    for split, mask in masks.items():
        probabilities = model.predict(x[mask], batch_size=args.batch_size, verbose=0)
        y_pred = probabilities.argmax(axis=1)
        report, cm, metrics = classification_tables(y[mask], y_pred, label_codes)
        report.to_csv(out_dir / f"{args.model}_{split}_classification_report.csv", index=False)
        cm.to_csv(out_dir / f"{args.model}_{split}_confusion_matrix.csv")
        metric_rows.append({"split": split, "windows": int(mask.sum()), **{k: round(v, 4) for k, v in metrics.items()}})
    model.save(out_dir / f"{args.model}_tensorflow_model")
    return pd.DataFrame(metric_rows), model, history.history


def train_with_torch(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
    out_dir: Path,
    label_codes: list[str],
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")

    class CnnModel(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, n_classes))

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return self.head(self.net(batch.transpose(1, 2)))

    class LstmModel(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=3, hidden_size=96, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, n_classes))

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.lstm(batch)
            encoded = torch.cat([hidden[-2], hidden[-1]], dim=1)
            return self.head(encoded)

    model: nn.Module = CnnModel(len(label_codes)) if args.model == "cnn" else LstmModel(len(label_codes))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[masks["train"]]), torch.from_numpy(y[masks["train"]])), batch_size=args.batch_size, shuffle=True)
    val_x = torch.from_numpy(x[masks["val"]]).to(device)
    val_y = torch.from_numpy(y[masks["val"]]).to(device)
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_accuracy = float((val_logits.argmax(dim=1) == val_y).float().mean().item())
        history["loss"].append(total_loss / total_seen)
        history["accuracy"].append(total_correct / total_seen)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"epoch {epoch + 1}/{args.epochs} loss={history['loss'][-1]:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metric_rows = []
    model.eval()
    for split, mask in masks.items():
        loader = DataLoader(TensorDataset(torch.from_numpy(x[mask]), torch.from_numpy(y[mask])), batch_size=args.batch_size)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        report, cm, metrics = classification_tables(y[mask], y_pred, label_codes)
        report.to_csv(out_dir / f"{args.model}_{split}_classification_report.csv", index=False)
        cm.to_csv(out_dir / f"{args.model}_{split}_confusion_matrix.csv")
        metric_rows.append({"split": split, "windows": int(mask.sum()), **{k: round(v, 4) for k, v in metrics.items()}})
    torch.save(model.state_dict(), out_dir / f"{args.model}_torch_state_dict.pt")
    return pd.DataFrame(metric_rows), model, history


def save_history(history: dict[str, Any], out_path: Path) -> None:
    hist = pd.DataFrame(history)
    hist.to_csv(out_path.with_suffix(".csv"), index=False)
    plt.figure(figsize=(8, 4.5))
    if "loss" in hist:
        plt.plot(hist.index + 1, hist["loss"], label="train loss")
    if "val_loss" in hist:
        plt.plot(hist.index + 1, hist["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Raw sequence model training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_backend_missing_report(out_dir: Path, report_path: Path, error: Exception) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    status = backend_status()
    lines = [
        "# Raw Sequence Deep Learning Baseline",
        "",
        "Training was not run because no usable deep-learning backend was available in the active Python environment.",
        "",
        "## Backend check",
        "",
        markdown_table(status),
        "",
        "## Error",
        "",
        f"`{type(error).__name__}: {error}`",
        "",
        "## Ready-to-run command",
        "",
        "```powershell",
        "python sequence_windowing.py",
        "python raw_sequence_deep_baseline.py --model cnn --backend auto",
        "python raw_sequence_deep_baseline.py --model lstm --backend auto",
        "```",
        "",
        "Install PyTorch or TensorFlow in the Python environment used to run the script, then rerun one of the commands above.",
        "",
        "Generated by `raw_sequence_deep_baseline.py`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        backend = choose_backend(args.backend)
    except Exception as exc:
        write_backend_missing_report(out_dir, args.report, exc)
        raise

    x, y, metadata, label_codes = load_sequence_data(args.sequences, args.metadata)
    masks = {split: metadata["split"].eq(split).to_numpy() for split in ["train", "val", "test"]}
    x, train_mean, train_std = normalize_from_train(x, masks["train"])
    np.savez(out_dir / "normalization_stats.npz", mean=train_mean, std=train_std)

    if backend == "tensorflow":
        metrics, model_obj, history = train_with_tensorflow(x, y, masks, args, out_dir, label_codes)
    else:
        metrics, model_obj, history = train_with_torch(x, y, masks, args, out_dir, label_codes)

    metrics.to_csv(out_dir / f"{args.model}_{backend}_metrics.csv", index=False)
    save_history(history, out_dir / f"{args.model}_{backend}_training_curve.png")

    lines = [
        "# Raw Sequence Deep Learning Baseline",
        "",
        f"Backend: `{backend}`",
        f"Model: `{args.model}`",
        f"Sequence tensor: `{args.sequences}`",
        "",
        "This model trains directly on raw `(time, x, y, z)` windows rather than handcrafted statistical features. Windows keep the existing subject-wise split, and normalization statistics are fit from train windows only.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Saved outputs",
        "",
        f"- Metrics: `{out_dir / f'{args.model}_{backend}_metrics.csv'}`",
        f"- Training history CSV: `{out_dir / f'{args.model}_{backend}_training_curve.csv'}`",
        f"- Training curve: `{out_dir / f'{args.model}_{backend}_training_curve.png'}`",
        f"- Test classification report: `{out_dir / f'{args.model}_test_classification_report.csv'}`",
        f"- Test confusion matrix: `{out_dir / f'{args.model}_test_confusion_matrix.csv'}`",
        f"- Normalization stats: `{out_dir / 'normalization_stats.npz'}`",
        "",
        "Generated by `raw_sequence_deep_baseline.py`.",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 1D CNN or LSTM on raw WISDM sequence windows.")
    parser.add_argument("--sequences", type=Path, default=Path("data/processed/sequences/phone_accel_rawseq_5p0s_50overlap.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/sequences/phone_accel_rawseq_5p0s_50overlap_metadata.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/raw_sequence_deep_baseline"))
    parser.add_argument("--report", type=Path, default=Path("raw_sequence_deep_baseline_results.md"))
    parser.add_argument("--backend", choices=["auto", "tensorflow", "torch"], default="auto")
    parser.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
