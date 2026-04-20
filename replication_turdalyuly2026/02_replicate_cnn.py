from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import (
    classification_outputs,
    ensure_dirs,
    fold_masks,
    load_prepared_window,
    markdown_table,
    normalize_sequence_train_only,
    parse_float_list,
    parse_str_list,
    save_metric_bundle,
    summarize_fold_metrics,
    timed,
)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def write_missing_backend_report(reports_dir: Path) -> None:
    ensure_dirs(reports_dir)
    lines = [
        "# CNN Replication Not Run",
        "",
        "PyTorch is not installed in the active Python environment, so the calibration CNN was not trained.",
        "",
        "Install PyTorch, then rerun:",
        "",
        "```powershell",
        "python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 4.0",
        "```",
        "",
        "The classical feature benchmark can still be run with `03_train_feature_models.py`.",
        "",
    ]
    (reports_dir / "cnn_backend_missing.md").write_text("\n".join(lines), encoding="utf-8")


def write_missing_cuda_report(reports_dir: Path) -> None:
    import torch

    ensure_dirs(reports_dir)
    lines = [
        "# CNN Replication Not Run",
        "",
        "PyTorch is installed, but CUDA is not visible to PyTorch in the active Python environment.",
        "",
        "For a fair runtime setup, this replication package requires GPU execution by default. CPU execution is allowed only for smoke tests when `--allow-cpu` is passed explicitly.",
        "",
        "## CUDA Check",
        "",
        f"- `torch.__version__`: `{torch.__version__}`",
        f"- `torch.version.cuda`: `{torch.version.cuda}`",
        f"- `torch.cuda.is_available()`: `{torch.cuda.is_available()}`",
        "",
        "Install a CUDA-enabled PyTorch build and rerun:",
        "",
        "```powershell",
        "python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 4.0",
        "```",
        "",
    ]
    (reports_dir / "cnn_cuda_missing.md").write_text("\n".join(lines), encoding="utf-8")


def build_model(n_channels: int, n_classes: int):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
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
        nn.Flatten(),
        nn.Dropout(0.35),
        nn.Linear(128, 96),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(96, n_classes),
    )


def train_one_fold(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, eval_mask: np.ndarray, args: argparse.Namespace, fold: int):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed + fold)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is not available to PyTorch. Install a CUDA-enabled torch build, or pass --allow-cpu only for smoke tests.")
    x_norm, mean, std = normalize_sequence_train_only(x, train_mask)
    model = build_model(x.shape[2], len(config.GROUP_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_ds = TensorDataset(torch.from_numpy(x_norm[train_mask]), torch.from_numpy(y[train_mask]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb.transpose(1, 2))
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(xb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += len(xb)
        history_rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": total_loss / max(total_seen, 1),
                "train_accuracy": total_correct / max(total_seen, 1),
            }
        )

    model.eval()
    eval_ds = TensorDataset(torch.from_numpy(x_norm[eval_mask]), torch.from_numpy(y[eval_mask]))
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            logits = model(xb.to(device).transpose(1, 2))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return y_pred, pd.DataFrame(history_rows), mean, std, str(device)


def run_window_modality(window_seconds: float, modality: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, metadata, _, fold_subjects = load_prepared_window(args.data_dir, window_seconds)
    channel_idx = list(config.MODALITY_CHANNELS[modality])
    x = x[:, :, channel_idx]
    y = metadata["y"].to_numpy(dtype=np.int64)
    fold_metrics = []
    per_class_frames = []
    history_frames = []
    confusion_dir = args.results_dir / "cnn" / "confusion_matrices"
    norm_dir = args.results_dir / "cnn" / "normalization"
    ensure_dirs(confusion_dir, norm_dir)

    folds = sorted(fold_subjects["fold"].unique())
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    for fold in folds:
        with timed(f"CNN {modality} {window_seconds:.1f}s fold {fold}"):
            train_mask, eval_mask = fold_masks(metadata, fold_subjects, int(fold))
            y_pred, history, mean, std, device = train_one_fold(x, y, train_mask, eval_mask, args, int(fold))
        report, cm, metrics = classification_outputs(y[eval_mask], y_pred)
        cm.to_csv(confusion_dir / f"cnn_{modality}_{window_seconds:.1f}s_fold{fold}.csv")
        np.savez(norm_dir / f"cnn_{modality}_{window_seconds:.1f}s_fold{fold}_normalization.npz", mean=mean, std=std)
        report.insert(0, "fold", int(fold))
        report.insert(0, "modality", modality)
        report.insert(0, "window_seconds", window_seconds)
        per_class_frames.append(report)
        history.insert(0, "modality", modality)
        history.insert(0, "window_seconds", window_seconds)
        history_frames.append(history)
        fold_metrics.append(
            {
                "model": "cnn_replication",
                "modality": modality,
                "window_seconds": window_seconds,
                "fold": int(fold),
                "eval_subjects": int(metadata.loc[eval_mask, "subject_id"].nunique()),
                "eval_windows": int(eval_mask.sum()),
                "device": device,
                **metrics,
            }
        )

    fold_df = pd.DataFrame(fold_metrics)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    history.to_csv(args.results_dir / "cnn" / f"cnn_{modality}_{window_seconds:.1f}s_training_history.csv", index=False)
    summary = summarize_fold_metrics(fold_df, ["model", "modality", "window_seconds"])
    return fold_df, per_class, summary


def write_report(summary: pd.DataFrame, reports_dir: Path) -> None:
    ensure_dirs(reports_dir)
    if summary.empty:
        calibration = "No CNN summary was generated."
    else:
        primary = summary[summary["window_seconds"].eq(4.0)]
        if "modality" in primary.columns:
            primary = primary[primary["modality"].eq("accel_gyro")]
        if primary.empty:
            calibration = "The 4.0 s accel+gyro primary CNN benchmark was not run."
        else:
            value = float(primary.iloc[0]["macro_f1_mean"])
            delta = value - config.PAPER_TARGET_MACRO_F1_MEAN
            within = abs(delta) <= config.PAPER_TARGET_MACRO_F1_STD
            calibration = (
                f"4.0 s CNN macro-F1 mean = {value:.4f}; paper target = {config.PAPER_TARGET_MACRO_F1_MEAN:.4f} "
                f"+/- {config.PAPER_TARGET_MACRO_F1_STD:.4f}. Calibration within one reported std: {within}."
            )
    lines = [
        "# Turdalyuly 2026 CNN Calibration",
        "",
        calibration,
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
    ]
    (reports_dir / "cnn_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-style PyTorch CNN calibration on prepared Turdalyuly 2026 windows.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--window-sizes", default="4.0")
    parser.add_argument("--modalities", default="accel_gyro", help="Comma-separated modalities: accel,gyro,accel_gyro.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution only for smoke tests. Full academic replication should use CUDA.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None, help="Smoke-test helper. Full replication leaves this unset.")
    args = parser.parse_args()

    ensure_dirs(args.results_dir / "cnn", args.reports_dir)
    if not torch_available():
        write_missing_backend_report(args.reports_dir)
        print(f"PyTorch is not installed; wrote {args.reports_dir / 'cnn_backend_missing.md'}")
        return
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        write_missing_cuda_report(args.reports_dir)
        print(f"CUDA is not visible to PyTorch; wrote {args.reports_dir / 'cnn_cuda_missing.md'}")
        return

    all_fold = []
    all_per_class = []
    all_summary = []
    modalities = parse_str_list(args.modalities, ["accel_gyro"])
    unknown_modalities = sorted(set(modalities) - set(config.MODALITY_CHANNELS))
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {unknown_modalities}. Choose from {sorted(config.MODALITY_CHANNELS)}")
    for window_seconds in parse_float_list(args.window_sizes, [4.0]):
        for modality in modalities:
            fold_df, per_class, summary = run_window_modality(window_seconds, modality, args)
            all_fold.append(fold_df)
            all_per_class.append(per_class)
            all_summary.append(summary)
    fold_metrics = pd.concat(all_fold, ignore_index=True)
    per_class = pd.concat(all_per_class, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    save_metric_bundle(args.results_dir / "cnn", "cnn_replication", fold_metrics, per_class, summary)
    write_report(summary, args.reports_dir)
    print(f"Wrote {args.reports_dir / 'cnn_calibration_report.md'}")


if __name__ == "__main__":
    main()
