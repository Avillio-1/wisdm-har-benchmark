from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import config


EPS = 1e-12


def log(message: str) -> None:
    stamp = pd.Timestamp.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    log(f"{label} ...")
    try:
        yield
    finally:
        log(f"{label} finished in {time.perf_counter() - start:.1f}s")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def window_slug(window_seconds: float) -> str:
    return str(window_seconds).replace(".", "p")


def parse_float_list(value: str | None, default: Iterable[float]) -> list[float]:
    if not value:
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str | None, default: Iterable[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def artifact_paths(data_dir: Path, window_seconds: float) -> dict[str, Path]:
    slug = window_slug(window_seconds)
    prefix = f"phone_accel_gyro_{slug}s"
    return {
        "tensor": data_dir / f"{prefix}_windows.npz",
        "metadata": data_dir / f"{prefix}_metadata.parquet",
        "features": data_dir / f"{prefix}_features.parquet",
        "fold_subjects": data_dir / f"{prefix}_fold_subjects.csv",
        "fold_class_counts": data_dir / f"{prefix}_fold_class_counts.csv",
        "window_summary": data_dir / f"{prefix}_window_summary.csv",
    }


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    table = df.head(max_rows).copy() if max_rows else df.copy()
    if table.empty:
        return "_No rows._"
    table = table.fillna("")
    columns = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(col), *(len(row[i]) for row in rows)) if rows else len(col)
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_prepared_window(data_dir: Path, window_seconds: float) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = artifact_paths(data_dir, window_seconds)
    missing = [path for path in paths.values() if not path.exists() and path != paths["window_summary"]]
    if missing:
        raise FileNotFoundError("Missing prepared artifacts: " + ", ".join(str(path) for path in missing))
    payload = np.load(paths["tensor"], allow_pickle=True)
    x = payload["X"].astype(np.float32)
    metadata = pd.read_parquet(paths["metadata"]).sort_values("window_id").reset_index(drop=True)
    features = pd.read_parquet(paths["features"]).sort_values("window_id").reset_index(drop=True)
    fold_subjects = pd.read_csv(paths["fold_subjects"])
    if len(metadata) != len(x):
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match tensor windows ({len(x)}).")
    if len(features) != len(x):
        raise ValueError(f"Feature rows ({len(features)}) do not match tensor windows ({len(x)}).")
    return x, metadata, features, fold_subjects


def fold_masks(metadata: pd.DataFrame, fold_subjects: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    eval_subjects = set(fold_subjects.loc[fold_subjects["fold"].eq(fold), "subject_id"].astype(int))
    eval_mask = metadata["subject_id"].astype(int).isin(eval_subjects).to_numpy()
    train_mask = ~eval_mask
    overlap = set(metadata.loc[train_mask, "subject_id"].astype(int)) & set(metadata.loc[eval_mask, "subject_id"].astype(int))
    if overlap:
        raise ValueError(f"Subject leakage in fold {fold}: {sorted(overlap)}")
    return train_mask, eval_mask


def make_fold_subjects(metadata: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    groups = metadata["subject_id"].to_numpy()
    y = metadata["y"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    rows = []
    dummy_x = np.zeros((len(metadata), 1), dtype=np.float32)
    for fold, (_, eval_idx) in enumerate(splitter.split(dummy_x, y, groups), start=1):
        for subject_id in sorted(pd.Series(groups[eval_idx]).astype(int).unique()):
            rows.append({"fold": fold, "subject_id": int(subject_id)})
    return pd.DataFrame(rows)


def fold_class_counts(metadata: pd.DataFrame, fold_subjects: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for fold in sorted(fold_subjects["fold"].unique()):
        _, eval_mask = fold_masks(metadata, fold_subjects, int(fold))
        counts = (
            metadata.loc[eval_mask]
            .groupby("group_label", observed=True)
            .size()
            .rename("eval_windows")
            .reindex(config.GROUP_LABELS, fill_value=0)
            .reset_index()
            .rename(columns={"index": "group_label"})
        )
        counts.insert(0, "fold", int(fold))
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def normalize_sequence_train_only(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=(0, 1), keepdims=True)
    std = x[train_mask].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.squeeze(), std.squeeze()


def classification_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[int] | None = None,
    target_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    labels = list(labels if labels is not None else range(len(config.GROUP_LABELS)))
    target_names = list(target_names if target_names is not None else config.GROUP_LABELS)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index().rename(columns={"index": "class"})
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=target_names, columns=target_names)
    return report, cm, metrics


def summarize_fold_metrics(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["accuracy", "macro_f1", "weighted_f1"]
    rows = []
    for keys, group in metrics.groupby(group_cols, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["folds"] = int(group["fold"].nunique())
        row["eval_windows_total"] = int(group["eval_windows"].sum())
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_feature_columns(features: pd.DataFrame, include_frequency: bool) -> list[str]:
    cols = []
    for col in features.columns:
        if col in config.METADATA_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        if not include_frequency and "_freq_" in col:
            continue
        cols.append(col)
    return cols


def save_metric_bundle(
    out_dir: Path,
    stem: str,
    fold_metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    ensure_dirs(out_dir)
    fold_metrics.to_csv(out_dir / f"{stem}_fold_metrics.csv", index=False)
    per_class.to_csv(out_dir / f"{stem}_per_class.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
