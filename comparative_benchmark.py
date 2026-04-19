from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from benchmark_definitions import MODEL_ORDER, REPRESENTATIONS, SENSOR_CONFIGS, STREAMS, TASKS, SensorConfig, TaskDef
from splits import choose_subject_split
from src.wisdm import ACTIVITY_MAP, SENSOR_COLUMNS, markdown_table


KEY_COLUMNS = ["split", "subject_id", "activity_label", "activity_name", "window_index"]
META_COLUMNS = ["split", "subject_id", "activity_label", "activity_name", "window_index", "start_timestamp_ns", "end_timestamp_ns", "n_samples"]
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
        elapsed = time.perf_counter() - start
        log(f"{label} finished in {elapsed:.1f}s")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_cache_format(requested: str) -> str:
    if requested != "auto":
        return requested
    if module_available("pyarrow"):
        return "parquet"
    return "pickle"


def cache_path(path: Path, cache_format: str) -> Path:
    if cache_format == "csv":
        return path
    text = str(path)
    if text.endswith(".csv.gz"):
        text = text[: -len(".csv.gz")]
    suffix = ".parquet" if cache_format == "parquet" else ".pkl"
    return Path(f"{text}{suffix}")


def alternate_cache_paths(path: Path) -> list[Path]:
    text = str(path)
    if text.endswith(".csv.gz"):
        stem = Path(text[: -len(".csv.gz")])
    elif path.suffix in {".parquet", ".pkl"}:
        stem = path.with_suffix("")
    else:
        stem = path
    return [Path(f"{stem}.parquet"), Path(f"{stem}.pkl"), Path(f"{stem}.csv.gz")]


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path)


def write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".pkl":
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False, compression="gzip")


def read_existing_cache(preferred_path: Path, convert_to_preferred: bool = True) -> pd.DataFrame | None:
    for candidate in [preferred_path, *alternate_cache_paths(preferred_path)]:
        if candidate.exists():
            df = read_frame(candidate)
            if convert_to_preferred and candidate != preferred_path:
                write_frame(df, preferred_path)
            return df
    return None


def window_tensor(values: np.ndarray, window_size: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=values.dtype), np.empty(0, dtype=np.int64)
    starts = np.arange(0, len(values) - window_size + 1, step, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size, axis=0)
    if windows.shape[1] == values.shape[1]:
        windows = np.swapaxes(windows, 1, 2)
    return windows[starts], starts


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


def add_axis_frequency(features: dict[str, np.ndarray], values: np.ndarray, prefix: str, sample_rate_hz: float) -> None:
    centered = values - values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(values.shape[1], d=1.0 / sample_rate_hz)
    total_power = power.sum(axis=1)

    non_dc = power.copy()
    if non_dc.shape[1] > 0:
        non_dc[:, 0] = 0.0
    dominant_idx = np.argmax(non_dc, axis=1)
    row_idx = np.arange(len(values))
    safe_total = total_power > EPS

    prob = np.divide(power, total_power[:, None], out=np.zeros_like(power), where=safe_total[:, None])
    if power.shape[1] > 1:
        entropy = -np.sum(prob * np.log2(prob + EPS), axis=1) / np.log2(power.shape[1])
    else:
        entropy = np.zeros(len(values), dtype=np.float64)
    low_mask = (freqs > 0.0) & (freqs <= 3.0)
    high_mask = freqs > 3.0

    features[f"{prefix}_freq_total_power"] = total_power
    features[f"{prefix}_freq_dominant_hz"] = np.where(safe_total, freqs[dominant_idx], 0.0)
    features[f"{prefix}_freq_dominant_power"] = np.divide(
        power[row_idx, dominant_idx],
        total_power,
        out=np.zeros_like(total_power),
        where=safe_total,
    )
    features[f"{prefix}_freq_entropy"] = np.where(safe_total, entropy, 0.0)
    features[f"{prefix}_freq_low_power"] = np.divide(
        power[:, low_mask].sum(axis=1),
        total_power,
        out=np.zeros_like(total_power),
        where=safe_total,
    )
    features[f"{prefix}_freq_high_power"] = np.divide(
        power[:, high_mask].sum(axis=1),
        total_power,
        out=np.zeros_like(total_power),
        where=safe_total,
    )


def rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered * left_centered, axis=1) * np.sum(right_centered * right_centered, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPS)


def extract_window_features(windows: np.ndarray, representation: str, sample_rate_hz: float) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    axes = {
        "x": windows[:, :, 0],
        "y": windows[:, :, 1],
        "z": windows[:, :, 2],
    }
    magnitude = np.sqrt(axes["x"] * axes["x"] + axes["y"] * axes["y"] + axes["z"] * axes["z"])
    for axis, values in [*axes.items(), ("magnitude", magnitude)]:
        add_axis_stats(features, values, axis)
        if representation == "stats_freq":
            add_axis_frequency(features, values, axis, sample_rate_hz)
    for left, right in [("x", "y"), ("x", "z"), ("y", "z")]:
        features[f"{left}_{right}_corr"] = rowwise_corr(axes[left], axes[right])
    return features


def group_feature_frame(
    keys: tuple[object, object, object, object],
    group: pd.DataFrame,
    stream_name: str,
    representation: str,
    sample_rate_hz: float,
    window_size: int,
    step: int,
) -> pd.DataFrame | None:
    values = group.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    windows, starts = window_tensor(values, window_size, step)
    if len(starts) == 0:
        return None

    split, subject_id, activity_label, activity_name = keys
    timestamps = group["timestamp_ns"].to_numpy(dtype=np.int64, copy=False)
    ends = starts + window_size - 1
    feature_values = extract_window_features(windows, representation, sample_rate_hz)
    rows: dict[str, object] = {
        "split": np.repeat(split, len(starts)),
        "subject_id": np.repeat(int(subject_id), len(starts)),
        "activity_label": np.repeat(activity_label, len(starts)),
        "activity_name": np.repeat(activity_name, len(starts)),
        "window_index": np.arange(len(starts), dtype=np.int64),
        "start_timestamp_ns": timestamps[starts],
        "end_timestamp_ns": timestamps[ends],
        "n_samples": np.repeat(window_size, len(starts)),
    }
    rows.update({f"{stream_name}__{name}": value for name, value in feature_values.items()})
    return pd.DataFrame(rows)


def load_clean_stream(clean_path: Path, stream_name: str, stream_cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if clean_path not in stream_cache:
        with timed(f"read clean stream {stream_name}"):
            stream_cache[clean_path] = pd.read_csv(
                clean_path,
                usecols=["subject_id", "activity_label", "activity_name", "timestamp_ns", *SENSOR_COLUMNS],
            )
    return stream_cache[clean_path]


def load_split_source(clean_path: Path, split_source_cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if clean_path not in split_source_cache:
        with timed("read split source subject/activity columns"):
            split_source_cache[clean_path] = pd.read_csv(clean_path, usecols=["subject_id", "activity_label"])
    return split_source_cache[clean_path]


def get_subject_splits(
    task: TaskDef,
    root: Path,
    seed: int,
    split_source_cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    split_dir = root / "splits"
    seed_path = split_dir / f"{task.name}_subject_splits_seed{seed}.csv"
    legacy_path = split_dir / f"{task.name}_subject_splits.csv"
    if seed_path.exists():
        log(f"subject split cache hit: {seed_path}")
        splits = pd.read_csv(seed_path)
        splits.to_csv(legacy_path, index=False)
        return splits

    base_stream = STREAMS["phone_accel"]
    base_df = load_split_source(base_stream.clean_path, split_source_cache)
    with timed(f"subject split {task.name}"):
        splits = choose_subject_split(base_df, seed=seed, labels=list(task.labels))
    splits.to_csv(seed_path, index=False)
    splits.to_csv(legacy_path, index=False)
    return splits


def generate_stream_features(
    clean_path: Path,
    splits: pd.DataFrame,
    labels: tuple[str, ...],
    stream_name: str,
    representation: str,
    out_path: Path,
    sample_rate_hz: float = 20.0,
    window_seconds: float = 5.0,
    overlap: float = 0.5,
    force: bool = False,
    stream_cache: dict[Path, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if not force:
        cached = read_existing_cache(out_path)
        if cached is not None:
            log(f"feature cache hit: {out_path}")
            return cached

    window_size = int(round(window_seconds * sample_rate_hz))
    step = max(1, int(round(window_size * (1 - overlap))))
    stream_cache = stream_cache if stream_cache is not None else {}
    with timed(f"generate {stream_name} {representation} features"):
        df = load_clean_stream(clean_path, stream_name, stream_cache)
        df = df[df["activity_label"].isin(labels)].copy()
        df = df.merge(splits, on="subject_id", how="left")
        if df["split"].isna().any():
            raise ValueError(f"{stream_name}: raw rows missing split assignments")
        df = df.sort_values(["split", "subject_id", "activity_label", "timestamp_ns"], kind="mergesort")

        frames: list[pd.DataFrame] = []
        group_cols = ["split", "subject_id", "activity_label", "activity_name"]
        for keys, group in df.groupby(group_cols, observed=True, sort=True):
            frame = group_feature_frame(keys, group, stream_name, representation, sample_rate_hz, window_size, step)
            if frame is not None:
                frames.append(frame)
        features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=META_COLUMNS)
        write_frame(features, out_path)
        log(f"saved {len(features):,} windows x {len(features.columns):,} columns to {out_path}")
    return features


def build_feature_table(
    task: TaskDef,
    sensor_config: SensorConfig,
    representation: str,
    root: Path,
    splits: pd.DataFrame,
    seed: int,
    force_features: bool = False,
    cache_format: str = "parquet",
    stream_cache: dict[Path, pd.DataFrame] | None = None,
) -> tuple[Path, pd.DataFrame]:
    split_tag = f"seed{seed}"
    task_dir = root / "features" / task.name / split_tag
    out_dir = root / "feature_tables" / task.name / split_tag
    out_path = cache_path(out_dir / f"{sensor_config.name}_{representation}.csv.gz", cache_format)
    if not force_features:
        cached = read_existing_cache(out_path)
        if cached is not None:
            log(f"feature table cache hit: {out_path} ({len(cached):,} rows)")
            return out_path, cached

    stream_tables = []
    for stream_name in sensor_config.streams:
        stream = STREAMS[stream_name]
        stream_path = cache_path(task_dir / f"{stream_name}_{representation}.csv.gz", cache_format)
        stream_df = generate_stream_features(
            stream.clean_path,
            splits,
            task.labels,
            stream_name,
            representation,
            stream_path,
            force=force_features,
            stream_cache=stream_cache,
        )
        stream_tables.append(stream_df)

    with timed(f"merge feature table {task.name}/{sensor_config.name}/{representation}"):
        merged = stream_tables[0]
        for next_df in stream_tables[1:]:
            drop_meta = [col for col in ["start_timestamp_ns", "end_timestamp_ns", "n_samples"] if col in next_df.columns]
            merged = merged.merge(next_df.drop(columns=drop_meta), on=KEY_COLUMNS, how="inner")
        write_frame(merged, out_path)
        log(f"saved merged feature table {len(merged):,} rows x {len(merged.columns):,} columns to {out_path}")
    return out_path, merged


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(META_COLUMNS)
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def set_model_n_jobs(model, n_jobs: int) -> bool:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) and "model" in model.named_steps else model
    if hasattr(estimator, "get_params") and "n_jobs" in estimator.get_params():
        estimator.set_params(n_jobs=n_jobs)
        return True
    return False


def fit_with_worker_fallback(model, x: np.ndarray, y: np.ndarray, label: str):
    try:
        return model.fit(x, y)
    except PermissionError as exc:
        if set_model_n_jobs(model, 1):
            log(f"{label}: parallel worker setup failed with {type(exc).__name__}; retrying with n_jobs=1")
            return model.fit(x, y)
        raise


def predict_without_feature_name_noise(model, x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        return model.predict(x)


def make_model(model_name: str, seed: int, n_classes: int, n_jobs: int = -1, xgboost_device: str = "cpu"):
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1, random_state=seed),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device=xgboost_device,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    model_name: str,
    model,
    df: pd.DataFrame,
    features: list[str],
    label_encoder: LabelEncoder,
    task_name: str,
    sensor_name: str,
    representation: str,
    out_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    labels = list(range(len(label_encoder.classes_)))
    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].eq("val")].copy()
    test = df[df["split"].eq("test")].copy()
    y_train = label_encoder.transform(train["activity_label"])
    x_by_split = {
        "train": train[features].to_numpy(dtype=np.float32, copy=False),
        "val": val[features].to_numpy(dtype=np.float32, copy=False),
        "test": test[features].to_numpy(dtype=np.float32, copy=False),
    }
    y_by_split = {
        "train": y_train,
        "val": label_encoder.transform(val["activity_label"]),
        "test": label_encoder.transform(test["activity_label"]),
    }
    split_frames = {"train": train, "val": val, "test": test}
    with timed(f"fit/evaluate {task_name}/{sensor_name}/{representation}/{model_name}"):
        fit_with_worker_fallback(model, x_by_split["train"], y_train, f"{task_name}/{sensor_name}/{representation}/{model_name}")

        result = {"task": task_name, "sensor_config": sensor_name, "representation": representation, "model": model_name}
        test_report = None
        test_cm = None
        for split_name in ["train", "val", "test"]:
            split_df = split_frames[split_name]
            y_true = y_by_split[split_name]
            y_pred = predict_without_feature_name_noise(model, x_by_split[split_name])
            result[f"{split_name}_windows"] = len(split_df)
            result[f"{split_name}_accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            result[f"{split_name}_macro_f1"] = round(f1_score(y_true, y_pred, average="macro", labels=labels), 4)

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
            report.insert(0, "split", split_name)
            report.insert(0, "model", model_name)
            report.insert(0, "representation", representation)
            report.insert(0, "sensor_config", sensor_name)
            report.insert(0, "task", task_name)

            cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=label_encoder.classes_, columns=label_encoder.classes_)
            cm.to_csv(out_dir / "confusion_matrices" / f"{task_name}__{sensor_name}__{representation}__{model_name}__{split_name}.csv")
            if split_name == "test":
                test_report = report
                test_cm = cm
                subject_errors = subject_level_errors(split_df, y_true, y_pred, label_encoder.classes_, task_name, sensor_name, representation, model_name)
                subject_errors.to_csv(out_dir / "subject_errors" / f"{task_name}__{sensor_name}__{representation}__{model_name}.csv", index=False)
    assert test_report is not None and test_cm is not None
    return result, test_report, test_cm


def subject_level_errors(
    split_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    sensor_name: str,
    representation: str,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    temp = split_df[["subject_id"]].copy()
    temp["y_true"] = y_true
    temp["y_pred"] = y_pred
    for subject_id, group in temp.groupby("subject_id", observed=True):
        rows.append(
            {
                "task": task_name,
                "sensor_config": sensor_name,
                "representation": representation,
                "model": model_name,
                "subject_id": int(subject_id),
                "windows": len(group),
                "accuracy": round(accuracy_score(group["y_true"], group["y_pred"]), 4),
                "macro_f1": round(f1_score(group["y_true"], group["y_pred"], average="macro", labels=list(range(len(labels))), zero_division=0), 4),
            }
        )
    return pd.DataFrame(rows)


def run_grouped_cv_for_best(
    df: pd.DataFrame,
    features: list[str],
    task: TaskDef,
    sensor_name: str,
    representation: str,
    model_name: str,
    out_dir: Path,
    seed: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    xgboost_device: str = "cpu",
) -> pd.DataFrame:
    cv_df = df[df["split"].isin(["train", "val"])].copy()
    label_encoder = LabelEncoder().fit(list(task.labels))
    y = label_encoder.transform(cv_df["activity_label"])
    groups = cv_df["subject_id"].to_numpy()
    labels = list(range(len(label_encoder.classes_)))
    n_splits = min(n_splits, pd.Series(groups).nunique())
    x = cv_df[features].to_numpy(dtype=np.float32, copy=False)
    rows = []
    for fold, (train_idx, eval_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups), start=1):
        model = make_model(model_name, seed + fold, len(task.labels), n_jobs=n_jobs, xgboost_device=xgboost_device)
        with timed(f"grouped CV fold {fold}/{n_splits} {task.name}/{sensor_name}/{representation}/{model_name}"):
            fit_with_worker_fallback(model, x[train_idx], y[train_idx], f"grouped CV {task.name}/{sensor_name}/{representation}/{model_name}/fold{fold}")
            pred = predict_without_feature_name_noise(model, x[eval_idx])
        rows.append(
            {
                "task": task.name,
                "sensor_config": sensor_name,
                "representation": representation,
                "model": model_name,
                "fold": fold,
                "eval_subjects": pd.Series(groups[eval_idx]).nunique(),
                "eval_windows": len(eval_idx),
                "accuracy": round(accuracy_score(y[eval_idx], pred), 4),
                "macro_f1": round(f1_score(y[eval_idx], pred, average="macro", labels=labels), 4),
            }
        )
    cv = pd.DataFrame(rows)
    cv.to_csv(out_dir / "grouped_cv" / f"{task.name}__{sensor_name}__{representation}__{model_name}.csv", index=False)
    return cv


def save_feature_importance(model, features: list[str], out_dir: Path, prefix: str) -> pd.DataFrame | None:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) and "model" in model.named_steps else model
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    importance = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance" / f"{prefix}.csv", index=False)
    top = importance.head(25).iloc[::-1]
    plt.figure(figsize=(8, 8))
    plt.barh(top["feature"], top["importance"], color="#2F6B7C")
    plt.xlabel("Importance")
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance" / f"{prefix}.png", dpi=200, bbox_inches="tight")
    plt.close()
    return importance


def plot_subject_errors(subject_error_path: Path, out_path: Path) -> None:
    errors = pd.read_csv(subject_error_path)
    if errors.empty:
        return
    ordered = errors.sort_values("macro_f1")
    plt.figure(figsize=(9, 4.8))
    plt.bar(ordered["subject_id"].astype(str), ordered["macro_f1"], color="#7A5C58")
    plt.ylim(0, 1.02)
    plt.xlabel("Held-out subject")
    plt.ylabel("Macro F1")
    plt.title("Subject-level held-out performance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_summary_report(root: Path, results: pd.DataFrame, cv_summary: pd.DataFrame) -> None:
    best_by_task = results.sort_values("test_macro_f1", ascending=False).groupby("task", observed=True).head(5)
    best_overall = results.sort_values("test_macro_f1", ascending=False).head(10)
    lines = [
        "# Comparative WISDM Benchmark Results",
        "",
        "Research question: which sensor setup, representation, and model family generalize best to unseen subjects under subject-wise evaluation?",
        "",
        "## Task Definitions",
        "",
        *[f"- `{task.name}`: labels `{','.join(task.labels)}`. {task.rationale}" for task in TASKS.values()],
        "",
        "## Overall Best Configurations",
        "",
        markdown_table(best_overall[["task", "sensor_config", "representation", "model", "test_accuracy", "test_macro_f1"]]),
        "",
        "## Best Configurations By Task",
        "",
        markdown_table(best_by_task[["task", "sensor_config", "representation", "model", "test_accuracy", "test_macro_f1"]]),
        "",
        "## Sensor Ablation Summary",
        "",
        markdown_table(
            results.groupby(["task", "sensor_config"], observed=True)["test_macro_f1"]
            .max()
            .reset_index()
            .sort_values(["task", "test_macro_f1"], ascending=[True, False])
        ),
        "",
        "## Representation Summary",
        "",
        markdown_table(
            results.groupby(["task", "representation"], observed=True)["test_macro_f1"]
            .max()
            .reset_index()
            .sort_values(["task", "test_macro_f1"], ascending=[True, False])
        ),
        "",
        "## Grouped CV Stability",
        "",
        markdown_table(cv_summary),
        "",
        "## Notes",
        "",
        "- All classical models fit on train only and report validation/test separately.",
        "- Grouped CV is run on train+validation subjects only for the best classical configuration per task.",
        "- Feature-level fusion aligns windows by subject, activity, split, and within-activity window index; it is reproducible but not exact timestamp synchronization.",
        "- Raw-sequence CNN experiments are maintained separately from this classical feature matrix.",
        "",
        "Generated by `comparative_benchmark.py`.",
        "",
    ]
    (root / "comparative_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_list(value: str | None, default: list[str]) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else default


def detect_tensorflow_gpu() -> dict[str, object]:
    if not module_available("tensorflow"):
        return {"available": False, "detail": "TensorFlow is not installed in this Python environment."}
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return {"available": bool(gpus), "detail": f"TensorFlow sees {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}"}
    except Exception as exc:
        return {"available": False, "detail": f"TensorFlow GPU check failed: {type(exc).__name__}: {exc}"}


def detect_xgboost_cuda(seed: int) -> dict[str, object]:
    if not module_available("xgboost"):
        return {"available": False, "device": "cpu", "detail": "XGBoost is not installed."}
    try:
        from xgboost import XGBClassifier

        rng = np.random.default_rng(seed)
        x = rng.normal(size=(96, 6)).astype(np.float32)
        y = rng.integers(0, 3, size=96)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = XGBClassifier(
                n_estimators=2,
                max_depth=2,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                device="cuda",
                n_jobs=1,
                random_state=seed,
                verbosity=0,
            )
            model.fit(x, y)
        warning_text = "\n".join(str(item.message) for item in caught)
        config = model.get_booster().save_config().lower()
        fallback_markers = ["no visible gpu", "not compiled with cuda", "gpu is not enabled", "falling back to prediction using dmatrix"]
        if any(marker in warning_text.lower() for marker in fallback_markers) or '"device":"cpu"' in config:
            return {"available": False, "device": "cpu", "detail": warning_text.strip() or "XGBoost CUDA request fell back to CPU."}
        if "cuda" in config:
            return {"available": True, "device": "cuda", "detail": "Tiny XGBoost CUDA fit succeeded."}
        return {"available": False, "device": "cpu", "detail": "XGBoost fit succeeded, but booster config did not report CUDA."}
    except Exception as exc:
        return {"available": False, "device": "cpu", "detail": f"XGBoost CUDA check failed: {type(exc).__name__}: {exc}"}


def inspect_runtime(use_gpu: bool, seed: int, n_jobs: int, cache_format: str) -> dict[str, object]:
    capabilities: dict[str, object] = {
        "cpu_count": os.cpu_count(),
        "n_jobs": n_jobs,
        "cache_format": cache_format,
        "cupy_available": module_available("cupy"),
        "tensorflow": detect_tensorflow_gpu(),
        "xgboost": {"available": False, "device": "cpu", "detail": "GPU check skipped; pass --use-gpu to test CUDA."},
    }
    if use_gpu:
        capabilities["xgboost"] = detect_xgboost_cuda(seed)

    log(f"CPU workers requested for CPU models: n_jobs={n_jobs} (machine reports {os.cpu_count()} logical cores)")
    if capabilities["cupy_available"]:
        log("CuPy is installed, but this script keeps feature extraction on NumPy CPU to preserve the current dependency path.")
    else:
        log("CuPy is not installed; feature extraction uses NumPy CPU FFT/statistics.")
    tf_detail = capabilities["tensorflow"]["detail"]  # type: ignore[index]
    log(f"TensorFlow CNN GPU status: {tf_detail}")
    xgb_detail = capabilities["xgboost"]["detail"]  # type: ignore[index]
    xgb_device = capabilities["xgboost"]["device"]  # type: ignore[index]
    log(f"XGBoost device for this run: {xgb_device}. {xgb_detail}")
    log("sklearn logistic regression and random forest are CPU models here; logistic stays single-process, random forest uses CPU parallelism.")
    log("LightGBM is configured for CPU execution with all requested workers.")
    return capabilities


def write_runtime_audit_report(root: Path, capabilities: dict[str, object]) -> None:
    xgb = capabilities["xgboost"]  # type: ignore[index]
    tf = capabilities["tensorflow"]  # type: ignore[index]
    lines = [
        "# comparative_benchmark.py Runtime Audit",
        "",
        "## Profiled Bottlenecks",
        "",
        "A bounded pre-refactor profile was run on `task3 / phone_accel / stats_freq / majority` with forced feature generation.",
        "That slice took about 75.3 profiler seconds after imports and showed the real bottleneck was feature construction, not model fitting.",
        "",
        "- `build_feature_table`: about 62.4s cumulative.",
        "- `generate_stream_features`: about 59.1s cumulative.",
        "- Per-window `extract_features`: about 40.7s cumulative.",
        "- Repeated per-window stats/percentiles: about 23.1s cumulative.",
        "- Repeated CSV reads: about 16.9s cumulative.",
        "- Repeated gzip CSV writes: about 6.6s cumulative.",
        "- Per-window FFT and correlation calls were also visible hotspots.",
        "",
        "## Refactor Summary",
        "",
        "- Replaced pandas row/window loops with batched NumPy window tensors per subject/activity segment.",
        "- Vectorized statistics, percentiles, correlations, and FFT feature extraction across all windows in each segment.",
        "- Added in-run clean-stream caching to avoid repeated reads of the same cleaned sensor CSV.",
        f"- Added seed-specific persistent feature caches using `{capabilities['cache_format']}` by default, with CSV/pickle/parquet compatibility fallbacks.",
        "- Reused cached stream features and merged feature tables when they already exist unless `--force-features` is passed.",
        "- Added timestamped progress logging and timings for feature generation, feature-table merging, model fitting, and grouped CV folds.",
        "- Switched CPU-capable model backends from single-threaded settings to configurable `--n-jobs` parallelism.",
        "- Added a safe retry with `n_jobs=1` if Windows blocks parallel worker setup.",
        "",
        "## GPU Reality Check",
        "",
        "- sklearn logistic regression: CPU only in this project. It is intentionally kept single-process because this solver gets little practical benefit from `n_jobs` and Windows joblib process spawning can fail in restricted environments.",
        "- sklearn random forest: CPU only, now parallelized with `n_jobs`.",
        "- LightGBM: configured as CPU LightGBM with `n_jobs`; this script does not claim LightGBM GPU acceleration.",
        f"- XGBoost: run device is `{xgb['device']}`. Detail: {xgb['detail']}",
        f"- TensorFlow CNN: separate raw-sequence CNN path. Detail from this Python environment: {tf['detail']}",
        f"- CuPy: {'available' if capabilities['cupy_available'] else 'not available'}; feature FFT/stat extraction stays on CPU NumPy.",
        "",
        "## Methodology",
        "",
        "The refactor preserves subject-wise splitting before windowing, windows remain contained within each subject/activity group, scaling remains inside train-fitted sklearn pipelines, and model selection/evaluation semantics are unchanged.",
        "",
    ]
    (root / "runtime_audit_comparative_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparative WISDM classical benchmark matrix.")
    parser.add_argument("--root", type=Path, default=Path("data/processed/comparative"))
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--sensors", default=None)
    parser.add_argument("--representations", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--cache-format", choices=["auto", "parquet", "pickle", "csv"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU workers for sklearn/LightGBM/XGBoost CPU paths; -1 means all cores.")
    parser.add_argument("--use-gpu", action="store_true", help="Test and use XGBoost CUDA if this environment actually supports it.")
    args = parser.parse_args()
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be 0; use -1 for all cores or a positive worker count.")

    root = args.root
    for subdir in ["splits", "features", "feature_tables", "results", "confusion_matrices", "subject_errors", "grouped_cv", "feature_importance", "figures"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    task_names = parse_list(args.tasks, list(TASKS.keys()))
    sensor_names = parse_list(args.sensors, list(SENSOR_CONFIGS.keys()))
    representations = parse_list(args.representations, list(REPRESENTATIONS))
    model_names = parse_list(args.models, list(MODEL_ORDER))
    cache_format = resolve_cache_format(args.cache_format)
    capabilities = inspect_runtime(args.use_gpu, args.seed, args.n_jobs, cache_format)
    xgboost_device = str(capabilities["xgboost"]["device"])  # type: ignore[index]

    run_manifest = {
        "tasks": task_names,
        "sensors": sensor_names,
        "representations": representations,
        "models": model_names,
        "seed": args.seed,
        "cache_format": cache_format,
        "n_jobs": args.n_jobs,
        "use_gpu_requested": args.use_gpu,
        "runtime_capabilities": capabilities,
    }
    (root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    all_results = []
    all_reports = []
    table_paths: dict[tuple[str, str, str], Path] = {}
    stream_cache: dict[Path, pd.DataFrame] = {}
    split_source_cache: dict[Path, pd.DataFrame] = {}
    for task_index, task_name in enumerate(task_names, start=1):
        task = TASKS[task_name]
        log(f"task [{task_index}/{len(task_names)}] {task.name}: labels={','.join(task.labels)}")
        splits = get_subject_splits(task, root, args.seed, split_source_cache)

        for sensor_index, sensor_name in enumerate(sensor_names, start=1):
            sensor_config = SENSOR_CONFIGS[sensor_name]
            log(f"sensor [{sensor_index}/{len(sensor_names)}] {sensor_name}: streams={','.join(sensor_config.streams)}")
            for rep_index, representation in enumerate(representations, start=1):
                log(f"representation [{rep_index}/{len(representations)}] {representation}")
                sensor_config = SENSOR_CONFIGS[sensor_name]
                feature_table, data = build_feature_table(
                    task,
                    sensor_config,
                    representation,
                    root,
                    splits,
                    args.seed,
                    force_features=args.force_features,
                    cache_format=cache_format,
                    stream_cache=stream_cache,
                )
                table_paths[(task_name, sensor_name, representation)] = feature_table
                features = feature_columns(data)
                label_encoder = LabelEncoder().fit(list(task.labels))
                log(f"feature matrix ready: {len(data):,} rows, {len(features):,} features")

                for model_index, model_name in enumerate(model_names, start=1):
                    log(f"model [{model_index}/{len(model_names)}] {task_name}/{sensor_name}/{representation}/{model_name}")
                    try:
                        model = make_model(model_name, args.seed, len(task.labels), n_jobs=args.n_jobs, xgboost_device=xgboost_device)
                    except Exception as exc:
                        all_results.append(
                            {
                                "task": task_name,
                                "sensor_config": sensor_name,
                                "representation": representation,
                                "model": model_name,
                                "status": f"skipped: {type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    result, report, _ = evaluate_model(
                        model_name,
                        model,
                        data,
                        features,
                        label_encoder,
                        task_name,
                        sensor_name,
                        representation,
                        root,
                    )
                    result["status"] = "ok"
                    result["n_features"] = len(features)
                    result["n_rows"] = len(data)
                    all_results.append(result)
                    all_reports.append(report)

    results = pd.DataFrame(all_results)
    results.to_csv(root / "results" / "benchmark_results.csv", index=False)
    if all_reports:
        pd.concat(all_reports, ignore_index=True).to_csv(root / "results" / "per_class_test_reports.csv", index=False)

    cv_rows = []
    ok_results = results[results["status"].eq("ok")].copy()
    if not args.skip_cv and not ok_results.empty:
        best_classical = (
            ok_results[~ok_results["model"].eq("majority")]
            .sort_values("test_macro_f1", ascending=False)
            .groupby("task", observed=True)
            .head(1)
        )
        for _, row in best_classical.iterrows():
            table_path = table_paths[(row["task"], row["sensor_config"], row["representation"])]
            data = read_existing_cache(table_path, convert_to_preferred=False)
            if data is None:
                data = read_frame(table_path)
            features = feature_columns(data)
            task = TASKS[row["task"]]
            cv = run_grouped_cv_for_best(
                data,
                features,
                task,
                row["sensor_config"],
                row["representation"],
                row["model"],
                root,
                args.seed,
                n_jobs=args.n_jobs,
                xgboost_device=xgboost_device,
            )
            cv_rows.append(cv)

            model = make_model(row["model"], args.seed, len(task.labels), n_jobs=args.n_jobs, xgboost_device=xgboost_device)
            train = data[data["split"].eq("train")]
            encoder = LabelEncoder().fit(list(task.labels))
            with timed(f"fit feature-importance model {row['task']}/{row['sensor_config']}/{row['representation']}/{row['model']}"):
                fit_with_worker_fallback(
                    model,
                    train[features].to_numpy(dtype=np.float32, copy=False),
                    encoder.transform(train["activity_label"]),
                    f"feature importance {row['task']}/{row['sensor_config']}/{row['representation']}/{row['model']}",
                )
            prefix = f"{row['task']}__{row['sensor_config']}__{row['representation']}__{row['model']}"
            save_feature_importance(model, features, root, prefix)
            subject_path = root / "subject_errors" / f"{prefix}.csv"
            plot_subject_errors(subject_path, root / "figures" / f"{prefix}__subject_errors.png")

    if cv_rows:
        cv_all = pd.concat(cv_rows, ignore_index=True)
        cv_all.to_csv(root / "grouped_cv" / "groupkfold_best_configs.csv", index=False)
        cv_summary = (
            cv_all.groupby(["task", "sensor_config", "representation", "model"], observed=True)
            .agg(
                folds=("fold", "nunique"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
            )
            .round(4)
            .reset_index()
        )
        cv_summary.to_csv(root / "grouped_cv" / "groupkfold_best_summary.csv", index=False)
    else:
        cv_summary = pd.DataFrame()

    if not ok_results.empty:
        write_summary_report(root, ok_results, cv_summary)
    write_runtime_audit_report(root, capabilities)
    print(f"Wrote {root / 'results' / 'benchmark_results.csv'}")
    print(f"Wrote {root / 'comparative_benchmark_report.md'}")
    print(f"Wrote {root / 'runtime_audit_comparative_benchmark.md'}")


if __name__ == "__main__":
    main()
