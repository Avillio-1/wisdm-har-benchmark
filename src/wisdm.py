from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "wisdm-dataset"
RAW_COLUMNS = ["subject_id", "activity_label", "timestamp_ns", "x", "y", "z"]

ACTIVITY_MAP = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}
ACTIVITY_ORDER = list(ACTIVITY_MAP.keys())
SENSOR_COLUMNS = ["x", "y", "z"]
METADATA_COLUMNS = [
    "window_id",
    "subject_id",
    "activity_label",
    "activity_name",
    "device",
    "sensor",
    "start_timestamp_ns",
    "end_timestamp_ns",
    "n_samples",
    "window_seconds",
    "overlap",
    "split",
]


@dataclass(frozen=True)
class RawFileInfo:
    path: Path
    subject_id: int
    sensor: str
    device: str


def slug_for_selection(device: str, sensor: str) -> str:
    return f"{device}_{sensor}".lower().replace("-", "_")


def parse_raw_path(path: Path) -> RawFileInfo:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 4 or parts[0] != "data":
        raise ValueError(f"Unexpected raw filename: {path.name}")
    return RawFileInfo(
        path=path,
        subject_id=int(parts[1]),
        sensor=parts[2].lower(),
        device=parts[3].lower(),
    )


def iter_raw_files(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    device: str | None = None,
    sensor: str | None = None,
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    if device and sensor:
        root = dataset_dir / "raw" / device / sensor
        files = root.glob("data_*_*.txt")
    else:
        files = (dataset_dir / "raw").glob("**/data_*_*.txt")
    return sorted(path for path in files if path.is_file())


def read_raw_file(path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    info = parse_raw_path(path)
    df = pd.read_csv(
        path,
        names=RAW_COLUMNS,
        header=None,
        dtype=str,
        nrows=nrows,
        on_bad_lines="skip",
    )
    df["source_file"] = str(path)
    df["source_subject_id"] = info.subject_id
    df["device"] = info.device
    df["sensor"] = info.sensor
    return standardize_raw_frame(df)


def standardize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_COLUMNS:
        if col not in out:
            out[col] = pd.NA

    out["activity_label"] = out["activity_label"].astype("string").str.strip().str.upper()
    out["z"] = (
        out["z"]
        .astype("string")
        .str.replace(";", "", regex=False)
        .str.strip()
    )

    for col in ["subject_id", "source_subject_id", "timestamp_ns"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in SENSOR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["activity_name"] = out["activity_label"].map(ACTIVITY_MAP).astype("string")
    out["is_known_activity"] = out["activity_label"].isin(ACTIVITY_MAP)
    out["timestamp_seconds"] = out["timestamp_ns"].astype("float64") / 1_000_000_000.0
    out["source_subject_mismatch"] = (
        out["source_subject_id"].notna()
        & out["subject_id"].notna()
        & (out["source_subject_id"] != out["subject_id"])
    )
    return out


def load_raw_dataset(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    device: str = "phone",
    sensor: str = "accel",
    nrows_per_file: int | None = None,
) -> pd.DataFrame:
    frames = [read_raw_file(path, nrows=nrows_per_file) for path in iter_raw_files(dataset_dir, device, sensor)]
    if not frames:
        raise FileNotFoundError(f"No raw files found for device={device!r}, sensor={sensor!r}")
    return pd.concat(frames, ignore_index=True)


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    work = df.copy()
    required = ["subject_id", "activity_label", "timestamp_ns", "x", "y", "z"]
    invalid_mask = work[required].isna().any(axis=1)
    invalid_mask |= ~work["is_known_activity"]
    invalid_mask |= work["timestamp_ns"].le(0).fillna(True)
    invalid_mask |= work["source_subject_mismatch"].fillna(False)

    invalid_rows = work.loc[invalid_mask].copy()
    clean = work.loc[~invalid_mask].copy()
    before_dedup = len(clean)
    dedup_subset = ["subject_id", "activity_label", "timestamp_ns", "x", "y", "z", "device", "sensor"]
    duplicate_mask = clean.duplicated(subset=dedup_subset, keep="first")
    duplicate_rows = clean.loc[duplicate_mask].copy()
    dropped_rows = pd.concat([invalid_rows, duplicate_rows], ignore_index=True)
    clean = clean.loc[~duplicate_mask].copy()
    clean = clean.sort_values(["subject_id", "activity_label", "timestamp_ns"]).reset_index(drop=True)

    counts = {
        "input_rows": int(len(work)),
        "invalid_rows": int(len(invalid_rows)),
        "duplicate_rows_removed": int(before_dedup - len(clean)),
        "output_rows": int(len(clean)),
    }
    return clean, dropped_rows, counts


def markdown_table(df: pd.DataFrame, max_rows: int | None = None, index: bool = False) -> str:
    table = df.head(max_rows).copy() if max_rows else df.copy()
    if index:
        table = table.reset_index()
    table = table.fillna("")
    columns = [str(c) for c in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(col), *(len(row[i]) for row in rows)) if rows else len(col)
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["activity_label", "activity_name"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("activity_label")
    )
    counts["percent"] = (counts["rows"] / counts["rows"].sum() * 100).round(2)
    return counts


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(METADATA_COLUMNS)
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def _axis_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {}
    q25, q75 = np.percentile(values, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_energy": float(np.mean(values**2)),
    }


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}
    axes = {axis: window[axis].to_numpy(dtype=float) for axis in SENSOR_COLUMNS}
    for axis, values in axes.items():
        features.update(_axis_features(values, axis))

    magnitude = np.sqrt(axes["x"] ** 2 + axes["y"] ** 2 + axes["z"] ** 2)
    features.update(_axis_features(magnitude, "magnitude"))

    if len(window) > 1:
        for left, right in [("x", "y"), ("x", "z"), ("y", "z")]:
            corr = np.corrcoef(axes[left], axes[right])[0, 1]
            features[f"{left}_{right}_corr"] = float(0.0 if np.isnan(corr) else corr)
    else:
        for left, right in [("x", "y"), ("x", "z"), ("y", "z")]:
            features[f"{left}_{right}_corr"] = 0.0
    return features


def make_windows(
    df: pd.DataFrame,
    window_seconds: float,
    overlap: float,
    sample_rate_hz: float = 20.0,
) -> pd.DataFrame:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    window_size = int(round(window_seconds * sample_rate_hz))
    if window_size < 2:
        raise ValueError("window size must contain at least two samples.")
    step = max(1, int(round(window_size * (1 - overlap))))

    rows: list[dict[str, object]] = []
    sort_cols = ["subject_id", "activity_label", "timestamp_ns"]
    ordered = df.sort_values(sort_cols)
    group_cols = ["subject_id", "activity_label", "activity_name", "device", "sensor"]
    window_id = 0
    for keys, group in ordered.groupby(group_cols, observed=True, sort=True):
        subject_id, activity_label, activity_name, device, sensor = keys
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1, step):
            window = group.iloc[start : start + window_size]
            row: dict[str, object] = {
                "window_id": window_id,
                "subject_id": int(subject_id),
                "activity_label": activity_label,
                "activity_name": activity_name,
                "device": device,
                "sensor": sensor,
                "start_timestamp_ns": int(window["timestamp_ns"].iloc[0]),
                "end_timestamp_ns": int(window["timestamp_ns"].iloc[-1]),
                "n_samples": int(len(window)),
                "window_seconds": float(window_seconds),
                "overlap": float(overlap),
            }
            row.update(extract_window_features(window))
            rows.append(row)
            window_id += 1
    return pd.DataFrame(rows)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.NA
    return out
