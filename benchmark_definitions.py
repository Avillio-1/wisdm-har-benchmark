from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskDef:
    name: str
    labels: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class SensorStream:
    name: str
    device: str
    sensor: str
    clean_path: Path


@dataclass(frozen=True)
class SensorConfig:
    name: str
    streams: tuple[str, ...]
    rationale: str


TASKS: dict[str, TaskDef] = {
    "task3": TaskDef(
        name="task3",
        labels=("A", "B", "E"),
        rationale="A priori sanity-check task: walking vs jogging vs standing, chosen as a simple locomotion/posture benchmark, not from clean test results.",
    ),
    "task6": TaskDef(
        name="task6",
        labels=("A", "B", "C", "D", "E", "F"),
        rationale="A priori medium-difficulty task: walking, jogging, stairs, sitting, standing, and typing. This mixes locomotion, posture, and a hand/phone interaction without consulting model performance.",
    ),
    "task18": TaskDef(
        name="task18",
        labels=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "O", "P", "Q", "R", "S"),
        rationale="Full hard benchmark using every WISDM activity label.",
    ),
}


STREAMS: dict[str, SensorStream] = {
    "phone_accel": SensorStream("phone_accel", "phone", "accel", Path("data/processed/phone_accel_clean.csv.gz")),
    "phone_gyro": SensorStream("phone_gyro", "phone", "gyro", Path("data/processed/phone_gyro_clean.csv.gz")),
    "watch_accel": SensorStream("watch_accel", "watch", "accel", Path("data/processed/watch_accel_clean.csv.gz")),
    "watch_gyro": SensorStream("watch_gyro", "watch", "gyro", Path("data/processed/watch_gyro_clean.csv.gz")),
}


SENSOR_CONFIGS: dict[str, SensorConfig] = {
    "phone_accel": SensorConfig("phone_accel", ("phone_accel",), "Phone accelerometer only."),
    "phone_accel_gyro": SensorConfig("phone_accel_gyro", ("phone_accel", "phone_gyro"), "Phone accelerometer plus phone gyroscope feature-level fusion."),
    "watch_accel": SensorConfig("watch_accel", ("watch_accel",), "Smartwatch accelerometer only."),
    "watch_accel_gyro": SensorConfig("watch_accel_gyro", ("watch_accel", "watch_gyro"), "Smartwatch accelerometer plus smartwatch gyroscope feature-level fusion."),
    "phone_watch_fusion": SensorConfig(
        "phone_watch_fusion",
        ("phone_accel", "phone_gyro", "watch_accel", "watch_gyro"),
        "Feature-level fusion of all phone and watch accelerometer/gyroscope streams. Windows are aligned by subject, activity, split, and within-activity window index.",
    ),
}


REPRESENTATIONS = ("stats", "stats_freq")


MODEL_ORDER = ("majority", "logistic_regression", "random_forest", "lightgbm", "xgboost")
