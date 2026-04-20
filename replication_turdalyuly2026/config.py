from __future__ import annotations

from pathlib import Path


REPLICATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPLICATION_ROOT.parent

DATA_DIR = REPLICATION_ROOT / "data"
RESULTS_DIR = REPLICATION_ROOT / "results"
REPORTS_DIR = REPLICATION_ROOT / "reports"

PHONE_ACCEL_CLEAN = PROJECT_ROOT / "data" / "processed" / "phone_accel_clean.csv.gz"
PHONE_GYRO_CLEAN = PROJECT_ROOT / "data" / "processed" / "phone_gyro_clean.csv.gz"

SAMPLE_RATE_HZ = 20.0
WINDOW_SIZES_SECONDS = (2.0, 4.0, 6.0)
OVERLAP = 0.5
N_SPLITS = 5
SEED = 20260420

PAPER_URL = "https://www.mdpi.com/2078-2489/17/4/368"
PAPER_TARGET_ACCURACY_MEAN = 0.4716
PAPER_TARGET_ACCURACY_STD = 0.0596
PAPER_TARGET_MACRO_F1_MEAN = 0.4626
PAPER_TARGET_MACRO_F1_STD = 0.0408
PAPER_REPORTED_TARGETS = {
    "table1_model_comparison_cnn_4s_accel_gyro": {
        "accuracy_mean": 0.4716,
        "accuracy_std": 0.0596,
        "macro_f1_mean": 0.4626,
        "macro_f1_std": 0.0408,
        "note": "Primary model-comparison row for smartphone CNN, 4.0 s window, phone accel+gyro.",
    },
    "table3_window_length_cnn_4s": {
        "accuracy_mean": 0.4568,
        "accuracy_std": 0.0553,
        "macro_f1_mean": 0.4473,
        "macro_f1_std": 0.0525,
        "note": "Window-length ablation row for 4.0 s smartphone CNN.",
    },
    "table4_sensor_ablation_accel_gyro": {
        "accuracy_mean": 0.4534,
        "accuracy_std": 0.0559,
        "macro_f1_mean": 0.4535,
        "macro_f1_std": 0.0643,
        "note": "Sensor-modality ablation row for accel+gyro smartphone CNN.",
    },
}

GROUP_LABELS = (
    "locomotion",
    "stairs",
    "static",
    "eat_drink",
    "sports",
    "upper_body",
)

WISDM_LABEL_TO_NAME = {
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

WISDM_LABEL_TO_GROUP = {
    "A": "locomotion",
    "B": "locomotion",
    "C": "stairs",
    "D": "static",
    "E": "static",
    "F": "upper_body",
    "G": "upper_body",
    "H": "eat_drink",
    "I": "eat_drink",
    "J": "eat_drink",
    "K": "eat_drink",
    "L": "eat_drink",
    "M": "sports",
    "O": "sports",
    "P": "sports",
    "Q": "upper_body",
    "R": "upper_body",
    "S": "upper_body",
}

GROUP_TO_ID = {label: idx for idx, label in enumerate(GROUP_LABELS)}

CHANNEL_COLUMNS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
MODALITY_CHANNELS = {
    "accel": (0, 1, 2),
    "gyro": (3, 4, 5),
    "accel_gyro": (0, 1, 2, 3, 4, 5),
}

METADATA_COLUMNS = {
    "window_id",
    "subject_id",
    "start_timestamp_ns",
    "end_timestamp_ns",
    "n_samples",
    "window_seconds",
    "overlap",
    "group_label",
    "y",
    "majority_fraction",
}
