# Sensor Selection and Feature Engineering for Human Activity Recognition: A Comparative Study on the WISDM Dataset

This repository contains the full ML pipeline, experiments, and IEEE Access paper for a comparative study of sensor configurations, feature representations, and model architectures for Human Activity Recognition (HAR) on the WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.

---

## Authors

- Nawaf Altamimi
- Suad Alnasser
- Mohammad Aldemaiji

---

## Paper

**Title:** Sensor Selection and Feature Engineering for Human Activity Recognition: A Comparative Study on the WISDM Dataset

**Format:** IEEE Access-style journal paper

**Key results:**

| Model | Type | Input | Macro-F1 |
|-------|------|-------|----------|
| Turdalyuly CNN (baseline) | Deep Learning | Raw sequences | 0.4626 |
| Majority Baseline | Classical | None | 0.073 |
| Logistic Regression | Classical | 126 statistical features | 0.572 |
| Random Forest | Classical | 126 statistical features | 0.607 |
| LightGBM | Classical | 126 stat + freq features | 0.663 |
| XGBoost | Classical | 126 stat + freq features | 0.670 |
| CNN+LSTM v1 | Deep Learning | Raw sequences | 0.5748 |
| CNN+LSTM v2 | Deep Learning | Raw sequences | 0.5956 |
| CNN+LSTM v3 | Deep Learning | Raw + FFT channels | 0.5666 |
| CNN+LSTM v4 | Deep Learning | Raw sequences | 0.5930 |
| Deep Feature Net | Deep Learning | 126 stat + freq features | 0.6713 |
| **Dual-Input Hybrid (proposed)** | **Deep Learning** | **Raw + 126 features** | **0.6720** |

All experiments use phone accelerometer + gyroscope, 4-second windows, 50% overlap, subject-wise evaluation protocol, macro-F1 as primary metric.

---

## Dataset

This project uses the **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset** (UCI ID 507).

- 51 subjects, 18 activities, ~20 Hz sampling rate
- Phone and smartwatch accelerometer + gyroscope streams
- Download from Kaggle: [shreeyashnaik/wisdm-smartphone-and-smartwatch-activity](https://www.kaggle.com/datasets/shreeyashnaik/wisdm-smartphone-and-smartwatch-activity)

Place the downloaded data into:

```
wisdm-dataset/
└── raw/
    ├── phone/
    │   ├── accel/
    │   └── gyro/
    └── watch/
        ├── accel/
        └── gyro/
```

---

## Six-Class Grouped Taxonomy (Turdalyuly et al. 2026)

| Group | Original WISDM Activities |
|-------|--------------------------|
| `locomotion` | walking, jogging |
| `stairs` | stairs |
| `static` | sitting, standing |
| `eat_drink` | soup, chips, pasta, drinking, sandwich |
| `sports` | kicking, catch, dribbling |
| `upper_body` | typing, teeth, writing, clapping, folding |

---

## Repository Structure

```
.
├── README.md
├── audit_dataset.py                     # Step 1: audit raw dataset
├── preprocess.py                        # Step 2: clean sensor streams
├── splits.py                            # Step 3: subject-wise splits
├── windowing.py                         # Step 4: feature windows
├── eda.py                               # Step 5: EDA (original)
├── comparative_benchmark.py             # Step 6: classical ML benchmark
├── sequence_windowing.py                # Raw sequence tensor builder
├── raw_sequence_deep_baseline.py        # CNN/LSTM on raw sequences
├── benchmark_definitions.py            # Task/sensor/model config
├── baselines.py                         # Baseline model implementations
├── deep_baseline.py                     # Deep learning baselines
├── grouped_cv_baselines.py             # GroupKFold CV
├── leakage_audit_clean3.py             # Leakage audit script
│
├── deep_learning/                       # All deep learning experiments
│   ├── cnn_lstm_turdalyuly_v1.py        # CNN+LSTM v1 (focal loss + class weights)
│   ├── cnn_lstm_turdalyuly_v2.py        # CNN+LSTM v2 (+ Gaussian noise, dropout)
│   ├── cnn_lstm_turdalyuly_v3.py        # CNN+LSTM v3 (+ FFT input channels)
│   ├── cnn_lstm_turdalyuly_v4.py        # CNN+LSTM v4 (+ L2 regularization)
│   ├── deep_feature_model.py            # Deep Feature Net (MLP on 126 features)
│   ├── dual_input_v1.py                 # Dual-Input Hybrid v1 (misaligned — negative result)
│   ├── dual_input_v2.py                 # Dual-Input Hybrid v2 (aligned — proposed model)
│   └── eda_wisdm.py                     # EDA for grouped taxonomy
│
├── replication_turdalyuly2026/          # Replication of Turdalyuly et al. 2026
│   ├── README.md
│   ├── config.py
│   ├── utils.py
│   ├── 01_prepare_windows.py
│   ├── 02_replicate_cnn.py
│   ├── 03_train_feature_models.py
│   ├── 04_make_report.py
│   ├── 05_audit_fairness.py
│   ├── reports/
│   └── results/
│
├── paper/
│   ├── paper.tex                        # Full IEEE Access LaTeX paper
│   ├── pipeline_diagram.png             # System overview diagram
│   ├── eda_01.png                       # 18-class distribution
│   ├── eda_02.png                       # Grouped taxonomy distribution
│   ├── eda_03.png                       # Raw signal samples per group
│   ├── eda_04.png                       # Subject variability
│   ├── eda_05.png                       # Feature boxplots
│   ├── eda_06.png                       # Axis correlation heatmaps
│   └── eda_07.png                       # Subject-class coverage
│
├── data/                                # Generated data (not tracked in Git)
│   ├── processed/
│   └── interim/
├── figures/
├── reports/
└── src/
    └── wisdm.py
```

---

## Setup

Python 3.10 or newer recommended.

Install core dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

Install benchmark dependencies:

```bash
pip install lightgbm xgboost pyarrow
```

Install deep learning dependencies:

```bash
pip install tensorflow seaborn
```

---

## Reproducing the Main Pipeline

Run all commands from the repository root.

### 1. Audit the raw dataset

```bash
python audit_dataset.py --report reports/audit/dataset_audit.md
```

### 2. Clean each sensor stream

```bash
python preprocess.py --device phone --sensor accel --report reports/preprocessing/preprocessing_report.md
python preprocess.py --device phone --sensor gyro --report reports/preprocessing/preprocessing_report_phone_gyro.md
python preprocess.py --device watch --sensor accel --report reports/preprocessing/preprocessing_report_watch_accel.md
python preprocess.py --device watch --sensor gyro --report reports/preprocessing/preprocessing_report_watch_gyro.md
```

### 3. Create subject-wise splits

```bash
python splits.py --clean-data data/processed/phone_accel_clean.csv.gz --out data/processed/subject_splits.csv --protocol reports/evaluation/evaluation_protocol.md
```

### 4. Create classical feature windows

```bash
python windowing.py --clean-data data/processed/phone_accel_clean.csv.gz --splits data/processed/subject_splits.csv --summary reports/windowing_eda/windowing_summary.md
```

### 5. Generate EDA

```bash
python eda.py --clean-data data/processed/phone_accel_clean.csv.gz --figures-dir figures --summary reports/windowing_eda/eda_summary.md
```

### 6. Run the classical benchmark

```bash
python comparative_benchmark.py --root data/processed/comparative_first_pass --n-jobs -1 --use-gpu
```

---

## Reproducing the Deep Learning Experiments

All scripts below run from the repository root and require the preprocessing pipeline (steps 1--3 above) to have been completed first.

### Generate EDA for grouped taxonomy

```bash
python deep_learning/eda_wisdm.py
```

### Run Turdalyuly replication (classical models)

```bash
python replication_turdalyuly2026/01_prepare_windows.py
python replication_turdalyuly2026/03_train_feature_models.py --window-sizes 4.0 --include-secondary
```

### CNN+LSTM experiments (v1 through v4)

```bash
python deep_learning/cnn_lstm_turdalyuly_v1.py
python deep_learning/cnn_lstm_turdalyuly_v2.py
python deep_learning/cnn_lstm_turdalyuly_v3.py
python deep_learning/cnn_lstm_turdalyuly_v4.py
```

### Deep Feature Network

```bash
python deep_learning/deep_feature_model.py
```

### Dual-Input Hybrid (proposed model)

```bash
python deep_learning/dual_input_v2.py
```

> **Note:** `dual_input_v1.py` is included as a negative result showing that temporal misalignment between the two input branches (raw sequences vs precomputed features) severely degrades performance (macro-F1 = 0.5106). The aligned v2 design fixes this by computing features inline from the same windows.

---

## Evaluation Protocol

All experiments follow a strict subject-wise evaluation protocol:

- **No random row splits** — subjects are divided at the person level
- **Splits:** 33 training subjects / 8 validation / 10 test
- **Scaling:** StandardScaler fitted on training data only, applied to val/test without refitting
- **Primary metric:** Macro-averaged F1
- **Classical benchmark:** additionally uses 5-fold GroupKFold by subject for stability check

---

## Deep Learning Architecture Summary

### CNN+LSTM v4 (final baseline)
- 3 × Conv1D blocks (32 → 64 → 64 filters) with BatchNorm + Dropout 0.4
- Bidirectional LSTM (64 units per direction)
- L2 regularization (λ = 0.001) on all layers
- Gaussian noise (σ = 0.1) augmentation
- Focal loss (γ = 2.0, α = 0.25) + class weights
- Adam optimizer (lr = 0.0005), early stopping (patience = 15)

### Deep Feature Network
- 4-layer MLP (256 → 256 → 128 → 64 units)
- BatchNorm + Dropout 0.3 per layer
- Same 126 features as XGBoost — isolates classifier contribution

### Dual-Input Hybrid (proposed)
- **Branch 1:** CNN+LSTM on raw sequences → 128-dim embedding
- **Branch 2:** 3-layer MLP on 126 features → 64-dim embedding
- **Merge:** Concatenate (192-dim) → Dense (128 → 64) → 6-class softmax
- Features computed **inline** from same windows as raw sequences (critical for alignment)

---

## Feature Engineering (126 features)

For each of 8 channels (accel x/y/z, gyro x/y/z, accel magnitude, gyro magnitude):

**Time-domain (9):** mean, std, min, max, median, IQR, range, RMS, energy

**Frequency-domain (6):** total power, dominant frequency (Hz), dominant power fraction, spectral entropy, low-band power (0--3 Hz), high-band power (>3 Hz)

**Cross-axis correlations (6):** accel x-y, x-z, y-z + gyro x-y, x-z, y-z

**Total:** 8 × 15 + 6 = **126 features per window**

---

## Compiling the Paper

The paper is written in LaTeX using the `IEEEtran` journal class. To compile:

```bash
cd paper/
pdflatex paper.tex
pdflatex paper.tex
```

Run twice to resolve cross-references and citations.

Required files in the same directory: `paper.tex` + all `eda_0*.png` + `pipeline_diagram.png`

---

## Key Findings

1. **Engineered features outperform raw sequences** on phone sensor data. XGBoost (0.6697) and the Deep Feature Network (0.6713) both exceed all CNN+LSTM variants (best: 0.5930), confirming that the 126-feature representation generalizes better than raw sequences with only 33 training subjects.

2. **The Dual-Input Hybrid achieves the best overall result** at macro-F1 = 0.6720, confirming that raw sequences and engineered features carry complementary information.

3. **Temporal alignment is critical** for dual-input models. The misaligned v1 design (precomputed features loaded from a separate file) achieved only 0.5106. The v2 fix (features computed inline from the same windows) achieved 0.6720.

4. **Static and upper_body classes are the hardest** on phone sensor data regardless of model, due to weak and ambiguous phone-pocket signals for fine-grained arm and hand movements.

---

## Acknowledgements

This work builds upon the benchmark and replication package established in:

> A. Turdalyuly et al., "Human Activity Recognition Using the WISDM Smartphone and Smartwatch Dataset," *Information*, vol. 17, no. 4, p. 368, Apr. 2026. DOI: 10.3390/info17040368.

Dataset:

> G. M. Weiss, K. Yoneda, and T. Hayajneh, "Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living," *IEEE Access*, vol. 7, pp. 133190–133202, 2019. DOI: 10.1109/ACCESS.2019.2940729.
