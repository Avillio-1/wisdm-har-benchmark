# Sensor Selection and Feature Engineering for Human Activity Recognition: A Comparative Study on the WISDM Dataset

This repository contains code, reports, result summaries, and IEEE-style paper material for a Human Activity Recognition (HAR) study on the WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.

The active reproducibility path in this checkout is the Turdalyuly et al. 2026 same-disclosed-protocol package under `replication_turdalyuly2026/`. Additional neural-model results are summarized for the paper, but should be kept tied to their supporting scripts or result artifacts before final submission.

## Authors

- Nawaf Altamimi
- Saud Alnasser
- Mohammad Aldemaiji

## Paper

**Title:** Sensor Selection and Feature Engineering for Human Activity Recognition: A Comparative Study on the WISDM Dataset

**Format:** IEEE Access-style journal paper

**Key results:**

| Model | Type | Input | Macro-F1 |
|---|---|---|---:|
| Turdalyuly CNN baseline | Deep Learning | Raw sequences | 0.4626 |
| Majority baseline | Classical | None | 0.0730 |
| Logistic regression | Classical | 126 statistical features | 0.5720 |
| Random forest | Classical | 126 statistical features | 0.6070 |
| LightGBM | Classical | 126 statistical + frequency features | 0.6630 |
| XGBoost | Classical | 126 statistical + frequency features | 0.6700 |
| CNN+LSTM v1 | Deep Learning | Raw sequences | 0.5748 |
| CNN+LSTM v2 | Deep Learning | Raw sequences | 0.5956 |
| CNN+LSTM v3 | Deep Learning | Raw + FFT channels | 0.5666 |
| CNN+LSTM v4 | Deep Learning | Raw sequences | 0.5930 |
| Deep Feature Net | Deep Learning | 126 statistical + frequency features | 0.6713 |
| **Dual-Input Hybrid (proposed)** | **Deep Learning** | **Raw + 126 features** | **0.6720** |

All headline experiments use phone accelerometer + gyroscope data, 4-second windows, 50% overlap, subject-wise evaluation, and macro-F1 as the primary metric. The committed replication package uses 5-fold GroupKFold by subject; any additional neural-model scripts should document their exact folds or train/validation/test split before the paper claims are finalized.

## Dataset

This project uses the **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset** (UCI ID 507).

- 51 subjects, 18 activities, approximately 20 Hz sampling rate
- Smartphone and smartwatch accelerometer + gyroscope streams
- Original UCI dataset page: <https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset>
- Kaggle mirror used by some project members: <https://www.kaggle.com/datasets/shreeyashnaik/wisdm-smartphone-and-smartwatch-activity>

Place the downloaded raw files under:

```text
wisdm-dataset/
+-- raw/
    +-- phone/
    |   +-- accel/
    |   +-- gyro/
    +-- watch/
        +-- accel/
        +-- gyro/
```

## Six-Class Grouped Taxonomy

The Turdalyuly same-protocol package maps the official WISDM activity labels into six grouped classes:

| Group | Original WISDM activities |
|---|---|
| `locomotion` | walking, jogging |
| `stairs` | stairs |
| `static` | sitting, standing |
| `eat_drink` | soup, chips, pasta, drinking, sandwich |
| `sports` | kicking, catch, dribbling |
| `upper_body` | typing, teeth, writing, clapping, folding |

The mapping is encoded in `replication_turdalyuly2026/config.py`.

## Repository Structure

```text
.
+-- README.md
+-- preprocess.py
+-- replication_turdalyuly2026/
|   +-- README.md
|   +-- requirements.txt
|   +-- config.py
|   +-- utils.py
|   +-- 01_prepare_windows.py
|   +-- 02_replicate_cnn.py
|   +-- 03_train_feature_models.py
|   +-- 04_make_report.py
|   +-- 05_audit_fairness.py
|   +-- reports/
|   +-- results/
|   +-- data/              # generated, ignored
+-- Paper/
|   +-- wisdm_har_ieee_access.tex
|   +-- wisdm_har_ieee.tex
|   +-- references.bib
|   +-- wisdm_related_work_report.md
+-- reports/
+-- figures/
+-- notebooks/
+-- src/
|   +-- wisdm.py
+-- data/                  # generated, ignored
+-- wisdm-dataset/         # raw dataset, ignored
```

Older first-pass scripts and reports may exist in earlier commits or local experiment history. The maintained reproduction path for the current Turdalyuly-style comparison is `replication_turdalyuly2026/`.

## Setup

Python 3.10 or newer is recommended.

Install the classical-model and data-processing dependencies:

```bash
python -m pip install -r replication_turdalyuly2026/requirements.txt
```

The CNN calibration requires PyTorch. For the recorded CUDA run, the replication package used:

```bash
python -m pip install --no-cache-dir --timeout 1000 --retries 10 torch --index-url https://download.pytorch.org/whl/cu128
```

If CUDA PyTorch is not available, use the CNN script's CPU mode only for smoke tests, not for the full recorded experiment.

## Reproducing the Maintained Pipeline

Run all commands from the repository root.

### 1. Clean the phone streams

```bash
python preprocess.py --device phone --sensor accel --report reports/preprocessing/preprocessing_report.md
python preprocess.py --device phone --sensor gyro --report reports/preprocessing/preprocessing_report_phone_gyro.md
```

These commands create:

```text
data/processed/phone_accel_clean.csv.gz
data/processed/phone_gyro_clean.csv.gz
```

### 2. Prepare aligned phone accelerometer + gyroscope windows

```bash
python replication_turdalyuly2026/01_prepare_windows.py
```

This creates exact timestamp-aligned phone accelerometer + gyroscope windows for 2.0 s, 4.0 s, and 6.0 s settings.

### 3. Train same-protocol feature models

```bash
python replication_turdalyuly2026/03_train_feature_models.py --window-sizes 4.0 --include-secondary
```

This evaluates:

- majority baseline
- logistic regression on statistical features
- random forest on statistical features
- LightGBM on statistical + frequency features
- XGBoost on statistical + frequency features

### 4. Run the CNN calibration

Primary 4.0 s phone accelerometer + gyroscope condition:

```bash
python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 4.0 --modalities accel_gyro
```

Full modality/window grid:

```bash
python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 2.0,4.0,6.0 --modalities accel,gyro,accel_gyro
```

### 5. Build reports and run the fairness audit

```bash
python replication_turdalyuly2026/04_make_report.py
python replication_turdalyuly2026/05_audit_fairness.py
```

The most important claim-safety document is:

```text
replication_turdalyuly2026/reports/fairness_audit.md
```

## Evaluation Protocol

The committed replication package uses:

- 5-fold `GroupKFold` by `subject_id`
- no subject overlap between training and evaluation folds
- phone accelerometer + gyroscope as the primary input condition
- 4.0 s windows with 50% overlap for the main comparison
- majority-vote labels over the six grouped classes
- training-fold-only normalization/scaling
- macro-F1 as the primary metric

The primary fold audit found 40-41 training subjects and 10-11 evaluation subjects per fold, with zero subject overlap and zero missing evaluation classes.

## Current Replication Result Snapshot

Feature-model benchmark on the primary 4.0 s phone accelerometer + gyroscope windows:

| Model | Feature set | Macro-F1 |
|---|---|---:|
| majority | none | 0.0730 +/- 0.0039 |
| logistic regression | statistical | 0.5720 +/- 0.0267 |
| random forest | statistical | 0.6061 +/- 0.0371 |
| LightGBM | statistical + frequency | 0.6628 +/- 0.0348 |
| XGBoost | statistical + frequency | 0.6697 +/- 0.0334 |

CNN calibration:

| Modality | Window | Macro-F1 |
|---|---:|---:|
| accel | 2.0 s | 0.5372 +/- 0.0487 |
| gyro | 2.0 s | 0.5204 +/- 0.0581 |
| accel + gyro | 2.0 s | 0.5515 +/- 0.0311 |
| accel | 4.0 s | 0.5670 +/- 0.0444 |
| gyro | 4.0 s | 0.5103 +/- 0.0234 |
| accel + gyro | 4.0 s | 0.5469 +/- 0.0564 |
| accel | 6.0 s | 0.5500 +/- 0.0295 |
| gyro | 6.0 s | 0.5515 +/- 0.0225 |
| accel + gyro | 6.0 s | 0.5627 +/- 0.0445 |

## Feature Engineering: 126 Features

For each 4-second phone accelerometer + gyroscope window, the feature representation contains 126 hand-crafted features.

There are 8 signals:

- accelerometer `x`, `y`, `z`
- gyroscope `x`, `y`, `z`
- accelerometer magnitude
- gyroscope magnitude

For each signal, 15 descriptors are computed:

- 9 time-domain descriptors: mean, standard deviation, minimum, maximum, median, interquartile range, range, RMS, energy
- 6 frequency-domain descriptors: total power, dominant frequency in Hz, dominant-power fraction, spectral entropy, low-band power from 0 to 3 Hz, high-band power above 3 Hz

The representation also includes 6 cross-axis correlations:

- accelerometer `x-y`, `x-z`, `y-z`
- gyroscope `x-y`, `x-z`, `y-z`

Total:

```text
8 signals * 15 descriptors + 6 correlations = 126 features per window
```

## Additional Neural Model Summaries

The following summaries describe the neural-model rows used in the paper-level comparison. Make sure the corresponding scripts or result files are present before treating these rows as fully reproducible from a fresh clone.

### CNN+LSTM v4

- 3 Conv1D blocks: 32 -> 64 -> 64 filters, with BatchNorm and Dropout 0.4
- Bidirectional LSTM with 64 units per direction
- L2 regularization with lambda = 0.001
- Gaussian noise augmentation with sigma = 0.1
- Focal loss with gamma = 2.0 and alpha = 0.25, plus class weights
- Adam optimizer with learning rate 0.0005
- early stopping with patience = 15

### Deep Feature Network

- 4-layer MLP: 256 -> 256 -> 128 -> 64 units
- BatchNorm and Dropout 0.3 per layer
- Uses the same 126 features as XGBoost to isolate classifier contribution

### Dual-Input Hybrid

- Branch 1: CNN+LSTM on raw sequences -> 128-dimensional embedding
- Branch 2: MLP on 126 features -> 64-dimensional embedding
- Merge: concatenate to 192 dimensions -> dense layers 128 -> 64 -> 6-class softmax
- Critical detail: engineered features are computed inline from the same windows as the raw sequence branch to avoid temporal misalignment

## Key Findings

1. Engineered features outperform raw sequences on the primary phone sensor condition. XGBoost reaches macro-F1 0.6697, while the committed 4.0 s phone accelerometer + gyroscope CNN calibration reaches 0.5469.
2. The paper-level Dual-Input Hybrid achieves the best overall reported result at macro-F1 0.6720.
3. Temporal alignment is critical for dual-input models. The negative v1 result shows that using precomputed features from a separate alignment path can severely degrade performance.
4. `static` and `upper_body` remain the hardest grouped classes on phone sensor data, which is expected because a pocket-carried phone weakly captures fine-grained arm and hand movement.

## Compiling the Paper

The current LaTeX sources live in `Paper/`.

Recommended compile sequence:

```bash
cd Paper
pdflatex wisdm_har_ieee_access.tex
bibtex wisdm_har_ieee_access
pdflatex wisdm_har_ieee_access.tex
pdflatex wisdm_har_ieee_access.tex
```

Required source files:

```text
Paper/wisdm_har_ieee_access.tex
Paper/references.bib
```

The LaTeX source also references figures and generated result plots outside `Paper/`. Missing figures are handled by the `safeincludegraphics` fallback, but final paper builds should include all intended image files.

## Claim Safety

Use this style of claim in the paper:

> Under the disclosed Turdalyuly et al. WISDM smartphone IMU protocol, using phone accelerometer + gyroscope signals, six grouped activity categories, 4.0 s windows, 50% overlap, majority-vote window labels, and 5-fold subject-wise GroupKFold evaluation, our XGBoost model using statistical and frequency-domain features achieved higher macro-F1 than the reported CNN benchmark.

Avoid these claims:

- Do not say the Turdalyuly hidden implementation was exactly reproduced.
- Do not imply the original authors' exact fold IDs or preprocessing code were available.
- Do not compare watch or phone+watch fusion as the headline Turdalyuly replication condition.
- Do not rely on accuracy alone; macro-F1 is the primary metric.

The exact audit verdict is:

```text
Protocol-fair proposed-method comparison, not exact hidden-code replication.
```

## Strongest Supporting Files

```text
replication_turdalyuly2026/reports/turdalyuly2026_replication_report.md
replication_turdalyuly2026/reports/fairness_audit.md
replication_turdalyuly2026/results/cnn/cnn_replication_summary.csv
replication_turdalyuly2026/results/feature_models/feature_models_summary.csv
replication_turdalyuly2026/results/feature_models/feature_models_fold_metrics.csv
replication_turdalyuly2026/results/feature_models/feature_models_per_class.csv
```

## Acknowledgements

This work builds upon the benchmark and disclosed protocol in:

> M. Turdalyuly, A. Zholdassova, T. Turdalykyzy, and A. Doshybekov, "Wearable Sensor-Free Adult Physical Activity Monitoring Using Smartphone IMU Signals: Cross-Subject Deep Learning with Window-Length and Sensor Modality Studies," *Information*, vol. 17, no. 4, p. 368, Apr. 2026. DOI: 10.3390/info17040368.

Dataset reference:

> G. M. Weiss, K. Yoneda, and T. Hayajneh, "Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living," *IEEE Access*, vol. 7, pp. 133190-133202, 2019. DOI: 10.1109/ACCESS.2019.2940729.
