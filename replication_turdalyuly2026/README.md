# Turdalyuly 2026 WISDM Replication Package

This folder contains an isolated replication and same-protocol benchmark for Turdalyuly et al. 2026, *Human Activity Recognition Using the WISDM Smartphone and Smartwatch Dataset*:

https://www.mdpi.com/2078-2489/17/4/368

The purpose of this package is narrow and academic: match the paper's disclosed WISDM smartphone IMU methodology as closely as possible, audit the remaining uncertainty, and compare our proposed models against the reported smartphone-only CNN benchmark under the same disclosed data condition.

## Bottom Line

We did the best available replication from the published paper, but the paper does not publish exact source code, fold subject IDs, or every preprocessing edge case. Therefore this repository should describe the work as a **same-disclosed-protocol comparison**, not as an exact clone of the authors' hidden implementation.

Primary benchmark target from Turdalyuly et al.:

| Source | Condition | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Table 1 | CNN, phone accelerometer + gyroscope, 4.0 s windows | 0.4716 +/- 0.0596 | 0.4626 +/- 0.0408 |

Our current same-protocol results:

| Model | Input | Window | Evaluation | Macro-F1 |
|---|---|---:|---|---:|
| Replication CNN calibration | phone accelerometer + gyroscope | 4.0 s | 5-fold GroupKFold by subject | 0.5469 +/- 0.0564 |
| XGBoost, statistical + frequency features | phone accelerometer + gyroscope | 4.0 s | same folds | 0.6697 +/- 0.0334 |

The CNN calibration is higher than the published CNN result, which is useful but also a warning: our implementation matches the disclosed protocol, but it is not numerically identical to the authors' unreleased implementation. The safest paper claim is that our proposed feature-based model outperforms the reported CNN under the disclosed smartphone IMU protocol.

## What Was Replicated

| Component | Replication choice |
|---|---|
| Dataset | WISDM Smartphone and Smartwatch Activity and Biometrics Dataset, UCI ID 507 |
| Raw sampling rate | 20 Hz |
| Subjects | 51 subjects |
| Original activities | 18 WISDM activity labels |
| Primary input | phone accelerometer + phone gyroscope |
| Channels | accelerometer x/y/z and gyroscope x/y/z |
| Window lengths | 2.0 s, 4.0 s, 6.0 s |
| Overlap | 50% |
| Window label | majority vote over grouped labels |
| Evaluation | 5-fold `GroupKFold` by `subject_id` |
| Normalization | fitted on each training fold only |
| Primary metric | macro-F1 |
| Secondary metrics | accuracy, weighted-F1, per-class precision/recall/F1, confusion matrices |

The primary comparison uses **phone accelerometer + phone gyroscope** because that is the same smartphone IMU condition used by the paper's reported CNN benchmark. Other sensor configurations should be treated as separate ablations, not as the headline comparison against this paper.

## Six-Class Taxonomy

The paper reports a reduced six-category task. This implementation fixes the following documented mapping from the official WISDM activity labels:

| Group | WISDM activities |
|---|---|
| `locomotion` | walking, jogging |
| `stairs` | stairs |
| `static` | sitting, standing |
| `eat_drink` | soup, chips, pasta, drinking, sandwich |
| `sports` | kicking, catch, dribbling |
| `upper_body` | typing, teeth, writing, clapping, folding |

This mapping is encoded in `config.py` so the paper, code, and results use the same taxonomy.

## Folder Layout

```text
replication_turdalyuly2026/
+-- README.md                     # this research and reproduction guide
+-- requirements.txt              # non-PyTorch Python dependencies
+-- .gitignore                    # keeps large/generated artifacts out of Git
+-- config.py                     # paper targets, paths, labels, constants
+-- utils.py                      # shared loading, metrics, reporting helpers
+-- 01_prepare_windows.py         # align phone accel/gyro and build windows/features
+-- 02_replicate_cnn.py           # PyTorch CNN calibration under the disclosed protocol
+-- 03_train_feature_models.py    # majority, logistic regression, RF, LightGBM, XGBoost
+-- 04_make_report.py             # combined paper-ready Markdown report
+-- 05_audit_fairness.py          # leakage, condition matching, and claim-safety audit
+-- reports/                      # small Markdown/CSV reports worth committing
+-- results/                      # small fold metrics and confusion matrices worth committing
+-- data/                         # ignored; regenerated tensors/features/fold tables
+-- smoke_*                       # ignored; local smoke-test artifacts
+-- _wording_tmp_*                # ignored; local temporary report-generation artifacts
```

The Git-friendly rule is:

- Track source scripts, `README.md`, `requirements.txt`, `reports/`, and small CSV metrics in `results/`.
- Do not track generated tensors, Parquet feature tables, raw/cleaned datasets, model checkpoints, cache files, or smoke-test outputs.

## Data Requirements

This package expects the main project preprocessing step to have already produced:

```text
data/processed/phone_accel_clean.csv.gz
data/processed/phone_gyro_clean.csv.gz
```

Those files are derived from the UCI WISDM Smartphone and Smartwatch Activity and Biometrics Dataset. The raw dataset itself should not be committed to Git. If the cleaned files are missing, run the main project preprocessing first:

```powershell
python preprocess.py --device phone --sensor accel --report reports/preprocessing/preprocessing_report.md
python preprocess.py --device phone --sensor gyro --report reports/preprocessing/preprocessing_report_phone_gyro.md
```

## Environment

Use Python 3.10 or newer. Install the classical-model dependencies from this folder:

```powershell
python -m pip install -r replication_turdalyuly2026/requirements.txt
```

The CNN calibration requires PyTorch. For the run recorded in this repository, CUDA-visible PyTorch was installed with:

```powershell
python -m pip install --no-cache-dir --timeout 1000 --retries 10 torch --index-url https://download.pytorch.org/whl/cu128
```

The CNN script requires CUDA by default. CPU execution is intentionally allowed only for smoke tests with `--allow-cpu`, because the full CNN run is long and the recorded experiment was run on GPU.

## Reproduction Commands

Run commands from the repository root.

### 1. Prepare aligned windows and feature tables

```powershell
python replication_turdalyuly2026/01_prepare_windows.py
```

Outputs:

- `data/alignment_summary.json`
- `data/phone_accel_gyro_*_windows.npz`
- `data/phone_accel_gyro_*_metadata.parquet`
- `data/phone_accel_gyro_*_features.parquet`
- `data/phone_accel_gyro_*_fold_subjects.csv`
- `data/phone_accel_gyro_*_fold_class_counts.csv`
- `reports/preparation_report.md`

### 2. Run the CNN calibration

Full modality/window grid:

```powershell
python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 2.0,4.0,6.0 --modalities accel,gyro,accel_gyro
```

Primary 4.0 s phone accelerometer + gyroscope calibration only:

```powershell
python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 4.0 --modalities accel_gyro
```

Outputs:

- `results/cnn/cnn_replication_summary.csv`
- `results/cnn/cnn_replication_fold_metrics.csv`
- `results/cnn/cnn_replication_per_class.csv`
- `results/cnn/confusion_matrices/`
- `reports/cnn_calibration_report.md`

### 3. Train same-protocol feature models

```powershell
python replication_turdalyuly2026/03_train_feature_models.py --window-sizes 4.0 --include-secondary
```

Models:

- majority baseline
- logistic regression on statistical features
- random forest on statistical features
- LightGBM on statistical + frequency features
- XGBoost on statistical + frequency features

Outputs:

- `results/feature_models/feature_models_summary.csv`
- `results/feature_models/feature_models_fold_metrics.csv`
- `results/feature_models/feature_models_per_class.csv`
- `results/feature_models/confusion_matrices/`
- `reports/feature_model_report.md`

### 4. Build the combined report

```powershell
python replication_turdalyuly2026/04_make_report.py
```

Main output:

```text
replication_turdalyuly2026/reports/turdalyuly2026_replication_report.md
```

### 5. Run the fairness audit

```powershell
python replication_turdalyuly2026/05_audit_fairness.py
```

Main output:

```text
replication_turdalyuly2026/reports/fairness_audit.md
```

The fairness audit is the most important document to consult before writing claims in the IEEE paper.

## Smoke Test

For a small local check that does not regenerate the full package:

```powershell
python replication_turdalyuly2026/01_prepare_windows.py --data-dir replication_turdalyuly2026/smoke_data --reports-dir replication_turdalyuly2026/smoke_reports --window-sizes 4.0 --max-subjects 5 --force
python replication_turdalyuly2026/03_train_feature_models.py --data-dir replication_turdalyuly2026/smoke_data --results-dir replication_turdalyuly2026/smoke_results --reports-dir replication_turdalyuly2026/smoke_reports --window-sizes 4.0 --models majority,logistic_regression --max-folds 2 --n-estimators 10
python replication_turdalyuly2026/02_replicate_cnn.py --data-dir replication_turdalyuly2026/smoke_data --results-dir replication_turdalyuly2026/smoke_results --reports-dir replication_turdalyuly2026/smoke_reports --window-sizes 4.0 --modalities accel_gyro --max-folds 1 --allow-cpu
python replication_turdalyuly2026/04_make_report.py --data-dir replication_turdalyuly2026/smoke_data --results-dir replication_turdalyuly2026/smoke_results --reports-dir replication_turdalyuly2026/smoke_reports --window-sizes 4.0
```

Smoke-test outputs are ignored by Git.

## Current Prepared Data Summary

The full preparation run produced exact timestamp-aligned phone accelerometer + gyroscope windows for all 51 subjects.

| Window | Total windows | Notes |
|---:|---:|---|
| 2.0 s | 136,268 | 50% overlap |
| 4.0 s | 68,094 | primary comparison window |
| 6.0 s | 45,374 | secondary window-length analysis |

The primary 4.0 s fold audit found no subject leakage:

| Fold | Train subjects | Evaluation subjects | Evaluation windows | Subject overlap | Missing eval classes |
|---:|---:|---:|---:|---:|---:|
| 1 | 41 | 10 | 13,561 | 0 | 0 |
| 2 | 41 | 10 | 13,673 | 0 | 0 |
| 3 | 41 | 10 | 13,562 | 0 | 0 |
| 4 | 40 | 11 | 13,648 | 0 | 0 |
| 5 | 41 | 10 | 13,650 | 0 | 0 |

## Current Result Snapshot

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

Feature-model benchmark on the primary 4.0 s phone accelerometer + gyroscope windows:

| Model | Feature set | Macro-F1 |
|---|---|---:|
| majority | none | 0.0730 +/- 0.0039 |
| logistic regression | statistical | 0.5720 +/- 0.0267 |
| random forest | statistical | 0.6061 +/- 0.0371 |
| LightGBM | statistical + frequency | 0.6628 +/- 0.0348 |
| XGBoost | statistical + frequency | 0.6697 +/- 0.0334 |

Best-model per-class F1 for XGBoost:

| Class | F1 |
|---|---:|
| locomotion | 0.9157 |
| sports | 0.8019 |
| stairs | 0.8011 |
| eat_drink | 0.5824 |
| upper_body | 0.5458 |
| static | 0.3713 |

The low static-class F1 is important for the paper discussion: the proposed method improves macro-F1 overall, but not every grouped class is equally solved.

## Fairness And Claim Rules

Use this wording style in the paper:

> Under the disclosed Turdalyuly et al. WISDM smartphone IMU protocol, using phone accelerometer + gyroscope signals, six grouped activity categories, 4.0 s windows, 50% overlap, majority-vote window labels, and 5-fold subject-wise GroupKFold evaluation, our XGBoost model using statistical and frequency-domain features achieved higher macro-F1 than the reported CNN benchmark.

Avoid these claims:

- Do not say we perfectly reproduced the authors' CNN.
- Do not imply the authors' exact fold IDs or preprocessing code were available.
- Do not compare watch or phone+watch fusion results as the headline Turdalyuly comparison.
- Do not describe the classical feature models as the same model family as the CNN.
- Do not rely on accuracy alone; macro-F1 is the primary metric.

The exact current audit verdict is:

```text
Protocol-fair proposed-method comparison, not exact hidden-code replication.
```

## Known Replication Limits

The following limits should be acknowledged in the IEEE paper:

- The authors' exact implementation code was not available.
- The exact fold subject IDs were not published.
- The exact CNN layer widths and some preprocessing edge cases were not fully specified.
- Timestamp alignment details were underspecified; this package uses an exact inner join on subject and timestamp.
- The six-class mapping was named in the paper but not fully enumerated as a label-code table, so this package documents its fixed interpretation.
- Our CNN calibration result is higher than the published CNN result, so the CNN calibration should be treated as a sanity check rather than proof of exact reproduction.

## Paper-Writing Pointers

Use this package for these IEEE paper sections:

- **Dataset Exploration / Preprocessing**: cite `reports/preparation_report.md` and `data/*_window_summary.csv`.
- **Methodology**: cite `config.py`, `01_prepare_windows.py`, `02_replicate_cnn.py`, and `03_train_feature_models.py`.
- **Experimental Results**: cite `reports/turdalyuly2026_replication_report.md`.
- **Discussion / Limitations**: cite `reports/fairness_audit.md`.
- **Reproducibility**: cite this README, the command sequence above, and the committed result CSVs.

The strongest evidence files to keep in the GitHub repository are:

```text
replication_turdalyuly2026/reports/turdalyuly2026_replication_report.md
replication_turdalyuly2026/reports/fairness_audit.md
replication_turdalyuly2026/results/cnn/cnn_replication_summary.csv
replication_turdalyuly2026/results/feature_models/feature_models_summary.csv
replication_turdalyuly2026/results/feature_models/feature_models_fold_metrics.csv
replication_turdalyuly2026/results/feature_models/feature_models_per_class.csv
```

## Quick Integrity Checklist

Before pushing or using these results in the paper:

- `git status --short --untracked-files=all replication_turdalyuly2026` should show source, reports, and small CSV result files, but not `data/`, `smoke_*`, `_wording_tmp_*`, `__pycache__/`, `.npz`, or `.parquet` files.
- `reports/fairness_audit.md` should show zero subject overlap in every fold.
- `reports/fairness_audit.md` should show zero missing evaluation classes for the primary 4.0 s folds.
- `results/feature_models/feature_models_summary.csv` should contain the XGBoost 4.0 s macro-F1 result.
- `results/cnn/cnn_replication_summary.csv` should contain the CNN calibration result for `accel_gyro`, `4.0`.
