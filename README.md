# WISDM Human Activity Recognition Benchmark

This project builds a leakage-safe comparative benchmark for human activity recognition on the WISDM smartphone and smartwatch dataset. The main research question is:

Which sensor setup, feature representation, and model family generalize best to unseen subjects under subject-wise evaluation?

The project started with dataset auditing, preprocessing, windowing, EDA, and baseline modeling, then expanded into a stronger benchmark across multiple task definitions, sensor ablations, feature representations, classical ML models, and raw-sequence deep learning baselines.

## Current Status

The main completed benchmark is the classical ML comparative first pass:

- Tasks: `task3`, `task6`, and `task18`
- Sensor setups: phone accelerometer, phone accelerometer plus gyroscope, watch accelerometer, watch accelerometer plus gyroscope, and phone plus watch fusion
- Representations: statistical features and statistical plus frequency-domain features
- Models: majority baseline, logistic regression, random forest, LightGBM, and XGBoost
- Evaluation: subject-wise train/validation/test split, plus GroupKFold stability checks for best configurations

The main output folder is:

```text
data/processed/comparative_first_pass/
```

The main findings summary is:

```text
data/processed/comparative_first_pass/first_pass_findings.md
```

## Project Layout

```text
.
+-- README.md
+-- audit_dataset.py
+-- preprocess.py
+-- splits.py
+-- windowing.py
+-- sequence_windowing.py
+-- baselines.py
+-- grouped_cv_baselines.py
+-- deep_baseline.py
+-- raw_sequence_deep_baseline.py
+-- benchmark_definitions.py
+-- comparative_benchmark.py
+-- src/
|   +-- wisdm.py
+-- wisdm-dataset/
|   +-- raw WISDM source files
+-- data/
|   +-- interim/
|   +-- processed/
+-- figures/
+-- notebooks/
|   +-- eda.ipynb
+-- reports/
|   +-- audit/
|   +-- benchmark/
|   +-- evaluation/
|   +-- modeling/
|   +-- preprocessing/
|   +-- windowing_eda/
+-- Paper/
    +-- wisdm_related_work_report.md
    +-- wisdm_papers.csv
    +-- wisdm_papers.md
```

## Data

The expected raw dataset location is:

```text
wisdm-dataset/
```

The shared helper module `src/wisdm.py` expects WISDM raw files under:

```text
wisdm-dataset/raw/<device>/<sensor>/
```

where `device` is `phone` or `watch`, and `sensor` is `accel` or `gyro`.

Cleaned sensor streams are saved to:

```text
data/processed/phone_accel_clean.csv.gz
data/processed/phone_gyro_clean.csv.gz
data/processed/watch_accel_clean.csv.gz
data/processed/watch_gyro_clean.csv.gz
```

## Task Definitions

The benchmark task definitions live in `benchmark_definitions.py`.

| Task | Labels | Purpose |
|---|---|---|
| `task3` | `A`, `B`, `E` | Sanity-check task: walking, jogging, standing |
| `task6` | `A`, `B`, `C`, `D`, `E`, `F` | A priori medium task: walking, jogging, stairs, sitting, standing, typing |
| `task18` | all WISDM labels in `ACTIVITY_MAP` except missing label gaps | Full hard benchmark |

The 3-class task is retained as a sanity check, not as a test-selected best case. The 6-class and 18-class tasks are defined a priori.

## Methodological Guardrails

This project is intentionally stricter than many WISDM examples:

- Split by subject before model evaluation.
- Avoid naive random row splits.
- Do not let windows cross activity boundaries.
- Fit scaling and preprocessing transforms on training data only.
- Use validation data for model selection.
- Keep the test split untouched for final reporting.
- Report macro F1 as the primary metric, with accuracy as secondary.
- Include per-class F1, confusion matrices, subject-level error analysis, and grouped-CV stability where possible.

These choices are documented in:

```text
reports/evaluation/evaluation_protocol.md
reports/evaluation/evaluation_protocol_clean3.md
reports/audit/methodology_audit_clean3_report.md
```

## Setup

Use Python 3.10 or newer if possible.

Core dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib joblib
```

Benchmark dependencies:

```bash
python -m pip install lightgbm xgboost pyarrow
```

Optional deep-learning dependencies:

```bash
python -m pip install tensorflow
```

or, if using the PyTorch backend in `raw_sequence_deep_baseline.py`:

```bash
python -m pip install torch
```

`pyarrow` is optional but recommended because the comparative benchmark can cache feature tables as Parquet.

## Reproducing The Pipeline

Run commands from the project root.

### 1. Audit The Raw Dataset

```bash
python audit_dataset.py --report reports/audit/dataset_audit.md
```

### 2. Clean Each Sensor Stream

```bash
python preprocess.py --device phone --sensor accel --report reports/preprocessing/preprocessing_report.md
python preprocess.py --device phone --sensor gyro --report reports/preprocessing/preprocessing_report_phone_gyro.md
python preprocess.py --device watch --sensor accel --report reports/preprocessing/preprocessing_report_watch_accel.md
python preprocess.py --device watch --sensor gyro --report reports/preprocessing/preprocessing_report_watch_gyro.md
```

### 3. Create Subject-Wise Splits

```bash
python splits.py --clean-data data/processed/phone_accel_clean.csv.gz --out data/processed/subject_splits.csv --protocol reports/evaluation/evaluation_protocol.md
```

For a scoped label set, pass labels explicitly:

```bash
python splits.py --include-labels A,B,E --protocol reports/evaluation/evaluation_protocol_clean3.md
```

### 4. Create Classical Feature Windows

```bash
python windowing.py --clean-data data/processed/phone_accel_clean.csv.gz --splits data/processed/subject_splits.csv --summary reports/windowing_eda/windowing_summary.md
```

The comparative benchmark also generates and caches its own task/sensor feature tables under its output root.

### 5. Generate EDA

```bash
python eda.py --clean-data data/processed/phone_accel_clean.csv.gz --figures-dir figures --summary reports/windowing_eda/eda_summary.md
```

### 6. Run The Comparative Classical Benchmark

```bash
python comparative_benchmark.py --root data/processed/comparative_first_pass --n-jobs -1 --use-gpu
```

GPU behavior is intentionally conservative:

- XGBoost uses CUDA only if the environment passes a real CUDA fit check.
- sklearn logistic regression and random forest are CPU-only.
- LightGBM is configured as CPU LightGBM in this project.
- FFT/statistical feature generation uses NumPy CPU unless a future GPU-backed path is explicitly added and documented.

To run a smaller smoke test:

```bash
python comparative_benchmark.py --root data/processed/comparative_smoke --tasks task3 --sensors phone_accel --representations stats --skip-cv
```

### 7. Run Raw-Sequence Deep Baselines

Create raw sequence tensors:

```bash
python sequence_windowing.py --clean-data data/processed/phone_accel_clean.csv.gz --splits data/processed/subject_splits.csv --dataset-name phone_accel
```

Train a CNN or LSTM:

```bash
python raw_sequence_deep_baseline.py --backend auto --model cnn --sequences data/processed/sequences/phone_accel_rawseq_5p0s_50overlap.npz --metadata data/processed/sequences/phone_accel_rawseq_5p0s_50overlap_metadata.csv
```

The raw-sequence CNN exists as a separate baseline path. It is not yet fully integrated into the entire 3-task x 5-sensor comparative matrix.

## Main Results

From `data/processed/comparative_first_pass/first_pass_findings.md`:

| Task | Best configuration | Accuracy | Macro F1 |
|---|---|---:|---:|
| `task3` | phone plus watch fusion, `stats_freq`, XGBoost | 0.9943 | 0.9944 |
| `task6` | watch accelerometer plus gyroscope, `stats_freq`, XGBoost | 0.8440 | 0.8469 |
| `task18` | watch accelerometer plus gyroscope, `stats_freq`, LightGBM | 0.6473 | 0.6634 |

Grouped CV stability for the best configuration per task:

| Task | Best configuration | CV macro F1 mean | CV macro F1 std |
|---|---|---:|---:|
| `task3` | phone plus watch fusion, `stats_freq`, XGBoost | 0.9957 | 0.0035 |
| `task6` | watch accelerometer plus gyroscope, `stats_freq`, XGBoost | 0.8535 | 0.0300 |
| `task18` | watch accelerometer plus gyroscope, `stats_freq`, LightGBM | 0.7233 | 0.0220 |

High-level takeaways:

- The 3-class task is a useful sanity check but is nearly solved.
- The 6-class task is a better medium-difficulty benchmark.
- The 18-class task remains the hard benchmark.
- Frequency-domain features helped consistently.
- Watch accelerometer plus gyroscope was strongest on the harder tasks.
- Subject-level variability is meaningful and should stay in the analysis.

## Reports

Key project reports:

- `reports/audit/dataset_audit.md`
- `reports/audit/methodology_audit_clean3_report.md`
- `reports/benchmark/experiment_definitions.md`
- `reports/benchmark/runtime_audit_comparative_benchmark.md`
- `reports/evaluation/evaluation_protocol.md`
- `reports/evaluation/grouped_cv_clean3_report.md`
- `reports/modeling/baseline_results_clean3.md`
- `reports/modeling/raw_sequence_cnn_clean3_results.md`
- `reports/windowing_eda/eda_summary.md`

The literature review package for paper writing is in:

```text
Paper/
```

Start with:

```text
Paper/wisdm_related_work_report.md
```

## Output Artifacts

Important outputs under `data/processed/comparative_first_pass/`:

```text
results/benchmark_results.csv
results/per_class_test_reports.csv
confusion_matrices/
subject_errors/
grouped_cv/groupkfold_best_summary.csv
feature_importance/
figures/
comparative_benchmark_report.md
first_pass_findings.md
run_manifest.json
```

The older `data/processed/comparative/` folder is retained as experiment history. The profiling and smoke-test folders were removed during cleanup because they were temporary runtime diagnostics and can be regenerated.

## Notes For Future Work

The most useful next steps are:

- Integrate raw-sequence CNN results into the same comparative result matrix as the classical models.
- Add a recurrent raw-sequence baseline such as GRU or BiLSTM under the same subject-wise splits.
- Add SHAP or richer feature-importance analysis for the best classical model if runtime allows.
- Continue reporting macro F1, per-class F1, and subject-level error rather than relying on accuracy alone.
