# comparative_benchmark.py Runtime Audit

## What Was Profiled

A bounded pre-refactor profile was run on:

`task3 / phone_accel / stats_freq / majority / skip-cv / force-features`

The wall-clock run took about 82 seconds. The profiler recorded about 75.3 seconds after imports.

## Real Bottlenecks

- Feature construction dominated runtime, not the majority model.
- `build_feature_table` took about 62.4 seconds cumulative.
- `generate_stream_features` took about 59.1 seconds cumulative.
- Per-window `extract_features` took about 40.7 seconds cumulative.
- Repeated per-window stats and percentile calls took about 23.1 seconds cumulative.
- Repeated CSV reads took about 16.9 seconds cumulative.
- Gzip CSV writes took about 6.6 seconds cumulative.
- Per-window FFT and correlation calls were also visible hotspots.

## Fixes Implemented

- Replaced pandas row/window loops with batched NumPy window tensors per subject/activity segment.
- Vectorized window statistics, percentiles, correlations, and FFT feature extraction.
- Preserved `float64` feature extraction so generated features match the old implementation to numerical noise.
- Added in-run clean-stream caching so the same cleaned sensor CSV is not re-read repeatedly.
- Added seed-specific subject split caches.
- Added seed-specific persistent feature caches with `--cache-format auto`, using parquet when `pyarrow` is available.
- Added compatibility fallback reads for existing parquet, pickle, and gzip CSV caches.
- Reused each feature table for every model in the same task/sensor/representation loop.
- Added timestamped progress logging and timings for task, sensor, representation, feature generation, model fitting, and grouped CV stages.
- Added `--n-jobs` for CPU model backends where it is actually useful.
- Added a safe worker fallback: if Windows denies parallel worker setup, the model retries with `n_jobs=1` instead of crashing.
- Added `--use-gpu` XGBoost CUDA detection using a tiny actual CUDA fit before enabling `device="cuda"`.

## GPU Reality Check

- sklearn logistic regression is CPU-only here. It is intentionally kept single-process because this solver gets little practical speedup from `n_jobs` and Windows joblib process spawning can fail.
- sklearn random forest is CPU-only. The script tries `n_jobs=-1` by default and falls back to `n_jobs=1` if worker setup is blocked.
- LightGBM is configured as CPU LightGBM with `n_jobs`; the script does not claim LightGBM GPU acceleration.
- XGBoost can use CUDA on this machine when `--use-gpu` is passed; the tiny runtime check succeeded and `device="cuda"` was used in the smoke test.
- TensorFlow is not installed in the active Python environment used by `comparative_benchmark.py`; the separate CNN path cannot use TensorFlow GPU from this environment.
- CuPy is not installed, so FFT/stat feature extraction stays on NumPy CPU.

## Validation

- Refactored bounded forced feature run completed in about 20 to 26 seconds depending on whether the seed split cache already existed.
- The feature-generation stage itself dropped from roughly 59 seconds cumulative to about 11 seconds including the cleaned CSV read; once the stream is already loaded, the vectorized feature extraction is only a few seconds on the profiled slice.
- Cached reruns of the same slice now complete in about 10 seconds, mostly import/startup and report overhead.
- Old and new `task3 / phone_accel / stats_freq` features have identical shape, columns, and metadata rows.
- Worst numeric absolute difference versus the old implementation after restoring `float64` extraction: about `4.7e-10`.
- A compact all-model smoke test completed for majority, logistic regression, random forest, LightGBM, and CUDA XGBoost.

## Methodology

The runtime refactor does not change the scientific meaning of the experiments. Subject-wise splitting still happens before windowing, windows are still generated inside each subject/activity segment, windows cannot cross activity boundaries, sklearn scaling remains inside train-fitted pipelines, and the task/sensor/representation/model matrix is unchanged.
