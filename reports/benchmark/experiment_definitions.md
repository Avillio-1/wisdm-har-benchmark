# Comparative WISDM Benchmark Definitions

## Research question

Which sensor setup, representation, and model family generalize best to unseen subjects under subject-wise evaluation?

## A priori task definitions

The benchmark uses three tasks chosen before looking at clean test-set performance:

- `task3`: walking (`A`), jogging (`B`), standing (`E`). This is the sanity-check locomotion/posture task.
- `task6`: walking (`A`), jogging (`B`), stairs (`C`), sitting (`D`), standing (`E`), typing (`F`). This is the medium-difficulty task combining locomotion, posture, and a hand/phone interaction.
- `task18`: all 18 WISDM activities. This is the hard benchmark.

The earlier reduced 3-class result chosen by test-set F1 is treated as contaminated and is not used to define these tasks.

## Sensor ablations

- `phone_accel`: phone accelerometer only.
- `phone_accel_gyro`: phone accelerometer plus phone gyroscope.
- `watch_accel`: smartwatch accelerometer only.
- `watch_accel_gyro`: smartwatch accelerometer plus smartwatch gyroscope.
- `phone_watch_fusion`: phone accelerometer, phone gyroscope, watch accelerometer, and watch gyroscope.

Fusion is performed at the feature-window level using subject, activity, split, and within-activity window index. This is a reproducible approximation rather than exact timestamp-level sensor synchronization.

## Representations

- `stats`: time-domain handcrafted window statistics.
- `stats_freq`: time-domain statistics plus frequency-domain FFT features.

## Model families

- Majority class baseline.
- Logistic regression.
- Random forest.
- LightGBM.
- XGBoost.
- Raw-sequence CNN is retained as the deep learning baseline where raw sequence tensors are generated and trained separately.

## Evaluation protocol

- Subjects are split before windowing.
- No subject appears in more than one split.
- Windows are generated within `(subject_id, activity_label)` groups, so no window crosses activity boundaries.
- Classical scaling is fit on train only through sklearn pipelines.
- CNN normalization is fit on train-only sequence windows.
- Validation is for model choice and sanity checks.
- Test is held out for final reporting.
- GroupKFold stability uses subject groups and is run on train+validation subjects unless explicitly marked otherwise.
