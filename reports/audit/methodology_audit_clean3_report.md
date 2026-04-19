# Methodology Audit and Clean 3-Class Results

## What was wrong

- The previous 3-class task was chosen by keeping classes whose original 18-class CNN test F1 was above 0.60. That uses test-set information to define the task, so those reduced-task results are contaminated and should not be treated as unbiased evidence.
- The reduced task was previously rerun mainly for the CNN, not for all baselines on the exact same clean task.
- The old classical baseline script fit on train+validation by default. That is acceptable only after model selection, but it does not satisfy the stricter requirement that scaling and fitting use train only.
- Feature windowing previously assigned split labels after window generation. Because windows were grouped by subject and activity this did not create leakage, but the implementation did not make "split before windowing" explicit.

## What was fixed

- The 3-class task is now defined a priori as a domain task: walking vs jogging vs standing. This uses labels `A`, `B`, and `E` and is not selected from clean test performance.
- A fresh subject-wise split was created with seed `20260418` and saved at `data/processed/clean3/subject_splits_clean3_seed20260418.csv`.
- Feature windows are now generated after assigning subjects to train/validation/test, and each split is windowed separately.
- Raw CNN sequence windows already carried split assignments before windowing; they were regenerated for the clean split.
- All windows are generated within `(subject_id, activity_label)` groups, so no window crosses a subject or activity boundary.
- Classical baselines now fit on train only. Validation is reported for model choice, and test is held out for final reporting.
- CNN normalization is fit on train sequence windows only.
- A leakage audit was added and passed: `leakage_audit_clean3.md`.
- GroupKFold stability was added on train+validation subjects only, leaving the clean test split untouched.

## Clean task setup

- Sensor stream: phone accelerometer.
- Classes: walking (`A`), jogging (`B`), standing (`E`).
- Window size: 5 seconds.
- Overlap: 50%.
- Subject-wise split: 36 train subjects, 8 validation subjects, 7 test subjects.
- Clean windows: 11,287 train, 2,713 validation, 1,926 test.

## Leakage checks

All checks passed.

- Subject overlap across train/validation/test: zero.
- Missing split assignments: zero.
- Labels present: only `A`, `B`, and `E`.
- Feature windows and sequence windows both match the same split counts.
- Tensor rows match sequence metadata rows.
- Classical scalers are inside train-fitted sklearn pipelines.
- CNN normalization stats are saved from train windows only.

## Main clean test results

| model | input | validation macro F1 | test accuracy | test macro F1 |
| --- | --- | ---: | ---: | ---: |
| random forest | handcrafted features | 0.9772 | 0.9756 | 0.9744 |
| LightGBM | handcrafted features | 0.9754 | 0.9268 | 0.9231 |
| logistic regression | handcrafted features | 0.9665 | 0.9117 | 0.9074 |
| TensorFlow CNN | raw sequence | 0.9470 | 0.8733 | 0.8712 |
| majority class | label prior | 0.1666 | 0.3593 | 0.1762 |

The clean winner is random forest, not the CNN.

## Per-class test F1

| model | walking (`A`) | jogging (`B`) | standing (`E`) |
| --- | ---: | ---: | ---: |
| majority class | 0.5286 | 0.0000 | 0.0000 |
| logistic regression | 0.8632 | 0.8667 | 0.9922 |
| random forest | 0.9662 | 0.9621 | 0.9950 |
| LightGBM | 0.8926 | 0.8825 | 0.9943 |
| TensorFlow CNN | 0.7914 | 0.9250 | 0.8973 |

## Test confusion matrices

Rows are true labels. Columns are predicted labels.

### Majority Class

| true \ pred | A | B | E |
| --- | ---: | ---: | ---: |
| A | 692 | 0 | 0 |
| B | 528 | 0 | 0 |
| E | 706 | 0 | 0 |

### Logistic Regression

| true \ pred | A | B | E |
| --- | ---: | ---: | ---: |
| A | 533 | 152 | 7 |
| B | 7 | 520 | 1 |
| E | 3 | 0 | 703 |

### Random Forest

| true \ pred | A | B | E |
| --- | ---: | ---: | ---: |
| A | 658 | 33 | 1 |
| B | 7 | 520 | 1 |
| E | 5 | 0 | 701 |

### LightGBM

| true \ pred | A | B | E |
| --- | ---: | ---: | ---: |
| A | 582 | 109 | 1 |
| B | 24 | 503 | 1 |
| E | 6 | 0 | 700 |

### TensorFlow CNN

| true \ pred | A | B | E |
| --- | ---: | ---: | ---: |
| A | 461 | 74 | 157 |
| B | 9 | 518 | 1 |
| E | 3 | 0 | 703 |

## GroupKFold stability

GroupKFold was run on train+validation windows only, grouped by `subject_id`, so the held-out clean test split was not used.

| model | folds | mean accuracy | accuracy std | mean macro F1 | macro F1 std |
| --- | ---: | ---: | ---: | ---: | ---: |
| random forest | 5 | 0.9615 | 0.0238 | 0.9624 | 0.0226 |
| LightGBM | 5 | 0.9547 | 0.0273 | 0.9556 | 0.0262 |
| logistic regression | 5 | 0.9521 | 0.0261 | 0.9528 | 0.0251 |
| majority class | 5 | 0.3273 | 0.0265 | 0.1642 | 0.0102 |

The grouped CV results agree with the clean test split: the classical feature baselines are stable, and random forest is the strongest overall.

## Conclusion

The earlier reduced-task CNN result should be treated as contaminated because class selection used test-set F1. After redefining the 3-class task without test-set information and rerunning all baselines under a fresh subject-wise split, the clean result is lower and more credible. The best clean model is random forest on handcrafted 5-second window features, with 0.9756 test accuracy and 0.9744 test macro F1. The TensorFlow CNN is valid under the cleaned protocol, but it underperforms the feature-based classical baselines on the fresh test split.

## Key artifacts

- Clean protocol: `evaluation_protocol_clean3.md`
- Leakage audit: `leakage_audit_clean3.md`
- Clean baseline report: `baseline_results_clean3.md`
- Clean CNN report: `raw_sequence_cnn_clean3_results.md`
- GroupKFold report: `grouped_cv_clean3_report.md`
- Clean report: `methodology_audit_clean3_report.md`
