# Phone vs Watch Accelerometer Reduced CNN Report

## Task

This comparison uses the reduced 3-class activity task:

| label | activity |
| --- | --- |
| A | walking |
| B | jogging |
| E | standing |

Both models use:

- Raw accelerometer windows shaped `(window, time, x/y/z)`.
- 5 second windows with 50% overlap.
- The same subject-wise train/validation/test split.
- The same TensorFlow 1D CNN architecture and training settings.
- Train-only normalization statistics.

## Dataset sizes

| stream | train windows | val windows | test windows | total windows |
| --- | ---: | ---: | ---: | ---: |
| phone accelerometer | 11,463 | 2,392 | 2,071 | 15,926 |
| watch accelerometer | 9,070 | 1,704 | 1,493 | 12,267 |

## Test metrics

| stream | test accuracy | test macro F1 | test weighted F1 |
| --- | ---: | ---: | ---: |
| phone accelerometer | 0.9464 | 0.9511 | 0.9461 |
| watch accelerometer | 0.9350 | 0.9347 | 0.9347 |

## Per-class test metrics

| stream | label | activity | precision | recall / per-class accuracy | F1 | support |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| phone | A | walking | 0.8737 | 0.9987 | 0.9320 | 762 |
| phone | B | jogging | 1.0000 | 0.8556 | 0.9222 | 762 |
| phone | E | standing | 0.9982 | 1.0000 | 0.9991 | 547 |
| watch | A | walking | 0.9674 | 0.8350 | 0.8963 | 497 |
| watch | B | jogging | 1.0000 | 0.9698 | 0.9847 | 497 |
| watch | E | standing | 0.8574 | 1.0000 | 0.9232 | 499 |

## Interpretation

Both streams work well for the reduced 3-class task. The phone accelerometer has the best overall test macro F1, while the watch accelerometer is especially strong for jogging. The watch model loses more walking recall and standing precision, suggesting some walking windows are being confused with standing or another retained class.

The phone stream also has more usable reduced-task windows after cleaning and windowing, which may contribute to the small performance advantage.

## Saved artifacts

- Phone CNN results: `raw_sequence_cnn_reduced_results.md`
- Watch CNN results: `raw_sequence_cnn_watch_reduced_results.md`
- Phone tensors: `data/processed/sequences/phone_accel_rawseq_5p0s_50overlap_A_B_E.npz`
- Watch tensors: `data/processed/sequences/watch_accel_rawseq_5p0s_50overlap_A_B_E.npz`
- Watch CNN artifacts: `data/processed/raw_sequence_deep_baseline_watch_reduced/cnn/`
