# Reduced Class Task Report

## Class filtering decision

The original 18-class TensorFlow CNN had several classes with weak held-out subject F1. Using a threshold of `F1 >= 0.60`, the retained classes are:

| label | activity | original CNN test F1 |
| --- | --- | ---: |
| A | walking | 0.6402 |
| B | jogging | 0.8237 |
| E | standing | 0.6648 |

All other classes were removed for this reduced task.

Important caveat: this filtering rule uses test-set performance from the original 18-class experiment, so it should be described as a scoped project reframing rather than an unbiased model-selection procedure. The original 18-class results should remain in the report for transparency.

## Reduced dataset

The reduced raw-sequence dataset keeps the same subject-wise train/validation/test split and regenerates raw accelerometer tensors only for walking, jogging, and standing.

| split | windows |
| --- | ---: |
| train | 11,463 |
| validation | 2,392 |
| test | 2,071 |

Tensor shape:

```text
X: (15926, 100, 3)
y: (15926,)
labels: A, B, E
window: 5 seconds, 50% overlap
```

## TensorFlow CNN results

| split | accuracy | macro F1 | weighted F1 |
| --- | ---: | ---: | ---: |
| train | 0.9971 | 0.9971 | 0.9971 |
| validation | 0.9452 | 0.9451 | 0.9451 |
| test | 0.9464 | 0.9511 | 0.9461 |

## Test per-class metrics

| label | activity | precision | recall / per-class accuracy | F1 | support |
| --- | --- | ---: | ---: | ---: | ---: |
| A | walking | 0.8737 | 0.9987 | 0.9320 | 762 |
| B | jogging | 1.0000 | 0.8556 | 0.9222 | 762 |
| E | standing | 0.9982 | 1.0000 | 0.9991 | 547 |

## Interpretation

The reduced task is much cleaner and the raw-window CNN performs well under held-out subject evaluation. Most remaining errors are walking/jogging confusions, while standing is nearly perfectly separated from the two locomotion classes.

## Saved artifacts

- Reduced sequence tensors: `data/processed/sequences/phone_accel_rawseq_5p0s_50overlap_A_B_E.npz`
- Reduced metadata: `data/processed/sequences/phone_accel_rawseq_5p0s_50overlap_A_B_E_metadata.csv`
- CNN metrics and reports: `data/processed/raw_sequence_deep_baseline_reduced/cnn/`
- Run report: `raw_sequence_cnn_reduced_results.md`
