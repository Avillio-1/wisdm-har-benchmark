# Deep Learning Baseline Comparison

## Raw-sequence TensorFlow CNN

The raw sequence CNN was trained with TensorFlow 2.10.1 under Python 3.10 on CPU.

- Input: raw phone accelerometer windows, shape `(window, time, axis)`
- Tensor shape: `(93389, 100, 3)`
- Window setting: 5 seconds, 100 samples, 50% overlap
- Split strategy: subject-wise train/validation/test
- Architecture: Conv1D -> BatchNorm -> MaxPool blocks, global average pooling, dense head
- Normalization: train-window mean/std only
- Early stopping: enabled; best validation loss occurred at epoch 3

## Metrics

| model | input | test accuracy | test macro F1 |
| --- | --- | ---: | ---: |
| random forest | handcrafted window features | 0.3324 | 0.3157 |
| LightGBM | handcrafted window features | 0.3226 | 0.3094 |
| logistic regression | handcrafted window features | 0.2921 | 0.2604 |
| raw-sequence TensorFlow CNN | raw `(time, x, y, z)` windows | 0.2798 | 0.2443 |
| sklearn MLP | handcrafted window features | 0.2846 | 0.2763 |
| majority class | label prior only | 0.0562 | 0.0059 |

## Interpretation

The CNN improves far beyond the majority-class baseline, but it does not beat the classical feature baselines. It reaches 0.588 train accuracy by epoch 11, while validation accuracy peaks around 0.287, which indicates subject-generalization difficulty rather than simple underfitting.

Per-class results show strong performance on jogging, standing, walking, and kicking, but very weak or zero recall for some hand/object activities such as sitting and soup. This suggests the raw accelerometer stream alone may need a better architecture, more regularization, subject-invariant augmentation, or multi-sensor fusion to generalize across held-out subjects.

## Recommendation

Keep random forest or LightGBM as the current project baseline. Treat the TensorFlow CNN as the first deep-learning baseline and improve it next with one of:

- LSTM or GRU sequence model on the same tensors.
- CNN plus temporal attention.
- Data augmentation: jitter, scaling, random crop, time masking.
- Subject-grouped cross-validation to reduce split-specific conclusions.
- Multi-stream fusion with phone gyroscope or watch accelerometer.

Saved TensorFlow outputs are in `data/processed/raw_sequence_deep_baseline/cnn/`.
