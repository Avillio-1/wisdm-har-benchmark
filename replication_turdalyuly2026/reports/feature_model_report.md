# Turdalyuly 2026 Feature-Model Benchmark

Best 4.0 s phone accelerometer + gyroscope feature model: xgboost (stats_freq), macro-F1 mean 0.6697. Delta versus Turdalyuly CNN target 0.4626: +0.2071.

## Summary

| model               | feature_set | window_seconds | folds | eval_windows_total | accuracy_mean       | accuracy_std         | macro_f1_mean       | macro_f1_std         | weighted_f1_mean    | weighted_f1_std      |
| ------------------- | ----------- | -------------- | ----- | ------------------ | ------------------- | -------------------- | ------------------- | -------------------- | ------------------- | -------------------- |
| xgboost             | stats_freq  | 4.0            | 5     | 68094              | 0.6316453905542497  | 0.027378837783687403 | 0.6696767547374132  | 0.03337593079028169  | 0.6284661199346031  | 0.0241490048840832   |
| lightgbm            | stats_freq  | 4.0            | 5     | 68094              | 0.6183535395964774  | 0.02681223951619036  | 0.6628021049261827  | 0.034844920774834656 | 0.6177398644561055  | 0.022344372190710577 |
| random_forest       | stats       | 4.0            | 5     | 68094              | 0.5646688704359928  | 0.025779938264400424 | 0.606061490266457   | 0.037135289198763834 | 0.561723037506028   | 0.021970796338121254 |
| logistic_regression | stats       | 4.0            | 5     | 68094              | 0.518312633352408   | 0.021104863001123176 | 0.5719610587288855  | 0.026665886978308392 | 0.5147760932052481  | 0.016196072318836286 |
| majority            | none        | 4.0            | 5     | 68094              | 0.28074776670998497 | 0.018964955443055535 | 0.07301127988431652 | 0.003878514226246859 | 0.12342785411407078 | 0.014659318848043962 |

## Claim Safety

- The headline comparison is valid only for the same smartphone IMU condition: phone accelerometer + gyroscope input.
- Models use the same prepared windows and GroupKFold subject folds as the CNN calibration.
- If the CNN calibration is far from the paper target, report this as an in-repository replication rather than exact paper reproduction.
