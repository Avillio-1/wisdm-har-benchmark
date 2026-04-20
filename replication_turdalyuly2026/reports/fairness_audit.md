# Fairness Audit: Turdalyuly et al. 2026 Replication

Source paper: https://www.mdpi.com/2078-2489/17/4/368

**Verdict:** Protocol-fair proposed-method comparison, not exact hidden-code replication.

The current package matches the disclosed dataset, sensor condition, grouped task, window length, overlap, majority-vote labeling, subject-wise GroupKFold evaluation, train-fold normalization, and CNN training hyperparameters. It does **not** exactly reproduce the published CNN number, so the paper should not claim a perfect replication of the authors' hidden implementation.

## Matched Disclosed Conditions

| condition         | paper                                      | ours                                                                       | status                         |
| ----------------- | ------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------ |
| Dataset           | WISDM507                                   | WISDM Smartphone and Smartwatch raw phone streams                          | matched disclosed dataset      |
| Sensor input      | phone accel + gyro, 6 channels             | phone accel + gyro, 6 channels                                             | matched                        |
| Window length     | 4.0 s primary                              | 4.0 s                                                                      | matched                        |
| Overlap           | 50%                                        | 50%                                                                        | matched                        |
| Labeling          | majority vote                              | majority vote                                                              | matched                        |
| Evaluation        | 5-fold GroupKFold by subject               | 5-fold GroupKFold by subject                                               | matched                        |
| Normalization     | train-fold z-score only                    | train-fold z-score only for CNN; sklearn train-fold pipelines for features | matched for implemented models |
| Optimizer         | AdamW, lr=1e-3                             | AdamW, lr=1e-3 for CNN                                                     | matched                        |
| Epochs/batch size | 10 epochs, train batch 256, eval batch 512 | 10 epochs, train batch 256, eval batch 512                                 | matched                        |
| GPU               | CUDA mixed precision when available        | CUDA required by default; fold metrics recorded device=cuda                | matched/improved enforcement   |

## Published CNN Target Checks

| paper_result                              | paper_macro_f1 | paper_std | our_macro_f1      | delta               | within_one_paper_std |
| ----------------------------------------- | -------------- | --------- | ----------------- | ------------------- | -------------------- |
| table1_model_comparison_cnn_4s_accel_gyro | 0.4626         | 0.0408    | 0.546923274802144 | 0.08432327480214397 | False                |
| table3_window_length_cnn_4s               | 0.4473         | 0.0525    | 0.546923274802144 | 0.099623274802144   | False                |
| table4_sensor_ablation_accel_gyro         | 0.4535         | 0.0643    | 0.546923274802144 | 0.09342327480214396 | False                |

## Same-Protocol Proposed-Method Results

| model               | feature_set | window_seconds | folds | eval_windows_total | accuracy_mean      | accuracy_std       | macro_f1_mean      | macro_f1_std       | weighted_f1_mean   | weighted_f1_std    |
| ------------------- | ----------- | -------------- | ----- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| xgboost             | stats_freq  | 4.0            | 5     | 68094              | 0.6316453905542497 | 0.0273788377836874 | 0.6696767547374132 | 0.0333759307902816 | 0.6284661199346031 | 0.0241490048840832 |
| lightgbm            | stats_freq  | 4.0            | 5     | 68094              | 0.6183535395964774 | 0.0268122395161903 | 0.6628021049261827 | 0.0348449207748346 | 0.6177398644561055 | 0.0223443721907105 |
| random_forest       | stats       | 4.0            | 5     | 68094              | 0.5646688704359928 | 0.0257799382644004 | 0.606061490266457  | 0.0371352891987638 | 0.561723037506028  | 0.0219707963381212 |
| logistic_regression | stats       | 4.0            | 5     | 68094              | 0.518312633352408  | 0.0211048630011231 | 0.5719610587288855 | 0.0266658869783083 | 0.5147760932052481 | 0.0161960723188362 |
| majority            | none        | 4.0            | 5     | 68094              | 0.2807477667099849 | 0.0189649554430555 | 0.0730112798843165 | 0.0038785142262468 | 0.1234278541140707 | 0.0146593188480439 |

## Fold Leakage Checks

| fold | train_subjects | eval_subjects | eval_windows | subject_overlap | missing_eval_classes |
| ---- | -------------- | ------------- | ------------ | --------------- | -------------------- |
| 1    | 41             | 10            | 13561        | 0               | 0                    |
| 2    | 41             | 10            | 13673        | 0               | 0                    |
| 3    | 41             | 10            | 13562        | 0               | 0                    |
| 4    | 40             | 11            | 13648        | 0               | 0                    |
| 5    | 41             | 10            | 13650        | 0               | 0                    |

## Fold Class Coverage

| fold | group_label | eval_windows |
| ---- | ----------- | ------------ |
| 1    | locomotion  | 1335         |
| 1    | stairs      | 673          |
| 1    | static      | 1615         |
| 1    | eat_drink   | 3989         |
| 1    | sports      | 1968         |
| 1    | upper_body  | 3981         |
| 2    | locomotion  | 1352         |
| 2    | stairs      | 775          |
| 2    | static      | 1809         |
| 2    | eat_drink   | 3564         |
| 2    | sports      | 2063         |
| 2    | upper_body  | 4110         |
| 3    | locomotion  | 1348         |
| 3    | stairs      | 671          |
| 3    | static      | 1728         |
| 3    | eat_drink   | 3907         |
| 3    | sports      | 1853         |
| 3    | upper_body  | 4055         |
| 4    | locomotion  | 1543         |
| 4    | stairs      | 875          |
| 4    | static      | 1295         |
| 4    | eat_drink   | 3649         |
| 4    | sports      | 2250         |
| 4    | upper_body  | 4036         |
| 5    | locomotion  | 1536         |
| 5    | stairs      | 742          |
| 5    | static      | 1633         |
| 5    | eat_drink   | 4031         |
| 5    | sports      | 2230         |
| 5    | upper_body  | 3478         |

## Alignment Summary

| accel_rows_after_subject_filter | exact_aligned_rows | gyro_rows_after_subject_filter | label_mismatch_rows_dropped | subjects | unknown_label_rows_dropped | usable_aligned_rows |
| ------------------------------- | ------------------ | ------------------------------ | --------------------------- | -------- | -------------------------- | ------------------- |
| 4734730                         | 2727154            | 3544385                        | 0                           | 51       | 0                          | 2727154             |

## Window Summary

| window_seconds | group_label | windows | subjects | candidate_artifact                |
| -------------- | ----------- | ------- | -------- | --------------------------------- |
| 4.0            | eat_drink   | 19140   | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | locomotion  | 7114    | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | sports      | 10364   | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | stairs      | 3736    | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | static      | 8080    | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | upper_body  | 19660   | 51       | phone_accel_gyro_4p0s_windows.npz |

## Unresolved Replication Risks

| issue                                                             | risk                                                                                                                        | action                                                                                                        |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Exact author code unavailable                                     | CNN architecture, layer widths, initialization, fold ordering, and preprocessing edge cases cannot be guaranteed identical. | Do not claim an exact clone; claim independently implemented reproduction under disclosed conditions.         |
| Official supplementary file not retrievable from this environment | MDPI blocked direct supplementary ZIP download attempts, so we could not inspect possible hidden code/configs.              | If a browser download succeeds, compare the official supplement against this package before final submission. |
| 6-class mapping not fully enumerated in the paper text            | Our mapping is a documented interpretation of the named categories using the official WISDM activity key.                   | Report the full mapping in the paper and appendix.                                                            |
| Timestamp alignment details are underspecified                    | The paper says aligned by timestamp, but does not disclose exact-match vs tolerance/resampling behavior.                    | Report exact timestamp inner join and aligned-row counts; avoid claiming identical preprocessing.             |
| Published CNN result is not reproduced numerically                | Our CNN macro-F1 is higher than all reported 4.0 s CNN targets, so the CNN is not a perfect replica.                        | Use CNN as calibration only; base the contribution on same-protocol proposed method comparison.               |
| Classical feature models are a different method family            | They are not a replication of the paper's CNN and should not be described as the same model.                                | Frame XGBoost stats+freq as our proposed model evaluated under the same disclosed data/protocol.              |

## Safe Academic Claim

Under the disclosed Turdalyuly et al. WISDM507 smartphone IMU condition (phone accelerometer + gyroscope), reduced 6-class taxonomy, 4.0 s windows, 50% overlap, majority-vote labeling, and 5-fold subject-wise GroupKFold evaluation, our statistical/frequency feature model outperforms the reported CNN benchmark. Because the authors' exact preprocessing code, fold assignment, and architecture widths are not disclosed and our CNN calibration is numerically higher than theirs, we present this as a same-protocol comparison rather than an exact reproduction of their hidden implementation.
