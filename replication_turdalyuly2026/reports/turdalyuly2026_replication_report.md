# Turdalyuly et al. 2026 WISDM507 Replication Benchmark

Source paper: https://www.mdpi.com/2078-2489/17/4/368

## Goal

Replicate the disclosed Turdalyuly et al. 2026 WISDM507 smartphone IMU condition, then test whether classical feature models beat their CNN benchmark on the same phone accelerometer + gyroscope input.

## Replicated Conditions

- Dataset: WISDM507 / WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.
- Input: phone accelerometer + phone gyroscope, six synchronized channels.
- Windowing: 50% overlap, majority-vote labels, 2.0 s, 4.0 s, 6.0 s windows.
- Grouping: locomotion, stairs, static, eat_drink, sports, upper_body.
- Evaluation: 5-fold GroupKFold by subject.
- Paper target: 4.0 s CNN macro-F1 0.4626 +/- 0.0408.

## Prepared Data

| window_seconds | group_label | windows | subjects | candidate_artifact                |
| -------------- | ----------- | ------- | -------- | --------------------------------- |
| 2.0            | eat_drink   | 38287   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | locomotion  | 14226   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | sports      | 20755   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | stairs      | 7479    | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | static      | 16173   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | upper_body  | 39348   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 4.0            | eat_drink   | 19140   | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | locomotion  | 7114    | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | sports      | 10364   | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | stairs      | 3736    | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | static      | 8080    | 51       | phone_accel_gyro_4p0s_windows.npz |
| 4.0            | upper_body  | 19660   | 51       | phone_accel_gyro_4p0s_windows.npz |
| 6.0            | eat_drink   | 12755   | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | locomotion  | 4736    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | sports      | 6902    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | stairs      | 2495    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | static      | 5374    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | upper_body  | 13112   | 51       | phone_accel_gyro_6p0s_windows.npz |

## CNN Calibration

The replicated 4.0 s CNN macro-F1 is 0.5469 +/- 0.0564; paper target is 0.4626 +/- 0.0408. Within one paper std: False.

| model           | modality   | window_seconds | folds | eval_windows_total | accuracy_mean      | accuracy_std       | macro_f1_mean      | macro_f1_std       | weighted_f1_mean   | weighted_f1_std    |
| --------------- | ---------- | -------------- | ----- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| cnn_replication | accel      | 2.0            | 5     | 136268             | 0.510687205664041  | 0.0336951019722482 | 0.5372042893761135 | 0.0486901667957562 | 0.5118560613002634 | 0.0314052502144996 |
| cnn_replication | gyro       | 2.0            | 5     | 136268             | 0.5399590721365876 | 0.0536331961082659 | 0.5204220635743055 | 0.0581387201155561 | 0.5196011652849661 | 0.0471993862671072 |
| cnn_replication | accel_gyro | 2.0            | 5     | 136268             | 0.522801697348201  | 0.0205001874283604 | 0.5515426067394253 | 0.0311416807980515 | 0.5244751903392055 | 0.0198666880090654 |
| cnn_replication | accel      | 4.0            | 5     | 68094              | 0.5274756634772542 | 0.0349247393539051 | 0.5669633395913495 | 0.0443898220815618 | 0.5251669069178051 | 0.0341083118641662 |
| cnn_replication | gyro       | 4.0            | 5     | 68094              | 0.5724254149284549 | 0.0205215090689513 | 0.5103294216466685 | 0.0233525533075669 | 0.5259658030257093 | 0.0298665951693474 |
| cnn_replication | accel_gyro | 4.0            | 5     | 68094              | 0.5235214423317733 | 0.0572872672783491 | 0.546923274802144  | 0.0564400589439379 | 0.5196053567982013 | 0.0603436687576888 |
| cnn_replication | accel      | 6.0            | 5     | 45374              | 0.5133199601103734 | 0.0313648008006849 | 0.5500404070414464 | 0.0295274782211356 | 0.5127762088828606 | 0.0273734737906061 |
| cnn_replication | gyro       | 6.0            | 5     | 45374              | 0.5977035716777931 | 0.0085954356876837 | 0.5515213432657351 | 0.0225214092870576 | 0.5575384817108907 | 0.0089884013549716 |
| cnn_replication | accel_gyro | 6.0            | 5     | 45374              | 0.5230287635933453 | 0.0440890268506257 | 0.5626933839371974 | 0.0445324618218434 | 0.5221150246203817 | 0.042118828746941  |

## Feature-Model Benchmark

The best 4.0 s phone accelerometer + gyroscope model is `xgboost` with macro-F1 0.6697 +/- 0.0334; this strongly exceeds the paper target by more than one reported standard deviation (delta +0.2071 versus 0.4626).

| model               | feature_set | window_seconds | folds | eval_windows_total | accuracy_mean      | accuracy_std       | macro_f1_mean      | macro_f1_std       | weighted_f1_mean   | weighted_f1_std    |
| ------------------- | ----------- | -------------- | ----- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| xgboost             | stats_freq  | 4.0            | 5     | 68094              | 0.6316453905542497 | 0.0273788377836874 | 0.6696767547374132 | 0.0333759307902816 | 0.6284661199346031 | 0.0241490048840832 |
| lightgbm            | stats_freq  | 4.0            | 5     | 68094              | 0.6183535395964774 | 0.0268122395161903 | 0.6628021049261827 | 0.0348449207748346 | 0.6177398644561055 | 0.0223443721907105 |
| random_forest       | stats       | 4.0            | 5     | 68094              | 0.5646688704359928 | 0.0257799382644004 | 0.606061490266457  | 0.0371352891987638 | 0.561723037506028  | 0.0219707963381212 |
| logistic_regression | stats       | 4.0            | 5     | 68094              | 0.518312633352408  | 0.0211048630011231 | 0.5719610587288855 | 0.0266658869783083 | 0.5147760932052481 | 0.0161960723188362 |
| majority            | none        | 4.0            | 5     | 68094              | 0.2807477667099849 | 0.0189649554430555 | 0.0730112798843165 | 0.0038785142262468 | 0.1234278541140707 | 0.0146593188480439 |

## Best Model Per-Class Performance

| class      | precision          | recall              | f1-score           | support |
| ---------- | ------------------ | ------------------- | ------------------ | ------- |
| locomotion | 0.9336322770569181 | 0.9002067367529977  | 0.915669206648556  | 1422.8  |
| sports     | 0.806205411858328  | 0.799681983348014   | 0.8018940646470452 | 2072.8  |
| stairs     | 0.8032030132149355 | 0.8070409798293146  | 0.8010505646912657 | 747.2   |
| eat_drink  | 0.5629804255435008 | 0.6111884443413462  | 0.5823927530170547 | 3828.0  |
| upper_body | 0.53105470124238   | 0.5644713181017758  | 0.5457659056738349 | 3932.0  |
| static     | 0.4946696666929439 | 0.30730274774473576 | 0.3712880337467227 | 1616.0  |

## Claim Safety Notes

- The headline claim must compare models using the same smartphone IMU input: phone accelerometer + gyroscope.
- The CNN calibration does not exactly reproduce the published value, so phrase results as an independently implemented reproduction under disclosed conditions, not a clone of the authors' hidden implementation.
- Do not use watch or phone+watch fusion as the headline win; those are broader ablations, not the same input condition.
- Report macro-F1 as primary and accuracy as secondary.
- Use the feature-model result as the proposed-method comparison; use the CNN result only as a calibration check.

## Reproduction Commands

```powershell
python replication_turdalyuly2026/01_prepare_windows.py
python replication_turdalyuly2026/02_replicate_cnn.py --window-sizes 2.0,4.0,6.0 --modalities accel,gyro,accel_gyro
python replication_turdalyuly2026/03_train_feature_models.py --window-sizes 4.0 --include-secondary
python replication_turdalyuly2026/04_make_report.py
python replication_turdalyuly2026/05_audit_fairness.py
```
