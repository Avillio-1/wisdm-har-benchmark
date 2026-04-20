# Turdalyuly 2026 Window Preparation

This package prepares WISDM507 phone accelerometer + gyroscope windows for a fair replication of Turdalyuly et al. 2026.

## Paper Conditions

- Paper URL: https://www.mdpi.com/2078-2489/17/4/368
- Sensor input: phone accelerometer + phone gyroscope (6 channels).
- Sample rate: 20 Hz.
- Overlap: 0.50.
- Evaluation: 5-fold GroupKFold by subject.
- Window label: majority vote over grouped activity labels.

## Alignment Summary

| accel_rows_after_subject_filter | gyro_rows_after_subject_filter | exact_aligned_rows | label_mismatch_rows_dropped | unknown_label_rows_dropped | usable_aligned_rows | subjects |
| ------------------------------- | ------------------------------ | ------------------ | --------------------------- | -------------------------- | ------------------- | -------- |
| 4734730                         | 3544385                        | 2727154            | 0                           | 0                          | 2727154             | 51       |

## Window Summary

| window_seconds | group_label | windows | subjects | candidate_artifact                |
| -------------- | ----------- | ------- | -------- | --------------------------------- |
| 2.0            | eat_drink   | 38287   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | locomotion  | 14226   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | sports      | 20755   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | stairs      | 7479    | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | static      | 16173   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 2.0            | upper_body  | 39348   | 51       | phone_accel_gyro_2p0s_windows.npz |
| 6.0            | eat_drink   | 12755   | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | locomotion  | 4736    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | sports      | 6902    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | stairs      | 2495    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | static      | 5374    | 51       | phone_accel_gyro_6p0s_windows.npz |
| 6.0            | upper_body  | 13112   | 51       | phone_accel_gyro_6p0s_windows.npz |

## Output Directory

`C:\Users\PC\Desktop\UNI\CS465\Project\replication_turdalyuly2026\data`
