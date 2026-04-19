# Dataset Audit

## Short dataset summary

Dataset: WISDM activity recognition raw sensor bundle.

- Participants: 51 subjects, with IDs 1600 through 1650 in the raw filenames.
- Raw streams: phone accelerometer, phone gyroscope, watch accelerometer, watch gyroscope.
- Raw schema: `subject_id`, `activity_label`, `timestamp_ns`, `x`, `y`, `z`; scripts add `device`, `sensor`, `source_file`, and `activity_name` metadata.
- Target label: `activity_label` / `activity_name`, an 18-class human activity label.
- Subject/user ID: `subject_id`.
- Timestamp column: `timestamp_ns`, a high-resolution sensor timestamp used for ordering.
- Sensor columns: `x`, `y`, `z`.
- Train/test split availability: no official split files were found in this bundle.
- Recommended main project task: 18-class activity classification from raw phone accelerometer windows using subject-wise evaluation.

## Raw files by stream

| device | sensor | files | subjects | rows    | mb     |
| ------ | ------ | ----- | -------- | ------- | ------ |
| phone  | accel  | 51    | 51       | 4804403 | 262.58 |
| phone  | gyro   | 51    | 51       | 3608635 | 215.78 |
| watch  | accel  | 51    | 51       | 3777046 | 206.45 |
| watch  | gyro   | 51    | 51       | 3440342 | 199.51 |

## Overall class distribution

| activity_label | activity_name | rows   | percent |
| -------------- | ------------- | ------ | ------- |
| A              | walking       | 886762 | 5.67    |
| B              | jogging       | 862281 | 5.52    |
| C              | stairs        | 841230 | 5.38    |
| D              | sitting       | 875030 | 5.6     |
| E              | standing      | 882587 | 5.65    |
| F              | typing        | 833208 | 5.33    |
| G              | teeth         | 871710 | 5.58    |
| H              | soup          | 869704 | 5.56    |
| I              | chips         | 861398 | 5.51    |
| J              | pasta         | 840358 | 5.38    |
| K              | drinking      | 901381 | 5.77    |
| L              | sandwich      | 857571 | 5.49    |
| M              | kicking       | 882417 | 5.65    |
| O              | catch         | 868766 | 5.56    |
| P              | dribbling     | 882716 | 5.65    |
| Q              | writing       | 871159 | 5.57    |
| R              | clapping      | 869905 | 5.57    |
| S              | folding       | 872243 | 5.58    |

## Recommended task class distribution: phone accelerometer

| activity_label | activity_name | rows   | percent |
| -------------- | ------------- | ------ | ------- |
| A              | walking       | 279817 | 5.82    |
| B              | jogging       | 268409 | 5.59    |
| C              | stairs        | 255645 | 5.32    |
| D              | sitting       | 264592 | 5.51    |
| E              | standing      | 269604 | 5.61    |
| F              | typing        | 246356 | 5.13    |
| G              | teeth         | 269609 | 5.61    |
| H              | soup          | 270756 | 5.64    |
| I              | chips         | 261360 | 5.44    |
| J              | pasta         | 249793 | 5.2     |
| K              | drinking      | 285190 | 5.94    |
| L              | sandwich      | 265781 | 5.53    |
| M              | kicking       | 278766 | 5.8     |
| O              | catch         | 272219 | 5.67    |
| P              | dribbling     | 272730 | 5.68    |
| Q              | writing       | 260497 | 5.42    |
| R              | clapping      | 268065 | 5.58    |
| S              | folding       | 265214 | 5.52    |

## Missing-value report

| column         | missing_rows | missing_percent |
| -------------- | ------------ | --------------- |
| subject_id     | 0            | 0.0             |
| activity_label | 0            | 0.0             |
| timestamp_ns   | 0            | 0.0             |
| x              | 0            | 0.0             |
| y              | 0            | 0.0             |
| z              | 0            | 0.0             |

## Duplicate-row report

| scope                       | duplicate_rows | duplicate_percent |
| --------------------------- | -------------- | ----------------- |
| within individual raw files | 263696         | 1.6871            |

## Quality checks by stream

| device | sensor | rows    | unknown_activity_rows | missing_required_rows | nonpositive_timestamp_rows | source_subject_mismatch_rows | exact_duplicate_rows | duplicate_timestamp_rows | nonmonotonic_timestamp_rows |
| ------ | ------ | ------- | --------------------- | --------------------- | -------------------------- | ---------------------------- | -------------------- | ------------------------ | --------------------------- |
| phone  | accel  | 4804403 | 0                     | 0                     | 0                          | 0                            | 69673                | 139346                   | 18                          |
| phone  | gyro   | 3608635 | 0                     | 0                     | 0                          | 0                            | 64250                | 128500                   | 18                          |
| watch  | accel  | 3777046 | 0                     | 0                     | 0                          | 0                            | 64904                | 129808                   | 18                          |
| watch  | gyro   | 3440342 | 0                     | 0                     | 0                          | 0                            | 64869                | 129738                   | 18                          |

## Data-quality issues list

- No official train/test split is provided; raw files are grouped by subject/device/sensor, so evaluation must create subject-wise splits.
- Raw timestamps are nanosecond-like sensor times, not human-readable calendar datetimes; use them for ordering/gaps, not wall-clock interpretation.
- The bundle contains multiple devices and sensors. A project should pick one stream first or carefully synchronize streams before sensor fusion.
- ARFF files are already windowed feature files, but the recommended project should create its own windows from raw samples to control leakage and preprocessing.
- Some exact duplicate rows were found within raw files and should be removed during preprocessing.

## Project framing recommendation

Use a subject-wise 18-class activity recognition task on raw phone accelerometer windows. This is clear, reproducible, and avoids the extra synchronization assumptions needed for multi-sensor fusion. Phone accelerometer is also a strong first task because every sample has one label, one subject ID, one timestamp, and three sensor axes.

Generated by `audit_dataset.py`.
