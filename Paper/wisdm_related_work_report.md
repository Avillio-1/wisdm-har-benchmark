# WISDM Related Work Support Package

## Purpose

This report summarizes papers that actually use the WISDM dataset and extracts the parts that matter for a fair comparison: task definition, sensors, split protocol, representation, model family, metrics, and limitations. The main caution is that WISDM results are often not apples-to-apples. Random window splits, selected-user subsets, dropped classes, and non-subject-wise protocols can produce much higher scores than evaluation on unseen subjects.

The companion tables are saved as:

- `Paper/wisdm_papers.csv`
- `Paper/wisdm_papers.md`

## Concise Related-Work Table

| Paper | Task | Sensors | Evaluation | Models | Representation | Reported result | Fairness note |
|---|---|---|---|---|---|---|---|
| Kwapisz et al. 2011 | Original 6-class HAR: walking, jogging, upstairs, downstairs, sitting, standing | Phone accelerometer | 10-fold cross-validation over examples/windows | J48, logistic regression, MLP | Time-domain handcrafted features | Best accuracy about 91.7% | Historical baseline, but not subject-wise. |
| Weiss et al. 2019 dataset release | 18 WISDM 2019 activities from 51 subjects | Phone/watch accelerometer and gyroscope | Dataset descriptor, no canonical split | N/A | Raw streams and transformed 10-second examples | N/A | Defines the dataset; does not define a fair benchmark by itself. |
| Peppas et al. 2020 | Classic WISDM phone-accelerometer HAR task | Phone accelerometer | 10-fold user-independent evaluation | CNN-style framework, CNN/LSTM/BiLSTM comparators | Forty time-domain features | 95.27% accuracy | Stronger fairness because users are held out. |
| Oluwalade et al. 2021 | 15-class WISDM 2019 task after dropping three missing-heavy classes | Phone/watch accelerometer and gyroscope ablations | Random 80/10/10 split | CNN, ConvLSTM, LSTM, BiLSTM | Raw 5-second windows | Best watch-accelerometer CNN about 94.47% accuracy | Useful ablation inspiration, but random split is easier. |
| Tan et al. 2023 | 18-class WISDM with 100-row windows, mixed-activity windows removed | Acceleration windows | Train/test split after windowing | CNN-BiGRU-SE-RELM | Raw 100x3 windows | 98.46% accuracy, 98.45% F1 | Very high score, but no evidence of subject-wise evaluation. |
| Alexan et al. 2024 | 7 selected activities from first five WISDM 2019 users | Watch accelerometer and gyroscope | Not presented as subject-wise | RF, SVM, kNN, CNN, VGG16, MobileNetV2, LSTM | 2D plot/image conversion | RF 91.18% accuracy and 91.26% F1 | Small selected subset; not comparable to full unseen-subject WISDM. |
| Turdalyuly et al. 2026 | 6 broad a priori WISDM 2019 categories | Phone/watch accelerometer and gyroscope plus fusion | 5-fold GroupKFold by subject | CNN1D, LSTM, GRU, BiLSTM, fusion networks | Raw IMU sequences | Best macro F1 about 46.0% | Closest recent fairness match; shows cross-subject WISDM is hard. |

## Research Landscape

The original WISDM work established the familiar six-activity phone-accelerometer benchmark: walking, jogging, stairs up, stairs down, sitting, and standing. It is useful historically because it made phone-based HAR concrete and showed that simple supervised models could exceed 90% accuracy. Its evaluation protocol, however, is not the same standard we want for our project. It used cross-validation over examples/windows, which is easier than holding out entire subjects.

The 2019 WISDM Smartphone and Smartwatch dataset broadened the problem substantially. It added smartwatch sensors, gyroscope streams, biometric framing, 51 subjects, and 18 activities. This is the dataset that makes our comparative question interesting: whether phone sensors, watch sensors, fusion, handcrafted features, frequency features, classical models, and raw-sequence neural networks generalize to unseen subjects.

Recent deep-learning papers often report very high WISDM performance, but many use easier protocols: random train/test splits, selected-user subsets, class dropping after inspecting missingness, or windows created before splitting. Those choices are not automatically wrong for engineering demos, but they are risky for scientific claims about generalization. In contrast, the more subject-aware papers show more modest results, especially when the task is broad and multi-class.

The most useful pattern for our project is not "deep learning always wins." It is that WISDM performance depends heavily on task scope, sensors, representation, and evaluation protocol. Classical models remain competitive when the feature pipeline is strong. Deep models are worth including, especially CNN and recurrent baselines, but they should be treated as one model family among several rather than the default winner.

## Lessons For Our Project

Scoped subsets like 6-class tasks are common. The original WISDM paper used a 6-class task, and recent work also remaps WISDM into coarser 6-class categories. This supports keeping our clean 3-class sanity task and adding an a priori 6-class task. The key is to say why the subset exists before evaluating on the test set.

The full 18-class task should stay. Many papers avoid the full WISDM 2019 difficulty by dropping classes or selecting users. Keeping the 18-class benchmark gives our study a harder and more honest endpoint.

Multimodal phone/watch fusion is worth testing, but not overselling. Oluwalade et al. found strong watch-accelerometer results under a random split. Turdalyuly et al. found that adding watch streams did not automatically improve cross-subject macro F1. Our paper should frame fusion as an empirical question, not a guaranteed improvement.

FFT and frequency-domain features are reasonable to include. WISDM work often relies on engineered window features, and activity signals are periodic enough that frequency summaries are plausible. Our handcrafted plus frequency representation is a useful, defensible ablation.

Classical models remain important. The original paper used decision trees, logistic regression, and MLPs. Alexan et al. found random forest stronger than several neural/image variants on their selected subset. Our random forest, LightGBM, XGBoost, and logistic regression baselines are not filler; they are central comparators.

Deep-learning baselines worth reproducing are CNN, LSTM/BiLSTM, and possibly ConvLSTM or GRU. A 1D CNN on raw windows is the best first deep baseline because it is simpler, fast enough, and common in the literature. LSTM/BiLSTM/GRU are useful next steps if runtime allows, but they should be evaluated under the same subject-wise protocol as all other models.

Preprocessing and evaluation pitfalls repeat across papers. The most important pitfalls are random row/window splits, windowing before subject split, windows crossing activity boundaries, scaling before splitting, dropping classes based on test behavior, and comparing accuracy-only results on imbalanced tasks. Our report should explicitly state how each is avoided.

## How Our Benchmark Differs

Our benchmark is designed around unseen-subject generalization. We split by subject before windowing, avoid windows crossing activity boundaries, fit scaling only on training data, and select hyperparameters using validation data rather than test results. This is stricter than many high-score WISDM papers.

Our task definitions are a priori. The 3-class walking/jogging/standing task is retained as a sanity-check benchmark, not as a test-selected best case. The 6-class task is added because scoped 6-class HAR problems are common in WISDM literature. The 18-class task is preserved as the hard benchmark.

Our sensor ablations are broader than the original phone-only benchmark. We compare phone accelerometer, phone accelerometer plus gyroscope, watch accelerometer, watch accelerometer plus gyroscope, and phone-plus-watch fusion when feasible.

Our representation and model comparison is wider than a single deep model. We compare statistical features, statistical plus frequency features, logistic regression, random forest, LightGBM, XGBoost, and a raw-sequence CNN. This lets us answer whether gains come from sensors, features, model family, or protocol.

Our main metric should be macro F1, with accuracy as secondary. Many WISDM papers emphasize accuracy, but macro F1 is safer for imbalanced activities and harder class subsets. Confusion matrices and per-class F1 should be reported for every task.

## Concrete Recommendation For Our Paper

Definitely keep these experiments:

- The clean 3-class sanity task, the a priori 6-class task, and the full 18-class hard task.
- Subject-wise evaluation as the headline protocol, with GroupKFold stability where runtime permits.
- Sensor ablations for phone accelerometer, phone accelerometer plus gyroscope, watch accelerometer, watch accelerometer plus gyroscope, and phone-plus-watch fusion.
- Classical baselines: majority class, logistic regression, random forest, LightGBM, and XGBoost.
- Raw-window 1D CNN as the primary deep-learning baseline.
- Per-class F1, macro F1, accuracy, confusion matrices, and subject-level error analysis.

The most valuable extra experiment to add is a recurrent raw-sequence baseline, preferably GRU or BiLSTM, on the same clean subject-wise splits. This is more useful than adding a flashy architecture because CNN/LSTM/GRU families appear repeatedly in WISDM papers and are easy for reviewers to understand.

Claims to avoid:

- Do not claim superiority over papers that used random splits unless the protocol difference is clearly stated.
- Do not claim phone/watch fusion is inherently better; present it as an ablation result.
- Do not claim full WISDM 2019 activity recognition is solved if strong results only appear on the 3-class or 6-class tasks.
- Do not compare our macro F1 directly to accuracy-only papers without noting the metric mismatch.
- Do not treat high random-split deep-learning results as evidence that subject-wise generalization should be easy.

The paper's strongest defensible claim is narrower and better: under leakage-safe subject-wise evaluation, we systematically compare task scope, sensor setup, feature representation, and model family on WISDM, showing which choices generalize best to unseen subjects and where common easier protocols can be misleading.

## Sources Used

- Kwapisz, Weiss, and Moore, [Activity Recognition using Cell Phone Accelerometers](https://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf).
- UCI Machine Learning Repository, [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset).
- Peppas et al., [A Real-Time Activity Recognition Framework Using Smartphone Accelerometers](https://www.mdpi.com/2076-3417/10/24/9040).
- Oluwalade et al., [Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data](https://www.scitepress.org/Papers/2021/104675/104675.pdf).
- Tan et al., [Human Activity Recognition Using a Hybrid Neural Network and a Regularized Extreme Learning Machine](https://www.mdpi.com/1424-8220/23/18/7974).
- Alexan et al., [Human Activity Recognition Using Machine Learning and Wearable Sensors](https://www.mdpi.com/2076-3417/14/19/8846).
- Turdalyuly et al., [Wearable Sensor-Free Adult Physical Activity Monitoring Using Smartphone IMU Signals and Deep Learning](https://www.mdpi.com/2078-2489/17/4/148).

## Note On Screened Papers

Several recent WISDM papers report high deep-learning scores but do not expose enough task and split detail in accessible metadata to support a fair row in the main table. Those papers may still be useful for architecture inspiration, but they should not be used as numerical comparators until the exact split protocol, class set, sensor streams, and windowing order are verified from the full text.
