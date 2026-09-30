## 3. Dataset and Exploratory Data Analysis

### 3.1 Dataset composition

The study uses the **Zindi Weekendz Swahili Audio Classification** dataset, consisting of 4,200 labelled audio clips distributed across 12 Swahili spoken-word classes. Each class contains exactly 350 examples, giving a balanced class distribution. The classes represent common words including *hapana* (no), *kumi* (ten), *mbili* (two), *moja* (one), *nane* (eight), *ndio* (yes), *nne* (four), *saba* (seven), *sita* (six), *tano* (five), *tatu* (three), and *tisa* (nine). 

A fixed stratified split was created for all experiments using random seed 42. The resulting dataset contains 2,940 training clips (70%), 630 validation clips (15%), and 630 internal test clips (15%). Each class contributes 245 training examples, while the validation and test sets contain either 52 or 53 examples per class because 15% of 350 is not an integer. The saved split was used consistently rather than regenerating splits for individual models. 

The official competition test set was not used as the research evaluation set because it contains no labels. Instead, the labelled dataset was divided into the fixed internal train, validation, and test partitions. 

### 3.2 Class distribution

The class distribution is perfectly balanced, with all 12 classes containing 350 labelled examples. The maximum-to-minimum class-count ratio is therefore 1.0. Consequently, class weighting and oversampling were not introduced as part of preprocessing, since the label distribution provides no evidence of a class-frequency imbalance problem. 

This is useful for subsequent model comparison because differences in macro-F1 or per-class performance are less likely to be explained simply by unequal class frequencies.

### 3.3 Audio characteristics

All 4,200 labelled files were found to have the same basic audio format: **16 kHz sampling rate, mono channel configuration, and 16-bit PCM encoding**. No sample-rate mismatches, channel inconsistencies, zero-length clips, or audio-length inconsistencies were detected during the full labelled-file check. 

The main source of variation was therefore **duration**. Clip duration ranged from 2.48 seconds to 62.24 seconds, with a median of 4.32 seconds and a mean of 4.82 seconds. The 95th percentile was 7.30 seconds, indicating that most recordings were relatively short while a small number formed a substantial long-duration tail. 

The long tail was not restricted to a particular class. There were 64 clips lasting at least 10 seconds and nine clips lasting at least 60 seconds, distributed across multiple classes. These recordings were retained rather than automatically classified as noise or removed. 

### 3.4 Temporal and energy characteristics

An additional investigation examined low-energy regions using two descriptive rules. At sample level, the proportion of samples below 1% of each clip's peak amplitude had a median of 0.8483. A separate frame-based analysis used 25 ms frames with a 10 ms hop and classified frames as low-energy when their RMS was below 1% of the clip's maximum frame RMS. Under this rule, the median low-energy frame proportion was 0.6203, with median leading and trailing low-energy regions of 0.21 and 0.105 seconds respectively. 

These measurements were treated only as **energy characteristics**, rather than direct measurements of silence. The analysis did not validate the threshold against speech/non-speech annotations, so automatic silence removal was not justified solely from these statistics. 

### 3.5 Implications for fixed-length processing

The substantial duration variation creates an important preprocessing issue for models that require fixed-length inputs. Several candidate windows were investigated on the training partition:

| Window  | Clips requiring padding | Clips requiring truncation | Duration retained if truncated |
| ------- | ----------------------: | -------------------------: | -----------------------------: |
| 4 s     |                  32.99% |                     63.23% |                         80.76% |
| 5 s     |                  73.95% |                     26.05% |                         89.75% |
| 6 s     |                  88.74% |                     10.85% |                         93.37% |
| **8 s** |              **96.46%** |                  **3.27%** |                     **96.01%** |



The 8-second window therefore provided a useful compromise for subsequent experimentation: only 3.27% of training clips exceeded the window, while 96.01% of the total clip duration would be retained if longer recordings were truncated. Importantly, this was treated as a **candidate supported by the EDA**, rather than assuming beforehand that 8 seconds was optimal.

### 3.6 MFCC investigation

Mel-frequency cepstral coefficients (MFCCs) were investigated as a compact representation of the audio signal. The candidate configuration used a 25 ms Hann window, 10 ms hop, 512-point FFT and 40 HTK-mel filters. The analysis considered 13, 20 and 40 MFCC coefficients. The resulting number of analysis frames varied substantially with clip duration, ranging from 246 to 6,222 frames, with a median of 430 frames. 

The investigation also confirmed that delta and delta-delta features can be calculated from the static MFCC representation. These provide information about temporal changes in the spectral characteristics rather than only describing the average spectral shape. 

Because retaining every frame for every recording would result in substantially larger feature representations, aggregation was investigated as a practical option for the classical models. The EDA itself did not claim that a particular MFCC configuration was necessarily the most accurate; this was left to subsequent preprocessing and baseline experiments. 

### 3.7 Data quality

Several data-quality checks were performed. No unreadable labelled files or duplicate labelled IDs were found. Exact PCM SHA-256 hashing produced 4,200 unique hashes for the 4,200 labelled clips, indicating no identical audio payloads. However, near-duplicate recordings involving transformations such as time shifts or gain changes were not tested. 

No speaker metadata was available in the released tables, and filenames were not treated as speaker identifiers. Consequently, a speaker-disjoint evaluation could not be constructed from the available metadata. This is retained as a limitation of the experimental design rather than introducing an unsupported speaker grouping. 

### 3.8 EDA-driven preprocessing decisions

Overall, the EDA established four main preprocessing considerations:

1. **No class rebalancing was required**, because all classes contain the same number of examples.
2. **No resampling or channel conversion was required**, because all labelled recordings already share the same 16 kHz mono format.
3. **Variable duration required explicit handling**, motivating investigation of fixed-length representations.
4. **MFCC-based features provided a compact representation** suitable for the classical baselines, with temporal derivatives and statistical aggregation investigated experimentally.

The final feature configuration was therefore selected through the subsequent preprocessing experiments rather than being treated as an assumption derived solely from the EDA.

---

# 4. Preprocessing and Classical Baselines

For the classical models, the audio recordings were transformed from variable-length waveforms into fixed-dimensional MFCC-based feature vectors. The selected representation was **13 static MFCC coefficients together with delta and delta-delta coefficients**, using an **8-second analysis window**. Mean and standard deviation aggregation were then applied to the temporal feature sequence.

This produced a fixed-dimensional representation for each recording while retaining information about both the spectral characteristics and their temporal variation. The 8-second window was motivated by the duration analysis: it substantially reduced truncation compared with shorter candidate windows while avoiding the much larger sequences that would result from padding every recording to the maximum duration. 

The preprocessing pipeline was fitted or selected using the training/validation procedure, with the internal test partition kept separate until final evaluation. This ensured that the held-out test set did not influence preprocessing selection.

Two classical classifiers were then evaluated using the same extracted representation:

* **Logistic Regression (LR)** — used as a linear probabilistic classification baseline.
* **Linear Support Vector Machine (SVM)** — used as a second linear baseline based on a different classification objective.

Using the same audio representation for both classifiers makes the comparison primarily a comparison of the classification approaches rather than allowing differences in feature engineering to confound the baseline comparison.

### 4.1 Baseline results

On the held-out internal test set, the classical baselines achieved:

| Model               |  Accuracy |  Macro-F1 |
| ------------------- | --------: | --------: |
| Logistic Regression | **0.381** | **0.380** |
| Linear SVM          | **0.395** | **0.386** |

The Linear SVM produced slightly higher accuracy and macro-F1 than Logistic Regression on the internal test set. Because the classes are balanced, accuracy and macro-F1 provide complementary but relatively straightforward measures of overall classification performance.

These two models establish the **classical baseline level** against which the sequential neural approaches developed by the other group members can be compared. Importantly, their purpose is not to represent the expected state of the art, but to provide a reference point for determining whether models capable of learning temporal dependencies from the audio representation provide measurable improvements.

