# EDA Findings

Numbers in this file were computed from `data/raw` and from `results/eda/audio_metadata_train.csv`. The saved split `data/splits/shared_split.csv` was read and was not rewritten. No classifier was trained. The official `Test.csv` file has no labels and was not used as an evaluation set.

## 1. Dataset composition

The labeled file has 4200 clips and 12 classes. The shared split counts are {'train': 2940, 'test': 630, 'validation': 630}. That file matches a fresh in-memory stratified split with random seed 42 under the installed scikit-learn. The fresh split was not saved. Official test ids inside the shared split: 0. Duplicate split ids: 0. Labeled ids missing from the split: 0.

## 2. Class balance

Labeled examples per class: {'hapana': 350, 'kumi': 350, 'mbili': 350, 'moja': 350, 'nane': 350, 'ndio': 350, 'nne': 350, 'saba': 350, 'sita': 350, 'tano': 350, 'tatu': 350, 'tisa': 350}. Minimum 350, maximum 350, ratio of maximum to minimum 1.0. Counts are equal: True.

Shared-split ranges: {'test': {'n_clips': 630, 'min_class_count': 52, 'max_class_count': 53, 'n_classes': 12}, 'train': {'n_clips': 2940, 'min_class_count': 245, 'max_class_count': 245, 'n_classes': 12}, 'validation': {'n_clips': 630, 'min_class_count': 52, 'max_class_count': 53, 'n_classes': 12}}.

Every labeled class has the same count, so class weighting or oversampling is not justified by the label distribution. Validation and internal-test counts differ by at most one clip per class because 15 percent of 350 is not an integer. That is a split-rounding effect, not a class-imbalance problem.

Class weighting justified by these label counts: False. Oversampling justified by these label counts: False.

## 3. Audio format

On the 4200 labeled clips, sample-rate counts are {'16000': 4200}, channel counts are {'1': 4200}, sample-width counts are {'2': 4200}, and compression counts are {'NONE': 4200}. A second read of every labeled WAV found 0 sample-rate mismatches and 0 length mismatches against the Phase 2 table. Zero-length clips: 0.

## 4. Duration characteristics

Labeled-pool duration in seconds, with linear percentile interpolation and sample standard deviation: minimum 2.4800, maximum 62.2400, mean 4.8166, median 4.3200, standard deviation 3.3635. Percentiles: 1st 2.8000, 5th 3.2000, 10th 3.3600, 25th 3.8400, 50th 4.3200, 75th 5.0400, 90th 6.1600, 95th 7.2992, 99th 11.8440.

The same summaries on the shared training split only (n=2940) are minimum 2.4800, maximum 62.2400, mean 4.7598, median 4.3200, standard deviation 2.9859, 95th 7.2800, 99th 10.8800.

Counts at or above round duration marks in the labeled pool. These marks are descriptive. No clip was removed.

- at least 5 s: 1074 clips (25.57% of 4200)
- at least 6 s: 471 clips (11.21% of 4200)
- at least 8 s: 160 clips (3.81% of 4200)
- at least 10 s: 64 clips (1.52% of 4200)
- at least 15 s: 30 clips (0.71% of 4200)
- at least 20 s: 21 clips (0.50% of 4200)
- at least 30 s: 13 clips (0.31% of 4200)
- at least 60 s: 9 clips (0.21% of 4200)

Per class, labeled pool:

- hapana (no): n=350, min=2.6400 s, median=4.4000 s, mean=4.9143 s, max=62.0800 s, n>=10 s=6, n>=60 s=1
- kumi (ten): n=350, min=2.4800 s, median=4.2800 s, mean=4.8406 s, max=62.2080 s, n>=10 s=8, n>=60 s=1
- mbili (two): n=350, min=2.7200 s, median=4.2400 s, mean=4.9009 s, max=62.2080 s, n>=10 s=7, n>=60 s=1
- moja (one): n=350, min=2.6400 s, median=4.2400 s, mean=4.9199 s, max=62.0800 s, n>=10 s=5, n>=60 s=1
- nane (eight): n=350, min=2.4800 s, median=4.3520 s, mean=5.0059 s, max=62.2400 s, n>=10 s=5, n>=60 s=2
- ndio (yes): n=350, min=2.5600 s, median=4.2400 s, mean=4.7492 s, max=62.0800 s, n>=10 s=4, n>=60 s=1
- nne (four): n=350, min=2.6400 s, median=4.2400 s, mean=4.6916 s, max=61.9520 s, n>=10 s=5, n>=60 s=1
- saba (seven): n=350, min=2.6400 s, median=4.3200 s, mean=4.6673 s, max=29.3120 s, n>=10 s=2, n>=60 s=0
- sita (six): n=350, min=2.8000 s, median=4.3200 s, mean=4.8564 s, max=62.0800 s, n>=10 s=5, n>=60 s=1
- tano (five): n=350, min=2.7200 s, median=4.2400 s, mean=4.7200 s, max=41.8560 s, n>=10 s=4, n>=60 s=0
- tatu (three): n=350, min=2.5600 s, median=4.3200 s, mean=4.7853 s, max=27.6000 s, n>=10 s=8, n>=60 s=0
- tisa (nine): n=350, min=2.8000 s, median=4.4000 s, mean=4.7472 s, max=25.3600 s, n>=10 s=5, n>=60 s=0

Class medians range from 4.2400 s to 4.4000 s. Class means range from 4.6673 s to 5.0059 s.

## 5. Long-tail duration findings

Clips of at least 10 s: 64, spread across 12 classes. The largest class count among those clips is 8. Counts by class: {'hapana': 6, 'kumi': 8, 'mbili': 7, 'moja': 5, 'nane': 5, 'ndio': 4, 'nne': 5, 'saba': 2, 'sita': 5, 'tano': 4, 'tatu': 8, 'tisa': 5}.

Clips of at least 60 s: 9, spread across 8 classes. The largest class count among those clips is 2. Counts by class: {'hapana': 1, 'kumi': 1, 'mbili': 1, 'moja': 1, 'nane': 2, 'ndio': 1, 'nne': 1, 'sita': 1}.

The maximum labeled duration is 62.2400 s. These clips were not removed, and this file does not call them noise.

## 6. Temporal and energy findings

Two separate descriptive rules were used.

Sample-level rule, already stored in Phase 2: proportion of samples below 1% of that clip's peak absolute amplitude. Labeled-pool median 0.8483, maximum 0.9853. This was not established as silence.

Frame rule, computed in this phase: a 25 ms frame every 10 ms is counted when its RMS is below 1% of that clip's maximum frame RMS. Labeled-pool frame proportion median 0.6203, maximum 0.9826. Leading low-energy time median 0.2100 s, maximum 7.0100 s. Trailing low-energy time median 0.1050 s, maximum 39.1900 s. Clips where every analysis frame met the low-energy rule: 0.

By duration bin:

- under_5s: n=3126, median leading=0.2100 s, median trailing=0.1200 s, median frame low-energy proportion=0.6268
- 5_to_8s: n=914, median leading=0.2350 s, median trailing=0.0900 s, median frame low-energy proportion=0.6023
- 8_to_10s: n=96, median leading=0.1950 s, median trailing=0.1000 s, median frame low-energy proportion=0.6016
- at_least_10s: n=64, median leading=0.1750 s, median trailing=0.1250 s, median frame low-energy proportion=0.7945

Median frame low-energy proportion by class ranges from 0.5602 to 0.6983.

Representative clips were chosen without a new random draw: the shortest labeled clip, the clip closest to the labeled median duration, the longest labeled clip, and the clip between the 25th and 75th duration percentiles with the highest sample-level peak-rule proportion. Ids are in `results/eda/representative_examples.csv`.

## 7. Spectral and MFCC findings

Candidate MFCC settings, not an adopted pipeline: 25 ms Hann window, 10 ms hop, FFT length 512, 40 HTK-mel filters from 0 Hz to the Nyquist frequency, log floor 1e-10, orthonormal DCT-II with c0 kept, pre-emphasis 0.0, no lifter. Frame counts below use that framing formula on every labeled clip. The formula was checked by computing MFCC matrices on 15 clips. Shape mismatches: none.

Labeled-pool analysis-frame counts: minimum 246, median 430.0, maximum 6222.

- 13 coefficients: median sequence is 430.0 frames by 13 values (93.9 MB float32 if every labeled clip were stored at the median length; 1358.9 MB at the maximum length). Mean pooling would be 0.2184 MB.
- 20 coefficients: median sequence is 430.0 frames by 20 values (144.5 MB float32 if every labeled clip were stored at the median length; 2090.6 MB at the maximum length). Mean pooling would be 0.3360 MB.
- 40 coefficients: median sequence is 430.0 frames by 40 values (289.0 MB float32 if every labeled clip were stored at the median length; 4181.2 MB at the maximum length). Mean pooling would be 0.6720 MB.

On one representative clip (`id_54acpxzvh5dm.wav`), static 13-MFCC shape [246, 13], delta shape [246, 13], delta-delta shape [246, 13]. Stacking static, delta, and delta-delta multiplies the coefficient axis by 3. That stacking was computed once to test feasibility. It was not adopted.

## 8. Fixed-length implications

Round windows on the shared training split:

- 4 s on the shared training split: 32.99% shorter (padding), 3.78% equal to the window, 63.23% longer (truncation), 80.76% of training-clip duration retained if the excess is cut off.
- 5 s on the shared training split: 73.95% shorter (padding), 0.00% equal to the window, 26.05% longer (truncation), 89.75% of training-clip duration retained if the excess is cut off.
- 6 s on the shared training split: 88.74% shorter (padding), 0.41% equal to the window, 10.85% longer (truncation), 93.37% of training-clip duration retained if the excess is cut off.
- 8 s on the shared training split: 96.46% shorter (padding), 0.27% equal to the window, 3.27% longer (truncation), 96.01% of training-clip duration retained if the excess is cut off.

Empirical training-split percentile windows, included because they come from the measured distribution rather than from round numbers:

- shared_train_split_p50: 4.3200 s, 49.32% shorter, 47.62% longer, 84.58% of duration retained if truncated.
- shared_train_split_p75: 5.0400 s, 73.95% shorter, 24.39% longer, 89.97% of duration retained if truncated.
- shared_train_split_p90: 6.1600 s, 89.83% shorter, 9.66% longer, 93.72% of duration retained if truncated.
- shared_train_split_p95: 7.2800 s, 94.97% shorter, 4.90% longer, 95.37% of duration retained if truncated.
- shared_train_split_p99: 10.8800 s, 98.95% shorter, 0.99% longer, 97.09% of duration retained if truncated.

No window was selected. No audio was padded or truncated on disk.

## 9. Data quality findings

Unreadable files in the full labeled read: 0. Duplicate labeled ids in the metadata table: 0. Exact PCM SHA-256 values: 4200 unique hashes out of 4200 labeled clips. Duplicate hash groups: 0. Pairwise near-duplicate comparison was not run. Identical PCM payloads are the only duplicates this check can detect. Time shifts and gain changes were not compared.

Clips with peak absolute amplitude at least 32767: 150. Shortest and longest ids are listed in `results/eda/data_quality.json`. No speaker labels exist in the released tables, and filenames were not treated as speaker ids.

## 10. Implications for preprocessing

Observed: labeled class counts are equal, and the shared split stays within one clip of equality per class.

Interpretation: class weights and oversampling are not supported by the label counts.

Open for Phase 4: none on this specific point. The supported statement is to not add class weighting because of label frequency.

Observed: every labeled clip that was read is 16 kHz, mono, 16-bit PCM.

Interpretation: resampling or downmixing is not required to make the labeled files match one another.

Open for Phase 4: a model could still change the sample rate. The files themselves do not require that change.

Observed: durations vary from 2.4800 s to 62.2400 s, with a long tail that is not confined to one class.

Interpretation: a fixed number of time steps will require padding, truncation, or both if a later model needs a fixed length. Cutting every clip to a short window would discard part of the long recordings. Padding to the maximum would create sequences of 6222 analysis frames under the candidate 10 ms hop.

Open for Phase 4: the window length, where a longer clip is cut, and what value is used for padding.

Observed: under the documented frame-RMS rule, median leading and trailing low-energy times are 0.2100 s and 0.1050 s. The sample-level peak rule has a high median proportion. Neither rule was validated as speech-versus-silence.

Interpretation: many samples are far below the clip peak. That does not by itself identify silence or justify trimming.

Open for Phase 4: whether any low-energy trim is used, and the rule if it is.

Observed: 13, 20, and 40 MFCCs are computable with the candidate settings, and delta features are a small extra calculation. Storing every frame of every clip costs much more than mean or mean-and-standard-deviation pooling.

Interpretation: a classical model will need some aggregation or a fixed window, because the sequences are not the same length. This investigation does not show which representation classifies the words more accurately.

Open for Phase 4: coefficient count, deltas, lifter, pre-emphasis, and aggregation.
