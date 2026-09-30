# Preprocessing pipeline

This document describes the reusable feature pipeline added in Phase 4. It does not report classifier results. No duration, MFCC size, delta setting, normalization setting, or aggregation was selected by an experiment.

Phase 3 measurements are unchanged. They are evidence for later classical-baseline runs, not decisions made here.

The implementation status of `PreprocessingConfig` is: Phase 4 implementation defaults. Not an experimentally selected preprocessing configuration.

## 1. Pipeline stages

`extract_features(audio_path, config)` runs these stages in order:

1. Load the WAV into memory. A `Word_id` is read from `Swahili_words.zip`. A filesystem path is read from that path. The source bytes are not written back.
2. Validate the header. The labeled files are 16 kHz, mono, 16-bit PCM. A file that already matches is not resampled and not downmixed.
3. Optional per-clip peak normalization. Off unless `normalize_amplitude` is True.
4. Duration handling. Either leave the clip unchanged, or pad/truncate it to `duration_seconds`.
5. MFCC extraction through `src.preprocessing.mfcc.mfcc_matrix`.
6. Optional delta and delta-delta through `src.preprocessing.mfcc.delta_features`.
7. Aggregate the frames to one vector.

`extract_features_batch(items, config)` repeats that for each row and returns `(X, metadata)`. `X` is numeric. Ids and labels stay in `metadata`.

No stage draws a random crop. `random_seed` is stored on the configuration and is not read by any stage.

## 2. Available configuration options

| Field | Role |
| --- | --- |
| `target_sample_rate` | Expected sample rate in Hz. A different file raises `AudioFormatError`. There is no resampler. |
| `channel_mode` | `require_mono` rejects extra channels. `mean_downmix` averages channels only when the file has more than one. A mono file is not downmixed in either mode. |
| `duration_seconds` | Target length used when `duration_mode` is `fixed`. Any positive value is accepted. |
| `duration_mode` | `fixed` pads or truncates. `none` keeps the original samples. |
| `padding_mode` | Where the zeros are placed: `end`, `start`, or `center`. |
| `truncation_mode` | Which region is kept: `start`, `end`, or `center`. |
| `n_mfcc` | Number of DCT coefficients, including c0. Must be <= `n_mels`. |
| `n_mels` | Number of HTK-mel bands. |
| `n_fft` | FFT length. `window_length` must be <= `n_fft`. |
| `hop_length` | Hop in samples. |
| `window_length` | Hann window in samples. |
| `include_delta` | Append first differences. Default False. |
| `include_delta_delta` | Append second differences. Requires `include_delta`. Default False. |
| `aggregation` | `mean`, or `mean_std`. |
| `normalize_amplitude` | Per-clip peak scaling. Default False. |
| `random_seed` | Recorded for reproducibility. Not used, because nothing in the pipeline is random. |

`truncation_mode` is included because Phase 3 left the cut position open. It is an implementation option, not a result.

## 3. Default configuration

These are the values constructed by `PreprocessingConfig()`:

| Field | Default |
| --- | --- |
| `target_sample_rate` | 16000 |
| `channel_mode` | `require_mono` |
| `duration_seconds` | 6.0 |
| `duration_mode` | `fixed` |
| `padding_mode` | `end` |
| `truncation_mode` | `start` |
| `n_mfcc` | 13 |
| `n_mels` | 40 |
| `n_fft` | 512 |
| `hop_length` | 160 samples (10 ms at 16 kHz) |
| `window_length` | 400 samples (25 ms at 16 kHz) |
| `include_delta` | False |
| `include_delta_delta` | False |
| `aggregation` | `mean_std` |
| `normalize_amplitude` | False |
| `random_seed` | 42 |

The 25 ms window, 10 ms hop, 512-point FFT, 40 mel bands, and 13 coefficients are the Phase 3 candidate that was checked on the labeled clips. Phase 3 also computed 20 and 40 coefficients and did not choose among them. Six seconds is one of the two durations this phase had to be able to run. It is not a chosen window.

Calling `PreprocessingConfig()` therefore does not mean that this configuration won an experiment.

## 4. Supported duration settings

`duration_mode="fixed"` sets the length to `round(duration_seconds * sample_rate)` samples.

- A shorter clip is padded with the constant 0.
- `padding_mode="end"` appends the zeros. `"start"` prepends them. `"center"` splits them, and an odd count puts the extra zero at the end.
- A longer clip is truncated. `truncation_mode="start"` keeps the beginning and drops the end. `"end"` keeps the end. `"center"` keeps the middle, and an odd number of discarded samples is dropped from the end.
- An exact match is copied.
- `duration_mode="none"` returns a copy of the samples and ignores the target length.

No padded or truncated audio is saved. The fill value is 0, not a mean computed from a split.

Phase 4 checks run 6 seconds and 8 seconds. Other positive durations are accepted by the same code. On the shared training split, Phase 3 recorded:

- 6 seconds: 88.74% of clips are shorter and would be padded, 10.85% are longer and would be truncated.
- 8 seconds: 96.46% would be padded, 3.27% would be truncated.

Those percentages are not a reason to prefer either window. Both remain available.

Low-energy trimming is not a stage. Phase 3 measured low-energy frames and did not establish that they are silence.

## 5. MFCC settings

The pipeline calls the existing `mfcc_matrix` helper. That helper uses a Hann window, an HTK-mel filterbank, a log floor of `1e-10`, and an orthonormal DCT-II. It keeps c0. It does not apply pre-emphasis or a lifter. Those two omissions are properties of the reused helper, not a claim that pre-emphasis or liftering were tested.

Before aggregation the matrix has shape `(n_frames, n_mfcc)`, or wider when deltas are added. A clip that produces zero frames raises `FeatureExtractionError` instead of returning a zero vector. With the default 400-sample window, a fixed duration of 6 or 8 seconds at 16 kHz does produce frames.

`n_mfcc`, `n_mels`, `n_fft`, `hop_length`, and `window_length` can all be changed. `window_length` is in samples. The defaults equal 25 ms and 10 ms only at 16 kHz.

## 6. Delta and delta-delta options

Deltas use the Phase 3 HTK-style regression with width 2.

- Static only: `include_delta=False`. This is the default.
- Static plus delta: `include_delta=True` and `include_delta_delta=False`.
- Static plus delta plus delta-delta: both flags True.

Delta-delta cannot be requested without delta. Neither flag is turned on automatically. Phase 3 showed that the extra blocks can be computed. It did not compare them in a classifier.

The blocks are concatenated in the order static, delta, delta-delta.

## 7. Aggregation options

Aggregation is per clip and deterministic.

- `mean`: arithmetic mean over frames.
- `mean_std`: that mean, followed by the population standard deviation of each coefficient (`ddof=0`, divisor equal to the number of frames).

A coefficient that does not vary has standard deviation 0. A single frame also has standard deviation 0. Those zeros are kept. They are not imputed and not removed.

If the matrix contains a NaN or an infinite value, aggregation raises `FeatureExtractionError`. The invalid entry is not dropped and is not replaced.

For `mean_std`, the first half of the vector is the means in coefficient order, and the second half is the standard deviations in the same order.

## 8. Expected feature dimensions

For 13 MFCC coefficients:

| Streams | Coefficient count | `mean` | `mean_std` |
| --- | --- | --- | --- |
| Static | 13 | 13 | 26 |
| Static + delta | 26 | 26 | 52 |
| Static + delta + delta-delta | 39 | 39 | 78 |

`feature_dimension(config)` returns these widths, and the Phase 4 checks compare extracted matrices against them.

Delta alone is 26 streams, not 39. The 39-stream vector is the full static + delta + delta-delta stack. A note that lists 39 streams for "MFCC + delta" without delta-delta does not match this implementation. The widths above were checked in code.

Duration does not change the aggregated width. A 6 second run and an 8 second run with the same MFCC and aggregation settings have the same dimension. Their values can still differ.

## 9. Leakage considerations

Every current transform uses only the clip it is applied to.

- Peak normalization divides by that clip's own peak. A zero peak is left unchanged. No gain is estimated from a split.
- Padding uses the constant 0.
- Means and standard deviations are computed inside the clip.
- `uses_dataset_statistics(config)` is False for every configuration. There is no fit step to run on the training split, and none should be run on validation or the internal test.
- Labels are returned in `metadata` and are not columns of `X`.
- The batch function does not look at the internal test split unless the caller passes those ids. The Phase 4 check does not pass them.
- The official unlabeled `Test.csv` file is not read.

If a later transform needs a statistic such as a training-set mean, that statistic has to be fit on the training split only and then applied to validation and test. This pipeline does not have that step.

The shared split file is not rewritten.

## 10. What remains to be decided experimentally

### Supported options

The pipeline can run any of these. The Phase 4 checks show that the code produces finite, repeatable vectors of the widths in section 8.

- Fixed durations, including 6 seconds and 8 seconds, or no duration change.
- Padding at the start, end, or center, and truncation that keeps the start, end, or center.
- 13 MFCC coefficients, and other legal `n_mfcc` / `n_mels` / `n_fft` / window / hop settings.
- Static coefficients, static plus delta, or static plus delta plus delta-delta.
- Mean aggregation, or mean and standard deviation.
- Peak normalization on or off.
- Optional on-disk cache of the aggregated vectors.

Phase 3 already supports two narrower statements, and this phase does not reopen them: labeled clips are already 16 kHz mono 16-bit PCM, so they are not resampled or downmixed to match each other; labeled class counts are equal, so class weighting is not justified by label frequency.

### Final experimental decisions

None. In particular, this phase does not conclude that:

- 6 seconds or 8 seconds is the duration to use
- deltas or delta-delta should be included
- peak normalization helps
- mean is better or worse than mean and standard deviation
- the default padding or truncation position is the right cut
- 13 coefficients are better than 20 or 40

Those comparisons belong to the classical baseline experiments. No logistic regression, SVM, neural network, or hyperparameter search was run to write this file.

### Cache

Caching is off unless `extract_features_batch` is given `cache_dir`. The directory must not be inside `data/raw` or `data/splits`. Each configuration gets its own subdirectory, named by the SHA-256 of the full configuration JSON. Each file stores one aggregated vector, named with the source id, a hash of the source string, and the first 16 hex characters of the WAV SHA-256. A configuration change does not reuse another configuration's vectors. Intermediate MFCC matrices are not cached.

### Checks

`python scripts/run_phase4_validation.py` runs the pipeline on a small deterministic subset of the shared training split: one clip from each class, plus a training clip shorter than 6 seconds and one longer than 8 seconds. It writes `results/eda/phase4_validation.csv` and `results/eda/phase4_validation.json`.

`python -m unittest discover -s tests -t .` runs the synthetic-audio tests.
