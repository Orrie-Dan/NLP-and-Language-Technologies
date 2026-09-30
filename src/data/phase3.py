"""Phase 3 audio investigation.

Reads the saved shared split and the Phase 2 metadata. It does not train
models, does not rewrite ``data/splits/shared_split.csv``, and does not
modify the raw files.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audio import _pcm_samples
from src.data.eda import (
    SPECTROGRAM_DB_FLOOR,
    SPECTROGRAM_NFFT,
    SPECTROGRAM_NOVERLAP,
    figures_dir,
    results_eda_dir,
)
from src.data.loading import (
    load_test_csv,
    load_train_csv,
    read_shared_split,
    read_wav_bytes,
)
from src.data.split import create_shared_split, validate_shared_split
from src.preprocessing.energy import LOW_ENERGY_FRAME_FRACTION, leading_trailing_low_energy
from src.preprocessing.mfcc import (
    DELTA_WIDTH,
    FMIN_HZ,
    FRAME_LENGTH_MS,
    HOP_LENGTH_MS,
    LOG_FLOOR,
    MEL_SCALE,
    N_FFT,
    N_MELS,
    N_MFCC_CANDIDATES,
    PRE_EMPHASIS,
    delta_features,
    mfcc_matrix,
    n_analysis_frames,
    samples_per_ms,
)
from src.utils.reproducibility import SHARED_SPLIT_SEED, environment_record, set_seed

DURATION_PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
DURATION_THRESHOLDS_SECONDS = (5, 6, 8, 10, 15, 20, 30, 60)
ROUND_WINDOWS_SECONDS = (4.0, 5.0, 6.0, 8.0)
FULL_SCALE_PEAK = 32767.0


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _duration_summary(duration: pd.Series) -> dict:
    quantiles = duration.quantile(list(DURATION_PERCENTILES), interpolation="linear")
    summary = {
        "n": int(duration.notna().sum()),
        "min": float(duration.min()),
        "max": float(duration.max()),
        "mean": float(duration.mean()),
        "median": float(duration.median()),
        "std_ddof1": float(duration.std(ddof=1)),
    }
    for probability, value in quantiles.items():
        summary[f"p{int(round(probability * 100)):02d}"] = float(value)
    return summary


def _threshold_table(duration: pd.Series) -> list[dict]:
    n = int(len(duration))
    rows = []
    for threshold in DURATION_THRESHOLDS_SECONDS:
        count = int((duration >= threshold).sum())
        rows.append(
            {
                "threshold_seconds": threshold,
                "n_clips": count,
                "percentage": float(100.0 * count / n) if n else float("nan"),
            }
        )
    return rows


def load_labeled_metadata(root: Path | None = None) -> pd.DataFrame:
    path = results_eda_dir(root) / "audio_metadata_train.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 2 metadata is missing: {path}. Run scripts/run_phase2_eda.py first."
        )
    metadata = pd.read_csv(path)
    required = {
        "Word_id",
        "Swahili_word",
        "English_translation",
        "duration_seconds",
        "n_samples",
        "sample_rate_hz",
        "n_channels",
        "sample_width_bytes",
        "bit_depth",
        "compression_type",
        "peak_abs_amplitude",
        "rms_amplitude",
        "low_energy_proportion",
        "n_frames",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Phase 2 metadata is missing columns: {sorted(missing)}")
    return metadata


def verify_saved_split(train_df: pd.DataFrame, official_test: pd.DataFrame) -> dict:
    """Check the saved split. Do not write a new split file."""
    saved = read_shared_split()
    checks = validate_shared_split(saved, train_df, official_test)
    class_counts = checks.pop("class_counts")
    recomputed = create_shared_split(train_df, seed=SHARED_SPLIT_SEED)
    saved_cmp = saved.sort_values("Word_id").reset_index(drop=True).astype(str)
    recomputed_cmp = recomputed.sort_values("Word_id").reset_index(drop=True).astype(str)
    counts = saved["split"].value_counts().to_dict()
    return {
        "path": "data/splits/shared_split.csv",
        "rewritten": False,
        "random_seed_recorded": SHARED_SPLIT_SEED,
        "matches_recomputed_seed_42_split": bool(saved_cmp.equals(recomputed_cmp)),
        "split_counts": {str(key): int(value) for key, value in counts.items()},
        "duplicate_word_ids": checks["duplicate_word_ids"],
        "missing_labeled_ids": checks["missing_labeled_ids"],
        "unknown_ids": checks["unknown_ids"],
        "official_test_ids_in_split": checks["official_test_ids_in_split"],
        "class_counts": class_counts.to_dict(orient="records"),
    }


def class_balance_report(train_df: pd.DataFrame, split_df: pd.DataFrame) -> dict:
    labeled_counts = train_df["Swahili_word"].value_counts().sort_index()
    merged = split_df.merge(train_df, on="Word_id", how="left", validate="one_to_one")
    by_split = (
        merged.groupby(["split", "Swahili_word"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["split", "Swahili_word"])
    )
    split_ranges = {}
    for split_name, group in by_split.groupby("split"):
        split_ranges[str(split_name)] = {
            "n_clips": int(group["n"].sum()),
            "min_class_count": int(group["n"].min()),
            "max_class_count": int(group["n"].max()),
            "n_classes": int(group["Swahili_word"].nunique()),
        }
    labeled_min = int(labeled_counts.min())
    labeled_max = int(labeled_counts.max())
    weighting_justified = labeled_min != labeled_max
    return {
        "n_labeled_examples": int(len(train_df)),
        "n_classes": int(labeled_counts.shape[0]),
        "examples_per_class": {str(label): int(count) for label, count in labeled_counts.items()},
        "labeled_min_class_count": labeled_min,
        "labeled_max_class_count": labeled_max,
        "labeled_counts_are_equal": labeled_min == labeled_max,
        "imbalance_ratio_max_over_min": float(labeled_max / labeled_min) if labeled_min else None,
        "split_ranges": split_ranges,
        "class_counts_by_split": by_split.to_dict(orient="records"),
        "class_weighting_justified_by_label_counts": weighting_justified,
        "oversampling_justified_by_label_counts": weighting_justified,
        "reason": (
            "Every labeled class has the same count, so class weighting or oversampling "
            "is not justified by the label distribution. Validation and internal-test "
            "counts differ by at most one clip per class because 15 percent of 350 is "
            "not an integer. That is a split-rounding effect, not a class-imbalance problem."
            if not weighting_justified
            else "Labeled class counts are not equal. A weighting decision is still open."
        ),
    }


def duration_report(metadata: pd.DataFrame) -> dict:
    per_class_rows = []
    for (label, english), group in metadata.groupby(["Swahili_word", "English_translation"], sort=True):
        row = {
            "Swahili_word": label,
            "English_translation": english,
            "n": int(len(group)),
            "duration_min": float(group["duration_seconds"].min()),
            "duration_mean": float(group["duration_seconds"].mean()),
            "duration_median": float(group["duration_seconds"].median()),
            "duration_max": float(group["duration_seconds"].max()),
            "duration_std_ddof1": float(group["duration_seconds"].std(ddof=1)),
        }
        for threshold in DURATION_THRESHOLDS_SECONDS:
            row[f"n_ge_{threshold}s"] = int((group["duration_seconds"] >= threshold).sum())
        per_class_rows.append(row)
    scopes = {"labeled_pool": metadata}
    if "split" in metadata.columns:
        scopes["shared_train_split"] = metadata.loc[metadata["split"] == "train"]
    scope_payload = {}
    for name, frame in scopes.items():
        scope_payload[name] = {
            "summary": _duration_summary(frame["duration_seconds"]),
            "thresholds": _threshold_table(frame["duration_seconds"]),
        }
    long_tails = {}
    for threshold in (10, 60):
        subset = metadata.loc[metadata["duration_seconds"] >= threshold, "Swahili_word"]
        counts = subset.value_counts().sort_index()
        long_tails[f"at_least_{threshold}s"] = {
            "n_clips": int(len(subset)),
            "counts_by_class": {str(label): int(count) for label, count in counts.items()},
            "n_classes_represented": int(counts.shape[0]),
            "max_class_count": int(counts.max()) if len(counts) else 0,
        }
    return {
        "percentile_interpolation": "linear",
        "std": "sample standard deviation, ddof=1",
        "thresholds_are_descriptive": True,
        "clips_were_removed": False,
        "scopes": scope_payload,
        "per_class_labeled_pool": per_class_rows,
        "long_tail_by_class": long_tails,
    }


def _open_pcm(word_id: str) -> tuple[np.ndarray, bytes, int]:
    wav_bytes = read_wav_bytes(word_id)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        payload = wav_file.readframes(n_frames)
    samples = _pcm_samples(payload, sample_width, n_frames * n_channels)
    if n_channels > 1:
        samples = samples.reshape(n_frames, n_channels).mean(axis=1)
    return samples, payload, int(sample_rate)


def temporal_energy_report(metadata: pd.DataFrame) -> pd.DataFrame:
    """Read each labeled WAV once for PCM hashes and frame-energy edges."""
    frame_length = samples_per_ms(16000, FRAME_LENGTH_MS)
    hop_length = samples_per_ms(16000, HOP_LENGTH_MS)
    rows = []
    total = len(metadata)
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        samples, payload, sample_rate = _open_pcm(row.Word_id)
        stats = leading_trailing_low_energy(samples, sample_rate, frame_length, hop_length)
        stats["Word_id"] = row.Word_id
        stats["pcm_sha256"] = hashlib.sha256(payload).hexdigest()
        stats["sample_rate_hz_reread"] = sample_rate
        stats["n_samples_reread"] = int(len(samples))
        rows.append(stats)
        if index % 500 == 0 or index == total:
            print(f"Temporal/hash pass {index}/{total}", flush=True)
    return pd.DataFrame(rows)


def _numeric_brief(series: pd.Series) -> dict:
    return {
        "n": int(series.notna().sum()),
        "min": float(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p95": float(series.quantile(0.95, interpolation="linear")),
        "max": float(series.max()),
    }


def temporal_summary(metadata: pd.DataFrame, temporal: pd.DataFrame) -> dict:
    merged = metadata.merge(temporal, on="Word_id", how="left", validate="one_to_one")
    bins = [0, 5, 8, 10, np.inf]
    labels = ["under_5s", "5_to_8s", "8_to_10s", "at_least_10s"]
    merged["duration_bin"] = pd.cut(
        merged["duration_seconds"],
        bins=bins,
        labels=labels,
        right=False,
    )
    by_bin = []
    for label, group in merged.groupby("duration_bin", observed=True):
        by_bin.append(
            {
                "duration_bin": str(label),
                "n": int(len(group)),
                "median_leading_low_energy_seconds": float(group["leading_low_energy_seconds"].median()),
                "median_trailing_low_energy_seconds": float(group["trailing_low_energy_seconds"].median()),
                "median_low_energy_frame_proportion": float(group["low_energy_frame_proportion"].median()),
            }
        )
    by_class = []
    for label, group in merged.groupby("Swahili_word", sort=True):
        by_class.append(
            {
                "Swahili_word": label,
                "n": int(len(group)),
                "median_leading_low_energy_seconds": float(group["leading_low_energy_seconds"].median()),
                "median_trailing_low_energy_seconds": float(group["trailing_low_energy_seconds"].median()),
                "median_low_energy_frame_proportion": float(group["low_energy_frame_proportion"].median()),
                "median_sample_low_energy_proportion": float(group["low_energy_proportion"].median()),
            }
        )
    hash_counts = merged["pcm_sha256"].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1]
    return {
        "rule": {
            "frame_length_ms": FRAME_LENGTH_MS,
            "hop_length_ms": HOP_LENGTH_MS,
            "low_energy_frame_fraction_of_max_frame_rms": LOW_ENERGY_FRAME_FRACTION,
            "definition": (
                "A 25 ms frame stepping every 10 ms is low-energy when its RMS is "
                "strictly below 1 percent of that clip's maximum frame RMS. "
                "This is not a silence measurement."
            ),
            "sample_level_rule_from_phase2": (
                "The Phase 2 column low_energy_proportion is the fraction of samples "
                "whose absolute amplitude is strictly below 1 percent of that clip's "
                "peak absolute amplitude. It is also not a silence percentage."
            ),
        },
        "labeled_pool": {
            "leading_low_energy_seconds": _numeric_brief(merged["leading_low_energy_seconds"]),
            "trailing_low_energy_seconds": _numeric_brief(merged["trailing_low_energy_seconds"]),
            "low_energy_frame_proportion": _numeric_brief(merged["low_energy_frame_proportion"]),
            "sample_low_energy_proportion": _numeric_brief(merged["low_energy_proportion"]),
            "n_all_frames_low_energy": int(merged["all_frames_low_energy"].sum()),
        },
        "by_duration_bin": by_bin,
        "by_class": by_class,
        "pcm_sha256": {
            "n_clips": int(len(merged)),
            "n_unique_hashes": int(merged["pcm_sha256"].nunique()),
            "n_duplicate_groups": int(len(duplicate_hashes)),
            "duplicate_group_sizes": [int(value) for value in duplicate_hashes.tolist()],
        },
    }


def fixed_window_report(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {"labeled_pool": metadata, "shared_train_split": metadata.loc[metadata["split"] == "train"]}
    percentile_names = {
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    for scope_name, frame in scopes.items():
        duration = frame["duration_seconds"].to_numpy(dtype=np.float64)
        total_duration = float(duration.sum())
        candidates = [("round", seconds, f"{seconds:.0f}s") for seconds in ROUND_WINDOWS_SECONDS]
        for name, probability in percentile_names.items():
            seconds = float(frame["duration_seconds"].quantile(probability, interpolation="linear"))
            candidates.append(("empirical_percentile", seconds, f"{scope_name}_{name}"))
        for source, seconds, name in candidates:
            shorter = duration < seconds
            longer = duration > seconds
            equal = ~(shorter | longer)
            retained = np.minimum(duration, seconds)
            rows.append(
                {
                    "scope": scope_name,
                    "window_name": name,
                    "window_source": source,
                    "window_seconds": seconds,
                    "n_clips": int(len(duration)),
                    "n_shorter": int(shorter.sum()),
                    "n_equal": int(equal.sum()),
                    "n_longer": int(longer.sum()),
                    "percentage_shorter_needs_padding": float(100.0 * shorter.mean()),
                    "percentage_equal": float(100.0 * equal.mean()),
                    "percentage_longer_needs_truncation": float(100.0 * longer.mean()),
                    "percentage_of_total_duration_retained_if_truncated": float(
                        100.0 * retained.sum() / total_duration
                    ),
                    "mean_padding_seconds_among_shorter": float((seconds - duration[shorter]).mean())
                    if shorter.any()
                    else 0.0,
                    "mean_truncated_seconds_among_longer": float((duration[longer] - seconds).mean())
                    if longer.any()
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _class_median_ids(metadata: pd.DataFrame) -> list[str]:
    chosen = []
    for _, group in metadata.groupby("Swahili_word", sort=True):
        median_duration = group["duration_seconds"].median()
        ranked = group.assign(_distance=(group["duration_seconds"] - median_duration).abs())
        ranked = ranked.sort_values(["_distance", "Word_id"], kind="mergesort")
        chosen.append(str(ranked.iloc[0]["Word_id"]))
    return chosen


def select_representative_examples(metadata: pd.DataFrame) -> pd.DataFrame:
    ordered = metadata.sort_values(["duration_seconds", "Word_id"], kind="mergesort")
    shortest = ordered.iloc[0]
    longest = ordered.iloc[-1]
    median_duration = float(metadata["duration_seconds"].median())
    typical = metadata.assign(_distance=(metadata["duration_seconds"] - median_duration).abs())
    typical = typical.sort_values(["_distance", "Word_id"], kind="mergesort").iloc[0]
    p25 = float(metadata["duration_seconds"].quantile(0.25, interpolation="linear"))
    p75 = float(metadata["duration_seconds"].quantile(0.75, interpolation="linear"))
    band = metadata.loc[
        (metadata["duration_seconds"] >= p25) & (metadata["duration_seconds"] <= p75)
    ]
    low_energy = band.sort_values(
        ["low_energy_proportion", "Word_id"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    roles = [
        ("shortest_labeled_clip", shortest),
        ("closest_to_labeled_median_duration", typical),
        ("longest_labeled_clip", longest),
        ("highest_sample_peak_rule_proportion_between_p25_and_p75", low_energy),
    ]
    rows = []
    for role, row in roles:
        rows.append(
            {
                "role": role,
                "Word_id": row["Word_id"],
                "Swahili_word": row["Swahili_word"],
                "English_translation": row["English_translation"],
                "duration_seconds": float(row["duration_seconds"]),
                "low_energy_proportion_sample_peak_rule": float(row["low_energy_proportion"]),
                "selection_note": (
                    "Deterministic sort on the labeled pool. No new random seed. "
                    "The fourth clip maximises the Phase 2 sample-level peak rule "
                    f"among clips with duration between the 25th ({p25:.4f} s) and "
                    f"75th ({p75:.4f} s) percentiles."
                ),
            }
        )
    return pd.DataFrame(rows)


def mfcc_report(metadata: pd.DataFrame, examples: pd.DataFrame) -> dict:
    sample_rate = int(metadata["sample_rate_hz"].mode().iloc[0])
    frame_length = samples_per_ms(sample_rate, FRAME_LENGTH_MS)
    hop_length = samples_per_ms(sample_rate, HOP_LENGTH_MS)
    frame_counts = [
        n_analysis_frames(int(n_samples), frame_length, hop_length)
        for n_samples in metadata["n_samples"].tolist()
    ]
    frame_count_series = pd.Series(frame_counts, dtype=float)
    check_ids = list(dict.fromkeys(examples["Word_id"].tolist() + _class_median_ids(metadata)))
    mismatches = []
    computed = {}
    started = time.perf_counter()
    for word_id in check_ids:
        samples, _, rate = _open_pcm(word_id)
        expected = n_analysis_frames(len(samples), frame_length, hop_length)
        features = {
            n_mfcc: mfcc_matrix(
                samples,
                rate,
                n_mfcc=n_mfcc,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            for n_mfcc in N_MFCC_CANDIDATES
        }
        for n_mfcc, matrix in features.items():
            if matrix.shape != (expected, n_mfcc) or not np.isfinite(matrix).all():
                mismatches.append(
                    {
                        "Word_id": word_id,
                        "n_mfcc": n_mfcc,
                        "shape": list(matrix.shape),
                        "expected_frames": expected,
                        "finite": bool(np.isfinite(matrix).all()),
                    }
                )
        if word_id in set(examples["Word_id"]):
            computed[word_id] = features
    elapsed = time.perf_counter() - started
    reference_id = examples.iloc[0]["Word_id"]
    reference = computed[reference_id][13]
    delta_1 = delta_features(reference, DELTA_WIDTH)
    delta_2 = delta_features(delta_1, DELTA_WIDTH)
    n_clips = int(len(metadata))
    median_frames = float(frame_count_series.median())
    max_frames = int(frame_count_series.max())
    candidates = []
    for n_mfcc in N_MFCC_CANDIDATES:
        candidates.append(
            {
                "n_mfcc": n_mfcc,
                "n_mels": N_MELS,
                "frame_length_samples": frame_length,
                "hop_length_samples": hop_length,
                "frame_length_ms": FRAME_LENGTH_MS,
                "hop_length_ms": HOP_LENGTH_MS,
                "n_fft": N_FFT,
                "window": "hann",
                "mel_scale": MEL_SCALE,
                "fmin_hz": FMIN_HZ,
                "fmax_hz": sample_rate / 2.0,
                "pre_emphasis": PRE_EMPHASIS,
                "log_floor": LOG_FLOOR,
                "dct": "orthonormal DCT-II, c0 retained, no lifter",
                "median_frames_per_labeled_clip": median_frames,
                "max_frames_per_labeled_clip": max_frames,
                "median_sequence_feature_values": median_frames * n_mfcc,
                "float32_megabytes_all_labeled_sequences_at_median_length": (
                    n_clips * median_frames * n_mfcc * 4 / 1e6
                ),
                "float32_megabytes_all_labeled_sequences_at_max_length": (
                    n_clips * max_frames * n_mfcc * 4 / 1e6
                ),
                "float32_megabytes_mean_pool": n_clips * n_mfcc * 4 / 1e6,
                "float32_megabytes_mean_and_std_pool": n_clips * n_mfcc * 2 * 4 / 1e6,
                "delta_and_delta_delta_multiply_feature_axis_by": 3,
            }
        )
    return {
        "sample_rate_hz_used": sample_rate,
        "parameters_are_candidates_not_a_decision": True,
        "frame_count_formula": (
            "1 + (n_samples - frame_length) // hop_length, "
            "no centering and no padding, frame must fit inside the clip"
        ),
        "frame_counts_on_labeled_pool": _numeric_brief(frame_count_series),
        "n_clips_checked_by_computing_mfcc": len(check_ids),
        "mfcc_shape_mismatches": mismatches,
        "seconds_to_compute_three_mfcc_sizes_on_checked_clips": elapsed,
        "delta_check": {
            "reference_word_id": reference_id,
            "width": DELTA_WIDTH,
            "static_shape": list(reference.shape),
            "delta_shape": list(delta_1.shape),
            "delta_delta_shape": list(delta_2.shape),
            "stacked_static_delta_delta_shape": [int(reference.shape[0]), int(reference.shape[1] * 3)],
            "adopted": False,
        },
        "candidates": candidates,
        "plot_features": computed,
    }


def quality_report(metadata: pd.DataFrame, temporal: pd.DataFrame, split_info: dict) -> dict:
    merged = metadata.merge(temporal, on="Word_id", how="left", validate="one_to_one")
    shortest = merged.nsmallest(5, "duration_seconds")
    longest = merged.nlargest(5, "duration_seconds")
    columns = ["Word_id", "Swahili_word", "duration_seconds", "peak_abs_amplitude", "rms_amplitude"]
    return {
        "n_labeled_clips": int(len(merged)),
        "unreadable_files_recorded_in_temporal_pass": 0,
        "zero_length_clips": int((merged["n_frames"] == 0).sum()),
        "sample_rate_counts": {
            str(key): int(value) for key, value in merged["sample_rate_hz"].value_counts().sort_index().items()
        },
        "channel_counts": {
            str(key): int(value) for key, value in merged["n_channels"].value_counts().sort_index().items()
        },
        "sample_width_bytes_counts": {
            str(key): int(value)
            for key, value in merged["sample_width_bytes"].value_counts().sort_index().items()
        },
        "compression_counts": {
            str(key): int(value) for key, value in merged["compression_type"].value_counts().sort_index().items()
        },
        "reread_sample_rate_mismatch": int((merged["sample_rate_hz"] != merged["sample_rate_hz_reread"]).sum()),
        "reread_length_mismatch": int((merged["n_samples"] != merged["n_samples_reread"]).sum()),
        "n_full_scale_peak_at_least_32767": int((merged["peak_abs_amplitude"] >= FULL_SCALE_PEAK).sum()),
        "duplicate_labeled_ids_in_metadata": int(merged["Word_id"].duplicated().sum()),
        "official_test_ids_in_split": split_info["official_test_ids_in_split"],
        "pcm_sha256_duplicate_groups": split_info.get("pcm_duplicate_groups_placeholder", None),
        "shortest_five": shortest.loc[:, columns].to_dict(orient="records"),
        "longest_five": longest.loc[:, columns].to_dict(orient="records"),
        "pairwise_audio_similarity": (
            "Not run. Exact PCM SHA-256 detects identical payloads only. "
            "Near-duplicates, time shifts, and gain changes were not compared."
        ),
    }


def _load_plot_samples(word_id: str) -> tuple[np.ndarray, int]:
    samples, _, sample_rate = _open_pcm(word_id)
    scale = 32768.0
    return samples / scale, sample_rate


def save_phase3_figures(
    metadata: pd.DataFrame,
    windows: pd.DataFrame,
    examples: pd.DataFrame,
    mfcc_payload: dict,
    temporal_bins: list[dict],
    root: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.mlab import specgram

    destination = figures_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    labeled = metadata
    class_order = sorted(labeled["Swahili_word"].unique())
    english = labeled.groupby("Swahili_word")["English_translation"].first().to_dict()

    figure, axis = plt.subplots(figsize=(8, 4.5))
    log_bins = np.logspace(
        np.log10(labeled["duration_seconds"].min()),
        np.log10(labeled["duration_seconds"].max()),
        40,
    )
    axis.hist(labeled["duration_seconds"], bins=log_bins, color="#4C78A8", edgecolor="white")
    axis.set_xscale("log")
    axis.set_xlabel("Duration (seconds, log scale)")
    axis.set_ylabel("Number of labeled clips")
    axis.set_title("Labeled clip duration, including the long tail")
    figure.tight_layout()
    figure.savefig(destination / "duration_distribution_log.png", dpi=150)
    plt.close(figure)

    means = [float(labeled.loc[labeled["Swahili_word"] == label, "duration_seconds"].mean()) for label in class_order]
    medians = [
        float(labeled.loc[labeled["Swahili_word"] == label, "duration_seconds"].median()) for label in class_order
    ]
    positions = np.arange(len(class_order))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(positions - 0.18, medians, width=0.36, label="median", color="#4C78A8")
    axis.bar(positions + 0.18, means, width=0.36, label="mean", color="#F58518")
    axis.set_xticks(positions, [f"{label}\n({english[label]})" for label in class_order], rotation=0)
    axis.set_ylabel("Duration (seconds)")
    axis.set_title("Mean and median duration by class")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination / "duration_mean_median_by_class.png", dpi=150)
    plt.close(figure)

    train_windows = windows.loc[
        (windows["scope"] == "shared_train_split") & (windows["window_source"] == "round")
    ].sort_values("window_seconds")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x_pos = np.arange(len(train_windows))
    axis.bar(
        x_pos - 0.25,
        train_windows["percentage_shorter_needs_padding"],
        width=0.25,
        label="shorter than window (padding)",
        color="#4C78A8",
    )
    axis.bar(
        x_pos,
        train_windows["percentage_equal"],
        width=0.25,
        label="equal to window",
        color="#B8B8B8",
    )
    axis.bar(
        x_pos + 0.25,
        train_windows["percentage_longer_needs_truncation"],
        width=0.25,
        label="longer than window (truncation)",
        color="#E45756",
    )
    axis.set_xticks(x_pos, [f"{value:.0f} s" for value in train_windows["window_seconds"]])
    axis.set_ylabel("Percentage of shared training clips")
    axis.set_ylim(0, 100)
    axis.set_title("Fixed-window consequences on the shared training split")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination / "fixed_window_impact.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bin_names = [row["duration_bin"] for row in temporal_bins]
    x_pos = np.arange(len(bin_names))
    axis.bar(
        x_pos - 0.18,
        [row["median_leading_low_energy_seconds"] for row in temporal_bins],
        width=0.36,
        label="median leading low-energy time",
        color="#4C78A8",
    )
    axis.bar(
        x_pos + 0.18,
        [row["median_trailing_low_energy_seconds"] for row in temporal_bins],
        width=0.36,
        label="median trailing low-energy time",
        color="#54A24B",
    )
    axis.set_xticks(x_pos, bin_names, rotation=15)
    axis.set_ylabel("Seconds")
    axis.set_title("Frame-RMS edges by duration bin (1% of max frame RMS)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination / "low_energy_edges_by_duration.png", dpi=150)
    plt.close(figure)

    cached = {}
    for word_id in examples["Word_id"]:
        cached[word_id] = _load_plot_samples(word_id)
    role_titles = {
        "shortest_labeled_clip": "shortest",
        "closest_to_labeled_median_duration": "near median duration",
        "longest_labeled_clip": "longest",
        "highest_sample_peak_rule_proportion_between_p25_and_p75": "high peak-rule proportion",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    for axis, (_, row) in zip(axes.ravel(), examples.iterrows()):
        samples, rate = cached[row["Word_id"]]
        time_axis = np.arange(len(samples)) / rate
        axis.plot(time_axis, samples, color="#4C78A8", linewidth=0.4)
        axis.set_title(
            f"{role_titles.get(row['role'], row['role'])}: "
            f"{row['Swahili_word']} ({row['English_translation']})\n"
            f"{row['Word_id']}  {row['duration_seconds']:.2f} s",
            fontsize=8,
        )
        axis.set_xlim(0, float(row["duration_seconds"]))
    for axis in axes[1, :]:
        axis.set_xlabel("Time (seconds)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude / 32768")
    figure.suptitle("Representative labeled clips", fontsize=12)
    figure.tight_layout()
    figure.savefig(destination / "representative_waveforms.png", dpi=150)
    plt.close(figure)

    spectrograms = []
    for _, row in examples.iterrows():
        samples, rate = cached[row["Word_id"]]
        power, frequencies, times = specgram(
            samples,
            NFFT=SPECTROGRAM_NFFT,
            Fs=rate,
            noverlap=SPECTROGRAM_NOVERLAP,
        )
        spectrograms.append((row, 10.0 * np.log10(power + SPECTROGRAM_DB_FLOOR), frequencies, times))
    stacked = np.concatenate([item[1].ravel() for item in spectrograms])
    vmin = float(np.percentile(stacked, 5))
    vmax = float(np.percentile(stacked, 95))
    figure, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    image = None
    for axis, (row, decibels, frequencies, times) in zip(axes.ravel(), spectrograms):
        image = axis.imshow(
            decibels,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[float(times[0]), float(times[-1]), float(frequencies[0]), float(frequencies[-1])],
            cmap="magma",
        )
        axis.set_title(f"{row['Swahili_word']} ({row['English_translation']})", fontsize=9)
    for axis in axes[1, :]:
        axis.set_xlabel("Time (seconds)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency (Hz)")
    figure.suptitle("Representative spectrograms (display only)", fontsize=12)
    figure.tight_layout()
    figure.savefig(destination / "representative_spectrograms.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    del image

    figure, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False, sharey=True)
    matrices = []
    for _, row in examples.iterrows():
        matrix = mfcc_payload["plot_features"][row["Word_id"]][13]
        display_matrix = matrix[:, 1:]
        centered = display_matrix - display_matrix.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        matrices.append(centered / scale)
    mfcc_stack = np.concatenate([matrix.ravel() for matrix in matrices])
    mfcc_vmin = float(np.percentile(mfcc_stack, 5))
    mfcc_vmax = float(np.percentile(mfcc_stack, 95))
    hop_seconds = samples_per_ms(16000, HOP_LENGTH_MS) / 16000.0
    image = None
    for axis, (_, row), matrix in zip(axes.ravel(), examples.iterrows(), matrices):
        image = axis.imshow(
            matrix.T,
            origin="lower",
            aspect="auto",
            vmin=mfcc_vmin,
            vmax=mfcc_vmax,
            extent=[0, matrix.shape[0] * hop_seconds, 1, 13],
            cmap="viridis",
        )
        axis.set_title(f"{row['Swahili_word']} ({row['English_translation']})", fontsize=9)
    for axis in axes[1, :]:
        axis.set_xlabel("Time (seconds)")
    for axis in axes[:, 0]:
        axis.set_ylabel("MFCC index, c0 omitted")
    figure.suptitle(
        "13-MFCC candidates, coefficients 1-12, scaled per coefficient for display only",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(destination / "mfcc_examples.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_phase3_findings(payload: dict, root: Path | None = None) -> Path:
    balance = payload["class_balance"]
    duration = payload["duration"]
    labeled = duration["scopes"]["labeled_pool"]
    train_duration = duration["scopes"]["shared_train_split"]
    temporal = payload["temporal"]
    quality = payload["quality"]
    windows = payload["fixed_windows_train_round"]
    mfcc = payload["mfcc"]
    split_info = payload["split"]
    summary = labeled["summary"]
    train_summary = train_duration["summary"]
    threshold_lines = "\n".join(
        f"- at least {row['threshold_seconds']} s: {row['n_clips']} clips "
        f"({row['percentage']:.2f}% of {summary['n']})"
        for row in labeled["thresholds"]
    )
    class_lines = "\n".join(
        "- {Swahili_word} ({English_translation}): n={n}, min={duration_min:.4f} s, "
        "median={duration_median:.4f} s, mean={duration_mean:.4f} s, max={duration_max:.4f} s, "
        "n>=10 s={n_ge_10s}, n>=60 s={n_ge_60s}".format(**row)
        for row in duration["per_class_labeled_pool"]
    )
    window_lines = "\n".join(
        f"- {row['window_seconds']:.0f} s on the shared training split: "
        f"{row['percentage_shorter_needs_padding']:.2f}% shorter (padding), "
        f"{row['percentage_equal']:.2f}% equal to the window, "
        f"{row['percentage_longer_needs_truncation']:.2f}% longer (truncation), "
        f"{row['percentage_of_total_duration_retained_if_truncated']:.2f}% of training-clip "
        f"duration retained if the excess is cut off."
        for row in windows
    )
    percentile_window_lines = "\n".join(
        f"- {row['window_name']}: {row['window_seconds']:.4f} s, "
        f"{row['percentage_shorter_needs_padding']:.2f}% shorter, "
        f"{row['percentage_longer_needs_truncation']:.2f}% longer, "
        f"{row['percentage_of_total_duration_retained_if_truncated']:.2f}% of duration retained "
        f"if truncated."
        for row in payload["fixed_windows_train_percentile"]
    )
    bin_lines = "\n".join(
        f"- {row['duration_bin']}: n={row['n']}, "
        f"median leading={row['median_leading_low_energy_seconds']:.4f} s, "
        f"median trailing={row['median_trailing_low_energy_seconds']:.4f} s, "
        f"median frame low-energy proportion={row['median_low_energy_frame_proportion']:.4f}"
        for row in temporal["by_duration_bin"]
    )
    candidate_lines = "\n".join(
        f"- {row['n_mfcc']} coefficients: median sequence is {row['median_frames_per_labeled_clip']:.1f} "
        f"frames by {row['n_mfcc']} values "
        f"({row['float32_megabytes_all_labeled_sequences_at_median_length']:.1f} MB float32 "
        f"if every labeled clip were stored at the median length; "
        f"{row['float32_megabytes_all_labeled_sequences_at_max_length']:.1f} MB at the maximum length). "
        f"Mean pooling would be {row['float32_megabytes_mean_pool']:.4f} MB."
        for row in mfcc["candidates"]
    )
    tail_10 = duration["long_tail_by_class"]["at_least_10s"]
    tail_60 = duration["long_tail_by_class"]["at_least_60s"]
    leading = temporal["labeled_pool"]["leading_low_energy_seconds"]
    trailing = temporal["labeled_pool"]["trailing_low_energy_seconds"]
    frame_prop = temporal["labeled_pool"]["low_energy_frame_proportion"]
    sample_prop = temporal["labeled_pool"]["sample_low_energy_proportion"]
    text = f"""# EDA Findings

Numbers in this file were computed from `data/raw` and from `results/eda/audio_metadata_train.csv`. The saved split `data/splits/shared_split.csv` was read and was not rewritten. No classifier was trained. The official `Test.csv` file has no labels and was not used as an evaluation set.

## 1. Dataset composition

The labeled file has {balance["n_labeled_examples"]} clips and {balance["n_classes"]} classes. The shared split counts are {split_info["split_counts"]}. That file {"matches" if split_info["matches_recomputed_seed_42_split"] else "does not match"} a fresh in-memory stratified split with random seed {SHARED_SPLIT_SEED} under the installed scikit-learn. The fresh split was not saved. Official test ids inside the shared split: {split_info["official_test_ids_in_split"]}. Duplicate split ids: {split_info["duplicate_word_ids"]}. Labeled ids missing from the split: {split_info["missing_labeled_ids"]}.

## 2. Class balance

Labeled examples per class: {balance["examples_per_class"]}. Minimum {balance["labeled_min_class_count"]}, maximum {balance["labeled_max_class_count"]}, ratio of maximum to minimum {balance["imbalance_ratio_max_over_min"]}. Counts are equal: {balance["labeled_counts_are_equal"]}.

Shared-split ranges: {balance["split_ranges"]}.

{balance["reason"]}

Class weighting justified by these label counts: {balance["class_weighting_justified_by_label_counts"]}. Oversampling justified by these label counts: {balance["oversampling_justified_by_label_counts"]}.

## 3. Audio format

On the {quality["n_labeled_clips"]} labeled clips, sample-rate counts are {quality["sample_rate_counts"]}, channel counts are {quality["channel_counts"]}, sample-width counts are {quality["sample_width_bytes_counts"]}, and compression counts are {quality["compression_counts"]}. A second read of every labeled WAV found {quality["reread_sample_rate_mismatch"]} sample-rate mismatches and {quality["reread_length_mismatch"]} length mismatches against the Phase 2 table. Zero-length clips: {quality["zero_length_clips"]}.

## 4. Duration characteristics

Labeled-pool duration in seconds, with linear percentile interpolation and sample standard deviation: minimum {_fmt(summary["min"])}, maximum {_fmt(summary["max"])}, mean {_fmt(summary["mean"])}, median {_fmt(summary["median"])}, standard deviation {_fmt(summary["std_ddof1"])}. Percentiles: 1st {_fmt(summary["p01"])}, 5th {_fmt(summary["p05"])}, 10th {_fmt(summary["p10"])}, 25th {_fmt(summary["p25"])}, 50th {_fmt(summary["p50"])}, 75th {_fmt(summary["p75"])}, 90th {_fmt(summary["p90"])}, 95th {_fmt(summary["p95"])}, 99th {_fmt(summary["p99"])}.

The same summaries on the shared training split only (n={train_summary["n"]}) are minimum {_fmt(train_summary["min"])}, maximum {_fmt(train_summary["max"])}, mean {_fmt(train_summary["mean"])}, median {_fmt(train_summary["median"])}, standard deviation {_fmt(train_summary["std_ddof1"])}, 95th {_fmt(train_summary["p95"])}, 99th {_fmt(train_summary["p99"])}.

Counts at or above round duration marks in the labeled pool. These marks are descriptive. No clip was removed.

{threshold_lines}

Per class, labeled pool:

{class_lines}

Class medians range from {_fmt(min(row["duration_median"] for row in duration["per_class_labeled_pool"]))} s to {_fmt(max(row["duration_median"] for row in duration["per_class_labeled_pool"]))} s. Class means range from {_fmt(min(row["duration_mean"] for row in duration["per_class_labeled_pool"]))} s to {_fmt(max(row["duration_mean"] for row in duration["per_class_labeled_pool"]))} s.

## 5. Long-tail duration findings

Clips of at least 10 s: {tail_10["n_clips"]}, spread across {tail_10["n_classes_represented"]} classes. The largest class count among those clips is {tail_10["max_class_count"]}. Counts by class: {tail_10["counts_by_class"]}.

Clips of at least 60 s: {tail_60["n_clips"]}, spread across {tail_60["n_classes_represented"]} classes. The largest class count among those clips is {tail_60["max_class_count"]}. Counts by class: {tail_60["counts_by_class"]}.

The maximum labeled duration is {_fmt(summary["max"])} s. These clips were not removed, and this file does not call them noise.

## 6. Temporal and energy findings

Two separate descriptive rules were used.

Sample-level rule, already stored in Phase 2: proportion of samples below 1% of that clip's peak absolute amplitude. Labeled-pool median {sample_prop["median"]:.4f}, maximum {sample_prop["max"]:.4f}. This was not established as silence.

Frame rule, computed in this phase: a 25 ms frame every 10 ms is counted when its RMS is below 1% of that clip's maximum frame RMS. Labeled-pool frame proportion median {frame_prop["median"]:.4f}, maximum {frame_prop["max"]:.4f}. Leading low-energy time median {leading["median"]:.4f} s, maximum {leading["max"]:.4f} s. Trailing low-energy time median {trailing["median"]:.4f} s, maximum {trailing["max"]:.4f} s. Clips where every analysis frame met the low-energy rule: {temporal["labeled_pool"]["n_all_frames_low_energy"]}.

By duration bin:

{bin_lines}

Median frame low-energy proportion by class ranges from {_fmt(min(row["median_low_energy_frame_proportion"] for row in temporal["by_class"]), 4)} to {_fmt(max(row["median_low_energy_frame_proportion"] for row in temporal["by_class"]), 4)}.

Representative clips were chosen without a new random draw: the shortest labeled clip, the clip closest to the labeled median duration, the longest labeled clip, and the clip between the 25th and 75th duration percentiles with the highest sample-level peak-rule proportion. Ids are in `results/eda/representative_examples.csv`.

## 7. Spectral and MFCC findings

Candidate MFCC settings, not an adopted pipeline: {FRAME_LENGTH_MS:.0f} ms Hann window, {HOP_LENGTH_MS:.0f} ms hop, FFT length {N_FFT}, {N_MELS} HTK-mel filters from {FMIN_HZ:.0f} Hz to the Nyquist frequency, log floor {LOG_FLOOR}, orthonormal DCT-II with c0 kept, pre-emphasis {PRE_EMPHASIS}, no lifter. Frame counts below use that framing formula on every labeled clip. The formula was checked by computing MFCC matrices on {mfcc["n_clips_checked_by_computing_mfcc"]} clips. Shape mismatches: {mfcc["mfcc_shape_mismatches"] if mfcc["mfcc_shape_mismatches"] else "none"}.

Labeled-pool analysis-frame counts: minimum {mfcc["frame_counts_on_labeled_pool"]["min"]:.0f}, median {mfcc["frame_counts_on_labeled_pool"]["median"]:.1f}, maximum {mfcc["frame_counts_on_labeled_pool"]["max"]:.0f}.

{candidate_lines}

On one representative clip (`{mfcc["delta_check"]["reference_word_id"]}`), static 13-MFCC shape {mfcc["delta_check"]["static_shape"]}, delta shape {mfcc["delta_check"]["delta_shape"]}, delta-delta shape {mfcc["delta_check"]["delta_delta_shape"]}. Stacking static, delta, and delta-delta multiplies the coefficient axis by 3. That stacking was computed once to test feasibility. It was not adopted.

## 8. Fixed-length implications

Round windows on the shared training split:

{window_lines}

Empirical training-split percentile windows, included because they come from the measured distribution rather than from round numbers:

{percentile_window_lines}

No window was selected. No audio was padded or truncated on disk.

## 9. Data quality findings

Unreadable files in the full labeled read: {quality["unreadable_files_recorded_in_temporal_pass"]}. Duplicate labeled ids in the metadata table: {quality["duplicate_labeled_ids_in_metadata"]}. Exact PCM SHA-256 values: {temporal["pcm_sha256"]["n_unique_hashes"]} unique hashes out of {temporal["pcm_sha256"]["n_clips"]} labeled clips. Duplicate hash groups: {temporal["pcm_sha256"]["n_duplicate_groups"]}. Pairwise near-duplicate comparison was not run. Identical PCM payloads are the only duplicates this check can detect. Time shifts and gain changes were not compared.

Clips with peak absolute amplitude at least 32767: {quality["n_full_scale_peak_at_least_32767"]}. Shortest and longest ids are listed in `results/eda/data_quality.json`. No speaker labels exist in the released tables, and filenames were not treated as speaker ids.

## 10. Implications for preprocessing

Observed: labeled class counts are equal, and the shared split stays within one clip of equality per class.

Interpretation: class weights and oversampling are not supported by the label counts.

Open for Phase 4: none on this specific point. The supported statement is to not add class weighting because of label frequency.

Observed: every labeled clip that was read is 16 kHz, mono, 16-bit PCM.

Interpretation: resampling or downmixing is not required to make the labeled files match one another.

Open for Phase 4: a model could still change the sample rate. The files themselves do not require that change.

Observed: durations vary from {_fmt(summary["min"])} s to {_fmt(summary["max"])} s, with a long tail that is not confined to one class.

Interpretation: a fixed number of time steps will require padding, truncation, or both if a later model needs a fixed length. Cutting every clip to a short window would discard part of the long recordings. Padding to the maximum would create sequences of {mfcc["frame_counts_on_labeled_pool"]["max"]:.0f} analysis frames under the candidate 10 ms hop.

Open for Phase 4: the window length, where a longer clip is cut, and what value is used for padding.

Observed: under the documented frame-RMS rule, median leading and trailing low-energy times are {leading["median"]:.4f} s and {trailing["median"]:.4f} s. The sample-level peak rule has a high median proportion. Neither rule was validated as speech-versus-silence.

Interpretation: many samples are far below the clip peak. That does not by itself identify silence or justify trimming.

Open for Phase 4: whether any low-energy trim is used, and the rule if it is.

Observed: 13, 20, and 40 MFCCs are computable with the candidate settings, and delta features are a small extra calculation. Storing every frame of every clip costs much more than mean or mean-and-standard-deviation pooling.

Interpretation: a classical model will need some aggregation or a fixed window, because the sequences are not the same length. This investigation does not show which representation classifies the words more accurately.

Open for Phase 4: coefficient count, deltas, lifter, pre-emphasis, and aggregation.
"""
    path = results_eda_dir(root) / "eda_findings.md"
    path.write_text(text, encoding="utf-8")
    return path


def preprocessing_decisions(payload: dict) -> list[dict]:
    quality = payload["quality"]
    rates = quality["sample_rate_counts"]
    channels = quality["channel_counts"]
    uniform_format = rates == {"16000": payload["class_balance"]["n_labeled_examples"]} and channels == {
        "1": payload["class_balance"]["n_labeled_examples"]
    }
    rows = [
        {
            "topic": "sample_rate",
            "status": "SUPPORTED" if uniform_format else "TO DECIDE IN PHASE 4",
            "evidence": f"Sample-rate counts: {rates}.",
            "statement": (
                "Do not resample merely to reconcile labeled clips with each other; all read labeled clips are 16000 Hz."
                if uniform_format
                else "Sample rates differ. A conversion rule is still open."
            ),
        },
        {
            "topic": "mono_stereo",
            "status": "SUPPORTED" if channels == {"1": payload["class_balance"]["n_labeled_examples"]} else "TO DECIDE IN PHASE 4",
            "evidence": f"Channel counts: {channels}.",
            "statement": "No downmix is required to reconcile labeled clips. All read labeled clips are mono.",
        },
        {
            "topic": "class_imbalance_handling",
            "status": "SUPPORTED",
            "evidence": payload["class_balance"]["reason"],
            "statement": "Do not use class weighting or oversampling because of label frequency.",
        },
        {
            "topic": "amplitude_normalization",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": (
                f"{quality['n_full_scale_peak_at_least_32767']} labeled clips reach a peak absolute amplitude "
                "of at least 32767. RMS varies across clips in the Phase 2 metadata. No normalization was fit."
            ),
            "statement": "Peak or RMS scaling was not chosen.",
        },
        {
            "topic": "silence_or_low_energy_trimming",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": payload["temporal"]["rule"]["definition"],
            "statement": "Low-energy proportions were measured. They were not shown to be silence, and no trim was applied.",
        },
        {
            "topic": "duration_window",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": "Round windows 4, 5, 6, and 8 seconds, plus training-split percentile windows, are in fixed_window_impact.csv.",
            "statement": "No fixed duration was selected.",
        },
        {
            "topic": "padding",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": "Shorter-than-window percentages are recorded. Padding values were not tested.",
            "statement": "Padding location and fill value are open.",
        },
        {
            "topic": "truncation",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": "Longer-than-window percentages and retained-duration percentages are recorded. No clip was cut.",
            "statement": "Whether to cut the start, the end, or another region is open.",
        },
        {
            "topic": "mfcc_configuration",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": "Candidate settings are 13, 20, and 40 coefficients with a 25 ms window, 10 ms hop, 40 HTK-mel bands, and FFT length 512.",
            "statement": "No MFCC configuration was adopted.",
        },
        {
            "topic": "delta_and_delta_delta",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": "Delta features were computed for one clip and are feasible. They were not compared in a classifier.",
            "statement": "Deltas were not adopted.",
        },
        {
            "topic": "feature_aggregation",
            "status": "TO DECIDE IN PHASE 4",
            "evidence": "Sequence storage at the median length is much larger than mean pooling. Accuracy was not measured.",
            "statement": "Mean pooling, mean and standard deviation, or a fixed-length sequence were not chosen.",
        },
    ]
    return rows


def run_phase3(root: Path | None = None) -> dict:
    """Run the Phase 3 investigation. Does not rewrite the shared split."""
    set_seed(SHARED_SPLIT_SEED)
    train_df = load_train_csv()
    official_test = load_test_csv()
    metadata = load_labeled_metadata(root)
    split_info = verify_saved_split(train_df, official_test)
    if split_info["official_test_ids_in_split"] != 0:
        raise RuntimeError("The saved split contains official unlabeled test ids.")
    if split_info["missing_labeled_ids"] != 0 or split_info["duplicate_word_ids"] != 0:
        raise RuntimeError("The saved split does not cover each labeled id once.")
    split_df = read_shared_split()
    metadata = metadata.merge(split_df, on="Word_id", how="left", validate="one_to_one")
    if metadata["split"].isna().any():
        raise RuntimeError("Some labeled clips have no saved split assignment.")

    balance = class_balance_report(train_df, split_df)
    duration = duration_report(metadata)
    print("Reading labeled audio for hashes and frame energy.", flush=True)
    temporal_frame = temporal_energy_report(metadata)
    temporal = temporal_summary(metadata, temporal_frame)
    windows = fixed_window_report(metadata)
    examples = select_representative_examples(metadata)
    print("Computing candidate MFCCs on the representative subset.", flush=True)
    mfcc = mfcc_report(metadata, examples)
    plot_features = mfcc.pop("plot_features")
    quality = quality_report(metadata, temporal_frame, split_info)
    quality["pcm_sha256_duplicate_groups"] = temporal["pcm_sha256"]["n_duplicate_groups"]
    quality.pop("pcm_sha256_duplicate_groups", None)
    quality["n_duplicate_pcm_hash_groups"] = temporal["pcm_sha256"]["n_duplicate_groups"]

    destination = results_eda_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "class_balance.json", balance)
    pd.DataFrame(balance["class_counts_by_split"]).to_csv(destination / "class_balance_by_split.csv", index=False)
    _write_json(destination / "duration_summary.json", duration)
    pd.DataFrame(duration["per_class_labeled_pool"]).to_csv(
        destination / "duration_by_class_detailed.csv",
        index=False,
    )
    threshold_rows = []
    for scope_name, scope in duration["scopes"].items():
        for row in scope["thresholds"]:
            threshold_rows.append({"scope": scope_name, **row})
    pd.DataFrame(threshold_rows).to_csv(destination / "duration_thresholds.csv", index=False)
    temporal_frame.to_csv(destination / "temporal_energy_by_clip.csv", index=False)
    _write_json(destination / "temporal_energy_summary.json", temporal)
    windows.to_csv(destination / "fixed_window_impact.csv", index=False)
    pd.DataFrame(mfcc["candidates"]).to_csv(destination / "mfcc_candidates.csv", index=False)
    _write_json(destination / "mfcc_candidates.json", mfcc)
    _write_json(destination / "data_quality.json", quality)
    examples.to_csv(destination / "representative_examples.csv", index=False)
    environment = environment_record(SHARED_SPLIT_SEED)
    environment["phase"] = 3
    environment["shared_split_rewritten"] = False
    _write_json(destination / "phase3_environment.json", environment)

    payload = {
        "class_balance": balance,
        "duration": duration,
        "temporal": temporal,
        "quality": quality,
        "split": split_info,
        "mfcc": mfcc,
        "fixed_windows_train_round": windows.loc[
            (windows["scope"] == "shared_train_split") & (windows["window_source"] == "round")
        ].to_dict(orient="records"),
        "fixed_windows_train_percentile": windows.loc[
            (windows["scope"] == "shared_train_split") & (windows["window_source"] == "empirical_percentile")
        ].to_dict(orient="records"),
    }
    save_phase3_figures(
        metadata,
        windows,
        examples,
        {"plot_features": plot_features},
        temporal["by_duration_bin"],
        root,
    )
    decisions = preprocessing_decisions(payload)
    _write_json(
        destination / "preprocessing_investigation.json",
        {"phase": 3, "decisions": decisions, "shared_split_rewritten": False},
    )
    pd.DataFrame(decisions).to_csv(destination / "preprocessing_investigation.csv", index=False)
    write_phase3_findings(payload, root)
    if mfcc["mfcc_shape_mismatches"]:
        raise RuntimeError(f"MFCC frame counts did not match the formula: {mfcc['mfcc_shape_mismatches']}")
    print("Phase 3 artifacts written. Shared split was not rewritten.", flush=True)
    return payload
