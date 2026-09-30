"""Phase 2 checks, shared split, and audio EDA.

This module reads the raw challenge files and writes summaries, figures, and
the shared split. It does not train models or extract model features.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audio import LOW_ENERGY_PEAK_FRACTION, collect_metadata, metadata_from_wav_bytes
from src.data.loading import (
    audio_zip_path,
    list_zip_members,
    load_sample_submission,
    load_test_csv,
    load_train_csv,
    project_root,
    raw_data_dir,
    read_wav_bytes,
)
from src.data.split import (
    HOLDOUT_FRACTION,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    create_shared_split,
    save_shared_split,
    validate_shared_split,
)
from src.utils.reproducibility import SHARED_SPLIT_SEED, environment_record, set_seed

FILENAME_PATTERN = re.compile(r"^id_[a-z0-9]+\.wav$")
SPECTROGRAM_NFFT = 512
SPECTROGRAM_NOVERLAP = 256
SPECTROGRAM_DB_FLOOR = 1e-12
# Round second marks used only to describe the long-duration tail.
# They are not clip-removal rules.
DURATION_TAIL_SECONDS = (10, 20, 30, 60)
# Histogram display limit so the main mass is visible. Clips at or above
# this value stay in the table and in the boxplot.
DURATION_HIST_DISPLAY_MAX_SECONDS = 10.0


def results_eda_dir(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / "results" / "eda"


def figures_dir(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / "results" / "figures"


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
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Series,)):
        return _json_ready(value.to_dict())
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _distribution(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _numeric_summary(series: pd.Series) -> dict:
    quantiles = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95], interpolation="linear")
    return {
        "n": int(series.notna().sum()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std_ddof1": float(series.std(ddof=1)),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
    }


def _translation_map(train_df: pd.DataFrame) -> dict:
    grouped = (
        train_df.groupby(["Swahili_word", "English_translation"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    per_swahili = train_df.groupby("Swahili_word")["English_translation"].nunique(dropna=False)
    per_english = train_df.groupby("English_translation")["Swahili_word"].nunique(dropna=False)
    return {
        "pairs": grouped.to_dict(orient="records"),
        "max_english_per_swahili": int(per_swahili.max()),
        "max_swahili_per_english": int(per_english.max()),
        "consistent_within_each_swahili_label": bool(per_swahili.max() == 1),
        "consistent_within_each_english_label": bool(per_english.max() == 1),
    }


def _speaker_audit(train_df: pd.DataFrame, test_df: pd.DataFrame, zip_names: list[str]) -> dict:
    columns = list(train_df.columns) + [column for column in test_df.columns if column not in train_df.columns]
    name_hits = [
        column
        for column in columns
        if any(token in column.lower() for token in ("speaker", "participant", "gender"))
    ]
    starter_path = raw_data_dir() / "Swahili_Audio_StarterNotebook.ipynb"
    starter_text = starter_path.read_text(encoding="utf-8").lower()
    starter_hits = [
        token
        for token in ("speaker", "participant", "gender")
        if token in starter_text
    ]
    nested_members = [name for name in zip_names if "/" in name or "\\" in name]
    return {
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "speaker_like_columns": name_hits,
        "zip_members_in_subdirectories": len(nested_members),
        "starter_notebook_path": starter_path.name,
        "starter_notebook_term_hits": starter_hits,
        "speaker_identifier_present": bool(name_hits or nested_members or starter_hits),
        "note": (
            "No speaker identifier was found in the CSV columns, zip paths, or starter notebook. "
            "Filenames were not treated as speaker ids."
        ),
    }


def build_dataset_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    zip_path: Path,
    zip_names: list[str],
) -> dict:
    train_ids = set(train_df["Word_id"])
    test_ids = set(test_df["Word_id"])
    submission_ids = set(submission_df["Word_id"])
    zip_ids = set(zip_names)
    class_counts = train_df["Swahili_word"].value_counts(dropna=False).sort_index()
    submission_class_columns = [column for column in submission_df.columns if column != "Word_id"]
    train_classes = set(train_df["Swahili_word"].dropna().unique())
    non_id_submission = submission_df.drop(columns=["Word_id"])
    infos = zipfile.ZipFile(zip_path).infolist()
    extensions = pd.Series([Path(name).suffix.lower() for name in zip_names]).value_counts().to_dict()
    token_lengths = pd.Series(
        [len(name[3:-4]) for name in zip_names if name.startswith("id_") and name.endswith(".wav")]
    )
    return {
        "train": {
            "n_rows": int(len(train_df)),
            "n_columns": int(train_df.shape[1]),
            "columns": list(train_df.columns),
            "dtypes": {column: str(dtype) for column, dtype in train_df.dtypes.items()},
            "missing_values": {column: int(train_df[column].isna().sum()) for column in train_df.columns},
            "duplicate_word_ids": int(train_df["Word_id"].duplicated().sum()),
            "duplicate_complete_rows": int(train_df.duplicated().sum()),
            "n_classes": int(train_df["Swahili_word"].nunique(dropna=False)),
            "class_labels": [str(label) for label in class_counts.index],
            "class_counts": {str(label): int(count) for label, count in class_counts.items()},
            "classes_have_equal_counts": bool(class_counts.nunique() == 1),
            "translations": _translation_map(train_df),
        },
        "test": {
            "n_rows": int(len(test_df)),
            "n_columns": int(test_df.shape[1]),
            "columns": list(test_df.columns),
            "dtypes": {column: str(dtype) for column, dtype in test_df.dtypes.items()},
            "missing_values": {column: int(test_df[column].isna().sum()) for column in test_df.columns},
            "duplicate_word_ids": int(test_df["Word_id"].duplicated().sum()),
            "overlap_with_train_ids": int(len(train_ids & test_ids)),
        },
        "sample_submission": {
            "n_rows": int(len(submission_df)),
            "n_columns": int(submission_df.shape[1]),
            "columns": list(submission_df.columns),
            "dtypes": {column: str(dtype) for column, dtype in submission_df.dtypes.items()},
            "ids_match_test_set": submission_ids == test_ids,
            "ids_in_same_order_as_test": bool(submission_df["Word_id"].equals(test_df["Word_id"])),
            "n_class_columns": len(submission_class_columns),
            "class_columns": submission_class_columns,
            "class_columns_match_train_labels": set(submission_class_columns) == train_classes,
            "all_class_entries_are_zero": bool((non_id_submission == 0).all().all()),
        },
        "audio_zip": {
            "path_name": zip_path.name,
            "compressed_file_size_bytes": int(zip_path.stat().st_size),
            "member_compressed_size_bytes": int(sum(info.compress_size for info in infos)),
            "member_uncompressed_size_bytes": int(sum(info.file_size for info in infos)),
            "n_members": len(zip_names),
            "extensions": {str(key): int(value) for key, value in extensions.items()},
            "n_members_matching_id_pattern": int(sum(bool(FILENAME_PATTERN.fullmatch(name)) for name in zip_names)),
            "filename_token_length_counts": _distribution(token_lengths) if len(token_lengths) else {},
            "train_ids_missing_from_zip": int(len(train_ids - zip_ids)),
            "test_ids_missing_from_zip": int(len(test_ids - zip_ids)),
            "zip_members_not_in_train_or_test": int(len(zip_ids - train_ids - test_ids)),
        },
        "speaker": _speaker_audit(train_df, test_df, zip_names),
    }


def save_dataset_summary(summary: dict, root: Path | None = None) -> None:
    destination = results_eda_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "dataset_summary.json", summary)

    rows: list[dict] = []
    train = summary["train"]
    for label, count in train["class_counts"].items():
        english = [
            pair["English_translation"]
            for pair in train["translations"]["pairs"]
            if pair["Swahili_word"] == label
        ]
        rows.append(
            {
                "section": "class_count",
                "key": label,
                "value": count,
                "detail": english[0] if len(english) == 1 else "|".join(str(item) for item in english),
            }
        )
    flat_checks = {
        "train_rows": train["n_rows"],
        "train_duplicate_word_ids": train["duplicate_word_ids"],
        "train_duplicate_complete_rows": train["duplicate_complete_rows"],
        "train_classes_equal": train["classes_have_equal_counts"],
        "translation_consistent_per_swahili": train["translations"]["consistent_within_each_swahili_label"],
        "test_rows": summary["test"]["n_rows"],
        "test_duplicate_word_ids": summary["test"]["duplicate_word_ids"],
        "train_test_id_overlap": summary["test"]["overlap_with_train_ids"],
        "submission_rows": summary["sample_submission"]["n_rows"],
        "submission_ids_match_test": summary["sample_submission"]["ids_match_test_set"],
        "submission_class_count": summary["sample_submission"]["n_class_columns"],
        "submission_classes_match_train": summary["sample_submission"]["class_columns_match_train_labels"],
        "zip_members": summary["audio_zip"]["n_members"],
        "zip_compressed_file_size_bytes": summary["audio_zip"]["compressed_file_size_bytes"],
        "train_ids_missing_from_zip": summary["audio_zip"]["train_ids_missing_from_zip"],
        "test_ids_missing_from_zip": summary["audio_zip"]["test_ids_missing_from_zip"],
        "zip_members_not_in_train_or_test": summary["audio_zip"]["zip_members_not_in_train_or_test"],
        "speaker_identifier_present": summary["speaker"]["speaker_identifier_present"],
    }
    for key, value in flat_checks.items():
        rows.append({"section": "check", "key": key, "value": value, "detail": ""})
    pd.DataFrame(rows).to_csv(destination / "dataset_summary.csv", index=False)


def _extreme_records(metadata: pd.DataFrame, column: str, largest: bool, n: int = 3) -> list[dict]:
    ordered = metadata.sort_values([column, "Word_id"], ascending=[not largest, True])
    # For largest=False, ascending duration then Word_id. For largest=True the
    # boolean above sorts duration descending. Re-sort explicitly for clarity.
    ordered = metadata.sort_values(
        [column, "Word_id"],
        ascending=[not largest, True],
        kind="mergesort",
    )
    subset = ordered.head(n)
    columns = [
        "Word_id",
        "Swahili_word",
        "English_translation",
        "duration_seconds",
        "n_frames",
        "sample_rate_hz",
        "low_energy_proportion",
        "peak_abs_amplitude",
        "rms_amplitude",
    ]
    return subset.loc[:, columns].to_dict(orient="records")


def build_audio_summary(metadata: pd.DataFrame) -> dict:
    by_class = (
        metadata.groupby(["Swahili_word", "English_translation"], dropna=False, as_index=False)
        .agg(
            n=("Word_id", "size"),
            duration_min=("duration_seconds", "min"),
            duration_median=("duration_seconds", "median"),
            duration_mean=("duration_seconds", "mean"),
            duration_max=("duration_seconds", "max"),
        )
        .sort_values("Swahili_word")
    )
    rates = _distribution(metadata["sample_rate_hz"])
    channels = _distribution(metadata["n_channels"])
    bit_depths = _distribution(metadata["bit_depth"])
    sample_widths = _distribution(metadata["sample_width_bytes"])
    compression = _distribution(metadata["compression_type"])
    unique_rate = metadata["sample_rate_hz"].nunique()
    tail_counts = {
        f"at_least_{cut}_seconds": int((metadata["duration_seconds"] >= cut).sum())
        for cut in DURATION_TAIL_SECONDS
    }
    if {int(value) for value in metadata["bit_depth"].unique()} == {16}:
        n_full_scale = int((metadata["peak_abs_amplitude"] >= 32767).sum())
    else:
        n_full_scale = None
    return {
        "n_clips": int(len(metadata)),
        "low_energy_rule": {
            "fraction_of_clip_peak": LOW_ENERGY_PEAK_FRACTION,
            "definition": (
                "Fraction of PCM samples whose absolute amplitude is strictly below "
                f"{LOW_ENERGY_PEAK_FRACTION:.0%} of that clip's peak absolute amplitude. "
                "If the peak is 0, the proportion is 1. This rule was fixed for description. "
                "It was not tuned on a validation split and is not a silence label."
            ),
        },
        "duration_seconds": _numeric_summary(metadata["duration_seconds"]),
        "n_frames": _numeric_summary(metadata["n_frames"].astype(float)),
        "rms_amplitude": _numeric_summary(metadata["rms_amplitude"]),
        "peak_abs_amplitude": _numeric_summary(metadata["peak_abs_amplitude"]),
        "low_energy_proportion": _numeric_summary(metadata["low_energy_proportion"]),
        "sample_rate_hz_counts": rates,
        "n_channels_counts": channels,
        "bit_depth_counts": bit_depths,
        "sample_width_bytes_counts": sample_widths,
        "compression_type_counts": compression,
        "n_zero_frame_clips": int((metadata["n_frames"] == 0).sum()),
        "n_zero_peak_clips": int((metadata["peak_abs_amplitude"] == 0).sum()),
        "n_clips_at_int16_full_scale": n_full_scale,
        "duration_tail_counts": tail_counts,
        "duration_tail_note": (
            "Counts are clips with duration greater than or equal to the stated "
            "number of seconds. The cut points describe the tail. No clip was removed."
        ),
        "format_is_uniform": bool(
            unique_rate == 1
            and metadata["n_channels"].nunique() == 1
            and metadata["bit_depth"].nunique() == 1
            and metadata["compression_type"].nunique() == 1
        ),
        "sampling_period_seconds": (
            float(1.0 / metadata["sample_rate_hz"].iloc[0]) if unique_rate == 1 else None
        ),
        "duration_by_class": by_class.to_dict(orient="records"),
        "shortest_clips": _extreme_records(metadata, "duration_seconds", largest=False),
        "longest_clips": _extreme_records(metadata, "duration_seconds", largest=True),
        "highest_low_energy_clips": _extreme_records(metadata, "low_energy_proportion", largest=True),
    }


def _choose_examples(metadata: pd.DataFrame) -> pd.DataFrame:
    """One clip per class: duration closest to that class median, then Word_id."""
    chosen_frames = []
    for _, group in metadata.groupby("Swahili_word", sort=True):
        median_duration = group["duration_seconds"].median()
        ranked = group.assign(_distance=(group["duration_seconds"] - median_duration).abs())
        ranked = ranked.sort_values(["_distance", "Word_id"], kind="mergesort")
        chosen_frames.append(ranked.iloc[0])
    chosen = pd.DataFrame(chosen_frames).drop(columns="_distance")
    return chosen.sort_values("Swahili_word").reset_index(drop=True)


def _load_pcm(word_id: str) -> tuple[np.ndarray, dict]:
    info = metadata_from_wav_bytes(word_id, read_wav_bytes(word_id))
    # metadata_from_wav_bytes already consumed the frames. Read once more for
    # the plot so the metadata function stays free of plotting state.
    import io
    import wave

    with wave.open(io.BytesIO(read_wav_bytes(word_id)), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        sample_width = wav_file.getsampwidth()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
    from src.data.audio import _pcm_samples

    samples = _pcm_samples(frames, sample_width, n_frames * n_channels)
    if n_channels > 1:
        samples = samples.reshape(n_frames, n_channels)[:, 0]
    scale = float(2 ** (8 * sample_width - 1))
    return samples / scale, info


def save_figures(metadata: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from matplotlib.mlab import specgram

    destination = figures_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    class_order = sorted(metadata["Swahili_word"].unique())
    english = (
        metadata.groupby("Swahili_word")["English_translation"].first().to_dict()
    )
    labels = [f"{label} ({english[label]})" for label in class_order]
    counts = metadata["Swahili_word"].value_counts().reindex(class_order)

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(class_order))
    axis.barh(positions, counts.to_numpy(), color="#4C78A8")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Number of labeled clips")
    axis.set_xlim(0, max(float(counts.max()) * 1.15, 1))
    axis.set_title("Class distribution in Train.csv")
    for position, count in zip(positions, counts.to_numpy()):
        axis.text(count + max(counts.max() * 0.02, 1), position, str(int(count)), va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination / "class_distribution.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )
    display_max = DURATION_HIST_DISPLAY_MAX_SECONDS
    shown = metadata.loc[metadata["duration_seconds"] < display_max, "duration_seconds"]
    n_omitted = int((metadata["duration_seconds"] >= display_max).sum())
    axes[0].hist(shown, bins=30, color="#4C78A8", edgecolor="white")
    median_duration = float(metadata["duration_seconds"].median())
    p95_duration = float(metadata["duration_seconds"].quantile(0.95, interpolation="linear"))
    axes[0].axvline(median_duration, color="#E45756", linestyle="--", label=f"median = {median_duration:.3f} s")
    axes[0].axvline(p95_duration, color="#F58518", linestyle=":", label=f"95th percentile = {p95_duration:.3f} s")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Number of labeled clips")
    axes[0].set_title(
        f"Durations below {display_max:.0f} s (n={len(shown)}). "
        f"{n_omitted} clips at or above {display_max:.0f} s are omitted here and shown in the boxplot."
    )
    axes[0].legend(frameon=False)
    box_data = [
        metadata.loc[metadata["Swahili_word"] == label, "duration_seconds"].to_numpy()
        for label in class_order
    ]
    axes[1].boxplot(box_data, tick_labels=labels, vert=True, showfliers=True)
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].set_ylabel("Duration (seconds)")
    axes[1].set_title("Duration by class")
    figure.tight_layout()
    figure.savefig(destination / "audio_duration_distribution.png", dpi=150)
    plt.close(figure)

    rate_counts = metadata["sample_rate_hz"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([str(rate) for rate in rate_counts.index], rate_counts.to_numpy(), color="#4C78A8")
    axis.set_xlabel("Sample rate (Hz)")
    axis.set_ylabel("Number of labeled clips")
    axis.set_title("Sample-rate distribution of labeled training clips")
    channel_counts = metadata["n_channels"].value_counts().sort_index().to_dict()
    depth_counts = metadata["bit_depth"].value_counts().sort_index().to_dict()
    compression_counts = metadata["compression_type"].value_counts().sort_index().to_dict()
    note = (
        f"Channels: {channel_counts}\n"
        f"Bit depth: {depth_counts}\n"
        f"Compression: {compression_counts}"
    )
    axis.text(
        0.98,
        0.95,
        note,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    figure.tight_layout()
    figure.savefig(destination / "sample_rate_distribution.png", dpi=150)
    plt.close(figure)

    examples = _choose_examples(metadata)
    waveform_cache: dict[str, tuple[np.ndarray, dict]] = {}
    for word_id in examples["Word_id"]:
        waveform_cache[word_id] = _load_pcm(word_id)

    max_duration = max(info["duration_seconds"] for _, info in waveform_cache.values())
    figure, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    for axis, (_, row) in zip(axes.ravel(), examples.iterrows()):
        samples, info = waveform_cache[row["Word_id"]]
        time_seconds = np.arange(info["n_frames"]) / info["sample_rate_hz"]
        axis.plot(time_seconds, samples, color="#4C78A8", linewidth=0.5)
        axis.set_title(f"{row['Swahili_word']} ({row['English_translation']})", fontsize=9)
        axis.set_xlim(0, max_duration)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (seconds)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude")
    figure.suptitle(
        "One labeled clip per class, closest to that class median duration",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(destination / "example_waveforms.png", dpi=150)
    plt.close(figure)

    spectrograms = []
    for _, row in examples.iterrows():
        samples, info = waveform_cache[row["Word_id"]]
        nfft = SPECTROGRAM_NFFT if len(samples) >= SPECTROGRAM_NFFT else len(samples)
        noverlap = min(SPECTROGRAM_NOVERLAP, max(nfft // 2, 0))
        if nfft < 2:
            raise ValueError(f"{row['Word_id']} is too short for a spectrogram plot.")
        power, frequencies, times = specgram(
            samples,
            NFFT=nfft,
            Fs=info["sample_rate_hz"],
            noverlap=noverlap,
        )
        decibels = 10.0 * np.log10(power + SPECTROGRAM_DB_FLOOR)
        spectrograms.append(
            {
                "row": row,
                "decibels": decibels,
                "frequencies": frequencies,
                "times": times,
                "nfft": nfft,
                "noverlap": noverlap,
            }
        )
    stacked = np.concatenate([item["decibels"].ravel() for item in spectrograms])
    vmin = float(np.percentile(stacked, 5))
    vmax = float(np.percentile(stacked, 95))
    figure, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    image = None
    for axis, item in zip(axes.ravel(), spectrograms):
        row = item["row"]
        image = axis.imshow(
            item["decibels"],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[
                float(item["times"][0]) if len(item["times"]) else 0.0,
                float(item["times"][-1]) if len(item["times"]) else 0.0,
                float(item["frequencies"][0]) if len(item["frequencies"]) else 0.0,
                float(item["frequencies"][-1]) if len(item["frequencies"]) else 0.0,
            ],
            cmap="magma",
        )
        axis.set_title(f"{row['Swahili_word']} ({row['English_translation']})", fontsize=9)
        axis.set_xlim(0, max_duration)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (seconds)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency (Hz)")
    figure.suptitle(
        f"Example spectrograms (NFFT={SPECTROGRAM_NFFT}, overlap={SPECTROGRAM_NOVERLAP}; display only)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.subplots_adjust(right=0.90)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    colorbar.set_label("Power (dB, display scale)")
    figure.savefig(destination / "example_spectrograms.png", dpi=150)
    plt.close(figure)

    examples_out = examples.loc[:, ["Word_id", "Swahili_word", "English_translation", "duration_seconds"]].copy()
    examples_out["selection_rule"] = "closest duration to class median; tie broken by Word_id"
    examples_out["spectrogram_nfft"] = [item["nfft"] for item in spectrograms]
    examples_out["spectrogram_noverlap"] = [item["noverlap"] for item in spectrograms]
    examples_out["spectrogram_db_floor"] = SPECTROGRAM_DB_FLOOR
    examples_out["spectrogram_color_limits_db"] = f"{vmin:.6f},{vmax:.6f}"
    return examples_out


def _fmt_seconds(value: float) -> str:
    return f"{value:.4f}"


def _count_phrase(counts: dict) -> str:
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def write_eda_findings(
    dataset_summary: dict,
    audio_summary: dict,
    split_summary: dict,
    root: Path | None = None,
) -> Path:
    train = dataset_summary["train"]
    test = dataset_summary["test"]
    submission = dataset_summary["sample_submission"]
    archive = dataset_summary["audio_zip"]
    speaker = dataset_summary["speaker"]
    duration = audio_summary["duration_seconds"]
    frames = audio_summary["n_frames"]
    low_energy = audio_summary["low_energy_proportion"]
    rms = audio_summary["rms_amplitude"]
    peak = audio_summary["peak_abs_amplitude"]
    class_lines = "\n".join(
        f"- {label}: {count}"
        for label, count in train["class_counts"].items()
    )
    translation_lines = "\n".join(
        f"- {pair['Swahili_word']} / {pair['English_translation']}: {pair['n']}"
        for pair in train["translations"]["pairs"]
    )
    duration_class_lines = "\n".join(
        "- {Swahili_word} ({English_translation}): n={n}, min={duration_min:.4f} s, "
        "median={duration_median:.4f} s, max={duration_max:.4f} s".format(**row)
        for row in audio_summary["duration_by_class"]
    )
    class_medians = [row["duration_median"] for row in audio_summary["duration_by_class"]]
    class_maxima = [row["duration_max"] for row in audio_summary["duration_by_class"]]
    tail = audio_summary["duration_tail_counts"]
    split_lines = []
    for split_name in ("train", "validation", "test"):
        rows = [
            row
            for row in split_summary["class_counts"]
            if row["split"] == split_name
        ]
        counts = [row["n"] for row in rows]
        detail = ", ".join(f"{row['Swahili_word']}={row['n']}" for row in rows)
        split_lines.append(
            f"- {split_name}: {split_summary['split_counts'].get(split_name, 0)} clips; "
            f"per-class counts range from {min(counts)} to {max(counts)}. {detail}."
        )
    shortest = audio_summary["shortest_clips"][0]
    longest = audio_summary["longest_clips"][0]
    highest_low = audio_summary["highest_low_energy_clips"][0]
    rate_phrase = _count_phrase(audio_summary["sample_rate_hz_counts"])
    channel_phrase = _count_phrase(audio_summary["n_channels_counts"])
    depth_phrase = _count_phrase(audio_summary["bit_depth_counts"])
    compression_phrase = _count_phrase(audio_summary["compression_type_counts"])

    if duration["min"] == duration["max"]:
        length_implication = (
            "Labeled clips already share one duration, so padding or truncation is not required "
            "to reconcile their lengths with each other."
        )
    else:
        length_implication = (
            "A later model that requires a fixed number of time steps will need a padding or "
            "truncation rule. That rule is not chosen in this phase. The duration distribution "
            "above is the measurement to use when choosing it."
        )
    if audio_summary["format_is_uniform"] and len(audio_summary["sample_rate_hz_counts"]) == 1:
        rate_implication = (
            "The labeled clips do not need resampling or channel conversion to match one another. "
            "This does not choose a sample rate for a future model."
        )
    else:
        rate_implication = (
            "Labeled clips do not all share one sample rate, channel count, or bit depth. "
            "A later pipeline will need an explicit conversion rule before comparing models. "
            "That rule is not chosen in this phase."
        )
    if train["classes_have_equal_counts"]:
        balance_implication = (
            "Unequal class weights are not motivated by the labeled class counts. "
            "Validation and internal-test counts can still differ by one clip per class "
            "when the requested fraction of 350 is not an integer."
        )
    else:
        balance_implication = (
            "Labeled class counts are not equal. Any class weighting should be justified from "
            "these counts later. No weighting was applied in this phase."
        )

    sampling_period = audio_summary["sampling_period_seconds"]
    if sampling_period is None:
        period_sentence = "A single sampling period is not defined, because more than one sample rate is present."
    else:
        period_sentence = (
            f"Where the sample rate is uniform, the sampling period is {sampling_period:.8f} seconds "
            f"(1 / {next(iter(audio_summary['sample_rate_hz_counts']))} Hz)."
        )

    text = f"""# EDA Findings

These statements were computed from `data/raw/Train.csv`, `data/raw/Test.csv`, `data/raw/SampleSubmission.csv`, and the WAV members of `data/raw/Swahili_words.zip`. No model was trained. The official `Test.csv` file was not used as an evaluation set and was not given labels.

## Dataset characteristics

`Train.csv` has {train["n_rows"]} rows and {train["n_columns"]} columns: {", ".join(train["columns"])}. Pandas dtypes are {train["dtypes"]}. Missing values are {train["missing_values"]}. Duplicate `Word_id` values: {train["duplicate_word_ids"]}. Duplicate complete rows: {train["duplicate_complete_rows"]}.

`Test.csv` has {test["n_rows"]} rows and columns {test["columns"]}. Missing values are {test["missing_values"]}. Duplicate ids: {test["duplicate_word_ids"]}. Ids shared with `Train.csv`: {test["overlap_with_train_ids"]}. This file has no labels.

`SampleSubmission.csv` has {submission["n_rows"]} rows and {submission["n_columns"]} columns. Ids match `Test.csv`: {submission["ids_match_test_set"]}. Ids are in the same order as `Test.csv`: {submission["ids_in_same_order_as_test"]}. Class columns ({submission["n_class_columns"]}): {", ".join(submission["class_columns"])}. Those columns match the training labels as a set: {submission["class_columns_match_train_labels"]}. All template entries in the class columns are zero: {submission["all_class_entries_are_zero"]}.

`Swahili_words.zip` has {archive["n_members"]} members. The zip file size on disk is {archive["compressed_file_size_bytes"]} bytes. The sum of member compressed sizes is {archive["member_compressed_size_bytes"]} bytes, and the sum of member uncompressed sizes is {archive["member_uncompressed_size_bytes"]} bytes. Extensions: {archive["extensions"]}. Members matching `id_` plus lowercase letters or digits plus `.wav`: {archive["n_members_matching_id_pattern"]} of {archive["n_members"]}. Filename-token length counts, for names with that prefix and suffix: {archive["filename_token_length_counts"]}. Train ids missing from the zip: {archive["train_ids_missing_from_zip"]}. Test ids missing from the zip: {archive["test_ids_missing_from_zip"]}. Zip members that are in neither CSV: {archive["zip_members_not_in_train_or_test"]}. The archive was not extracted to disk.

The shared research split was drawn only from the {train["n_rows"]} labeled rows. Seed: {split_summary["random_seed"]}. Fractions: train {TRAIN_FRACTION:.0%}, validation {VALIDATION_FRACTION:.0%}, internal test {TEST_FRACTION:.0%}. Method: scikit-learn `train_test_split`, first holding out {HOLDOUT_FRACTION:.0%} stratified on `Swahili_word` with `random_state` {split_summary["random_seed"]}, then splitting that holdout in half with the same seed and stratify column. scikit-learn version recorded for the run: {split_summary["packages"]["sklearn"]}. Split sizes: {split_summary["split_counts"]}. Official test ids present in the split: {split_summary["official_test_ids_in_split"]}. Labeled ids missing from the split: {split_summary["missing_labeled_ids"]}. Duplicate split ids: {split_summary["duplicate_word_ids"]}.

## Class distribution

Labeled class counts:

{class_lines}

Equal counts across classes: {train["classes_have_equal_counts"]}.

Swahili label and English translation pairs:

{translation_lines}

Each Swahili label maps to one English translation: {train["translations"]["consistent_within_each_swahili_label"]}. Each English translation maps to one Swahili label: {train["translations"]["consistent_within_each_english_label"]}.

Internal split counts:

{chr(10).join(split_lines)}

## Audio characteristics

Metadata was read for {audio_summary["n_clips"]} labeled training clips by decompressing each WAV from the zip in memory.

Duration in seconds (sample standard deviation uses ddof=1; percentiles use pandas linear interpolation): minimum {_fmt_seconds(duration["min"])}, maximum {_fmt_seconds(duration["max"])}, mean {_fmt_seconds(duration["mean"])}, median {_fmt_seconds(duration["median"])}, standard deviation {_fmt_seconds(duration["std_ddof1"])}. Percentiles: 5th {_fmt_seconds(duration["p05"])}, 25th {_fmt_seconds(duration["p25"])}, 50th {_fmt_seconds(duration["p50"])}, 75th {_fmt_seconds(duration["p75"])}, 95th {_fmt_seconds(duration["p95"])}.

Frame counts use the same summaries: minimum {frames["min"]:.0f}, maximum {frames["max"]:.0f}, mean {frames["mean"]:.4f}, median {frames["median"]:.4f}, standard deviation {frames["std_ddof1"]:.4f}.

Sample-rate counts (Hz): {rate_phrase}. Channel counts: {channel_phrase}. Bit-depth counts: {depth_phrase}. Compression-type counts: {compression_phrase}. Uniform across these fields: {audio_summary["format_is_uniform"]}. {period_sentence}

Peak absolute amplitude, in integer PCM units: minimum {peak["min"]:.4f}, median {peak["median"]:.4f}, maximum {peak["max"]:.4f}. RMS amplitude in the same units: minimum {rms["min"]:.4f}, median {rms["median"]:.4f}, maximum {rms["max"]:.4f}. Clips with zero frames: {audio_summary["n_zero_frame_clips"]}. Clips with peak amplitude 0: {audio_summary["n_zero_peak_clips"]}.

Low-energy proportion: {audio_summary["low_energy_rule"]["definition"]} Minimum {low_energy["min"]:.4f}, median {low_energy["median"]:.4f}, maximum {low_energy["max"]:.4f}, 5th percentile {low_energy["p05"]:.4f}, 95th percentile {low_energy["p95"]:.4f}.

Shortest labeled clip: `{shortest["Word_id"]}` ({shortest["Swahili_word"]} / {shortest["English_translation"]}), {_fmt_seconds(shortest["duration_seconds"])} seconds, {int(shortest["n_frames"])} frames. Longest labeled clip: `{longest["Word_id"]}` ({longest["Swahili_word"]} / {longest["English_translation"]}), {_fmt_seconds(longest["duration_seconds"])} seconds, {int(longest["n_frames"])} frames. Highest low-energy proportion: `{highest_low["Word_id"]}` ({highest_low["Swahili_word"]} / {highest_low["English_translation"]}), {highest_low["low_energy_proportion"]:.4f}.

Duration by class:

{duration_class_lines}

Class median durations range from {_fmt_seconds(min(class_medians))} s to {_fmt_seconds(max(class_medians))} s.

Waveform and spectrogram figures use one labeled clip per class. The clip is the one whose duration is closest to that class median, with `Word_id` order breaking ties. Spectrogram panels use NFFT {SPECTROGRAM_NFFT} and overlap {SPECTROGRAM_NOVERLAP} when the clip is long enough. A display floor of {SPECTROGRAM_DB_FLOOR} is added before the log, and the shared color scale is the 5th to 95th percentile of those displayed values. These settings are for the figure only. They are not model features.

## Sequential characteristics

Each labeled row is one clip with one class label. The audio measurement of that clip is a sequence of PCM frames. Sequence length in frames ranges from {frames["min"]:.0f} to {frames["max"]:.0f}. Duration ranges from {_fmt_seconds(duration["min"])} s to {_fmt_seconds(duration["max"])} s. {period_sentence}

The table structure gives one `Swahili_word` per `Word_id`. The center of the duration distribution is a few seconds (median {_fmt_seconds(duration["median"])} s, 95th percentile {_fmt_seconds(duration["p95"])} s). A long tail remains: {tail["at_least_10_seconds"]} clips last at least 10 s, {tail["at_least_20_seconds"]} at least 20 s, {tail["at_least_30_seconds"]} at least 30 s, and {tail["at_least_60_seconds"]} at least 60 s. The maximum is {_fmt_seconds(duration["max"])} s. These cut points describe that tail. No clip was removed. This phase did not measure whether each recording contains only its label word, nor whether a model that uses time order will score higher than one that does not.

## Data-quality observations

Missing labeled values, duplicate labeled ids, duplicate labeled rows, train/test id overlap, missing wav members, and extra zip members are the counts reported in Dataset characteristics. Speaker-like CSV columns: {speaker["speaker_like_columns"] if speaker["speaker_like_columns"] else "none"}. Zip members stored in subdirectories: {speaker["zip_members_in_subdirectories"]}. Starter-notebook text hits for speaker, participant, or gender: {speaker["starter_notebook_term_hits"] if speaker["starter_notebook_term_hits"] else "none"}. Speaker identifier present under those checks: {speaker["speaker_identifier_present"]}.

{speaker["note"]} A clip-level split therefore cannot be checked for speaker overlap. No speaker split was created.

Format exceptions among the labeled clips read for metadata: zero-frame clips = {audio_summary["n_zero_frame_clips"]}, zero-peak clips = {audio_summary["n_zero_peak_clips"]}, uniform sample rate / channels / bit depth / compression = {audio_summary["format_is_uniform"]}. Clips whose peak absolute amplitude reaches 16-bit full scale (at least 32767): {audio_summary["n_clips_at_int16_full_scale"]}. The value 32768 is the absolute value of the most negative 16-bit sample. Duration tail counts, with no clips removed: at least 10 s = {tail["at_least_10_seconds"]}, at least 20 s = {tail["at_least_20_seconds"]}, at least 30 s = {tail["at_least_30_seconds"]}, at least 60 s = {tail["at_least_60_seconds"]}.

## Implications for modelling

OBSERVATION: Labeled durations range from {_fmt_seconds(duration["min"])} s to {_fmt_seconds(duration["max"])} s, with a median of {_fmt_seconds(duration["median"])} s and a 95th percentile of {_fmt_seconds(duration["p95"])} s. Frame counts range from {frames["min"]:.0f} to {frames["max"]:.0f}.

IMPLICATION: {length_implication} Padding every clip to the maximum length would mean sequences of {frames["max"]:.0f} frames. A cutoff near the median would shorten every longer clip, including the {tail["at_least_10_seconds"]} clips that last at least 10 s. Neither choice was made here.

OBSERVATION: Sample-rate counts are {rate_phrase}. Channel counts are {channel_phrase}. Bit-depth counts are {depth_phrase}. Compression counts are {compression_phrase}.

IMPLICATION: {rate_implication}

OBSERVATION: The labeled pool has {train["n_classes"]} classes and equal counts = {train["classes_have_equal_counts"]}. Internal split sizes are {split_summary["split_counts"]}.

IMPLICATION: {balance_implication} The internal test split stays unused for tuning, model selection, preprocessing fits, threshold selection, and early stopping. Validation is the development split. `Test.csv` stays an unlabeled external file.

OBSERVATION: No speaker identifier was found in the released tables, filenames' documented fields, zip layout, or starter notebook. The shared split is stratified only by `Swahili_word`.

IMPLICATION: The same speaker may appear in more than one split, and this cannot be measured from the released metadata. Comparisons later should state that limitation. A speaker-disjoint split was not invented.

OBSERVATION: Class median durations range from {_fmt_seconds(min(class_medians))} s to {_fmt_seconds(max(class_medians))} s. Class maxima range from {_fmt_seconds(min(class_maxima))} s to {_fmt_seconds(max(class_maxima))} s, so the long recordings are not confined to one class.

IMPLICATION: Typical clip length is similar across classes. The large length differences are inside classes. A later length rule should use the full duration distribution, not class medians alone. No length rule was chosen here.

OBSERVATION: Low-energy proportion, under the fixed per-clip peak rule above, has median {low_energy["median"]:.4f} and maximum {low_energy["max"]:.4f}.

IMPLICATION: Amplitude summaries show that clips are not constant full-scale signals. They do not, by themselves, justify a denoising or silence-trimming pipeline. Any such step would need its own stated rule and would have to be fit without the internal test set.

OBSERVATION: The input files are WAV audio with a clip-level word label. No document text is provided as model input.

IMPLICATION: Text TF-IDF is not a baseline for these files. Classical baselines still need an audio representation. That representation was not chosen or extracted in this phase. The five models to compare were not chosen in this phase.
"""
    path = results_eda_dir(root) / "eda_findings.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_phase2(root: Path | None = None) -> dict:
    """Run loading checks, audio EDA, and the shared split. Do not train models."""
    set_seed(SHARED_SPLIT_SEED)
    train_df = load_train_csv()
    test_df = load_test_csv()
    submission_df = load_sample_submission()
    zip_path = audio_zip_path()
    zip_names = list_zip_members(zip_path)

    dataset_summary = build_dataset_summary(train_df, test_df, submission_df, zip_path, zip_names)
    save_dataset_summary(dataset_summary, root)

    metadata = collect_metadata(train_df["Word_id"].tolist(), zip_path)
    metadata = metadata.merge(train_df, on="Word_id", how="left", validate="one_to_one")
    if metadata["Swahili_word"].isna().any():
        raise RuntimeError("Audio metadata could not be joined back to every labeled row.")
    eda_dir = results_eda_dir(root)
    eda_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(eda_dir / "audio_metadata_train.csv", index=False)
    audio_summary = build_audio_summary(metadata)
    examples = save_figures(metadata, root)
    audio_summary["example_clips"] = examples.to_dict(orient="records")
    audio_summary["figure_files"] = [
        "class_distribution.png",
        "audio_duration_distribution.png",
        "sample_rate_distribution.png",
        "example_waveforms.png",
        "example_spectrograms.png",
    ]
    _write_json(eda_dir / "audio_metadata_summary.json", audio_summary)
    pd.DataFrame(audio_summary["duration_by_class"]).to_csv(
        eda_dir / "duration_by_class.csv",
        index=False,
    )

    environment = environment_record(SHARED_SPLIT_SEED)
    _write_json(eda_dir / "environment.json", environment)

    split_df = create_shared_split(train_df, seed=SHARED_SPLIT_SEED)
    split_checks = validate_shared_split(split_df, train_df, test_df)
    class_counts = split_checks.pop("class_counts")
    problems = []
    if split_checks["duplicate_word_ids"] != 0:
        problems.append("duplicate Word_id values in the split")
    if split_checks["missing_labeled_ids"] != 0 or split_checks["unknown_ids"] != 0:
        problems.append("split ids do not match the labeled ids")
    if split_checks["official_test_ids_in_split"] != 0:
        problems.append("official unlabeled test ids entered the research split")
    if set(split_checks["allowed_split_names"]) != {"train", "validation", "test"}:
        problems.append("unexpected split names")
    if problems:
        raise RuntimeError("Shared split failed validation: " + "; ".join(problems))

    save_shared_split(split_df)
    split_summary = {
        **split_checks,
        "random_seed": SHARED_SPLIT_SEED,
        "fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "method": (
            "sklearn.model_selection.train_test_split on Train.csv only; "
            f"hold out {HOLDOUT_FRACTION} stratified on Swahili_word with random_state "
            f"{SHARED_SPLIT_SEED}; split that holdout in half with the same seed and stratify column."
        ),
        "packages": environment["packages"],
        "python_version": environment["python_version"],
        "class_counts": class_counts.to_dict(orient="records"),
        "official_test_used_for_split": False,
        "internal_test_reserved_for_final_evaluation": True,
    }
    _write_json(eda_dir / "split_summary.json", split_summary)
    class_counts.to_csv(eda_dir / "split_summary.csv", index=False)
    write_eda_findings(dataset_summary, audio_summary, split_summary, root)
    print("Phase 2 artifacts written.", flush=True)
    return {
        "dataset_summary": dataset_summary,
        "audio_summary": audio_summary,
        "split_summary": split_summary,
    }
