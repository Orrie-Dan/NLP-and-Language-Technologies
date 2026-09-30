"""Run the Phase 4 pipeline checks on a small shared-training subset.

The checks do not train a classifier, do not read ``Test.csv``, and do not
rewrite the shared split or the Phase 3 result files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import results_eda_dir
from src.data.loading import (
    audio_zip_path,
    load_train_csv,
    read_shared_split,
    read_wav_bytes,
    shared_split_path,
)
from src.preprocessing.config import CONFIGURATION_STATUS, PreprocessingConfig
from src.preprocessing.pipeline import (
    apply_duration,
    extract_features,
    extract_features_batch,
    feature_dimension,
    load_audio,
    uses_dataset_statistics,
)

# Documented widths for 13 coefficients. Delta is one extra block (26), not
# three. Delta plus delta-delta is the 39-block stack. mean_std doubles them.
_STATIC_MEAN = 13
_STATIC_MEAN_STD = 26
_DELTA_MEAN = 26
_DELTA_MEAN_STD = 52
_DELTA_DELTA_MEAN = 39
_DELTA_DELTA_MEAN_STD = 78

_PHASE3_FILES = (
    "preprocessing_investigation.json",
    "preprocessing_investigation.csv",
    "fixed_window_impact.csv",
    "mfcc_candidates.csv",
    "eda_findings.md",
)


def select_training_subset(root: Path | None = None) -> pd.DataFrame:
    """Pick a deterministic training-split subset.

    One clip per class, plus a clip shorter than 6 s and a clip longer than
    8 s when those are not already included. Ordering is by ``Word_id``.
    Validation and internal-test rows are not included.
    """
    split_df = read_shared_split()
    labels = load_train_csv()
    metadata_path = results_eda_dir(root) / "audio_metadata_train.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Phase 2 metadata is missing: {metadata_path}. Run scripts/run_phase2_eda.py first."
        )
    metadata = pd.read_csv(metadata_path)
    merged = split_df.merge(labels, on="Word_id", how="left", validate="one_to_one")
    merged = merged.merge(
        metadata.loc[:, ["Word_id", "duration_seconds"]],
        on="Word_id",
        how="left",
        validate="one_to_one",
    )
    train = merged.loc[merged["split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("The shared split has no training rows.")
    if train["Swahili_word"].isna().any() or train["duration_seconds"].isna().any():
        raise RuntimeError("Training rows are missing labels or durations.")
    ordered = train.sort_values(["Swahili_word", "Word_id"], kind="mergesort")
    per_class = ordered.groupby("Swahili_word", sort=True, as_index=False).head(1)
    short = (
        train.loc[train["duration_seconds"] < 6.0]
        .sort_values(["duration_seconds", "Word_id"], kind="mergesort")
        .head(1)
    )
    long = (
        train.loc[train["duration_seconds"] > 8.0]
        .sort_values(["duration_seconds", "Word_id"], kind="mergesort")
        .head(1)
    )
    if short.empty or long.empty:
        raise RuntimeError(
            "The training split needs at least one clip shorter than 6 s and one longer than 8 s."
        )
    subset = pd.concat([per_class, short, long], ignore_index=True)
    subset = subset.drop_duplicates("Word_id").sort_values("Word_id", kind="mergesort")
    subset = subset.reset_index(drop=True)
    if set(subset["split"].astype(str)) != {"train"}:
        raise RuntimeError("The Phase 4 subset is not limited to the training split.")
    if int(subset["Swahili_word"].nunique()) != int(train["Swahili_word"].nunique()):
        raise RuntimeError("The Phase 4 subset is missing a training class.")
    return subset


def run_phase4_validation(root: Path | None = None) -> dict:
    """Extract features for the training subset and record the checks.

    Writes ``results/eda/phase4_validation.csv`` and
    ``results/eda/phase4_validation.json``. Does not modify Phase 3 files.
    """
    if uses_dataset_statistics(PreprocessingConfig()):
        raise RuntimeError("The Phase 4 pipeline unexpectedly fits dataset statistics.")
    subset = select_training_subset(root)
    destination = results_eda_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    phase3_before = _snapshot_files(destination, _PHASE3_FILES)
    split_path = shared_split_path()
    split_before = _sha256_file(split_path)
    zip_before = _path_token(audio_zip_path())
    wav_before = {word_id: _sha256_bytes(read_wav_bytes(word_id)) for word_id in subset["Word_id"]}

    short_id = str(
        subset.sort_values(["duration_seconds", "Word_id"], kind="mergesort").iloc[0]["Word_id"]
    )
    long_id = str(
        subset.sort_values(["duration_seconds", "Word_id"], kind="mergesort").iloc[-1]["Word_id"]
    )
    rows: list[dict] = []
    stored: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name, config, expected in _feature_cases():
        print(f"Phase 4 check {name}", flush=True)
        predicted = feature_dimension(config)
        passed = predicted == expected
        detail = f"feature_dimension={predicted}; documented={expected}"
        features, metadata = extract_features_batch(
            subset.loc[:, ["Word_id", "Swahili_word"]],
            config,
        )
        finite = bool(features.size and np.isfinite(features).all())
        shape_ok = features.shape == (len(subset), expected)
        labels_ok = list(metadata.columns) == ["Word_id", "Swahili_word"] and metadata[
            "Swahili_word"
        ].tolist() == subset["Swahili_word"].tolist()
        repeat_id = str(subset.iloc[0]["Word_id"])
        first = extract_features(repeat_id, config)
        second = extract_features(repeat_id, config)
        reproducible = bool(np.array_equal(first, second) and np.array_equal(first, features[0]))
        nonzero = bool(np.any(features != 0))
        passed = passed and finite and shape_ok and labels_ok and reproducible and nonzero
        if not shape_ok:
            detail += f"; matrix shape {features.shape}"
        if not finite:
            detail += "; non-finite values"
        if not reproducible:
            detail += "; repeated extraction differed"
        if not labels_ok:
            detail += "; labels were not kept beside the matrix"
        if not nonzero:
            detail += "; features were all zero"
        stored[name] = (features, metadata)
        rows.append(
            {
                "check": name,
                "passed": passed,
                "n_examples": int(len(subset)),
                "n_classes": int(subset["Swahili_word"].nunique()),
                "duration_seconds": config.duration_seconds,
                "duration_mode": config.duration_mode,
                "aggregation": config.aggregation,
                "include_delta": config.include_delta,
                "include_delta_delta": config.include_delta_delta,
                "normalize_amplitude": config.normalize_amplitude,
                "feature_dimension": int(features.shape[1]) if features.ndim == 2 else None,
                "all_finite": finite,
                "reproducible": reproducible,
                "detail": detail,
            }
        )

    rows.append(_duration_geometry_row(short_id, long_id))
    rows.append(_unchanged_duration_row(short_id))
    rows.append(_normalization_row(short_id, stored["A_6s_static_mean"][0], subset))
    rows.append(_duration_changes_values_row(stored, subset, short_id))
    rows.extend(_integrity_rows(destination, phase3_before, split_path, split_before, zip_before, wav_before))

    table = pd.DataFrame(rows)
    all_passed = bool(table["passed"].all())
    all_finite = bool(table.loc[table["all_finite"].notna(), "all_finite"].all())
    reproducible = bool(table.loc[table["reproducible"].notna(), "reproducible"].all())
    passed_by_name = {row["check"]: bool(row["passed"]) for row in rows}
    payload = {
        "phase": 4,
        "configuration_status": CONFIGURATION_STATUS,
        "experimental_configuration_selected": False,
        "uses_dataset_statistics": False,
        "subset_split": "train",
        "n_examples": int(len(subset)),
        "n_classes": int(subset["Swahili_word"].nunique()),
        "word_ids": subset["Word_id"].tolist(),
        "classes": sorted(subset["Swahili_word"].astype(str).unique()),
        "dimension_convention": {
            "static_mean": _STATIC_MEAN,
            "static_mean_std": _STATIC_MEAN_STD,
            "delta_mean": _DELTA_MEAN,
            "delta_mean_std": _DELTA_MEAN_STD,
            "delta_and_delta_delta_mean": _DELTA_DELTA_MEAN,
            "delta_and_delta_delta_mean_std": _DELTA_DELTA_MEAN_STD,
            "note": (
                "13 MFCCs plus delta are 26 coefficient streams. "
                "13 MFCCs plus delta plus delta-delta are 39. "
                "mean_std doubles those widths to 52 and 78."
            ),
        },
        "source_audio_modified": not passed_by_name["J_source_wav_unchanged"],
        "shared_split_modified": not passed_by_name["shared_split_unchanged"],
        "phase3_results_modified": not passed_by_name["phase3_results_unchanged"],
        "all_finite": all_finite,
        "reproducible": reproducible,
        "all_passed": all_passed,
        "checks": rows,
    }
    _write_json(destination / "phase4_validation.json", payload)
    table.to_csv(destination / "phase4_validation.csv", index=False)
    if not all_passed:
        failed = table.loc[~table["passed"], "check"].tolist()
        raise RuntimeError("Phase 4 validation failed: " + ", ".join(failed))
    print(
        f"Phase 4 validation passed on {len(subset)} training clips "
        f"and {subset['Swahili_word'].nunique()} classes.",
        flush=True,
    )
    payload["table"] = table
    return payload


def _feature_cases() -> list[tuple[str, PreprocessingConfig, int]]:
    static = PreprocessingConfig(
        duration_mode="fixed",
        normalize_amplitude=False,
        include_delta=False,
        include_delta_delta=False,
        n_mfcc=13,
    )
    return [
        ("A_6s_static_mean", replace(static, duration_seconds=6.0, aggregation="mean"), _STATIC_MEAN),
        ("B_8s_static_mean", replace(static, duration_seconds=8.0, aggregation="mean"), _STATIC_MEAN),
        (
            "C_F_6s_static_mean_std",
            replace(static, duration_seconds=6.0, aggregation="mean_std"),
            _STATIC_MEAN_STD,
        ),
        (
            "delta_6s_mean",
            replace(static, duration_seconds=6.0, aggregation="mean", include_delta=True),
            _DELTA_MEAN,
        ),
        (
            "delta_6s_mean_std",
            replace(static, duration_seconds=6.0, aggregation="mean_std", include_delta=True),
            _DELTA_MEAN_STD,
        ),
        (
            "D_6s_delta_delta_mean",
            replace(
                static,
                duration_seconds=6.0,
                aggregation="mean",
                include_delta=True,
                include_delta_delta=True,
            ),
            _DELTA_DELTA_MEAN,
        ),
        (
            "D_F_6s_delta_delta_mean_std",
            replace(
                static,
                duration_seconds=6.0,
                aggregation="mean_std",
                include_delta=True,
                include_delta_delta=True,
            ),
            _DELTA_DELTA_MEAN_STD,
        ),
        (
            "I_8s_delta_delta_mean_std",
            replace(
                static,
                duration_seconds=8.0,
                aggregation="mean_std",
                include_delta=True,
                include_delta_delta=True,
            ),
            _DELTA_DELTA_MEAN_STD,
        ),
    ]


def _duration_geometry_row(short_id: str, long_id: str) -> dict:
    config_6 = PreprocessingConfig(duration_seconds=6.0, padding_mode="end", truncation_mode="start")
    config_8 = replace(config_6, duration_seconds=8.0)
    short = load_audio(short_id, config_6)
    long = load_audio(long_id, config_8)
    padded = apply_duration(short.samples, short.sample_rate, config_6)
    truncated = apply_duration(long.samples, long.sample_rate, config_8)
    target_6 = int(round(6.0 * short.sample_rate))
    target_8 = int(round(8.0 * long.sample_rate))
    pad_ok = (
        padded.shape == (target_6,)
        and np.array_equal(padded[: short.n_samples], short.samples)
        and np.all(padded[short.n_samples :] == 0.0)
    )
    truncate_ok = truncated.shape == (target_8,) and np.array_equal(
        truncated, long.samples[:target_8]
    )
    return _row(
        "padding_and_truncation",
        pad_ok and truncate_ok,
        (
            f"{short_id} padded to {padded.shape[0]} samples; "
            f"{long_id} truncated to {truncated.shape[0]} samples. "
            "Padding is zeros at the end. Truncation keeps the start. No WAV was written."
        ),
    )


def _unchanged_duration_row(word_id: str) -> dict:
    config = PreprocessingConfig(duration_mode="none", aggregation="mean", duration_seconds=6.0)
    loaded = load_audio(word_id, config)
    kept = apply_duration(loaded.samples, loaded.sample_rate, config)
    vector = extract_features(word_id, config)
    passed = (
        kept.shape == loaded.samples.shape
        and np.array_equal(kept, loaded.samples)
        and vector.shape == (_STATIC_MEAN,)
        and np.isfinite(vector).all()
    )
    return _row(
        "duration_mode_none",
        passed,
        f"{word_id} kept {loaded.n_samples} samples and produced a finite 13-vector.",
        all_finite=bool(np.isfinite(vector).all()),
        feature_dimension=_STATIC_MEAN,
        duration_mode="none",
        aggregation="mean",
    )


def _normalization_row(word_id: str, static_mean: np.ndarray, subset: pd.DataFrame) -> dict:
    config = PreprocessingConfig(duration_seconds=6.0, aggregation="mean", normalize_amplitude=True)
    vector = extract_features(word_id, config)
    position = subset["Word_id"].tolist().index(word_id)
    unnormalized = static_mean[position]
    finite = bool(np.isfinite(vector).all())
    changed = not np.array_equal(vector, unnormalized)
    return _row(
        "normalization_option",
        finite and changed and vector.shape == (_STATIC_MEAN,),
        (
            "Per-clip peak normalization is finite and changes the 6 s static mean vector. "
            "It is available and not selected."
        ),
        all_finite=finite,
        feature_dimension=_STATIC_MEAN,
        aggregation="mean",
        normalize_amplitude=True,
    )


def _duration_changes_values_row(
    stored: dict[str, tuple[np.ndarray, pd.DataFrame]],
    subset: pd.DataFrame,
    short_id: str,
) -> dict:
    position = subset["Word_id"].tolist().index(short_id)
    six = stored["A_6s_static_mean"][0][position]
    eight = stored["B_8s_static_mean"][0][position]
    same_width = six.shape == eight.shape == (_STATIC_MEAN,)
    different_values = not np.array_equal(six, eight)
    return _row(
        "I_duration_changes_values_not_width",
        same_width and different_values,
        f"{short_id}: 6 s and 8 s static means both have width 13 and are not the same vector.",
        feature_dimension=_STATIC_MEAN,
        all_finite=bool(np.isfinite(six).all() and np.isfinite(eight).all()),
    )


def _integrity_rows(
    destination: Path,
    phase3_before: dict[str, str],
    split_path: Path,
    split_before: str,
    zip_before: tuple[int, int],
    wav_before: dict[str, str],
) -> list[dict]:
    phase3_after = _snapshot_files(destination, _PHASE3_FILES)
    split_after = _sha256_file(split_path)
    zip_after = _path_token(audio_zip_path())
    wav_after = {word_id: _sha256_bytes(read_wav_bytes(word_id)) for word_id in wav_before}
    phase3_ok = phase3_after == phase3_before
    split_ok = split_after == split_before
    wav_ok = wav_after == wav_before
    zip_ok = zip_after == zip_before
    return [
        _row(
            "J_source_wav_unchanged",
            wav_ok and zip_ok,
            "SHA-256 of each subset WAV member and the zip size/mtime match the pre-check snapshot.",
        ),
        _row(
            "shared_split_unchanged",
            split_ok,
            "SHA-256 of data/splits/shared_split.csv matches the pre-check snapshot.",
        ),
        _row(
            "phase3_results_unchanged",
            phase3_ok,
            "SHA-256 of the Phase 3 investigation files matches the pre-check snapshot.",
        ),
    ]


def _row(
    check: str,
    passed: bool,
    detail: str,
    *,
    all_finite: bool | None = None,
    reproducible: bool | None = None,
    feature_dimension: int | None = None,
    duration_mode: str | None = None,
    aggregation: str | None = None,
    normalize_amplitude: bool | None = None,
) -> dict:
    return {
        "check": check,
        "passed": bool(passed),
        "n_examples": None,
        "n_classes": None,
        "duration_seconds": None,
        "duration_mode": duration_mode,
        "aggregation": aggregation,
        "include_delta": None,
        "include_delta_delta": None,
        "normalize_amplitude": normalize_amplitude,
        "feature_dimension": feature_dimension,
        "all_finite": all_finite,
        "reproducible": reproducible,
        "detail": detail,
    }


def _snapshot_files(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: _sha256_file(directory / name) for name in names}


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_token(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
