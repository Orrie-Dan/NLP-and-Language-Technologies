"""Frame-level MFCC sequences for the neural sequential models.

``extract_features`` aggregates each clip into one vector for the classical
baselines. Sequence models need the time axis, so this module returns the
``(n_frames, n_coefficients)`` matrix instead. It reuses the shared loader,
duration rule, and MFCC/delta helpers, so the acoustic front end is identical
to the classical pipeline.

Clips longer than ``duration_seconds`` are truncated with the shared rule.
Shorter clips are NOT zero-padded on the waveform: padding happens per batch
and is masked by the model, so the network never sees log-floor frames
manufactured from padded silence.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from src.data.loading import audio_zip_path
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.mfcc import DELTA_WIDTH, delta_features, mfcc_matrix
from src.preprocessing.pipeline import (
    _decode_wav,
    apply_duration,
    apply_peak_normalization,
)


def static_mfcc_sequence(
    samples: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Static MFCC frames for one decoded clip, truncated but never padded."""
    signal = apply_peak_normalization(samples, config.normalize_amplitude)
    max_samples = int(round(config.duration_seconds * sample_rate))
    if config.duration_mode == "fixed" and signal.shape[0] > max_samples:
        signal = apply_duration(signal, sample_rate, config)
    matrix = mfcc_matrix(
        signal,
        sample_rate,
        n_mfcc=config.n_mfcc,
        frame_length=config.window_length,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
    )
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("MFCC sequence is empty or non-finite.")
    return matrix.astype(np.float32)


def with_deltas(static: np.ndarray, n_streams: int) -> np.ndarray:
    """Append delta (n_streams>=2) and delta-delta (n_streams==3) blocks."""
    if n_streams not in (1, 2, 3):
        raise ValueError("n_streams must be 1, 2, or 3")
    parts = [static]
    if n_streams >= 2:
        delta = delta_features(static.astype(np.float64), DELTA_WIDTH)
        parts.append(delta.astype(np.float32))
        if n_streams == 3:
            parts.append(delta_features(delta, DELTA_WIDTH).astype(np.float32))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else static


def extract_static_sequences(
    word_ids: list[str],
    config: PreprocessingConfig,
    cache_path: Path | None = None,
    progress_every: int = 250,
) -> list[np.ndarray]:
    """Static MFCC sequences for ``word_ids`` in order, cached as one ``.npz``.

    The cache stores the configuration and id order and is rejected if either
    differs, so a stale cache cannot be loaded silently.
    """
    config_json = json.dumps(config.to_dict(), sort_keys=True)
    if cache_path is not None and Path(cache_path).is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["config"]) == config_json and list(cached["ids"]) == list(word_ids):
            offsets = cached["offsets"]
            data = cached["data"]
            return [data[offsets[i] : offsets[i + 1]] for i in range(len(word_ids))]
        print(f"Ignoring stale sequence cache {cache_path}", flush=True)

    sequences: list[np.ndarray] = []
    with zipfile.ZipFile(audio_zip_path()) as archive:
        for index, word_id in enumerate(word_ids, start=1):
            loaded = _decode_wav(archive.read(word_id), word_id, config)
            sequences.append(static_mfcc_sequence(loaded.samples, loaded.sample_rate, config))
            if progress_every and index % progress_every == 0:
                print(f"MFCC sequences {index}/{len(word_ids)}", flush=True)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        offsets = np.concatenate([[0], np.cumsum([len(s) for s in sequences])]).astype(np.int64)
        np.savez(
            cache_path,
            config=np.array(config_json),
            ids=np.array(word_ids),
            offsets=offsets,
            data=np.concatenate(sequences, axis=0),
        )
    return sequences


def fit_frame_standardizer(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-coefficient mean and std over every frame. Fit on training clips only."""
    frames = np.concatenate(sequences, axis=0).astype(np.float64)
    std = frames.std(axis=0)
    std[std == 0] = 1.0
    return frames.mean(axis=0).astype(np.float32), std.astype(np.float32)


def stack_frames(sequence: np.ndarray, k: int) -> np.ndarray:
    """Concatenate each run of ``k`` frames into one step (length / k, width * k).

    The final partial run is completed by repeating the last frame.
    """
    if k == 1:
        return sequence
    remainder = (-sequence.shape[0]) % k
    if remainder:
        sequence = np.concatenate([sequence, np.repeat(sequence[-1:], remainder, axis=0)])
    return sequence.reshape(-1, sequence.shape[1] * k)


def standardize(
    sequences: list[np.ndarray], mean: np.ndarray, std: np.ndarray
) -> list[np.ndarray]:
    return [((s - mean) / std).astype(np.float32) for s in sequences]
