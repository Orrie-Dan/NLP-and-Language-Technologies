"""WAV metadata for the Swahili audio clips.

Reads one zip member at a time. It does not extract the archive to disk and
does not compute model features (no MFCCs, mel spectrogram features, or
augmentation).
"""

from __future__ import annotations

import io
import wave
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed descriptive rule for EDA only. It was not selected by model scores
# and it is not a preprocessing threshold. For each clip, a sample is counted
# as low-energy when its absolute amplitude is strictly below this fraction
# of that clip's own peak absolute amplitude.
LOW_ENERGY_PEAK_FRACTION = 0.01


def _pcm_samples(frames: bytes, sample_width: int, n_values: int) -> np.ndarray:
    """Return integer PCM as float64. ``n_values`` includes every channel."""
    if sample_width == 1:
        unsigned = np.frombuffer(frames, dtype=np.uint8, count=n_values)
        return unsigned.astype(np.float64) - 128.0
    if sample_width == 2:
        signed = np.frombuffer(frames, dtype="<i2", count=n_values)
        return signed.astype(np.float64)
    if sample_width == 4:
        signed = np.frombuffer(frames, dtype="<i4", count=n_values)
        return signed.astype(np.float64)
    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def metadata_from_wav_bytes(word_id: str, wav_bytes: bytes) -> dict:
    """Read header fields and amplitude summaries from one WAV byte string."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        compression = wav_file.getcomptype()
        frames = wav_file.readframes(n_frames)

    n_values = n_frames * n_channels
    samples = _pcm_samples(frames, sample_width, n_values)
    peak = float(np.max(np.abs(samples))) if n_values else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if n_values else 0.0
    if n_values == 0:
        low_energy = float("nan")
    elif peak == 0.0:
        low_energy = 1.0
    else:
        threshold = LOW_ENERGY_PEAK_FRACTION * peak
        low_energy = float(np.mean(np.abs(samples) < threshold))

    duration = float(n_frames / sample_rate) if sample_rate else float("nan")
    return {
        "Word_id": word_id,
        "n_channels": int(n_channels),
        "sample_width_bytes": int(sample_width),
        "bit_depth": int(sample_width * 8),
        "sample_rate_hz": int(sample_rate),
        "n_frames": int(n_frames),
        "n_samples": int(n_values),
        "duration_seconds": duration,
        "compression_type": compression,
        "peak_abs_amplitude": peak,
        "rms_amplitude": rms,
        "low_energy_proportion": low_energy,
    }


def collect_metadata(
    word_ids: list[str],
    zip_path: Path,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Collect per-clip metadata for ``word_ids`` from ``zip_path``.

    The zip is opened once. Members are decompressed in memory and are not
    written to disk.
    """
    rows: list[dict] = []
    total = len(word_ids)
    with zipfile.ZipFile(zip_path) as archive:
        for index, word_id in enumerate(word_ids, start=1):
            try:
                wav_bytes = archive.read(word_id)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"{word_id} is not a member of {zip_path.name}"
                ) from exc
            rows.append(metadata_from_wav_bytes(word_id, wav_bytes))
            if progress_every and index % progress_every == 0:
                print(f"Read audio metadata {index}/{total}", flush=True)
    if total:
        print(f"Read audio metadata {total}/{total}", flush=True)
    return pd.DataFrame(rows)
