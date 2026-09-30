"""Investigation-only MFCC helper.

This is not the project feature pipeline. Parameters are recorded so a later
phase can adopt or reject them. No classifier is trained here.
"""

from __future__ import annotations

import numpy as np

# Speech-oriented settings for 16 kHz audio. They are candidates, not a decision.
FRAME_LENGTH_MS = 25.0
HOP_LENGTH_MS = 10.0
N_FFT = 512
N_MELS = 40
FMIN_HZ = 0.0
LOG_FLOOR = 1e-10
PRE_EMPHASIS = 0.0
N_MFCC_CANDIDATES = (13, 20, 40)
DELTA_WIDTH = 2
MEL_SCALE = "HTK"


def samples_per_ms(sample_rate_hz: int, milliseconds: float) -> int:
    return int(round(sample_rate_hz * milliseconds / 1000.0))


def n_analysis_frames(n_samples: int, frame_length: int, hop_length: int) -> int:
    """Frame count with no centering and no padding.

    A frame starts every ``hop_length`` samples and must lie entirely inside
    the clip. Clips shorter than ``frame_length`` yield zero frames.
    """
    if n_samples < frame_length or hop_length <= 0:
        return 0
    return 1 + (n_samples - frame_length) // hop_length


def hz_to_mel_htk(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz_htk(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate_hz: int,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin_hz: float = FMIN_HZ,
    fmax_hz: float | None = None,
) -> np.ndarray:
    """Triangular HTK-mel filterbank with shape (n_mels, n_fft // 2 + 1)."""
    if fmax_hz is None:
        fmax_hz = sample_rate_hz / 2.0
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel_htk(fmin_hz), hz_to_mel_htk(fmax_hz), n_mels + 2)
    hz_points = np.asarray(mel_to_hz_htk(mel_points), dtype=np.float64)
    bins = np.floor(n_fft * hz_points / sample_rate_hz).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)
    filters = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for mel_index in range(n_mels):
        left = int(bins[mel_index])
        center = int(bins[mel_index + 1])
        right = int(bins[mel_index + 2])
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)
        up = np.arange(left, center)
        down = np.arange(center, right)
        if center > left:
            filters[mel_index, up] = (up - left) / (center - left)
        if right > center:
            filters[mel_index, down] = (right - down) / (right - center)
    return filters


def _dct_ii_ortho(values: np.ndarray, n_mfcc: int) -> np.ndarray:
    """Orthonormal DCT-II along the last axis. Keeps the first ``n_mfcc`` rows."""
    n_mels = values.shape[-1]
    n_index = np.arange(n_mels, dtype=np.float64)
    k_index = np.arange(n_mfcc, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (n_index + 0.5) * k_index / n_mels)
    basis[0] *= np.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= np.sqrt(2.0 / n_mels)
    return values @ basis.T


def mfcc_matrix(
    samples: np.ndarray,
    sample_rate_hz: int,
    n_mfcc: int,
    frame_length: int | None = None,
    hop_length: int | None = None,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Return MFCC features with shape (n_frames, n_mfcc).

    No pre-emphasis and no liftering are applied. The Hann window covers
    ``frame_length`` samples and is zero-padded to ``n_fft`` before the FFT.
    """
    if frame_length is None:
        frame_length = samples_per_ms(sample_rate_hz, FRAME_LENGTH_MS)
    if hop_length is None:
        hop_length = samples_per_ms(sample_rate_hz, HOP_LENGTH_MS)
    if n_mfcc > n_mels:
        raise ValueError(f"n_mfcc={n_mfcc} exceeds n_mels={n_mels}")
    signal = np.asarray(samples, dtype=np.float64)
    if PRE_EMPHASIS != 0.0:
        raise ValueError("This investigation helper does not apply pre-emphasis.")
    n_frames = n_analysis_frames(len(signal), frame_length, hop_length)
    if n_frames == 0:
        return np.zeros((0, n_mfcc), dtype=np.float64)
    window = np.hanning(frame_length)
    filters = mel_filterbank(sample_rate_hz, n_fft=n_fft, n_mels=n_mels)
    mel_log = np.empty((n_frames, n_mels), dtype=np.float64)
    for frame_index in range(n_frames):
        start = frame_index * hop_length
        frame = signal[start : start + frame_length] * window
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[:frame_length] = frame
        power = (np.abs(np.fft.rfft(padded, n=n_fft)) ** 2) / n_fft
        mel = filters @ power
        mel_log[frame_index] = np.log(np.maximum(mel, LOG_FLOOR))
    return _dct_ii_ortho(mel_log, n_mfcc)


def delta_features(features: np.ndarray, width: int = DELTA_WIDTH) -> np.ndarray:
    """HTK-style first differences. ``features`` has shape (n_frames, n_coeff)."""
    if features.ndim != 2:
        raise ValueError("delta_features expects a 2D array")
    if len(features) == 0:
        return features.copy()
    denominator = 2.0 * sum(offset * offset for offset in range(1, width + 1))
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    combined = np.zeros_like(features)
    n_frames = len(features)
    for offset in range(1, width + 1):
        combined += offset * (
            padded[width + offset : width + offset + n_frames]
            - padded[width - offset : width - offset + n_frames]
        )
    return combined / denominator
