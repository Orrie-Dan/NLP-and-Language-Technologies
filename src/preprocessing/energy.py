"""Frame-energy summaries for preprocessing investigation.

A frame is low-energy when its RMS is strictly below 1% of that clip's
maximum frame RMS. This is not a silence label.
"""

from __future__ import annotations

import numpy as np

LOW_ENERGY_FRAME_FRACTION = 0.01


def frame_rms(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """RMS of non-overlapping-in-definition frames that step by ``hop_length``."""
    signal = np.ascontiguousarray(samples, dtype=np.float64)
    n_frames = 0 if len(signal) < frame_length else 1 + (len(signal) - frame_length) // hop_length
    if n_frames == 0:
        if len(signal) == 0:
            return np.zeros(0, dtype=np.float64)
        return np.array([float(np.sqrt(np.mean(np.square(signal))))], dtype=np.float64)
    shape = (n_frames, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    framed = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.sqrt(np.mean(np.square(framed), axis=1))


def leading_trailing_low_energy(
    samples: np.ndarray,
    sample_rate_hz: int,
    frame_length: int,
    hop_length: int,
) -> dict:
    """Measure edges under the frame-RMS rule.

    Leading time is ``hop_length / sample_rate`` times the number of opening
    frames below the threshold, stopping at the first frame that is not
    low-energy. Trailing time is defined from the end in the same way.
    If every frame is low-energy, ``all_frames_low_energy`` is true, leading
    time is the frame-grid duration, and trailing time is 0 so the two are
    not added together.
    """
    rms = frame_rms(samples, frame_length, hop_length)
    n_frames = int(len(rms))
    hop_seconds = hop_length / sample_rate_hz if sample_rate_hz else float("nan")
    if n_frames == 0 or not np.isfinite(rms).any():
        return {
            "n_analysis_frames": n_frames,
            "max_frame_rms": 0.0,
            "median_frame_rms": 0.0,
            "low_energy_frame_proportion": float("nan"),
            "leading_low_energy_seconds": float("nan"),
            "trailing_low_energy_seconds": float("nan"),
            "all_frames_low_energy": False,
        }
    max_rms = float(np.max(rms))
    if max_rms == 0.0:
        low = np.ones(n_frames, dtype=bool)
    else:
        low = rms < (LOW_ENERGY_FRAME_FRACTION * max_rms)
    high_indexes = np.flatnonzero(~low)
    all_low = len(high_indexes) == 0
    if all_low:
        leading_frames = n_frames
        trailing_frames = 0
    else:
        leading_frames = int(high_indexes[0])
        trailing_frames = int(n_frames - 1 - high_indexes[-1])
    return {
        "n_analysis_frames": n_frames,
        "max_frame_rms": max_rms,
        "median_frame_rms": float(np.median(rms)),
        "low_energy_frame_proportion": float(np.mean(low)),
        "leading_low_energy_seconds": float(leading_frames * hop_seconds),
        "trailing_low_energy_seconds": float(trailing_frames * hop_seconds),
        "all_frames_low_energy": bool(all_low),
    }
