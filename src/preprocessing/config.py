"""Phase 4 preprocessing configuration.

The field values on ``PreprocessingConfig`` are implementation defaults.
They record a reproducible starting point taken from the Phase 3 candidate
settings. They are not an experimentally selected configuration: duration,
deltas, amplitude normalization, and aggregation still have to be compared
when the classical baselines are trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.preprocessing.mfcc import (
    FRAME_LENGTH_MS,
    HOP_LENGTH_MS,
    N_FFT,
    N_MELS,
    samples_per_ms,
)

# Phase 3 read every labeled clip at this rate. The pipeline does not resample.
TARGET_SAMPLE_RATE_HZ = 16000

# Candidate analysis window from Phase 3, converted at the labeled sample rate.
# 25 ms and 10 ms are 400 and 160 samples only at 16 kHz.
DEFAULT_WINDOW_LENGTH = samples_per_ms(TARGET_SAMPLE_RATE_HZ, FRAME_LENGTH_MS)
DEFAULT_HOP_LENGTH = samples_per_ms(TARGET_SAMPLE_RATE_HZ, HOP_LENGTH_MS)

# Durations the Phase 4 checks must be able to run. Other positive durations
# are accepted. Neither value has been selected.
PHASE4_CHECK_DURATIONS_SECONDS = (6.0, 8.0)

CONFIGURATION_STATUS = (
    "Phase 4 implementation defaults. "
    "Not an experimentally selected preprocessing configuration."
)

_DURATION_MODES = ("fixed", "none")
_PADDING_MODES = ("end", "start", "center")
_TRUNCATION_MODES = ("start", "end", "center")
_CHANNEL_MODES = ("require_mono", "mean_downmix")
_AGGREGATIONS = ("mean", "mean_std")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings for one feature-extraction run.

    ``duration_seconds`` is used only when ``duration_mode`` is ``"fixed"``.
    ``"none"`` leaves the clip length unchanged. Padding and truncation are
    deterministic. No step in the pipeline reads ``random_seed``; it is stored
    so a later random step has an explicit place to record one, and so a
    cache key changes if the seed changes.

    ``channel_mode="require_mono"`` rejects any file that is not already mono.
    ``"mean_downmix"`` averages channels only when the file actually has more
    than one. A mono file is never downmixed.

    Amplitude normalization defaults to off. Phase 3 did not show that it
    helps classification.
    """

    target_sample_rate: int = TARGET_SAMPLE_RATE_HZ
    channel_mode: str = "require_mono"
    duration_seconds: float = 6.0
    duration_mode: str = "fixed"
    padding_mode: str = "end"
    truncation_mode: str = "start"
    n_mfcc: int = 13
    n_mels: int = N_MELS
    n_fft: int = N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    window_length: int = DEFAULT_WINDOW_LENGTH
    include_delta: bool = False
    include_delta_delta: bool = False
    aggregation: str = "mean_std"
    normalize_amplitude: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.channel_mode not in _CHANNEL_MODES:
            raise ValueError(
                f"channel_mode must be one of {_CHANNEL_MODES}, got {self.channel_mode!r}"
            )
        if self.duration_mode not in _DURATION_MODES:
            raise ValueError(
                f"duration_mode must be one of {_DURATION_MODES}, got {self.duration_mode!r}"
            )
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {_PADDING_MODES}, got {self.padding_mode!r}"
            )
        if self.truncation_mode not in _TRUNCATION_MODES:
            raise ValueError(
                f"truncation_mode must be one of {_TRUNCATION_MODES}, got {self.truncation_mode!r}"
            )
        if self.n_mfcc < 1:
            raise ValueError("n_mfcc must be at least 1")
        if self.n_mels < 1:
            raise ValueError("n_mels must be at least 1")
        if self.n_mfcc > self.n_mels:
            raise ValueError(f"n_mfcc={self.n_mfcc} exceeds n_mels={self.n_mels}")
        if self.n_fft < 2:
            raise ValueError("n_fft must be at least 2")
        if self.window_length < 1 or self.hop_length < 1:
            raise ValueError("window_length and hop_length must be positive sample counts")
        if self.window_length > self.n_fft:
            raise ValueError(
                "window_length must be <= n_fft. The reused MFCC helper zero-pads "
                "each window up to n_fft and does not accept a longer window."
            )
        if self.include_delta_delta and not self.include_delta:
            raise ValueError("include_delta_delta requires include_delta")
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_AGGREGATIONS}, got {self.aggregation!r}"
            )
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an int")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0")

    def to_dict(self) -> dict:
        """Return a JSON-ready copy of every field. Order is not significant."""
        return asdict(self)
