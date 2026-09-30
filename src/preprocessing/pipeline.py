"""Reusable audio feature pipeline for later classical baselines.

Stages, in order:

1. Load one WAV without writing it back.
2. Validate sample rate, bit depth, and channel count.
3. Optionally divide the clip by its own peak absolute amplitude.
4. Optionally pad or truncate to a fixed duration.
5. Extract MFCCs with the Phase 3 helper.
6. Optionally append delta and delta-delta.
7. Aggregate the frames to one fixed-size vector.

No stage estimates a statistic from more than the current clip. Labels are
not inputs. The official unlabeled ``Test.csv`` file is not read. Source
WAV bytes are not modified.

``random_seed`` on the configuration is not used. Padding, truncation, and
aggregation are deterministic. There is no random crop.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audio import _pcm_samples
from src.data.loading import raw_data_dir, read_wav_bytes, splits_dir
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.mfcc import DELTA_WIDTH, delta_features, mfcc_matrix

_CACHE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class AudioFormatError(ValueError):
    """The WAV header does not match the requested format.

    The pipeline does not resample, does not convert bit depth, and does not
    downmix unless ``channel_mode`` is explicitly ``"mean_downmix"``.
    """


class FeatureExtractionError(ValueError):
    """Feature extraction could not produce a finite fixed-size vector.

    Non-finite values are not dropped and are not replaced.
    """


def feature_streams(config: PreprocessingConfig) -> int:
    """Return how many coefficient blocks the configuration stacks.

    Static MFCCs are always included. Delta and delta-delta each add one
    block of ``n_mfcc`` coefficients. For 13 static coefficients the counts
    are 13, 26, and 39.
    """
    streams = 1
    if config.include_delta:
        streams += 1
    if config.include_delta_delta:
        streams += 1
    return streams


def feature_dimension(config: PreprocessingConfig) -> int:
    """Width of the aggregated vector for ``config``.

    ``mean`` returns one value per coefficient stream.
    ``mean_std`` appends the population standard deviation of each stream,
    so the width doubles. Delta alone is two blocks, not three: 13 MFCCs
    with delta and ``mean`` are 26 features, and with delta plus delta-delta
    they are 39. ``mean_std`` makes those 52 and 78.
    """
    width = config.n_mfcc * feature_streams(config)
    if config.aggregation == "mean_std":
        return width * 2
    return width


def uses_dataset_statistics(config: PreprocessingConfig | None = None) -> bool:
    """Return whether this configuration fits any parameter on a data split.

    The argument is accepted so callers can pass the config they intend to
    run. Every option implemented in this phase is a per-clip transform, so
    the result is always False. No fit on the training split is required,
    and validation or internal-test clips must not be used to compute one.
    """
    del config
    return False


def load_audio(source: str | Path, config: PreprocessingConfig | None = None) -> LoadedAudio:
    """Load one clip into a mono float64 array and validate its header.

    ``source`` may be a filesystem path or a ``Word_id`` stored in
    ``Swahili_words.zip``. Zip members are read into memory. Nothing is
    extracted and nothing is written.

    A file that already matches the expected format is not resampled and,
    when it is already mono, is not downmixed.
    """
    config = _config(config)
    wav_bytes, label = _read_source_bytes(source)
    return _decode_wav(wav_bytes, label, config)


def apply_peak_normalization(samples: np.ndarray, enabled: bool) -> np.ndarray:
    """Scale one clip by its own peak, or return the samples unchanged.

    When ``enabled`` is True and the peak absolute amplitude is positive,
    every sample is divided by that peak. The resulting peak absolute
    amplitude is 1. The divisor comes from this clip only. A clip whose
    peak is 0 is returned unchanged, because there is no positive scale.
    When ``enabled`` is False the samples are returned unchanged.

    This is not a dataset-level normalization. It was not selected as a
    useful preprocessing step; the flag only makes the comparison possible.
    """
    signal = np.array(samples, dtype=np.float64, copy=True)
    if not enabled:
        return signal
    if signal.size == 0:
        return signal
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return signal
    if not np.isfinite(peak):
        raise FeatureExtractionError("Peak amplitude is not finite. Normalization was not applied.")
    return signal / peak


def apply_duration(
    samples: np.ndarray,
    sample_rate_hz: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Pad, truncate, or keep one clip.

    ``duration_mode="none"`` returns a copy of ``samples``.

    ``duration_mode="fixed"`` uses ``round(duration_seconds * sample_rate)``
    samples. A shorter clip is zero-padded. A longer clip is cut. An exact
    match is copied. The fill value is the constant 0, not a mean estimated
    from any split.

    ``padding_mode`` chooses where the zeros go: ``"end"`` appends them,
    ``"start"`` prepends them, and ``"center"`` splits them. An odd pad
    count puts the extra zero at the end.

    ``truncation_mode`` chooses the region that is kept: ``"start"`` keeps
    the beginning, ``"end"`` keeps the end, and ``"center"`` keeps the
    middle. An odd number of discarded samples drops the extra sample from
    the end. No random crop is used.

    The returned array is a new buffer. The function does not write a WAV.
    """
    signal = np.asarray(samples, dtype=np.float64)
    if config.duration_mode == "none":
        return signal.copy()
    if sample_rate_hz <= 0:
        raise FeatureExtractionError("sample_rate_hz must be positive to apply a fixed duration")
    target = int(round(config.duration_seconds * sample_rate_hz))
    if target < 1:
        raise FeatureExtractionError("The fixed duration rounds to zero samples")
    n_samples = int(signal.shape[0])
    if n_samples == target:
        return signal.copy()
    if n_samples < target:
        return _pad(signal, target, config.padding_mode)
    return _truncate(signal, target, config.truncation_mode)


def aggregate_features(matrix: np.ndarray, aggregation: str) -> np.ndarray:
    """Reduce a ``(n_frames, n_coefficients)`` matrix to one vector.

    ``mean`` is the arithmetic mean over frames.
    ``mean_std`` concatenates that mean with the population standard
    deviation (divisor ``n_frames``, NumPy ``ddof=0``). A single frame, or
    a coefficient that does not vary, has standard deviation 0. Those zeros
    are kept. They are not treated as missing.

    A matrix with no frames, or with any NaN or infinite value, raises
    ``FeatureExtractionError``. Invalid entries are not removed.
    """
    features = np.asarray(matrix, dtype=np.float64)
    if features.ndim != 2:
        raise FeatureExtractionError(
            f"Aggregation expected a 2D MFCC matrix, got ndim={features.ndim}"
        )
    if features.shape[0] == 0:
        raise FeatureExtractionError(
            "Aggregation received no frames. Empty frames are not replaced with zeros."
        )
    if not np.isfinite(features).all():
        raise FeatureExtractionError(
            "Aggregation received NaN or infinite MFCC values. Those values are not dropped."
        )
    mean = features.mean(axis=0)
    if aggregation == "mean":
        vector = mean
    elif aggregation == "mean_std":
        std = features.std(axis=0, ddof=0)
        if not np.isfinite(std).all():
            raise FeatureExtractionError(
                "Standard deviation is not finite. It was not replaced or dropped."
            )
        vector = np.concatenate([mean, std])
    else:
        raise ValueError(f"aggregation must be 'mean' or 'mean_std', got {aggregation!r}")
    if not np.isfinite(vector).all():
        raise FeatureExtractionError(
            "Aggregated features are not finite. Non-finite values were not dropped."
        )
    return np.ascontiguousarray(vector, dtype=np.float64)


def extract_features(
    audio_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Return one fixed-size feature vector for ``audio_path``.

    The vector contains no label information. Calling this function does not
    modify the source audio.
    """
    config = _config(config)
    loaded = load_audio(audio_path, config)
    return _features_from_samples(loaded.samples, loaded.sample_rate, config)


def extract_features_batch(
    items: pd.DataFrame | list | tuple,
    config: PreprocessingConfig | None = None,
    *,
    id_column: str = "Word_id",
    label_column: str | None = "Swahili_word",
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract one feature row per example, in the given order.

    Returns ``(X, metadata)``. ``X`` has shape ``(n_examples, n_features)``
    and dtype float64. ``metadata`` contains ``id_column`` and, when
    ``label_column`` is not None, that label column. Labels are not written
    into ``X``.

    A dataframe is processed in its current row order. A list or tuple is
    treated as audio paths or zip member ids, and ``label_column`` is ignored
    because those sequences have no labels.

    ``cache_dir`` stores only the aggregated vectors, under a subdirectory
    named by a hash of the full configuration. The cache is refused if it
    points at ``data/raw`` or ``data/splits``. ``None`` disables caching.
    """
    config = _config(config)
    frame, sources, metadata = _batch_tables(items, id_column, label_column)
    n_examples = len(frame)
    width = feature_dimension(config)
    features = np.empty((n_examples, width), dtype=np.float64)
    cache_root = _prepare_cache(cache_dir, config) if cache_dir is not None else None
    for row_index, source in enumerate(sources):
        features[row_index] = _extract_maybe_cached(source, config, cache_root)
    if n_examples == 0:
        features = np.empty((0, width), dtype=np.float64)
    return features, metadata.reset_index(drop=True)


def _config(config: PreprocessingConfig | None) -> PreprocessingConfig:
    if config is None:
        return PreprocessingConfig()
    if not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be a PreprocessingConfig or None")
    return config


def _read_source_bytes(source: str | Path) -> tuple[bytes, str]:
    path = Path(source)
    if path.is_file():
        return path.read_bytes(), str(path)
    if len(path.parts) > 1:
        raise FileNotFoundError(f"Audio file not found: {path}")
    word_id = str(source)
    return read_wav_bytes(word_id), word_id


class LoadedAudio:
    """One validated clip. ``samples`` is a mono float64 copy of the PCM."""

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        n_channels_in_file: int,
        sample_width_bytes: int,
        source: str,
        wav_sha256: str,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.n_channels_in_file = n_channels_in_file
        self.sample_width_bytes = sample_width_bytes
        self.source = source
        self.wav_sha256 = wav_sha256

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


def _decode_wav(wav_bytes: bytes, source: str, config: PreprocessingConfig) -> LoadedAudio:
    digest = hashlib.sha256(wav_bytes).hexdigest()
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            compression = wav_file.getcomptype()
            payload = wav_file.readframes(n_frames)
    except wave.Error as exc:
        raise AudioFormatError(f"{source} is not a readable PCM WAV: {exc}") from exc
    if compression != "NONE":
        raise AudioFormatError(
            f"{source} compression is {compression!r}. This pipeline accepts uncompressed PCM only "
            "and does not transcode the file."
        )
    if sample_width != 2:
        raise AudioFormatError(
            f"{source} sample width is {sample_width} bytes. Expected 16-bit PCM (2 bytes). "
            "Bit depth is not converted."
        )
    if sample_rate != config.target_sample_rate:
        raise AudioFormatError(
            f"{source} sample rate is {sample_rate} Hz. Expected {config.target_sample_rate} Hz. "
            "This pipeline does not resample."
        )
    n_values = n_frames * n_channels
    pcm = _pcm_samples(payload, sample_width, n_values)
    samples = _to_mono(pcm, n_frames, n_channels, source, config.channel_mode)
    return LoadedAudio(
        samples=samples,
        sample_rate=int(sample_rate),
        n_channels_in_file=int(n_channels),
        sample_width_bytes=int(sample_width),
        source=source,
        wav_sha256=digest,
    )


def _to_mono(
    pcm: np.ndarray,
    n_frames: int,
    n_channels: int,
    source: str,
    channel_mode: str,
) -> np.ndarray:
    if n_channels == 1:
        return np.array(pcm, dtype=np.float64, copy=True)
    if channel_mode == "mean_downmix":
        interleaved = np.array(pcm, dtype=np.float64, copy=True).reshape(n_frames, n_channels)
        return interleaved.mean(axis=1)
    raise AudioFormatError(
        f"{source} has {n_channels} channels. channel_mode='require_mono' does not downmix. "
        "Pass a mono file, or set channel_mode='mean_downmix' to average the channels explicitly."
    )


def _pad(samples: np.ndarray, target: int, padding_mode: str) -> np.ndarray:
    pad = target - int(samples.shape[0])
    if padding_mode == "end":
        widths = (0, pad)
    elif padding_mode == "start":
        widths = (pad, 0)
    elif padding_mode == "center":
        left = pad // 2
        widths = (left, pad - left)
    else:
        raise ValueError(f"padding_mode must be 'end', 'start', or 'center', got {padding_mode!r}")
    return np.pad(samples, widths, mode="constant", constant_values=0.0)


def _truncate(samples: np.ndarray, target: int, truncation_mode: str) -> np.ndarray:
    n_samples = int(samples.shape[0])
    if truncation_mode == "start":
        start = 0
    elif truncation_mode == "end":
        start = n_samples - target
    elif truncation_mode == "center":
        start = (n_samples - target) // 2
    else:
        raise ValueError(
            f"truncation_mode must be 'start', 'end', or 'center', got {truncation_mode!r}"
        )
    return np.array(samples[start : start + target], dtype=np.float64, copy=True)


def _features_from_samples(
    samples: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    signal = apply_peak_normalization(samples, config.normalize_amplitude)
    signal = apply_duration(signal, sample_rate, config)
    matrix = _coefficient_matrix(signal, sample_rate, config)
    return aggregate_features(matrix, config.aggregation)


def _coefficient_matrix(
    samples: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    static = mfcc_matrix(
        samples,
        sample_rate,
        n_mfcc=config.n_mfcc,
        frame_length=config.window_length,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
    )
    parts = [static]
    if config.include_delta:
        delta = delta_features(static, DELTA_WIDTH)
        parts.append(delta)
        if config.include_delta_delta:
            parts.append(delta_features(delta, DELTA_WIDTH))
    stacked = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    expected = config.n_mfcc * feature_streams(config)
    if stacked.ndim != 2 or stacked.shape[1] != expected:
        raise FeatureExtractionError(
            f"MFCC matrix shape is {stacked.shape}; expected (_, {expected})."
        )
    if stacked.shape[0] == 0:
        raise FeatureExtractionError(
            "The clip produced no analysis frames. Each frame needs "
            f"{config.window_length} samples, and shorter clips are not filled with "
            "manufactured MFCC values. Increase duration_seconds or supply a longer clip."
        )
    if not np.isfinite(stacked).all():
        raise FeatureExtractionError(
            "MFCC or delta features contain NaN or infinite values. Those values are not dropped."
        )
    return stacked


def _batch_tables(
    items: pd.DataFrame | list | tuple,
    id_column: str,
    label_column: str | None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    if isinstance(items, pd.DataFrame):
        frame = items
        if id_column not in frame.columns:
            raise KeyError(f"id_column {id_column!r} is not in the dataframe")
        columns = [id_column]
        if label_column is not None:
            if label_column not in frame.columns:
                raise KeyError(f"label_column {label_column!r} is not in the dataframe")
            if label_column != id_column:
                columns.append(label_column)
        metadata = frame.loc[:, columns].copy()
        sources = [str(value) for value in frame[id_column].tolist()]
        return frame, sources, metadata
    if isinstance(items, (list, tuple)):
        sources = [str(value) for value in items]
        metadata = pd.DataFrame({id_column: sources})
        return metadata, sources, metadata
    raise TypeError("items must be a pandas DataFrame, list, or tuple")


def config_fingerprint(config: PreprocessingConfig) -> str:
    """Hash the full configuration so cache directories cannot be shared."""
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_cache(cache_dir: str | Path, config: PreprocessingConfig) -> Path:
    resolved = Path(cache_dir).resolve()
    raw = raw_data_dir().resolve()
    splits = splits_dir().resolve()
    if resolved == raw or raw in resolved.parents:
        raise ValueError(f"Refusing to cache features inside raw audio: {resolved}")
    if resolved == splits or splits in resolved.parents:
        raise ValueError(f"Refusing to cache features inside the shared split directory: {resolved}")
    directory = resolved / config_fingerprint(config)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "config.json"
    payload = json.dumps(config.to_dict(), sort_keys=True, indent=2)
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != config.to_dict():
            raise FeatureExtractionError(
                f"{manifest} does not match this configuration. "
                "Cached features from another configuration were not loaded."
            )
    else:
        manifest.write_text(payload + "\n", encoding="utf-8")
    return directory


def _cache_file(cache_root: Path, source: str, wav_sha256: str) -> Path:
    stem = _CACHE_NAME.sub("_", Path(source).name) or "audio"
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    return cache_root / f"{stem}__{source_hash}__{wav_sha256[:16]}.npy"


def _extract_maybe_cached(
    source: str,
    config: PreprocessingConfig,
    cache_root: Path | None,
) -> np.ndarray:
    if cache_root is None:
        return extract_features(source, config)
    wav_bytes, label = _read_source_bytes(source)
    digest = hashlib.sha256(wav_bytes).hexdigest()
    path = _cache_file(cache_root, label, digest)
    if path.is_file():
        cached = np.load(path)
        expected = (feature_dimension(config),)
        if cached.shape != expected or cached.dtype != np.float64:
            raise FeatureExtractionError(
                f"Cache file {path.name} has shape {cached.shape} and dtype {cached.dtype}; "
                f"expected shape {expected} and float64. It was not used."
            )
        if not np.isfinite(cached).all():
            raise FeatureExtractionError(
                f"Cache file {path.name} contains non-finite values. It was not used."
            )
        return np.array(cached, dtype=np.float64, copy=True)
    loaded = _decode_wav(wav_bytes, label, config)
    if loaded.wav_sha256 != digest:
        raise FeatureExtractionError("WAV hash changed while the file was being read.")
    vector = _features_from_samples(loaded.samples, loaded.sample_rate, config)
    np.save(path, vector)
    return vector
