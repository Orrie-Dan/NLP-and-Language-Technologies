"""Fast checks for the Phase 4 feature pipeline.

Uses synthetic WAV files. It does not read the competition audio and does not
train a classifier.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loading import raw_data_dir, shared_split_path, splits_dir
from src.preprocessing import (
    AudioFormatError,
    FeatureExtractionError,
    PreprocessingConfig,
    extract_features,
    extract_features_batch,
    feature_dimension,
    uses_dataset_statistics,
)
from src.preprocessing.mfcc import delta_features, mfcc_matrix
from src.preprocessing.pipeline import (
    aggregate_features,
    apply_duration,
    config_fingerprint,
    load_audio,
)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = 16000,
    n_channels: int = 1,
    sample_width: int = 2,
) -> None:
    if n_channels == 1:
        payload = samples
    else:
        payload = np.repeat(samples[:, None], n_channels, axis=1).reshape(-1)
    if sample_width == 2:
        pcm = np.clip(np.round(payload), -32768, 32767).astype("<i2")
        frames = pcm.tobytes()
    elif sample_width == 1:
        pcm = np.clip(np.round(payload), 0, 255).astype(np.uint8)
        frames = pcm.tobytes()
    else:
        raise ValueError("test helper supports 8-bit and 16-bit WAV only")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def _tone(seconds: float, sample_rate: int = 16000, amplitude: float = 8000.0) -> np.ndarray:
    times = np.arange(int(round(seconds * sample_rate)), dtype=np.float64)
    return amplitude * np.sin(2.0 * np.pi * 220.0 * times / sample_rate)


class PipelineTests(unittest.TestCase):
    def test_documented_feature_dimensions(self) -> None:
        base = PreprocessingConfig(n_mfcc=13, duration_mode="none")
        self.assertEqual(feature_dimension(replace(base, aggregation="mean")), 13)
        self.assertEqual(feature_dimension(replace(base, aggregation="mean_std")), 26)
        self.assertEqual(
            feature_dimension(replace(base, aggregation="mean", include_delta=True)),
            26,
        )
        self.assertEqual(
            feature_dimension(replace(base, aggregation="mean_std", include_delta=True)),
            52,
        )
        both = dict(include_delta=True, include_delta_delta=True)
        self.assertEqual(feature_dimension(replace(base, aggregation="mean", **both)), 39)
        self.assertEqual(feature_dimension(replace(base, aggregation="mean_std", **both)), 78)
        self.assertFalse(uses_dataset_statistics(base))

    def test_configuration_rejects_inconsistent_options(self) -> None:
        with self.assertRaises(ValueError):
            PreprocessingConfig(include_delta=False, include_delta_delta=True)
        with self.assertRaises(ValueError):
            PreprocessingConfig(window_length=513, n_fft=512)
        with self.assertRaises(ValueError):
            PreprocessingConfig(duration_seconds=0)
        with self.assertRaises(ValueError):
            PreprocessingConfig(n_mfcc=41, n_mels=40)

    def test_padding_truncation_and_unchanged_duration(self) -> None:
        rate = 16000
        padded = apply_duration(
            np.arange(4, dtype=np.float64),
            rate,
            PreprocessingConfig(duration_seconds=10 / rate, padding_mode="end"),
        )
        self.assertEqual(padded.shape, (10,))
        np.testing.assert_array_equal(padded[:4], np.arange(4))
        self.assertTrue(np.all(padded[4:] == 0.0))

        odd = apply_duration(
            np.arange(4, dtype=np.float64),
            rate,
            PreprocessingConfig(duration_seconds=9 / rate, padding_mode="center"),
        )
        np.testing.assert_array_equal(odd[:2], 0.0)
        np.testing.assert_array_equal(odd[2:6], np.arange(4))
        np.testing.assert_array_equal(odd[6:], 0.0)

        truncated = apply_duration(
            np.arange(11, dtype=np.float64),
            rate,
            PreprocessingConfig(duration_seconds=4 / rate, truncation_mode="start"),
        )
        np.testing.assert_array_equal(truncated, np.arange(4))
        centered = apply_duration(
            np.arange(11, dtype=np.float64),
            rate,
            PreprocessingConfig(duration_seconds=4 / rate, truncation_mode="center"),
        )
        np.testing.assert_array_equal(centered, np.arange(3, 7))

        original = np.arange(6, dtype=np.float64)
        kept = apply_duration(
            original,
            rate,
            PreprocessingConfig(duration_mode="none", duration_seconds=6.0),
        )
        np.testing.assert_array_equal(kept, original)
        self.assertIsNot(kept, original)

    def test_aggregation_keeps_zero_variance_and_rejects_nan(self) -> None:
        constant = np.ones((3, 2), dtype=np.float64)
        vector = aggregate_features(constant, "mean_std")
        np.testing.assert_array_equal(vector, np.array([1.0, 1.0, 0.0, 0.0]))
        self.assertEqual(vector.shape, (4,))

        broken = constant.copy()
        broken[0, 0] = np.nan
        with self.assertRaises(FeatureExtractionError):
            aggregate_features(broken, "mean")
        with self.assertRaises(FeatureExtractionError):
            aggregate_features(np.zeros((0, 13)), "mean_std")

    def test_extract_is_finite_reproducible_and_reuses_mfcc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            _write_wav(path, _tone(0.5))
            before = path.read_bytes()
            config = PreprocessingConfig(
                duration_mode="none",
                aggregation="mean",
                normalize_amplitude=False,
                include_delta=False,
            )
            loaded = load_audio(path, config)
            direct = mfcc_matrix(
                loaded.samples,
                loaded.sample_rate,
                n_mfcc=config.n_mfcc,
                frame_length=config.window_length,
                hop_length=config.hop_length,
                n_fft=config.n_fft,
                n_mels=config.n_mels,
            )
            first = extract_features(path, config)
            second = extract_features(path, config)
            np.testing.assert_allclose(first, direct.mean(axis=0))
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(first.shape, (feature_dimension(config),))
            self.assertEqual(path.read_bytes(), before)

            stacked_config = replace(
                config,
                include_delta=True,
                include_delta_delta=True,
                aggregation="mean_std",
            )
            delta = delta_features(direct)
            delta_delta = delta_features(delta)
            stacked = np.concatenate([direct, delta, delta_delta], axis=1)
            expected = np.concatenate([stacked.mean(axis=0), stacked.std(axis=0, ddof=0)])
            stacked_features = extract_features(path, stacked_config)
            np.testing.assert_allclose(stacked_features, expected)
            self.assertEqual(stacked_features.shape, (78,))
            self.assertTrue(np.isfinite(stacked_features).all())
            self.assertEqual(path.read_bytes(), before)

    def test_duration_settings_keep_dimension_and_change_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.wav"
            _write_wav(path, _tone(0.25))
            before = path.read_bytes()
            mean = dict(aggregation="mean", include_delta=False, include_delta_delta=False)
            six = extract_features(path, PreprocessingConfig(duration_seconds=6.0, **mean))
            eight = extract_features(path, PreprocessingConfig(duration_seconds=8.0, **mean))
            self.assertEqual(six.shape, (13,))
            self.assertEqual(eight.shape, (13,))
            self.assertTrue(np.isfinite(six).all())
            self.assertTrue(np.isfinite(eight).all())
            self.assertFalse(np.array_equal(six, eight))
            self.assertEqual(path.read_bytes(), before)

    def test_batch_keeps_labels_out_of_the_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.wav"
            second = root / "b.wav"
            _write_wav(first, _tone(0.4))
            _write_wav(second, _tone(0.4, amplitude=3000.0))
            config = PreprocessingConfig(duration_mode="none", aggregation="mean")
            frame = pd.DataFrame(
                {"Word_id": [str(first), str(second)], "Swahili_word": ["moja", "mbili"]},
                index=[5, 1],
            )
            features, metadata = extract_features_batch(frame, config)
            self.assertEqual(features.shape, (2, 13))
            self.assertEqual(features.dtype, np.float64)
            self.assertTrue(np.isfinite(features).all())
            self.assertEqual(metadata["Swahili_word"].tolist(), ["moja", "mbili"])
            np.testing.assert_array_equal(features[0], extract_features(first, config))
            self.assertEqual(list(metadata.columns), ["Word_id", "Swahili_word"])

    def test_format_errors_do_not_modify_or_resample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stereo = root / "stereo.wav"
            wrong_rate = root / "rate.wav"
            wrong_width = root / "width.wav"
            mono = root / "mono.wav"
            tone = _tone(0.2)
            _write_wav(stereo, tone, n_channels=2)
            _write_wav(wrong_rate, tone[:1600], sample_rate=8000)
            _write_wav(wrong_width, np.full(1600, 128.0), sample_width=1)
            _write_wav(mono, tone)
            before = {path: path.read_bytes() for path in (stereo, wrong_rate, wrong_width, mono)}
            with self.assertRaises(AudioFormatError) as stereo_error:
                load_audio(stereo, PreprocessingConfig(channel_mode="require_mono"))
            self.assertIn("does not downmix", str(stereo_error.exception))
            with self.assertRaises(AudioFormatError) as rate_error:
                load_audio(wrong_rate, PreprocessingConfig())
            self.assertIn("does not resample", str(rate_error.exception))
            with self.assertRaises(AudioFormatError):
                load_audio(wrong_width, PreprocessingConfig())
            required = load_audio(mono, PreprocessingConfig(channel_mode="require_mono"))
            downmix_setting = load_audio(mono, PreprocessingConfig(channel_mode="mean_downmix"))
            np.testing.assert_array_equal(required.samples, downmix_setting.samples)
            for path, payload in before.items():
                self.assertEqual(path.read_bytes(), payload)

    def test_peak_normalization_is_optional_and_per_clip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            _write_wav(path, _tone(0.4))
            base = PreprocessingConfig(duration_mode="none", aggregation="mean", normalize_amplitude=False)
            off = extract_features(path, base)
            on = extract_features(path, replace(base, normalize_amplitude=True))
            self.assertTrue(np.isfinite(on).all())
            self.assertFalse(np.array_equal(off, on))
            loaded = load_audio(path, base)
            scaled = loaded.samples / np.max(np.abs(loaded.samples))
            self.assertAlmostEqual(float(np.max(np.abs(scaled))), 1.0)
            direct = mfcc_matrix(
                scaled,
                loaded.sample_rate,
                n_mfcc=base.n_mfcc,
                frame_length=base.window_length,
                hop_length=base.hop_length,
                n_fft=base.n_fft,
                n_mels=base.n_mels,
            )
            np.testing.assert_allclose(on, direct.mean(axis=0))

    def test_cache_keys_follow_the_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tone.wav"
            _write_wav(path, _tone(0.4))
            cache = root / "cache"
            mean = PreprocessingConfig(duration_mode="none", aggregation="mean", n_mfcc=13)
            other = replace(mean, n_mfcc=12)
            self.assertNotEqual(config_fingerprint(mean), config_fingerprint(other))
            first, _ = extract_features_batch([str(path)], mean, label_column=None, cache_dir=cache)
            second, _ = extract_features_batch([str(path)], mean, label_column=None, cache_dir=cache)
            changed, _ = extract_features_batch([str(path)], other, label_column=None, cache_dir=cache)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(changed.shape, (1, 12))
            self.assertEqual(len(list(cache.glob("*/*.npy"))), 2)
            with self.assertRaises(ValueError):
                extract_features_batch(
                    [str(path)],
                    mean,
                    label_column=None,
                    cache_dir=raw_data_dir() / "feature_cache",
                )
            with self.assertRaises(ValueError):
                extract_features_batch(
                    [str(path)],
                    mean,
                    label_column=None,
                    cache_dir=splits_dir() / "feature_cache",
                )

    def test_shared_split_file_is_not_modified(self) -> None:
        split_path = shared_split_path()
        before = hashlib.sha256(split_path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            _write_wav(path, _tone(0.3))
            extract_features(
                path,
                PreprocessingConfig(duration_mode="none", aggregation="mean"),
            )
        after = hashlib.sha256(split_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
