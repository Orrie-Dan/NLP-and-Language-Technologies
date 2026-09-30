"""Audio preprocessing for the classical baselines.

Ownership: Dan.

``extract_features`` turns one clip into a fixed-size vector. The defaults on
``PreprocessingConfig`` are Phase 4 implementation defaults, not a selected
experimental configuration. Phase 3 investigation helpers remain in ``mfcc``
and ``energy``.
"""

from src.preprocessing.config import CONFIGURATION_STATUS, PreprocessingConfig
from src.preprocessing.pipeline import (
    AudioFormatError,
    FeatureExtractionError,
    extract_features,
    extract_features_batch,
    feature_dimension,
    load_audio,
    uses_dataset_statistics,
)

__all__ = [
    "CONFIGURATION_STATUS",
    "AudioFormatError",
    "FeatureExtractionError",
    "PreprocessingConfig",
    "extract_features",
    "extract_features_batch",
    "feature_dimension",
    "load_audio",
    "uses_dataset_statistics",
]
