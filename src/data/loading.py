"""Dataset loading for the Swahili audio challenge.

Ownership: Dan.

Paths are resolved from this file's location, so callers do not need a
machine-specific absolute path. The official ``Test.csv`` file is unlabeled
and is not the research test set. The shared split is read from
``data/splits/shared_split.csv``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

TRAIN_CSV_NAME = "Train.csv"
TEST_CSV_NAME = "Test.csv"
SAMPLE_SUBMISSION_NAME = "SampleSubmission.csv"
AUDIO_ZIP_NAME = "Swahili_words.zip"
SHARED_SPLIT_NAME = "shared_split.csv"


def project_root() -> Path:
    """Return the repository root (the directory that contains ``src``)."""
    return Path(__file__).resolve().parents[2]


def raw_data_dir() -> Path:
    return project_root() / "data" / "raw"


def splits_dir() -> Path:
    return project_root() / "data" / "splits"


def _csv_path(filename: str, path: Path | None) -> Path:
    resolved = raw_data_dir() / filename if path is None else Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required data file not found: {resolved}")
    return resolved


def load_train_csv(path: Path | None = None) -> pd.DataFrame:
    """Load labeled clips. Does not create a train/validation/test split."""
    return pd.read_csv(_csv_path(TRAIN_CSV_NAME, path))


def load_test_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the official unlabeled challenge ids. Not the research test set."""
    return pd.read_csv(_csv_path(TEST_CSV_NAME, path))


def load_sample_submission(path: Path | None = None) -> pd.DataFrame:
    """Load the official sample submission template."""
    return pd.read_csv(_csv_path(SAMPLE_SUBMISSION_NAME, path))


def audio_zip_path(path: Path | None = None) -> Path:
    resolved = raw_data_dir() / AUDIO_ZIP_NAME if path is None else Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio zip not found: {resolved}")
    return resolved


def list_zip_members(path: Path | None = None) -> list[str]:
    """Return zip member names without extracting the archive."""
    with zipfile.ZipFile(audio_zip_path(path)) as archive:
        return archive.namelist()


def read_wav_bytes(word_id: str, zip_path: Path | None = None) -> bytes:
    """Read one WAV member from the zip into memory. Does not write the file."""
    archive_path = audio_zip_path(zip_path)
    with zipfile.ZipFile(archive_path) as archive:
        try:
            return archive.read(word_id)
        except KeyError as exc:
            raise FileNotFoundError(
                f"{word_id} is not a member of {archive_path.name}"
            ) from exc


def shared_split_path(path: Path | None = None) -> Path:
    return splits_dir() / SHARED_SPLIT_NAME if path is None else Path(path)


def read_shared_split(path: Path | None = None) -> pd.DataFrame:
    """Load the shared split assignments (``Word_id``, ``split``).

    Models must use this file. They must not draw their own split.
    """
    resolved = shared_split_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Shared split not found: {resolved}. Create it with "
            "src.data.split.create_shared_split before modelling."
        )
    split_df = pd.read_csv(resolved)
    expected = ["Word_id", "split"]
    if list(split_df.columns) != expected:
        raise ValueError(
            f"{resolved} columns are {list(split_df.columns)}; expected {expected}."
        )
    return split_df
