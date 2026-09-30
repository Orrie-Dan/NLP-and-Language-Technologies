"""Shared stratified split of the labeled Swahili clips.

The official unlabeled challenge test file is never included. The split is
70% train, 15% validation, and 15% internal test, stratified on
``Swahili_word``, with ``random_state`` 42.

Fifteen percent of 350 clips per class is not an integer. Exact per-class
counts are whatever ``sklearn.model_selection.train_test_split`` assigns
under the recorded scikit-learn version. After creation, group members must
load ``data/splits/shared_split.csv`` rather than draw a new split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loading import shared_split_path
from src.utils.reproducibility import SHARED_SPLIT_SEED

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
HOLDOUT_FRACTION = VALIDATION_FRACTION + TEST_FRACTION


def create_shared_split(
    train_df: pd.DataFrame,
    seed: int = SHARED_SPLIT_SEED,
) -> pd.DataFrame:
    """Assign each labeled ``Word_id`` to train, validation, or test.

    The 30% holdout is split in half so the two held-out parts are each 15%
    of the labeled rows. Both calls use ``random_state=seed`` and stratify
    on ``Swahili_word``. ``train_df`` row order is the order returned by
    ``load_train_csv``.
    """
    if abs((TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION) - 1.0) > 1e-12:
        raise ValueError("Split fractions must sum to 1.")

    required = {"Word_id", "Swahili_word"}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f"Labeled table is missing columns: {sorted(missing)}")
    if train_df["Word_id"].duplicated().any():
        raise ValueError("Cannot split labeled data with duplicate Word_id values.")

    train_part, holdout = train_test_split(
        train_df,
        test_size=HOLDOUT_FRACTION,
        random_state=seed,
        stratify=train_df["Swahili_word"],
        shuffle=True,
    )
    validation_part, test_part = train_test_split(
        holdout,
        test_size=0.5,
        random_state=seed,
        stratify=holdout["Swahili_word"],
        shuffle=True,
    )

    pieces = [
        train_part.assign(split="train"),
        validation_part.assign(split="validation"),
        test_part.assign(split="test"),
    ]
    split_df = pd.concat(pieces, ignore_index=True)
    split_df = split_df.loc[:, ["Word_id", "split"]].sort_values("Word_id")
    return split_df.reset_index(drop=True)


def validate_shared_split(
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    official_test_df: pd.DataFrame,
) -> dict:
    """Check coverage, uniqueness, and exclusion of official test ids."""
    split_ids = split_df["Word_id"]
    labeled_ids = set(train_df["Word_id"])
    official_ids = set(official_test_df["Word_id"])
    assigned_ids = set(split_ids)
    counts = split_df["split"].value_counts().to_dict()
    merged = split_df.merge(train_df, on="Word_id", how="left", validate="one_to_one")
    class_counts = (
        merged.groupby(["split", "Swahili_word"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    return {
        "n_rows": int(len(split_df)),
        "duplicate_word_ids": int(split_ids.duplicated().sum()),
        "n_assigned": int(len(assigned_ids)),
        "n_labeled": int(len(labeled_ids)),
        "missing_labeled_ids": int(len(labeled_ids - assigned_ids)),
        "unknown_ids": int(len(assigned_ids - labeled_ids)),
        "official_test_ids_in_split": int(len(assigned_ids & official_ids)),
        "split_counts": {str(key): int(value) for key, value in counts.items()},
        "class_counts": class_counts,
        "allowed_split_names": sorted(split_df["split"].unique().tolist()),
    }


def save_shared_split(split_df: pd.DataFrame, path: Path | None = None) -> Path:
    destination = shared_split_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(destination, index=False)
    return destination
