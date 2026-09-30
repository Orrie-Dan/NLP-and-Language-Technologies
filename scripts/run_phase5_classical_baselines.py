"""Phase 5A — Classical sequential baselines.

This experiment compares two classical models:

1. Multinomial Logistic Regression
2. Linear SVM

The experiment has two stages:

Stage A
-------
Compare a small number of preprocessing/feature configurations on the
validation split.

Stage B
-------
Lock the selected preprocessing configuration, tune the classifier
hyperparameters using validation, and evaluate the locked models once
on the held-out internal test split.

The official unlabeled Zindi Test.csv is never used.

The shared split is read but never regenerated or modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data.loading import load_train_csv, read_shared_split
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import (
    extract_features_batch,
    feature_dimension,
)


SEED = 42

RESULTS_DIR = Path("results/classical")
FIGURES_DIR = Path("results/figures/classical")

PRIMARY_METRIC = "macro_f1"

# Clips passed to extract_features_batch at once. The per-clip Phase 4
# cache is unchanged. Override with --batch-size.
FEATURE_BATCH_SIZE = 100


# ---------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureExperiment:
    name: str
    duration_seconds: float
    n_mfcc: int
    include_delta: bool
    include_delta_delta: bool
    aggregation: str = "mean_std"
    normalize_amplitude: bool = False


FEATURE_CONFIGS = (
    FeatureExperiment(
        name="mfcc13_6s_meanstd",
        duration_seconds=6.0,
        n_mfcc=13,
        include_delta=False,
        include_delta_delta=False,
        aggregation="mean_std",
    ),
    FeatureExperiment(
        name="mfcc13_8s_meanstd",
        duration_seconds=8.0,
        n_mfcc=13,
        include_delta=False,
        include_delta_delta=False,
        aggregation="mean_std",
    ),
    FeatureExperiment(
        name="mfcc13_deltas_8s_meanstd",
        duration_seconds=8.0,
        n_mfcc=13,
        include_delta=True,
        include_delta_delta=True,
        aggregation="mean_std",
    ),
    FeatureExperiment(
        name="mfcc20_8s_meanstd",
        duration_seconds=8.0,
        n_mfcc=20,
        include_delta=False,
        include_delta_delta=False,
        aggregation="mean_std",
    ),
)


CLASSIFIER_C_VALUES = {
    "logistic_regression": (0.01, 0.1, 1.0, 10.0),
    "linear_svm": (0.01, 0.1, 1.0, 10.0),
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def class_label_column(columns: list[str]) -> str:
    """Return the class column from the Train.csv header just loaded.

    The released file has Word_id, Swahili_word, and English_translation.
    Swahili_word is the class. English_translation is the gloss and is not
    the classification target. The name is used only after it is found in
    the loaded header.
    """

    if "Swahili_word" in columns:
        return "Swahili_word"

    raise ValueError(
        "Train.csv does not contain the class column Swahili_word. "
        f"Columns found: {columns}"
    )


def load_shared_split() -> pd.DataFrame:
    """Join Train.csv labels onto the shared split in memory.

    shared_split.csv stays Word_id and split. This function does not write
    that file, Train.csv, Test.csv, or any audio.
    """

    split = read_shared_split()
    labels = load_train_csv()
    n_split = len(split)
    n_labels = len(labels)

    print(f"Loaded shared split: {n_split} rows", flush=True)
    print(f"Loaded labels from Train.csv: {n_labels} rows", flush=True)

    train_columns = list(labels.columns)

    if "Word_id" not in train_columns:
        raise ValueError(
            f"Train.csv has no Word_id column. Columns found: {train_columns}"
        )

    if labels["Word_id"].duplicated().any():
        raise ValueError("Duplicate Word_id values found in Train.csv.")

    if split["Word_id"].duplicated().any():
        raise ValueError("Duplicate Word_id values found in the shared split.")

    source_label = class_label_column(train_columns)
    label_table = labels.loc[:, ["Word_id", source_label]].rename(
        columns={source_label: "label"}
    )

    joined = split.merge(
        label_table,
        on="Word_id",
        how="left",
        validate="one_to_one",
    )

    if len(joined) != n_split:
        raise ValueError(
            "Joining labels changed the shared split row count from "
            f"{n_split} to {len(joined)}."
        )

    missing = int(joined["label"].isna().sum())

    if missing:
        raise ValueError(
            f"{missing} shared-split Word_id values have no label in Train.csv."
        )

    joined["split"] = joined["split"].astype(str).str.lower()
    expected_splits = {"train", "validation", "test"}
    actual_splits = set(joined["split"])

    if actual_splits != expected_splits:
        raise ValueError(
            "Expected train, validation, and test in the shared split, found "
            f"{sorted(actual_splits)}."
        )

    print("Joined labels successfully", flush=True)

    counts = joined["split"].value_counts()

    print(
        f"Train: {int(counts['train'])} | "
        f"Validation: {int(counts['validation'])} | "
        f"Test: {int(counts['test'])}",
        flush=True,
    )

    classes = ", ".join(sorted(joined["label"].astype(str).unique()))

    print(f"Classes: {classes}", flush=True)

    return joined.loc[:, ["Word_id", "split", "label"]].reset_index(drop=True)


def build_config(experiment: FeatureExperiment) -> PreprocessingConfig:
    """Convert an experiment definition into the actual Phase 4 config."""

    return PreprocessingConfig(
        duration_seconds=experiment.duration_seconds,
        duration_mode="fixed",
        padding_mode="end",
        truncation_mode="start",
        n_mfcc=experiment.n_mfcc,
        include_delta=experiment.include_delta,
        include_delta_delta=experiment.include_delta_delta,
        aggregation=experiment.aggregation,
        normalize_amplitude=experiment.normalize_amplitude,
        random_seed=SEED,
    )


def extract_split_features(
    split: pd.DataFrame,
    config: PreprocessingConfig,
    cache_dir: Path,
    split_name: str,
    batch_size: int = FEATURE_BATCH_SIZE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract features in small batches, preserving row order.

    Each batch calls the Phase 4 ``extract_features_batch`` cache. Cached
    clips are loaded and missing clips are computed. Only one batch is
    passed to that function at a time. The returned matrix follows
    ``split`` from top to bottom.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    n_rows = len(split)
    width = feature_dimension(config)
    features = np.empty((n_rows, width), dtype=np.float64)
    metadata_parts: list[pd.DataFrame] = []

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        batch = split.iloc[start:stop]
        batch_features, batch_metadata = extract_features_batch(
            batch[["Word_id", "label"]],
            config=config,
            id_column="Word_id",
            label_column="label",
            cache_dir=cache_dir,
        )
        if batch_features.shape != (stop - start, width):
            raise RuntimeError(
                f"{split_name} batch shape {batch_features.shape} does not "
                f"match {(stop - start, width)}."
            )
        features[start:stop] = batch_features
        metadata_parts.append(batch_metadata)
        print(
            f"Extracting {split_name} features: {stop}/{n_rows}",
            flush=True,
        )
        del batch
        del batch_features
        del batch_metadata
        gc.collect()

    metadata = (
        pd.concat(metadata_parts, ignore_index=True)
        if metadata_parts
        else pd.DataFrame(columns=["Word_id", "label"])
    )
    del metadata_parts
    gc.collect()

    if features.shape[0] != n_rows:
        raise RuntimeError(
            f"Feature row count mismatch: X has {features.shape[0]} rows "
            f"but split has {n_rows}."
        )

    if list(metadata["Word_id"]) != list(split["Word_id"]):
        raise RuntimeError(
            f"{split_name} feature row order does not match the split order."
        )

    if not np.isfinite(features).all():
        raise RuntimeError("Feature matrix contains NaN or infinite values.")

    return features, metadata


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def make_logistic_regression(C: float) -> Pipeline:
    """Create a reproducible multinomial logistic regression pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_linear_svm(C: float) -> Pipeline:
    """Create a reproducible linear SVM pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LinearSVC(
                    C=C,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate the main classification metrics."""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.unique(y_true),
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )[1]
        ),
    }


def save_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    stage: str,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "model": model_name,
                    "stage": stage,
                    "class": label,
                    **values,
                }
            )

    output = pd.DataFrame(rows)

    output.to_csv(
        RESULTS_DIR
        / f"{stage}_{model_name}_classification_report.csv",
        index=False,
    )


def save_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    stage: str,
) -> None:
    labels = sorted(pd.unique(pd.Series(y_true).astype(str)))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    matrix_df.to_csv(
        RESULTS_DIR
        / f"{stage}_{model_name}_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(matrix)

    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — {stage.title()} Confusion Matrix"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Add counts to cells.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / f"{stage}_{model_name}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Validation experiment
# ---------------------------------------------------------------------

def run_validation_experiment(
    split: pd.DataFrame,
    cache_root: Path,
    batch_size: int = FEATURE_BATCH_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Compare preprocessing configurations using validation only."""

    train = split[split["split"] == "train"].reset_index(drop=True)
    validation = split[
        split["split"] == "validation"
    ].reset_index(drop=True)

    y_train = train["label"]
    y_validation = validation["label"]

    results = []

    for experiment in FEATURE_CONFIGS:
        print("=" * 72)
        print(f"Feature configuration: {experiment.name}", flush=True)

        config = build_config(experiment)

        print(
            f"Duration={experiment.duration_seconds}s | "
            f"MFCC={experiment.n_mfcc} | "
            f"deltas={experiment.include_delta_delta} | "
            f"aggregation={experiment.aggregation}",
            flush=True,
        )

        feature_cache = cache_root / "features"

        X_train, _ = extract_split_features(
            train,
            config,
            feature_cache,
            "train",
            batch_size,
        )

        X_validation, _ = extract_split_features(
            validation,
            config,
            feature_cache,
            "validation",
            batch_size,
        )

        print(f"Train shape: {X_train.shape}")
        print(f"Validation shape: {X_validation.shape}")

        for model_name, c_values in CLASSIFIER_C_VALUES.items():

            for C in c_values:

                if model_name == "logistic_regression":
                    model = make_logistic_regression(C)
                else:
                    model = make_linear_svm(C)

                print(
                    f"  {model_name}, C={C}: training..."
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=ConvergenceWarning,
                    )

                    model.fit(X_train, y_train)

                predictions = model.predict(X_validation)

                metrics = evaluate_predictions(
                    y_validation,
                    predictions,
                )

                results.append(
                    {
                        "stage": "validation",
                        "feature_config": experiment.name,
                        "model": model_name,
                        "C": C,
                        "feature_dimension": feature_dimension(config),
                        **metrics,
                    }
                )

                print(
                    f"    accuracy={metrics['accuracy']:.4f}, "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )

                del model
                del predictions

        del X_train
        del X_validation
        del config
        gc.collect()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "validation_classical_results.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Select preprocessing configuration.
    #
    # For each feature representation:
    #   1. Find the best C for each classifier.
    #   2. Average the two classifiers' best validation macro-F1.
    #
    # This makes the representation choice independent of one single
    # classifier.
    # --------------------------------------------------------------

    best_per_model = (
        results_df
        .sort_values(
            ["feature_config", "model", PRIMARY_METRIC],
            ascending=[True, True, False],
        )
        .groupby(
            ["feature_config", "model"],
            as_index=False,
        )
        .first()
    )

    preprocessing_selection = (
        best_per_model
        .groupby("feature_config", as_index=False)
        .agg(
            mean_validation_macro_f1=("macro_f1", "mean"),
            min_validation_macro_f1=("macro_f1", "min"),
            max_validation_macro_f1=("macro_f1", "max"),
        )
        .sort_values(
            "mean_validation_macro_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    preprocessing_selection.to_csv(
        RESULTS_DIR / "preprocessing_selection.csv",
        index=False,
    )

    selected_feature_name = preprocessing_selection.iloc[0][
        "feature_config"
    ]

    selected_experiment = next(
        experiment
        for experiment in FEATURE_CONFIGS
        if experiment.name == selected_feature_name
    )

    selection_summary = {
        "primary_metric": PRIMARY_METRIC,
        "selection_rule": (
            "Select the preprocessing configuration with the highest "
            "mean validation macro-F1 across the best Logistic Regression "
            "and Linear SVM configurations."
        ),
        "selected_feature_config": selected_feature_name,
        "selected_feature_config_details": asdict(
            selected_experiment
        ),
        "validation_results": results_df.to_dict(
            orient="records"
        ),
        "preprocessing_selection": preprocessing_selection.to_dict(
            orient="records"
        ),
    }

    with open(
        RESULTS_DIR / "preprocessing_selection.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            selection_summary,
            handle,
            indent=2,
        )

    return results_df, selection_summary


# ---------------------------------------------------------------------
# Locked final classifier evaluation
# ---------------------------------------------------------------------

def run_locked_final_evaluation(
    split: pd.DataFrame,
    selected_experiment: FeatureExperiment,
    selection_summary: dict,
    cache_root: Path,
    batch_size: int = FEATURE_BATCH_SIZE,
) -> pd.DataFrame:
    """Tune classifiers on validation and evaluate once on internal test."""

    train = split[split["split"] == "train"].reset_index(drop=True)
    validation = split[
        split["split"] == "validation"
    ].reset_index(drop=True)
    test = split[split["split"] == "test"].reset_index(drop=True)

    config = build_config(selected_experiment)

    feature_cache = cache_root / "features"

    print("=" * 72)
    print("LOCKED FINAL CONFIGURATION")
    print("=" * 72)
    print(f"Feature configuration: {selected_experiment.name}", flush=True)

    print(json.dumps(asdict(selected_experiment), indent=2))

    X_train, _ = extract_split_features(
        train,
        config,
        feature_cache,
        "train",
        batch_size,
    )

    X_validation, _ = extract_split_features(
        validation,
        config,
        feature_cache,
        "validation",
        batch_size,
    )

    X_test, _ = extract_split_features(
        test,
        config,
        feature_cache,
        "test",
        batch_size,
    )

    y_train = train["label"]
    y_validation = validation["label"]
    y_test = test["label"]

    final_results = []

    for model_name, c_values in CLASSIFIER_C_VALUES.items():

        # ----------------------------------------------------------
        # Select C using validation.
        # ----------------------------------------------------------

        candidate_results = []

        for C in c_values:

            if model_name == "logistic_regression":
                model = make_logistic_regression(C)
            else:
                model = make_linear_svm(C)

            print(
                f"Validation tuning: {model_name}, C={C}"
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=ConvergenceWarning,
                )

                model.fit(X_train, y_train)

            validation_predictions = model.predict(
                X_validation
            )

            metrics = evaluate_predictions(
                y_validation,
                validation_predictions,
            )

            candidate_results.append(
                {
                    "model": model_name,
                    "C": C,
                    **metrics,
                }
            )

            del model
            del validation_predictions

        candidate_df = pd.DataFrame(candidate_results)

        candidate_df.to_csv(
            RESULTS_DIR
            / f"{model_name}_validation_hyperparameters.csv",
            index=False,
        )

        best_row = candidate_df.sort_values(
            PRIMARY_METRIC,
            ascending=False,
        ).iloc[0]

        best_C = float(best_row["C"])

        print(
            f"Selected {model_name} C={best_C} "
            f"using validation macro-F1="
            f"{best_row['macro_f1']:.4f}"
        )

        # ----------------------------------------------------------
        # Refit on TRAIN + VALIDATION after C has been selected.
        # ----------------------------------------------------------

        combined = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        y_combined = combined["label"]

        X_combined, _ = extract_split_features(
            combined,
            config,
            feature_cache,
            "train + validation",
            batch_size,
        )

        if model_name == "logistic_regression":
            final_model = make_logistic_regression(best_C)
        else:
            final_model = make_linear_svm(best_C)

        print(
            f"Refitting {model_name} on train + validation..."
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
            )

            final_model.fit(
                X_combined,
                y_combined,
            )

        # ----------------------------------------------------------
        # TEST IS USED ONLY NOW.
        # ----------------------------------------------------------

        test_predictions = final_model.predict(X_test)

        test_metrics = evaluate_predictions(
            y_test,
            test_predictions,
        )

        print(
            f"TEST {model_name}: "
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"weighted_f1={test_metrics['weighted_f1']:.4f}"
        )

        save_classification_report(
            y_test,
            test_predictions,
            model_name,
            "test",
        )

        save_confusion_matrix(
            y_test,
            test_predictions,
            model_name,
            "test",
        )

        final_results.append(
            {
                "model": model_name,
                "feature_config": selected_experiment.name,
                "feature_dimension": feature_dimension(config),
                "selected_C": best_C,
                **test_metrics,
            }
        )

        del combined
        del y_combined
        del X_combined
        del final_model
        del test_predictions
        gc.collect()

    del X_train
    del X_validation
    del X_test
    gc.collect()

    final_df = pd.DataFrame(final_results)

    final_df.to_csv(
        RESULTS_DIR / "final_classical_results.csv",
        index=False,
    )

    final_metadata = {
        "seed": SEED,
        "primary_metric": PRIMARY_METRIC,
        "selected_feature_config": asdict(
            selected_experiment
        ),
        "selection_rule": selection_summary[
            "selection_rule"
        ],
        "final_results": final_df.to_dict(
            orient="records"
        ),
        "test_usage": (
            "The internal test split was evaluated only after "
            "preprocessing and classifier hyperparameters were locked."
        ),
    }

    with open(
        RESULTS_DIR / "final_classical_results.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_metadata,
            handle,
            indent=2,
        )

    return final_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 5 classical baselines."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FEATURE_BATCH_SIZE,
        help=(
            "Number of clips passed to extract_features_batch at once. "
            f"Default: {FEATURE_BATCH_SIZE}."
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    print("=" * 72)
    print("PHASE 5A — CLASSICAL BASELINES")
    print("=" * 72)

    print("Loading existing shared split...")
    split = load_shared_split()

    print("\nSplit counts:")
    print(
        split["split"]
        .value_counts()
        .sort_index()
    )

    print("\nClass counts:")
    print(
        pd.crosstab(
            split["split"],
            split["label"],
        )
    )

    cache_root = RESULTS_DIR / "cache"

    print("\nStage A: preprocessing + classifier validation")
    print("-" * 72)
    print(f"Feature batch size: {args.batch_size}", flush=True)

    _, selection_summary = run_validation_experiment(
        split,
        cache_root,
        args.batch_size,
    )

    selected_name = selection_summary[
        "selected_feature_config"
    ]

    selected_experiment = next(
        experiment
        for experiment in FEATURE_CONFIGS
        if experiment.name == selected_name
    )

    print("\nSelected preprocessing configuration:")
    print(
        json.dumps(
            asdict(selected_experiment),
            indent=2,
        )
    )

    print("\nStage B: locked final evaluation")
    print("-" * 72)

    final_results = run_locked_final_evaluation(
        split,
        selected_experiment,
        selection_summary,
        cache_root,
        args.batch_size,
    )

    print("\n" + "=" * 72)
    print("FINAL CLASSICAL RESULTS")
    print("=" * 72)

    print(
        final_results[
            [
                "model",
                "feature_config",
                "selected_C",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "macro_precision",
                "macro_recall",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved under:")
    print(RESULTS_DIR)

    print("\nPhase 5A complete.")
    print("The official unlabeled Zindi Test set was not used.")


if __name__ == "__main__":
    main()