"""Shared evaluation metrics.

Every model should be scored with the same functions and the same held-out
test set. Validation metrics guide model and tuning decisions. Test metrics
are for final comparison. The metric keys match the classical baselines in
``results/classical/`` so tables can be joined directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
) -> dict:
    """Accuracy, macro/weighted F1, macro precision/recall, and macro ROC-AUC.

    ``y_true``/``y_pred`` are integer class indices. ``proba`` has one column
    per class index; when given, one-vs-rest macro ROC-AUC is added.
    """
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }
    if proba is not None:
        metrics["macro_roc_auc"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                          labels=np.arange(proba.shape[1]))
        )
    return metrics


def classification_report_frame(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], model: str, stage: str
) -> pd.DataFrame:
    """Per-class report in the same layout as the classical baseline CSVs."""
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = [
        {"model": model, "stage": stage, "class": label, **values}
        for label, values in report.items()
        if isinstance(values, dict)
    ]
    return pd.DataFrame(rows)


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
