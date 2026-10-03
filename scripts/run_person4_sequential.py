"""Person 4 — Transformer encoder (self-attention) over MFCC frame sequences.

Stage A (validation only)
    Greedy search: each stage varies one design factor around the current
    best configuration and keeps the variant with the highest validation
    macro-F1. Early stopping also uses validation macro-F1.

Stage B (locked)
    The selected configuration is evaluated once on the internal test split.
    It is then retrained with two more seeds (same locked configuration, early
    stopping on validation) to report seed variability on test.

The acoustic front end and the sequence preparation (training-only frame
standardization, 30 ms frame stacking) are shared with Person 2's recurrent
model, so architecture is the factor being compared. The official unlabeled
Zindi Test.csv is never read. The shared split is read and never modified.

Run from the repository root:
    python scripts/run_person4_sequential.py            # full run (GPU recommended)
    python scripts/run_person4_sequential.py --quick    # tiny smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from scripts.run_person2_recurrent import (
    AXIS,
    BLUE,
    FRONT_END,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    Data,
    plot_validation_experiments,
    style_matplotlib,
    subsample,
)
from scripts.run_phase5_classical_baselines import load_shared_split
from src.evaluation.metrics import (
    classification_metrics,
    classification_report_frame,
    confusion_frame,
)
from src.models.neural import TransformerConfig, train_transformer
from src.models.recurrent import TrainResult, pick_device, predict_proba
from src.preprocessing.sequences import extract_static_sequences
from src.utils.reproducibility import SHARED_SPLIT_SEED, environment_record

SEED = SHARED_SPLIT_SEED
EXTRA_SEEDS = (43, 44)
PRIMARY_METRIC = "macro_f1"
MODEL_NAME = "transformer"
LABEL = "Transformer model"

BASE = TransformerConfig(name="base")

# Each stage: (stage name, list of (run name, fields to change on the current best)).
# The feed-forward width is kept at 2 x d_model so capacity scales as one factor.
SEARCH_STAGES = [
    ("input_features", [
        ("static_mfcc_only", {"feature_streams": 1}),
        ("mfcc_deltas", {"feature_streams": 3}),
    ]),
    ("pooling", [
        ("mean_pooling", {"pooling": "mean"}),
        ("cls_token_pooling", {"pooling": "cls"}),
    ]),
    ("capacity", [
        ("d_model_64", {"d_model": 64, "ffn_dim": 128}),
        ("d_model_128", {"d_model": 128, "ffn_dim": 256}),
        ("d_model_256", {"d_model": 256, "ffn_dim": 512}),
    ]),
    ("depth", [
        ("layers_1", {"num_layers": 1}),
        ("layers_2", {"num_layers": 2}),
        ("layers_4", {"num_layers": 4}),
    ]),
    ("regularization", [
        ("dropout_0.1", {"dropout": 0.1}),
        ("dropout_0.3", {"dropout": 0.3}),
        ("dropout_0.5", {"dropout": 0.5}),
    ]),
]


def run_one(config: TransformerConfig, data: Data, seed: int, device: torch.device) -> TrainResult:
    seqs, _ = data.sequences(config.feature_streams, config.frame_stack)
    return train_transformer(
        config, seqs["train"], data.labels("train"), seqs["validation"],
        data.labels("validation"), len(data.classes), seed, device,
    )


def search(data: Data, device: torch.device, out: Path, base: TransformerConfig, stages=SEARCH_STAGES):
    """Greedy one-factor-at-a-time search on validation macro-F1."""
    (out / "histories").mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    best_config, best_score = base, -1.0
    for stage, variants in stages:
        print("=" * 72 + f"\nStage: {stage}", flush=True)
        stage_best = None
        for run_name, changes in variants:
            config = replace(best_config, name=run_name, **changes)
            key = config.architecture_key()
            if key in results:
                result, first_name = results[key]
                print(f"  {run_name}: identical to {first_name}, reusing", flush=True)
            else:
                result = run_one(config, data, SEED, device)
                results[key] = (result, run_name)
                first_name = run_name
                result.history.to_csv(out / "histories" / f"{run_name}.csv", index=False)
            score = result.best_val_metrics[PRIMARY_METRIC]
            rows.append({
                "stage": stage, "run": run_name, "same_as": first_name,
                **{k: v for k, v in config.to_dict().items() if k != "name"},
                "best_epoch": result.best_epoch, "epochs_run": len(result.history),
                "n_parameters": result.n_parameters, "train_seconds": round(result.seconds, 1),
                **{f"val_{k}": v for k, v in result.best_val_metrics.items()},
            })
            print(f"  {run_name}: val macro-F1 {score:.4f} (best epoch {result.best_epoch})", flush=True)
            if stage_best is None or score > stage_best[1]:
                stage_best = (config, score)
        if stage_best[1] > best_score:
            best_config, best_score = stage_best
        print(f"  -> carrying forward {best_config.name} ({best_score:.4f})", flush=True)
        pd.DataFrame(rows).to_csv(out / "validation_experiments.csv", index=False)
    selected = results[best_config.architecture_key()][0]
    return best_config, selected, pd.DataFrame(rows)


def evaluate_locked(config, result, data, device, out, figures):
    seqs, stats = data.sequences(config.feature_streams, config.frame_stack)
    classes = data.classes
    outputs = {}
    for name in ("validation", "test"):
        proba = predict_proba(result.model, seqs[name], device)
        y_true = data.labels(name)
        y_pred = proba.argmax(axis=1)
        outputs[name] = (y_true, y_pred, proba)
        frame = data.split.iloc[data.index[name]][["Word_id", "label"]].reset_index(drop=True)
        frame["predicted"] = [classes[i] for i in y_pred]
        frame["correct"] = frame["label"] == frame["predicted"]
        frame["confidence"] = proba.max(axis=1)
        for i, c in enumerate(classes):
            frame[f"p_{c}"] = proba[:, i]
        frame.to_csv(out / f"{name}_predictions.csv", index=False)

    y_true, y_pred, proba = outputs["test"]
    metrics = classification_metrics(y_true, y_pred, proba)
    val_metrics = classification_metrics(*outputs["validation"])
    final = {
        "model": MODEL_NAME, "selected_run": config.name, "seed": SEED,
        "config": config.to_dict(), "front_end": FRONT_END.to_dict(),
        "best_epoch": result.best_epoch, "n_parameters": result.n_parameters,
        "validation": val_metrics, "test": metrics,
        "test_usage": "The internal test split was evaluated only after the configuration was locked on validation.",
    }
    (out / f"final_{MODEL_NAME}_results.json").write_text(json.dumps(final, indent=2))
    pd.DataFrame([{"model": MODEL_NAME, "selected_run": config.name, **metrics}]).to_csv(
        out / f"final_{MODEL_NAME}_results.csv", index=False)
    classification_report_frame(y_true, y_pred, classes, MODEL_NAME, "test").to_csv(
        out / f"test_{MODEL_NAME}_classification_report.csv", index=False)
    confusion_frame(y_true, y_pred, classes).to_csv(out / f"test_{MODEL_NAME}_confusion_matrix.csv")
    (out / "standardizer.json").write_text(json.dumps(stats))
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save(result.model.state_dict(), out / "checkpoints" / "selected_model.pt")

    result.history.to_csv(out / "selected_history.csv", index=False)
    plot_learning_curves(result.history, result.best_epoch, figures / "learning_curves.png")
    plot_confusion(confusion_frame(y_true, y_pred, classes), figures / "test_confusion_matrix.png")
    plot_roc(y_true, proba, metrics["macro_roc_auc"], figures / "test_roc_curves.png")
    return metrics


def seed_robustness(config, first_result, data, device, out):
    seqs, _ = data.sequences(config.feature_streams, config.frame_stack)
    rows = []
    for seed in (SEED, *EXTRA_SEEDS):
        result = first_result if seed == SEED else run_one(config, data, seed, device)
        proba = predict_proba(result.model, seqs["test"], device)
        rows.append({
            "seed": seed, "best_epoch": result.best_epoch,
            "val_macro_f1": result.best_val_metrics["macro_f1"],
            **{f"test_{k}": v for k, v in classification_metrics(
                data.labels("test"), proba.argmax(axis=1), proba).items()},
        })
    frame = pd.DataFrame(rows)
    summary = frame.drop(columns="seed").agg(["mean", "std"]).reset_index(names="seed")
    frame = pd.concat([frame, summary], ignore_index=True)
    frame.to_csv(out / "seed_robustness.csv", index=False)
    return frame


def plot_learning_curves(history: pd.DataFrame, best_epoch: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (train_col, val_col, label) in zip(axes, [
        ("train_loss", "val_loss", "Cross-entropy loss"),
        ("train_accuracy", "val_accuracy", "Accuracy"),
    ]):
        ax.plot(history["epoch"], history[train_col], color=BLUE, label="Train")
        ax.plot(history["epoch"], history[val_col], color=ORANGE, label="Validation")
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(label)
        ax.legend(loc="best")
    axes[1].annotate(f"selected epoch {best_epoch}", xy=(best_epoch, axes[1].get_ylim()[0]),
                     xytext=(4, 6), textcoords="offset points", color=INK_2, fontsize=9)
    fig.suptitle(f"Selected {LABEL.lower()}: learning curves", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    values = matrix.to_numpy()
    image = ax.imshow(values, cmap="Blues", vmin=0)
    ax.grid(False)
    ticks = range(len(matrix))
    ax.set_xticks(ticks, matrix.columns, rotation=60, ha="right")
    ax.set_yticks(ticks, matrix.index)
    threshold = values.max() * 0.6
    for i in ticks:
        for j in ticks:
            if values[i, j]:
                ax.text(j, i, values[i, j], ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > threshold else INK)
    ax.set_xlabel("Predicted word")
    ax.set_ylabel("True word")
    ax.set_title(f"{LABEL}: internal test confusion matrix", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, proba: np.ndarray, macro_auc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    grid = np.linspace(0, 1, 201)
    tprs = []
    for k in range(proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == k, proba[:, k])
        tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(fpr, tpr, color=AXIS, linewidth=1, label="Each word (one-vs-rest)" if k == 0 else None)
    ax.plot(grid, np.mean(tprs, axis=0), color=BLUE, label=f"Macro average (AUC = {macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{LABEL}: internal test ROC", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_vs_classical(out: Path, path: Path) -> None:
    svm = ROOT / "results" / "classical" / "test_linear_svm_classification_report.csv"
    ours = pd.read_csv(out / f"test_{MODEL_NAME}_classification_report.csv")
    if not svm.is_file():
        return
    svm = pd.read_csv(svm)
    skip = {"macro avg", "weighted avg", "accuracy"}
    ours = ours[~ours["class"].isin(skip)].set_index("class")["f1-score"]
    svm = svm[~svm["class"].isin(skip)].set_index("class")["f1-score"].reindex(ours.index)
    order = ours.sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(y + 0.2, ours[order], height=0.38, color=BLUE, label=LABEL)
    ax.barh(y - 0.2, svm[order], height=0.38, color=ORANGE, label="Linear SVM (classical)")
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Internal test F1")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title(f"Per-word F1: {LABEL.lower()} vs best classical baseline", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def accuracy_by_duration(out: Path, path: Path) -> pd.DataFrame | None:
    metadata = ROOT / "results" / "eda" / "audio_metadata_train.csv"
    if not metadata.is_file():
        return None
    durations = pd.read_csv(metadata, usecols=["Word_id", "duration_seconds"])
    preds = pd.read_csv(out / "test_predictions.csv").merge(durations, on="Word_id", how="left")
    bins = [0, 4, 5, 8, np.inf]
    names = ["< 4 s", "4–5 s", "5–8 s", "≥ 8 s (truncated)"]
    preds["duration_bin"] = pd.cut(preds["duration_seconds"], bins=bins, labels=names, right=False)
    table = preds.groupby("duration_bin", observed=False)["correct"].agg(["mean", "size"]).reset_index()
    table.columns = ["duration_bin", "accuracy", "n_clips"]
    table.to_csv(out / "test_accuracy_by_duration.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(table))
    ax.bar(x, table["accuracy"].fillna(0), color=BLUE, width=0.55)
    for xi, (acc, n) in enumerate(zip(table["accuracy"], table["n_clips"])):
        if n:
            ax.text(xi, acc + 0.02, f"{acc:.2f}\n(n={n})", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, table["duration_bin"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{LABEL}: internal test accuracy by clip duration", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: a few clips per class, tiny model, 2 epochs. Writes to transformer_smoke/.")
    args = parser.parse_args()

    style_matplotlib()
    device = pick_device(args.device)
    suffix = f"{MODEL_NAME}_smoke" if args.quick else MODEL_NAME
    out = ROOT / "results" / "models" / suffix
    figures = ROOT / "results" / "figures" / suffix
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | output: {out.relative_to(ROOT)}", flush=True)
    if device.type == "cpu" and not args.quick:
        print("WARNING: a full run on CPU takes days. Use a GPU runtime (see the notebook).", flush=True)

    split = load_shared_split()
    base = BASE
    if args.quick:
        split = subsample(split, {"train": 4, "validation": 2, "test": 2})
        base = replace(BASE, d_model=16, num_heads=2, ffn_dim=32, num_layers=1,
                       max_epochs=2, patience=2, warmup_epochs=1)

    classes = sorted(split["label"].unique())

    started = time.time()
    # Same cache file name and front end as Person 2, so one extraction serves both.
    cache = out / "cache" / "static_mfcc13_8s.npz"
    static = extract_static_sequences(split["Word_id"].tolist(), FRONT_END, cache)
    lengths = np.array([len(s) for s in static])
    print(f"MFCC frames per clip: min {lengths.min()}, median {np.median(lengths):.0f}, max {lengths.max()}",
          flush=True)
    data = Data(split, static, classes)

    stages = SEARCH_STAGES
    if args.quick:
        # Keep the smoke test tiny: the capacity stage would otherwise build full-size models.
        stages = [(s, v) for s, v in SEARCH_STAGES if s != "capacity"]

    config, result, table = search(data, device, out, base, stages)
    (out / "selected_config.json").write_text(json.dumps({
        "selected": config.to_dict(),
        "selection_rule": "Greedy one-factor-at-a-time search; keep the highest validation macro-F1.",
        "selected_val_macro_f1": result.best_val_metrics["macro_f1"],
    }, indent=2))
    plot_validation_experiments(table, config.name, figures / "validation_experiments.png")

    print("=" * 72 + f"\nLocked configuration: {config.name}. Evaluating on internal test.", flush=True)
    metrics = evaluate_locked(config, result, data, device, out, figures)
    plot_per_class_vs_classical(out, figures / "test_per_class_f1_vs_classical.png")
    accuracy_by_duration(out, figures / "test_accuracy_by_duration.png")

    print("Seed robustness (locked configuration, seeds 42/43/44)", flush=True)
    robust = seed_robustness(config, result, data, device, out)

    env = environment_record(SEED)
    env.update({"torch": torch.__version__, "device": str(device), "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "total_minutes": round((time.time() - started) / 60, 1)})
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    print("=" * 72 + "\nFINAL TRANSFORMER RESULT (internal test, seed 42)", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nSeed robustness:\n" + robust.to_string(index=False))
    print("\nThe official unlabeled Zindi Test.csv was not used.")


if __name__ == "__main__":
    main()
