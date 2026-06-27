"""
evaluator.py — Evaluation pipeline:
  - Accuracy, F1 (weighted), Confusion Matrix
  - Classification Report
  - Plot: training curves + confusion matrix
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)

from config import DEVICE, PLOT_DIR, LOG_DIR

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


# ── Collect predictions ───────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].numpy()

        out    = model(input_ids, attn_mask)
        preds  = out["logits"].argmax(dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Compute & print metrics ───────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    split: str = "Test",
) -> dict:
    print(f"\n{'─'*50}")
    print(f"  Evaluation on: {split} set")
    print(f"{'─'*50}")

    preds, labels = predict(model, loader)

    acc      = accuracy_score(labels, preds)
    f1_w     = f1_score(labels, preds, average="weighted")
    cm       = confusion_matrix(labels, preds)
    report   = classification_report(labels, preds, target_names=LABEL_NAMES, digits=4)

    print(f"  Accuracy (weighted): {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (weighted)      : {f1_w:.4f}")
    print(f"\n{report}")

    results = {
        "split":        split,
        "accuracy":     acc,
        "f1_weighted":  f1_w,
        "confusion_matrix": cm.tolist(),
    }

    # Save results
    out_path = os.path.join(LOG_DIR, f"{split.lower()}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")

    return results


# ── Plot: Training Curves ─────────────────────────────────────────────────────
def plot_training_curves(history: dict, save: bool = True) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc", color="#2196F3")
    ax.plot(epochs, history["val_acc"],   label="Val Acc",   color="#F44336")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=0.3)

    # Learning Rate
    ax = axes[2]
    ax.plot(epochs, history["lr"], color="#4CAF50")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {path}")
    plt.close()


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(cm: list | np.ndarray, split: str = "Test", save: bool = True) -> None:
    cm = np.array(cm)
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Confusion Matrix — {split} Set", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalized"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, f"confusion_matrix_{split.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {path}")
    plt.close()
