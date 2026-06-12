"""
evaluate.py — Evaluation utilities for the heart disease prediction model.
Includes threshold tuning, metric reporting, and plot generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)


def tune_threshold(y_true, y_proba, min_recall: float = 0.90) -> float:
    """
    Find the lowest decision threshold that achieves `min_recall`
    while maximising precision. Returns the optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    candidates = [
        (p, r, t)
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if r >= min_recall
    ]
    if not candidates:
        print(f"Warning: could not achieve {min_recall:.0%} recall. Using 0.5.")
        return 0.5
    best = max(candidates, key=lambda x: x[0])
    print(f"Optimal threshold: {best[2]:.3f} | Precision: {best[0]:.3f} | Recall: {best[1]:.3f}")
    return float(best[2])


def print_report(y_true, y_pred, y_proba, threshold: float = 0.5) -> None:
    """Print full classification report and ROC-AUC."""
    auc = roc_auc_score(y_true, y_proba)
    print(f"\n{'='*50}")
    print(f"Decision threshold : {threshold:.3f}")
    print(f"ROC-AUC            : {auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["No disease", "Disease"]))


def plot_confusion_matrix(y_true, y_pred, save_path: str = "outputs/confusion_matrix.png") -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(y_true, y_proba, save_path: str = "outputs/roc_curve.png") -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — heart disease predictor")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall(y_true, y_proba, threshold: float,
                          save_path: str = "outputs/precision_recall.png") -> None:
    """Plot precision-recall curve with chosen threshold marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall[:-1], precision[:-1], lw=2)
    # Mark the chosen threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(recall[idx], precision[idx], s=80, color="red", zorder=5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
