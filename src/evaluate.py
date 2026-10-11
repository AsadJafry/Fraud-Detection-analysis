"""Evaluation metrics and plots for imbalanced fraud classification.

PR-AUC (average precision) is the headline metric. Accuracy is deliberately
not part of :func:`evaluate_model`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.plotting import FIGURE_DIR, save_fig

# Fixed colour per model so the same model looks the same in every figure.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def evaluate_model(y_true, y_score, threshold: float = 0.5) -> dict:
    """Score a model on one dataset.

    Parameters
    ----------
    y_true : array-like of 0/1
        True labels.
    y_score : array-like of float
        Predicted fraud probability (or any score where higher = more likely fraud).
    threshold : float
        Scores >= threshold are predicted as fraud.

    Returns
    -------
    dict
        Threshold-free metrics (``pr_auc``, ``roc_auc``), threshold metrics
        (``precision``, ``recall``, ``f1``), the confusion matrix as
        ``[[tn, fp], [fn, tp]]`` plus its four cells, and the dataset's size
        and base rate (the PR-AUC of a random model). All values are plain
        Python numbers so the dict can be saved as JSON.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
    }


def plot_pr_curve(results: dict, title: str = "Precision-recall curve (validation)",
                  name: str = "pr_curve"):
    """Plot precision-recall curves for several models and save the figure.

    Parameters
    ----------
    results : dict
        ``{model name: (y_true, y_score)}``.
    title : str
        Figure title.
    name : str
        File name (without .png) under reports/figures/.

    The dashed line is the base rate of the first dataset, which is the
    PR-AUC a random model would get.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    base_rate = None
    for (label, (y_true, y_score)), color in zip(results.items(), MODEL_COLORS):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.step(recall, precision, where="post", color=color, linewidth=2,
                label=f"{label} (PR-AUC {ap:.3f})")
        if base_rate is None:
            base_rate = float(np.mean(y_true))
    ax.axhline(base_rate, color="0.4", linestyle="--", linewidth=1,
               label=f"Random model (base rate {base_rate:.2%})")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title(title)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=1)
    save_fig(fig, name)
    return fig


def plot_confusion_matrix(cm, title: str, name: str):
    """Plot a 2x2 confusion matrix with counts and row percentages, then save it.

    Parameters
    ----------
    cm : array-like
        ``[[tn, fp], [fn, tp]]``, e.g. ``evaluate_model(...)["confusion_matrix"]``.
    title : str
        Figure title.
    name : str
        File name (without .png) under reports/figures/.

    Row percentages show, for each true class, how its transactions were
    classified (the fraud row's right cell is recall).
    """
    cm = np.asarray(cm)
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    labels = ["legit", "fraud"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1%} of row)", ha="center",
                    va="center", fontsize=10,
                    color="white" if row_pct[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], labels=[f"predicted {l}" for l in labels])
    ax.set_yticks([0, 1], labels=[f"actual {l}" for l in labels])
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_fig(fig, name)
    return fig


def save_metrics(metrics: dict, path) -> Path:
    """Write a metrics dict to JSON (pretty-printed) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
