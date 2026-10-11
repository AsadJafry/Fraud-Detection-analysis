"""Cost-based decision thresholds.

A model outputs a fraud score; the bank still has to decide above which score
to raise an alert. Each choice trades two costs:

- a **false negative** (missed fraud) costs ``cost_fn``, e.g. the average loss;
- a **false positive** (false alarm) costs ``cost_fp``, e.g. analyst review
  time plus customer friction.

Total cost = cost_fn * FN + cost_fp * FP. The best threshold minimises it.
"""

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = np.round(np.linspace(0.001, 0.999, 999), 3)


def cost_curve(y_true, y_score, cost_fn: float, cost_fp: float,
               thresholds=None) -> pd.DataFrame:
    """Total cost and error counts for each candidate threshold.

    A transaction is flagged when ``score >= threshold`` (the same rule as
    :func:`src.evaluate.evaluate_model`).

    Returns a DataFrame with one row per threshold: ``threshold``, ``tp``,
    ``fp``, ``fn``, ``n_alerts``, ``precision`` (NaN when nothing is flagged),
    ``recall`` and ``total_cost``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.asarray(DEFAULT_THRESHOLDS if thresholds is None else thresholds, dtype=float)

    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    # Rows with score < t are not flagged.
    fn = np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    tp = len(pos) - fn
    n_alerts = tp + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(n_alerts > 0, tp / n_alerts, np.nan)
    recall = tp / len(pos) if len(pos) else np.full(len(thresholds), np.nan)

    return pd.DataFrame({
        "threshold": thresholds,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_alerts": n_alerts,
        "precision": precision,
        "recall": recall,
        "total_cost": cost_fn * fn + cost_fp * fp,
    })


def best_threshold(y_true, y_score, cost_fn: float, cost_fp: float,
                   thresholds=None) -> tuple:
    """Return ``(threshold, row)`` for the cost-minimising threshold.

    When several neighbouring thresholds tie at the minimum cost (common when
    a model produces only a few distinct scores), the middle of the first
    tied run is returned, so the choice is not pushed against the edge of
    the flat region by a tiny shift in scores.
    """
    curve = cost_curve(y_true, y_score, cost_fn, cost_fp, thresholds)
    costs = curve["total_cost"].to_numpy()
    at_min = np.flatnonzero(costs == costs.min())
    run_end = at_min[0]
    while run_end + 1 < len(costs) and costs[run_end + 1] == costs.min():
        run_end += 1
    best_idx = (at_min[0] + run_end) // 2
    row = curve.iloc[best_idx]
    return float(row["threshold"]), row


def cost_of_flagging_nothing(y_true, cost_fn: float) -> float:
    """Total cost if no transaction is ever flagged: every fraud is missed."""
    return float(cost_fn * np.asarray(y_true).astype(int).sum())


def plot_cost_curves(curves: dict, title: str, name: str):
    """Plot total cost vs threshold for one or more models and mark each minimum.

    Parameters
    ----------
    curves : dict
        ``{model name: (cost_curve DataFrame, chosen threshold)}``.
    title, name : str
        Figure title, and file name (without .png) under reports/figures/.

    The y-axis is logarithmic because costs at extreme thresholds (flag
    everything / flag nothing) are orders of magnitude above the minimum.
    """
    import matplotlib.pyplot as plt

    from src.evaluate import MODEL_COLORS
    from src.plotting import save_fig

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, (curve, t_best)), color in zip(curves.items(), MODEL_COLORS):
        ax.plot(curve["threshold"], curve["total_cost"], color=color, linewidth=2, label=label)
        best = curve.loc[(curve["threshold"] - t_best).abs().idxmin()]
        ax.scatter([t_best], [best["total_cost"]], color=color, s=70, zorder=3,
                   edgecolor="white", linewidth=1.5)
        above = best["total_cost"] == 0   # keep a zero-cost label clear of the x-axis
        ax.annotate(f"min at {t_best:.3f}\ncost {best['total_cost']:,.0f}",
                    (t_best, best["total_cost"]), xytext=(8, 10 if above else -4),
                    textcoords="offset points", fontsize=8, color="0.25",
                    va="bottom" if above else "top")
    ax.axvline(0.5, color="0.5", linestyle=(0, (2, 2)), linewidth=1)
    ax.text(0.505, 0.97, "default 0.5", transform=ax.get_xaxis_transform(),
            fontsize=8, color="0.4", va="top")
    if min(curve["total_cost"].min() for curve, _ in curves.values()) > 0:
        ax.set_yscale("log")
    else:  # a perfect threshold (cost 0) cannot be drawn on a log axis
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=0)  # costs are never negative
    ax.set_xlim(0, 1)
    ax.set_xlabel("Decision threshold (flag if score >= threshold)")
    ax.set_ylabel("Total cost (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)
    save_fig(fig, name)
    return fig


def plot_threshold_sensitivity(table: pd.DataFrame, title: str, name: str):
    """Plot the cost-optimal threshold against the FN:FP cost ratio, per model.

    ``table`` needs columns ``model``, ``fn_fp_ratio`` and ``best_threshold``.
    """
    import matplotlib.pyplot as plt

    from src.evaluate import MODEL_COLORS
    from src.plotting import save_fig

    fig, ax = plt.subplots(figsize=(8, 4.8))
    markers = ["o", "s", "^", "D"]
    for (label, part), color, marker in zip(table.groupby("model", sort=False), MODEL_COLORS, markers):
        ax.plot(part["fn_fp_ratio"], part["best_threshold"], color=color, marker=marker,
                markersize=8, linewidth=2, label=label)
    ax.set_xscale("log")
    ratios = sorted(table["fn_fp_ratio"].unique())
    ax.set_xticks(ratios, labels=[f"{r:g}:1" for r in ratios])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cost ratio, missed fraud : false alarm (log scale)")
    ax.set_ylabel("Cost-optimal threshold")
    ax.set_title(title)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9)
    save_fig(fig, name)
    return fig
