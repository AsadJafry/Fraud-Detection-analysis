"""Shared plotting style and figure saving."""

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

# Fixed colours so a class always has the same colour across figures.
LEGIT_COLOR = "#2a78d6"  # blue
FRAUD_COLOR = "#eb6834"  # orange
CLASS_COLORS = {"legit": LEGIT_COLOR, "fraud": FRAUD_COLOR}


def set_style() -> None:
    """Apply a clean, readable matplotlib style used by every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    """Save ``fig`` to reports/figures/<name>.png at 150 dpi and return the path."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
