"""Paths, feature construction, and small output helpers (no ML helpers)."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
Q_FEATURES = FEATURES + [name + "^2" for name in FEATURES]


def ensure_dirs():
    FIGURES.mkdir(parents=True, exist_ok=True)


def expanded(raw, feature_set):
    raw = np.asarray(raw, dtype=float)
    if feature_set == "R":
        return raw
    if feature_set == "Q":
        return np.concatenate([raw, raw * raw], axis=1)
    raise ValueError("feature_set must be R or Q")


def with_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def markdown_table(headers, rows):
    def text(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(text, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(text, row)) + " |" for row in rows],
    ])


def prepared():
    return np.load(RESULTS / "prepared.npz", allow_pickle=False)


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 170,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18,
                         "font.size": 10, "figure.facecolor": "white"})
    return plt
