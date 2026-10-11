"""Task 2: plots of raw consecutive measurements, never scaled data."""
from src.load_data import load_raw
from src.prepare import clean
from src.common import DATA, RESULTS, FIGURES, ensure_dirs, setup_plot
import numpy as np


def main():
    ensure_dirs()
    plt = setup_plot()
    df, _, _ = clean(load_raw(DATA))
    first = df.iloc[:2000]
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, color in zip(["Ia", "Ib", "Ic"], ["#3976df", "#ee8f2c", "#16a080"]):
        ax.plot(first.index, first[name], label=name, color=color, linewidth=0.85)
    # One rectangle per contiguous fault run, aligned to sample cells.
    fault = first.fault.to_numpy() == 1
    edges = np.diff(np.r_[False, fault, False].astype(int))
    for j, (start, stop) in enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
        ax.axvspan(start - 0.5, stop - 0.5, color="#ef4444", alpha=0.13,
                   label="Fault interval" if j == 0 else None)
    ax.set(xlabel="Row number (consecutive samples)", ylabel="Current (A)", title="Three-phase currents: first 2,000 raw samples")
    ax.legend(loc="upper right", ncol=4); fig.tight_layout()
    fig.savefig(FIGURES / "currents_first_2000.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, color, name in [(0, "#3976df", "No fault"), (1, "#ef7049", "Fault")]:
        subset = df[df.fault == label]
        ax.scatter(subset.Ia, subset.Ib, s=5, alpha=0.35, color=color, label=name, rasterized=True)
    ax.set(xlabel="Ia (A)", ylabel="Ib (A)", title="Raw phase-current geometry")
    ax.legend(markerscale=3); fig.tight_layout(); fig.savefig(FIGURES / "ia_ib_scatter.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(df.Ia.min(), df.Ia.max(), 70)
    for label, color, name in [(0, "#3976df", "No fault"), (1, "#ef7049", "Fault")]:
        ax.hist(df.loc[df.fault == label, "Ia"], bins=bins, alpha=0.6, color=color, label=name)
    ax.set(xlabel="Ia (A)", ylabel="Number of samples", title="Ia distribution by fault label")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "ia_histogram.png"); plt.close(fig)
    (RESULTS / "eda.md").write_text("""# Task 2 - Exploratory data analysis

![First 2000 current samples](figures/currents_first_2000.png)

The three currents oscillate around zero, and the fault-labelled intervals have much larger excursions in the affected phases. Shading marks the labels rather than a threshold inferred from the current. Near a zero crossing, even a fault sample can have a small instantaneous current.

![Ia versus Ib](figures/ia_ib_scatter.png)

Normal samples concentrate near the centre, while faults extend outwards in both positive and negative directions. Some samples overlap near the centre because the input is an instant of an alternating waveform. The geometry is closer to an inner region surrounded by fault regions than two opposite half-planes.

![Ia histogram](figures/ia_histogram.png)

The no-fault distribution is concentrated around small current magnitudes; fault samples have wider positive and negative tails. The shared bin edges make the comparison consistent, and the vertical axis shows counts rather than normalised densities. The overlap explains why a rule using only signed Ia cannot classify every fault.

## Could one straight line separate Ia and Ib?

A single straight line is not a suitable separator: its two sides are half-planes, whereas fault samples occur on both sides of the normal current cluster. A fault increases the magnitude of the alternating current, not consistently its signed value, so both a large positive current and a large negative current can indicate a fault. Squared currents express magnitude regardless of sign and permit a curved boundary in the original measurement space. The six-feature models also use voltage, so this two-dimensional view is explanatory rather than proof about every six-dimensional boundary.
""")


if __name__ == "__main__":
    main()
