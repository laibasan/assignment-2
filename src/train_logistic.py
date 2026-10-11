"""Task 3: four learning rates per set, then an independent final fit."""
import numpy as np
import pandas as pd
import joblib
from src.logistic import fit_logistic, predict, cost
from src.metrics import accuracy
from src.common import RESULTS, FIGURES, Q_FEATURES, FEATURES, prepared, with_intercept, setup_plot, markdown_table, write_json

ALPHAS = [0.01, 0.1, 1.0, 10.0]
N_ITERS = 2000


def main():
    d = prepared(); plt = setup_plot(); rows = []; trials = []; final = {}
    for s in ["R", "Q"]:
        X = with_intercept(d[f"X_{s}_train"]); y = d["y_train"]
        Xv = with_intercept(d[f"X_{s}_val"])
        initial = cost(X, y, np.zeros(X.shape[1]))
        fig, ax = plt.subplots(figsize=(8, 4.5)); valid = []
        for alpha in ALPHAS:
            _, history = fit_logistic(X, y, alpha, N_ITERS)
            full = np.r_[initial, history]
            decreasing = bool(np.all(np.diff(full) < 0))
            trials.append({"feature_set": s, "alpha": alpha, "strictly_decreasing": decreasing,
                           "initial_cost": initial, "final_cost": float(history[-1]),
                           "largest_cost_change": float(np.diff(full).max())})
            if decreasing:
                valid.append(alpha)
            pd.DataFrame({"iteration": np.arange(N_ITERS + 1), "cost": full}).to_csv(RESULTS / f"cost_{s}_alpha_{alpha:g}.csv", index=False)
            ax.plot(np.arange(N_ITERS + 1), full, label=f"alpha = {alpha:g}")
            print(f"Set {s}, alpha={alpha:g}, strict decrease={decreasing}, final J={history[-1]:.8f}", flush=True)
        if not valid:
            raise RuntimeError(f"No learning rate strictly decreases for Set {s}")
        chosen = max(valid)
        theta, history = fit_logistic(X, y, chosen, N_ITERS)
        assert np.all(np.diff(np.r_[initial, history]) < 0)
        row = {"feature_set": s, "chosen_alpha": chosen, "final_cost": float(history[-1]),
               "training_accuracy": accuracy(y, predict(X, theta)),
               "validation_accuracy": accuracy(d["y_val"], predict(Xv, theta)), "iterations": N_ITERS}
        rows.append(row); final[s] = row
        joblib.dump({"feature_set": s, "features": FEATURES if s == "R" else Q_FEATURES,
                     "theta": theta, "mean": d[f"mean_{s}"], "std": d[f"std_{s}"],
                     "alpha": chosen, "iterations": N_ITERS}, RESULTS / f"logistic_{s}.joblib")
        np.save(RESULTS / f"theta_{s}.npy", theta)
        names = ["intercept"] + (FEATURES if s == "R" else Q_FEATURES)
        pd.DataFrame({"feature": names, "theta": theta}).to_csv(RESULTS / f"theta_{s}.csv", index=False)
        ax.set(xlabel="Iteration (0 = initial theta)", ylabel="Mean binary cross-entropy J", title=f"Logistic regression: Set {s}")
        ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / f"cost_set_{s}.png"); plt.close(fig)
        print(f"Final Set {s}: {row}\ntheta = {theta}", flush=True)
    pd.DataFrame(trials).to_csv(RESULTS / "learning_rate_trials.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "logistic_results.csv", index=False)
    write_json(RESULTS / "logistic_results.json", final)
    (RESULTS / "table_2_logistic.md").write_text("# Table 2 - Logistic regression\n\n" + markdown_table(
        ["Feature set", "Chosen alpha", "Final J", "Training accuracy", "Validation accuracy"],
        [[r["feature_set"], r["chosen_alpha"], f"{r['final_cost']:.8f}", f"{r['training_accuracy']:.6%}", f"{r['validation_accuracy']:.6%}"] for r in rows]) + "\n")
    theta = np.load(RESULTS / "theta_Q.npy")
    (RESULTS / "table_5_theta_Q.md").write_text("# Table 5 - Learned Set-Q weights\n\n" + markdown_table(
        ["Weight", "Value"], [[n, f"{t:.10f}"] for n, t in zip(["theta_0 (intercept)"] + Q_FEATURES, theta)]) + "\n")


if __name__ == "__main__":
    main()
