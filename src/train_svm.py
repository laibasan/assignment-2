"""Task 4: SVC is the only scikit-learn component used in this project."""
from sklearn.svm import SVC
import joblib
import pandas as pd
from src.common import RESULTS, FIGURES, prepared, setup_plot, markdown_table, write_json
from src.metrics import accuracy

CS = [0.1, 1, 10, 100]
GAMMAS = [0.1, 1]


def main():
    d = prepared(); X = d["X_R_train"]; Xv = d["X_R_val"]
    rows = []; best = {}; selected = {}
    # Ascending grid order breaks ties towards smaller C, then smaller gamma.
    for kernel in ["linear", "rbf"]:
        score_best = -1.0
        for C in CS:
            for gamma in ([None] if kernel == "linear" else GAMMAS):
                kwargs = {"kernel": kernel, "C": C}
                if gamma is not None:
                    kwargs["gamma"] = gamma
                model = SVC(**kwargs)
                model.fit(X, d["y_train"])
                score = accuracy(d["y_val"], model.predict(Xv))
                row = {"kernel": kernel, "C": C, "gamma": gamma,
                       "validation_accuracy": score, "support_vectors": int(model.n_support_.sum())}
                rows.append(row); print(row, flush=True)
                if score > score_best:
                    score_best = score; best[kernel] = model; selected[kernel] = row
        joblib.dump(best[kernel], RESULTS / f"svm_{kernel}.joblib")
    joblib.dump({"mean": d["mean_R"], "std": d["std_R"]}, RESULTS / "svm_scaler.joblib")
    pd.DataFrame(rows).to_csv(RESULTS / "svm_validation.csv", index=False)
    write_json(RESULTS / "svm_selected.json", selected)
    (RESULTS / "table_3_svm.md").write_text("# Table 3 - SVM validation\n\n" + markdown_table(
        ["Kernel", "C", "gamma", "Validation accuracy", "Support vectors"],
        [[r["kernel"], r["C"], "-" if r["gamma"] is None else r["gamma"], f"{r['validation_accuracy']:.6%}", r["support_vectors"]] for r in rows]) + "\n")
    plt = setup_plot(); fig, ax = plt.subplots(figsize=(8, 4.5))
    for kernel, gamma, label in [("linear", None, "Linear"), ("rbf", 0.1, "RBF, gamma=0.1"), ("rbf", 1, "RBF, gamma=1")]:
        subset = [r for r in rows if r["kernel"] == kernel and r["gamma"] == gamma]
        ax.plot([r["C"] for r in subset], [100 * r["validation_accuracy"] for r in subset], marker="o", label=label)
    ax.set(xscale="log", xlabel="C (log scale)", ylabel="Validation accuracy (%)", title="Validation alone chooses the SVM settings")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "svm_validation_vs_C.png"); plt.close(fig)


if __name__ == "__main__":
    main()
