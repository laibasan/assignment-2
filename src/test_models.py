"""Task 5: one final held-out evaluation, cached with artifact fingerprints.

Rerunning returns the cached result. Changed models/data raise instead of silently
using the test set again. Start a separately documented experiment to reevaluate.
"""
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
from src.common import RESULTS, prepared, with_intercept, markdown_table, write_json
from src.logistic import predict
from src.metrics import accuracy, confusion_matrix


def main():
    files = ["prepared.npz", "logistic_R.joblib", "logistic_Q.joblib", "svm_linear.joblib", "svm_rbf.joblib"]
    fingerprint = {f: hashlib.sha256((RESULTS / f).read_bytes()).hexdigest() for f in files}
    cache = RESULTS / "test_results.json"
    if cache.exists():
        previous = json.loads(cache.read_text())
        if previous["fingerprints"] != fingerprint:
            raise RuntimeError("Artifacts changed after the test evaluation. Refusing a second test evaluation.")
        print("Using the existing once-only test evaluation."); return previous
    d = prepared(); y = d["y_test"]; rows = []
    settings = json.loads((RESULTS / "svm_selected.json").read_text())
    models = [("Logistic regression, Set R", "R", "logistic_R.joblib"),
              ("Logistic regression, Set Q", "Q", "logistic_Q.joblib"),
              (f"Linear SVM (C={settings['linear']['C']:g})", "R", "svm_linear.joblib"),
              (f"RBF SVM (C={settings['rbf']['C']:g}, gamma={settings['rbf']['gamma']:g})", "R", "svm_rbf.joblib")]
    predictions = {}
    for i, (name, s, file) in enumerate(models):
        model = joblib.load(RESULTS / file)
        X = d[f"X_{s}_test"]
        pred = predict(with_intercept(X), model["theta"]) if file.startswith("logistic") else model.predict(X)
        cm = confusion_matrix(y, pred)
        assert int(cm.sum()) == len(y)
        tn, fp, fn, tp = cm.ravel().tolist()
        row = {"model": name, "accuracy": accuracy(y, pred), "TN": tn, "false_alarms": fp,
               "missed_faults": fn, "TP": tp, "confusion_matrix": cm.tolist()}
        rows.append(row); predictions[f"model_{i}"] = pred; print(row, flush=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in rows]).to_csv(RESULTS / "test_results.csv", index=False)
    np.savez_compressed(RESULTS / "test_predictions.npz", y_true=y, row_indices=d["test_idx"], **predictions)
    result = {"evaluation_count": 1, "test_rows": len(y), "fingerprints": fingerprint, "models": rows}
    write_json(cache, result)
    (RESULTS / "table_4_test.md").write_text("# Table 4 - Once-only test results\n\n" + markdown_table(
        ["Model", "Accuracy", "TN", "False alarms (FP)", "Missed faults (FN)", "TP"],
        [[r["model"], f"{r['accuracy']:.6%}", r["TN"], r["false_alarms"], r["missed_faults"], r["TP"]] for r in rows]) + "\n\nConfusion matrices use actual classes as rows and predicted classes as columns: [[TN, FP], [FN, TP]].\n")
    return result


if __name__ == "__main__":
    main()
