"""Export the actual saved models for exact client-side inference.

The RBF SVC expansion is b + sum_i dual_coef_i exp(-gamma ||x-sv_i||^2).
No model is retrained in JavaScript. Parity checks use validation rows only.
"""
import hashlib
import json
import joblib
import numpy as np
from src.common import ROOT, RESULTS, FEATURES, prepared, expanded, with_intercept, write_json
from src.logistic import hypothesis


def main():
    app = ROOT / "app"; app.mkdir(exist_ok=True)
    q = joblib.load(RESULTS / "logistic_Q.joblib")
    svc = joblib.load(RESULTS / "svm_rbf.joblib")
    scaler = joblib.load(RESULTS / "svm_scaler.joblib")
    d = prepared()
    payload = {"schema_version": 1, "feature_order": FEATURES,
               "logistic": {"theta": q["theta"].tolist(), "mean": q["mean"].tolist(), "std": q["std"].tolist()},
               "rbf": {"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist(),
                       "support_vectors": svc.support_vectors_.tolist(), "dual_coef": svc.dual_coef_[0].tolist(),
                       "intercept": float(svc.intercept_[0]), "gamma": float(svc._gamma),
                       "classes": svc.classes_.tolist(), "C": float(svc.C)},
               "test_results": json.loads((RESULTS / "test_results.json").read_text())["models"],
               "dataset_rows": len(d["y_train"]) + len(d["y_val"]) + len(d["y_test"]),
               "source_sha256": {f: hashlib.sha256((RESULTS / f).read_bytes()).hexdigest()
                                  for f in ["logistic_Q.joblib", "svm_rbf.joblib", "svm_scaler.joblib"]}}
    # JSON at full float precision; JS loads a static JS assignment, also works file://.
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (app / "models.js").write_text("window.FAULT_MODELS = " + serialized + ";\n")
    raw = d["raw_val"][:64]
    q_features = (expanded(raw, "Q") - q["mean"]) / q["std"]
    r_features = (raw - scaler["mean"]) / scaler["std"]
    expected = {"raw": raw.tolist(), "logistic_probability": hypothesis(with_intercept(q_features), q["theta"]).tolist(),
                "rbf_margin": svc.decision_function(r_features).tolist(), "rbf_prediction": svc.predict(r_features).tolist()}
    write_json(RESULTS / "web_parity_validation.json", expected)
    # Examples are from the training set, so frontend work does not inspect test rows.
    example_idx = [int(np.flatnonzero(d["y_train"] == y)[0]) for y in [0, 1]]
    examples = [{"label": "No-fault example" if y == 0 else "Fault example",
                 "actual_class": y, "values": d["raw_train"][i].tolist()} for y, i in zip([0, 1], example_idx)]
    (app / "examples.js").write_text("window.FAULT_EXAMPLES = " + json.dumps(examples) + ";\n")
    print(f"Exported {len(svc.support_vectors_)} RBF support vectors and Set-Q theta for browser inference.")


if __name__ == "__main__":
    main()
