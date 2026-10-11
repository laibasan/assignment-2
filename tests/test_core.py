"""Substantive mathematical and data-leakage checks; no sklearn helpers."""
import ast
import hashlib
import json
import unittest
import joblib
import numpy as np
from src.common import ROOT, RESULTS, FEATURES, expanded, with_intercept, prepared
from src.prepare import clean, split_indices, fit_scaler
from src.load_data import load_raw
from src.logistic import sigmoid, hypothesis, cost, fit_logistic, predict
from src.metrics import accuracy, confusion_matrix


class CoreTests(unittest.TestCase):
    def test_sigmoid_extremes_and_initial_cost(self):
        np.testing.assert_allclose(sigmoid(np.array([-1000., 0., 1000.])), [0., .5, 1.])
        X = np.array([[1., -1.], [1., 2.]])
        self.assertAlmostEqual(cost(X, np.array([0, 1]), np.zeros(2)), np.log(2))
        self.assertTrue(np.isfinite(cost(X, np.array([1, 0]), np.array([0, 1000]))))
        np.testing.assert_array_equal(predict(X, np.zeros(2)), [1, 1])

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(17)
        X = with_intercept(rng.normal(size=(12, 3))); y = rng.integers(0, 2, 12)
        theta = rng.normal(size=4); epsilon = 1e-6
        analytic = X.T @ (hypothesis(X, theta) - y) / len(y)
        numerical = np.empty(4)
        for j in range(4):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += epsilon; minus[j] -= epsilon
            numerical[j] = (cost(X, y, plus) - cost(X, y, minus)) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)
        # Also check the actual update in fit_logistic from its mandated zero start.
        fitted, _ = fit_logistic(X, y, 0.1, 1)
        np.testing.assert_allclose(fitted, 0.1 * (X.T @ (y - .5)) / len(y))

    def test_manual_metric_orientation(self):
        actual = np.array([0, 0, 0, 1, 1, 1, 1])
        guessed = np.array([0, 1, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(confusion_matrix(actual, guessed), [[2, 1], [1, 3]])
        self.assertAlmostEqual(accuracy(actual, guessed), 5 / 7)
        with self.assertRaises(ValueError): accuracy([], [])
        with self.assertRaises(ValueError): confusion_matrix([0, 2], [0, 1])

    def test_exact_split_and_training_only_scaling(self):
        d = prepared(); df, removed, missing = clean(load_raw(ROOT / "data"))
        self.assertEqual(removed, ["Unnamed: 7", "Unnamed: 8"])
        self.assertFalse(any(missing.values()))
        self.assertEqual(list(df[FEATURES].columns), FEATURES)
        indices = split_indices(len(df))
        for name, idx in zip(["train", "val", "test"], indices):
            np.testing.assert_array_equal(idx, d[name + "_idx"])
        combined = np.concatenate(indices)
        self.assertEqual(len(np.unique(combined)), len(df))
        raw = df[FEATURES].to_numpy()
        for s in ["R", "Q"]:
            all_features = expanded(raw, s)
            m, sd = fit_scaler(all_features[indices[0]])
            np.testing.assert_array_equal(m, d["mean_" + s])
            np.testing.assert_array_equal(sd, d["std_" + s])
            for name, idx in zip(["train", "val", "test"], indices):
                np.testing.assert_allclose((all_features[idx] - m) / sd, d[f"X_{s}_{name}"])
        np.testing.assert_array_equal(expanded(raw[:3], "Q")[:, 6:], raw[:3] ** 2)

    def test_chosen_rates_and_saved_models(self):
        import pandas as pd
        trials = pd.read_csv(RESULTS / "learning_rate_trials.csv")
        for s in ["R", "Q"]:
            model = joblib.load(RESULTS / f"logistic_{s}.joblib")
            valid = trials[(trials.feature_set == s) & trials.strictly_decreasing]
            self.assertEqual(model["alpha"], float(valid.alpha.max()))
            history = pd.read_csv(RESULTS / f"cost_{s}_alpha_{model['alpha']:g}.csv")
            self.assertEqual(len(history), 2001)
            self.assertTrue(np.all(np.diff(history.cost) < 0))
            self.assertAlmostEqual(history.cost.iloc[0], np.log(2))
            np.testing.assert_array_equal(model["theta"], np.load(RESULTS / f"theta_{s}.npy"))

    def test_cached_once_only_results(self):
        result = json.loads((RESULTS / "test_results.json").read_text())
        self.assertEqual(result["evaluation_count"], 1)
        saved = np.load(RESULTS / "test_predictions.npz", allow_pickle=False)
        for i, model in enumerate(result["models"]):
            # Recount stored predictions, without querying any final model again.
            self.assertEqual(sum(sum(row) for row in model["confusion_matrix"]), 2401)
            np.testing.assert_array_equal(confusion_matrix(saved["y_true"], saved[f"model_{i}"]), model["confusion_matrix"])
            self.assertEqual(accuracy(saved["y_true"], saved[f"model_{i}"]), model["accuracy"])
        for filename, expected in result["fingerprints"].items():
            self.assertEqual(hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest(), expected)

    def test_only_allowed_sklearn_import(self):
        for file in (ROOT / "src").glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
                    self.assertEqual(node.module, "sklearn.svm")
                    self.assertEqual([x.name for x in node.names], ["SVC"])
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.startswith("sklearn") for n in node.names))


if __name__ == "__main__": unittest.main()
