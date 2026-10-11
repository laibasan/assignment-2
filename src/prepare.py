"""Task 1: exact row split; train-only population standard deviations."""
import hashlib
import numpy as np
import pandas as pd
from src.load_data import load_raw
from src.common import DATA, RESULTS, FEATURES, Q_FEATURES, ensure_dirs, expanded, write_json, markdown_table


def split_indices(n):
    # These statements implement the exact split prescribed in the manual.
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def clean(raw):
    empty = [c for c in raw if raw[c].isna().all()]
    constant = [c for c in raw if c != "Output (S)" and c not in empty
                and raw[c].nunique(dropna=False) == 1]
    removed = empty + constant
    df = raw.drop(columns=removed).rename(columns={"Output (S)": "fault"}).copy()
    if set(df.columns) != set(FEATURES + ["fault"]):
        raise ValueError(f"Unexpected retained columns: {list(df.columns)}")
    for c in FEATURES + ["fault"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    missing = df.isna().sum().astype(int).to_dict()
    if any(missing.values()):
        raise ValueError(f"Missing measurements or labels: {missing}; refusing to change the prescribed rows")
    if not df.fault.isin([0, 1]).all():
        raise ValueError("fault must contain only 0 and 1")
    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Measurements must be finite")
    df["fault"] = df.fault.astype(int)
    return df, removed, missing


def fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    # A constant training feature contributes zeros and does not divide by zero.
    std = np.where(std == 0, 1.0, std)
    return mean, std


def main():
    ensure_dirs()
    raw = load_raw(DATA)
    preview = f"Shape: {raw.shape}\nColumns: {raw.columns.tolist()}\n\n{raw.head(5).to_string()}\n"
    print(preview, flush=True)
    (RESULTS / "raw_preview.txt").write_text(preview)
    df, removed, missing = clean(raw)
    values = df[FEATURES].to_numpy(dtype=float)
    labels = df.fault.to_numpy(dtype=int)
    split_names = ["train", "val", "test"]
    indices = split_indices(len(df))
    arrays = {}
    splits = {}
    for name, idx in zip(split_names, indices):
        arrays[name + "_idx"] = idx
        arrays["raw_" + name] = values[idx]
        arrays["y_" + name] = labels[idx]
        splits[name] = {"rows": len(idx), "faults": int(labels[idx].sum()),
                        "fault_fraction": float(labels[idx].mean())}
    for feature_set in ["R", "Q"]:
        features = expanded(values, feature_set)
        mean, std = fit_scaler(features[indices[0]])
        arrays["mean_" + feature_set] = mean
        arrays["std_" + feature_set] = std
        for name, idx in zip(split_names, indices):
            arrays[f"X_{feature_set}_{name}"] = (features[idx] - mean) / std
    np.savez_compressed(RESULTS / "prepared.npz", **arrays)
    report = {"raw_rows": len(raw), "raw_columns": len(raw.columns),
              "raw_column_names": list(raw.columns), "removed_columns": removed,
              "missing_after_cleaning": missing, "numeric_features": FEATURES,
              "fault_rows": int(labels.sum()), "no_fault_rows": int((labels == 0).sum()),
              "splits": splits, "seed": 42, "std_ddof": 0,
              "dataset_sha256": hashlib.sha256((DATA / "detect_dataset.csv").read_bytes()).hexdigest()}
    write_json(RESULTS / "data_preparation.json", report)
    rows = [["Raw rows / raw columns", f"{len(raw)} / {len(raw.columns)}"],
            ["Columns removed", ", ".join(removed)],
            ["Missing values after cleaning", ", ".join(f"{k}: {v}" for k, v in missing.items())],
            ["Fault rows / no-fault rows", f"{report['fault_rows']} / {report['no_fault_rows']}"],
            ["Train / validation / test rows", " / ".join(str(splits[n]["rows"]) for n in split_names)],
            ["Fault fractions (train / validation / test)", " / ".join(f"{splits[n]['fault_fraction']:.6f}" for n in split_names)]]
    (RESULTS / "table_1_data.md").write_text("# Table 1 - Data preparation\n\n" + markdown_table(["Item", "Value"], rows) + "\n")
    scaler_rows = []
    for s, names in [("R", FEATURES), ("Q", Q_FEATURES)]:
        for name, m, sd in zip(names, arrays["mean_" + s], arrays["std_" + s]):
            scaler_rows.append({"set": s, "feature": name, "training_mean": m, "training_std": sd})
    pd.DataFrame(scaler_rows).to_csv(RESULTS / "scaling_statistics.csv", index=False)
    print(report, flush=True)


if __name__ == "__main__":
    main()
