"""Reproducible learning experiment: run this in BITS Virtual Lab, not inside the app."""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from core import FEATURES, TARGET, evaluate

ROOT = Path(__file__).resolve().parent
SEED = 2026


def main():
    source = ROOT / "data" / "spambase.data"
    if not source.exists():
        raise FileNotFoundError("Extract the official UCI spambase.zip into data/ first. See STUDY_GUIDE.md.")
    frame = pd.read_csv(source, header=None, names=FEATURES + [TARGET])
    assert frame.shape == (4601, 58), "Unexpected source dimensions"
    assert frame.notna().all().all() and np.isfinite(frame.to_numpy()).all()
    # Conflicting labels for identical feature vectors cannot be resolved objectively.
    # Remove all such vectors, then deduplicate the remaining examples BEFORE splitting.
    conflict_mask = frame.groupby(FEATURES, dropna=False)[TARGET].transform("nunique") > 1
    unambiguous = frame.loc[~conflict_mask]
    clean = unambiguous.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    train, test = train_test_split(clean, test_size=0.2, stratify=clean[TARGET], random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    train_hashes = set(pd.util.hash_pandas_object(train[FEATURES], index=False))
    test_hashes = set(pd.util.hash_pandas_object(test[FEATURES], index=False))
    assert train_hashes.isdisjoint(test_hashes), "Identical feature rows crossed the split"
    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    test.to_csv(ROOT / "test_data.csv", index=False)
    train.to_csv(ROOT / "data" / "train_data.csv", index=False)
    # Evaluate exactly the CSV precision the app will read.
    test = pd.read_csv(ROOT / "test_data.csv")
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11, weights="distance")),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", gamma="scale")),
    }
    artifacts, records = {}, []
    for name, model in models.items():
        started = perf_counter()
        model.fit(train[FEATURES], train[TARGET])
        elapsed = perf_counter() - started
        metrics, pred, score = evaluate(model, test)
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, ROOT / "model" / filename, compress=3)
        restored = joblib.load(ROOT / "model" / filename)
        np.testing.assert_array_equal(pred, restored.predict(test[FEATURES]))
        artifacts[name] = filename
        records.append({"ML Model Name": name, **metrics})
        pd.DataFrame({"actual": test[TARGET], "predicted": pred, "score": score}).to_csv(
            ROOT / "results" / (filename.replace(".joblib", "_predictions.csv")), index=False)
        print(f"{name}: F1={metrics['F1']:.4f}, AUC={metrics['AUC']:.4f}, train={elapsed:.2f}s", flush=True)
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "streamlit", "joblib"]}
    manifest = {
        "dataset_url": "https://archive.ics.uci.edu/dataset/94/spambase",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
        "source_rows": len(frame), "conflicting_label_rows_removed": int(conflict_mask.sum()),
        "duplicate_rows_removed": len(unambiguous) - len(clean), "clean_rows": len(clean),
        "train_rows": len(train), "test_rows": len(test), "seed": SEED,
        "feature_count": len(FEATURES), "features": FEATURES, "target": TARGET,
        "source_class_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().items()},
        "train_class_counts": {str(k): int(v) for k, v in train[TARGET].value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in test[TARGET].value_counts().items()},
        "models": artifacts, "python": platform.python_version(), "versions": versions,
        "method": "One stratified holdout; fixed hyperparameters; no test-set tuning. Winner is descriptive, not a model-selection claim.",
    }
    (ROOT / "model" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(ROOT / "results" / "metrics.csv", index=False)
    (ROOT / "requirements.txt").write_text("\n".join(f"{p}=={v}" for p, v in versions.items()) + "\n", encoding="utf-8")
    from write_readme import write_readme
    write_readme(ROOT)
    print(f"Rows: source={len(frame)}, clean={len(clean)}, train={len(train)}, test={len(test)}")
    print("Saved five models, test CSV, measured results, README, and exact dependency versions.")


if __name__ == "__main__":
    main()
