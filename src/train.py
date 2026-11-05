from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import DATA_DIR, PROCESSED_DIR, process_subject
from src.feature_extraction import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_dataset():
    rows = []
    subjects = sorted(p.name for p in DATA_DIR.glob("subject_*") if p.is_dir())

    if not subjects:
        raise RuntimeError(f"No subject folders found under {DATA_DIR}")

    for subject in subjects:
        try:
            data = process_subject(subject)
            features = extract_features(data)
            features["subject"] = subject
            rows.append(features)
            print(f"Extracted features for {subject}")
        except Exception as exc:
            print(f"Skipping {subject}: {exc}")

    if not rows:
        raise RuntimeError("No usable subject features were extracted.")

    dataset = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED_DIR / "all_subject_features.csv", index=False)
    return dataset


def train_model():
    dataset = build_feature_dataset()
    feature_columns = [c for c in dataset.columns if c != "subject"]

    X = dataset[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if len(X) < 2 or X.shape[1] == 0:
        raise RuntimeError("Not enough usable rows/features to train the model.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X)

    results = dataset[["subject"]].copy()
    results["anomaly_score"] = model.decision_function(X)
    results["anomaly_label"] = np.where(
        model.predict(X) == -1, "Potential anomaly", "Typical pattern"
    )

    joblib.dump(
        {"model": model, "feature_columns": X.columns.tolist()},
        MODEL_DIR / "anomaly_model.joblib",
    )
    results.to_csv(PROCESSED_DIR / "anomaly_results.csv", index=False)

    print("\nTraining completed!")
    print(f"Subjects processed: {len(results)}")
    print(f"Model saved to: {MODEL_DIR / 'anomaly_model.joblib'}")
    print(f"Results saved to: {PROCESSED_DIR / 'anomaly_results.csv'}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    train_model()
