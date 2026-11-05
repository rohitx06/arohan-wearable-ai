from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "anomaly_model.joblib"


def predict_features(features: pd.DataFrame) -> pd.DataFrame:
    """Predict anomalies from a DataFrame containing the trained feature columns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    columns = bundle["feature_columns"]

    X = features.reindex(columns=columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    output = features.copy()
    output["anomaly_score"] = model.decision_function(X)
    output["anomaly_label"] = np.where(
        model.predict(X) == -1, "Potential anomaly", "Typical pattern"
    )
    return output
