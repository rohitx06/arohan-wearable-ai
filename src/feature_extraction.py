import numpy as np
import pandas as pd
from src.preprocessing import process_subject, DATA_DIR, PROCESSED_DIR


def extract_signal_features(signal, prefix: str) -> dict:
    values = np.asarray(signal, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_slope": np.nan,
        }

    slope = 0.0
    if len(values) > 1:
        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_slope": slope,
    }


def extract_features(processed_data: dict) -> pd.Series:
    features = {}

    for sensor in ["EDA", "TEMP", "BVP", "HR"]:
        df = processed_data.get(sensor)
        if df is not None and not df.empty:
            features.update(extract_signal_features(df.iloc[:, 0], sensor))

    acc = processed_data.get("ACC")
    if acc is not None and acc.shape[1] >= 3:
        xyz = acc.iloc[:, :3].to_numpy(dtype=float)
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        if len(xyz):
            magnitude = np.sqrt(np.sum(xyz ** 2, axis=1))
            features.update(extract_signal_features(magnitude, "ACC_magnitude"))
            for i in range(3):
                features.update(extract_signal_features(xyz[:, i], f"ACC_axis_{i+1}"))

    ibi = processed_data.get("IBI")
    if ibi is not None and ibi.shape[1] >= 2:
        # Empatica IBI commonly stores timestamp in col 1 and interval in col 2.
        features.update(extract_signal_features(ibi.iloc[:, 1], "IBI"))

    return pd.Series(features, dtype=float)


if __name__ == "__main__":
    subjects = sorted(p.name for p in DATA_DIR.glob("subject_*") if p.is_dir())
    if not subjects:
        raise SystemExit(f"No subject folders found under {DATA_DIR}")

    rows = []
    for subject in subjects:
        data = process_subject(subject)
        row = extract_features(data)
        row["subject"] = subject
        rows.append(row)
        print(f"Extracted features for {subject}")

    output = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "all_subject_features.csv"
    output.to_csv(path, index=False)
    print(f"\nSaved features to: {path}")
    print(output.head())
