from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "Subjects"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SENSORS = ["EDA", "TEMP", "BVP", "HR", "IBI", "ACC"]


def load_sensor(subject: str, sensor: str) -> pd.DataFrame:
    """Load a sensor CSV. Assumes files contain data rows only, no metadata rows."""
    path = DATA_DIR / subject / f"{sensor}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sensor file: {path}")
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def clean_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove fully empty columns and interpolate short gaps."""
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    # Interpolation is a simple MVP step; it may not be appropriate for long gaps.
    if not df.empty:
        df = df.interpolate(method="linear", limit_direction="both")
    return df


def process_subject(subject: str, save: bool = True) -> dict[str, pd.DataFrame]:
    processed = {}
    for sensor in SENSORS:
        try:
            raw = load_sensor(subject, sensor)
            cleaned = clean_sensor(raw)
            processed[sensor] = cleaned
            print(f"{subject} | {sensor}: {len(cleaned)} samples, {cleaned.shape[1]} columns")
        except FileNotFoundError as exc:
            print(exc)

    if save:
        output_dir = PROCESSED_DIR / subject
        output_dir.mkdir(parents=True, exist_ok=True)
        for sensor, df in processed.items():
            df.to_csv(output_dir / f"{sensor}.csv", index=False, header=False)

    return processed


if __name__ == "__main__":
    subjects = sorted(p.name for p in DATA_DIR.glob("subject_*") if p.is_dir())
    if not subjects:
        raise SystemExit(f"No subject folders found under {DATA_DIR}")
    process_subject(subjects[0])
    print(f"Processed files saved to: {PROCESSED_DIR / subjects[0]}")
