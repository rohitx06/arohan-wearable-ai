from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" / "Subjects"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SAMPLING_RATES = {"EDA": 4, "TEMP": 4, "BVP": 64, "HR": 1, "ACC": 32}
SENSOR_LABELS = {
    "EDA": "Electrodermal Activity (EDA)",
    "TEMP": "Skin Temperature",
    "BVP": "Blood Volume Pulse (BVP)",
    "HR": "Heart Rate",
    "IBI": "Inter-Beat Interval (IBI)",
    "ACC": "Acceleration",
}


st.set_page_config(
    page_title="AROHAN | Wearable AI",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_subjects():
    if not RAW_DIR.exists():
        return []
    return sorted(p.name for p in RAW_DIR.glob("subject_*") if p.is_dir())


@st.cache_data(show_spinner=False, persist="disk")
def load_sensor(subject, sensor):
    path = RAW_DIR / subject / f"{sensor}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def calculate_quality(df):
    if df is None or df.empty:
        return {"samples": 0, "columns": 0, "missing_percent": 100.0}
    missing = df.isna().sum().sum()
    total = max(df.shape[0] * df.shape[1], 1)
    return {
        "samples": len(df),
        "columns": df.shape[1],
        "missing_percent": round(missing / total * 100, 2),
    }


def signal_features(values, prefix):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    names = ["mean", "std", "min", "max", "median", "range"]
    if len(values) == 0:
        return {f"{prefix}_{name}": np.nan for name in names}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def build_window_features(subject, window_seconds):
    sensors = {
        s: load_sensor(subject, s)
        for s in ["EDA", "TEMP", "BVP", "HR", "ACC", "IBI"]
    }
    durations = []
    for sensor, rate in SAMPLING_RATES.items():
        df = sensors.get(sensor)
        if df is not None and len(df):
            durations.append(len(df) / rate)

    ibi = sensors.get("IBI")
    if ibi is not None and len(ibi) and ibi.shape[1] >= 2:
        ts = ibi.iloc[:, 0].dropna()
        if len(ts):
            durations.append(float(ts.max()))
