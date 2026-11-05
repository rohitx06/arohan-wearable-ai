# AROHAN Wearable AI MVP

A Streamlit MVP for visualizing Empatica E4 sensor recordings and exploring window-level anomaly detection.

## 1. Project structure

```text
arohan-wearable-ai/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── Subjects/
│   │       ├── subject_01/
│   │       │   ├── EDA.csv
│   │       │   ├── TEMP.csv
│   │       │   ├── BVP.csv
│   │       │   ├── HR.csv
│   │       │   ├── IBI.csv
│   │       │   └── ACC.csv
│   │       └── ...
│   └── processed/
├── models/
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── feature_extraction.py
    ├── train.py
    └── predict.py
```

Put your existing `Subjects` dataset under `data/raw/Subjects/`. The dataset itself is not included in this code bundle.

## 2. Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the venv Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Run preprocessing

```powershell
python -m src.preprocessing
```

This processes the first subject folder it finds and saves cleaned files under `data/processed/<subject>/`.

## 4. Extract subject-level features

```powershell
python -m src.feature_extraction
```

This processes all subject folders and creates `data/processed/all_subject_features.csv`.

## 5. Train the initial subject-level anomaly model

```powershell
python -m src.train
```

Outputs:
- `models/anomaly_model.joblib`
- `data/processed/all_subject_features.csv`
- `data/processed/anomaly_results.csv`

## 6. Run the dashboard

```powershell
streamlit run app.py
```

The dashboard includes overview metrics, sensor exploration, window-level anomaly timeline, feature analysis, data quality, and model information. It extracts window features and fits an Isolation Forest when needed.

## Important assumptions and limitations

- CSV files are assumed to contain numeric measurement rows only, without timestamp/frequency metadata rows.
- Nominal Empatica E4 sampling rates used by the dashboard are EDA 4 Hz, TEMP 4 Hz, BVP 64 Hz, HR 1 Hz, and ACC 32 Hz.
- IBI is assumed to have relative timestamp in column 1 and interval duration in column 2. Verify this against the dataset documentation.
- Window alignment is approximate unless original timestamps are available and verified.
- The current dataset is stress-related, not labelled for coronary artery disease (CAD). Do not describe anomaly outputs as CAD predictions or clinical risk scores.
- Isolation Forest identifies unusual feature patterns; it does not diagnose disease.
- For a reliable evaluation, use a subject-wise held-out test set and a dataset with suitable clinical labels for any future disease-prediction task.
