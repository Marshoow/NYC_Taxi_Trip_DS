<div align="center">

# 🚕 NYC Taxi Trip Duration

### Predicting New York City taxi ride durations using geospatial, temporal, and engineered features

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-7CB9E8?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)

[![Status](https://img.shields.io/badge/Status-Complete-2E7D32?style=flat-square)](.)
[![R²](https://img.shields.io/badge/Best%20R²-0.871-1565C0?style=flat-square)](.)
[![RMSE](https://img.shields.io/badge/Best%20RMSE-241s-6A1B9A?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Pipeline](#-pipeline)
- [Dataset](#-dataset)
- [Features](#-features)
- [Feature Selection](#-feature-selection)
- [Models](#-models)
- [Bug Fixes](#-bug-fixes)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Team](#-team)

---

## 🎯 Overview

This project tackles the **NYC Taxi Trip Duration** regression challenge — predicting how long a taxi ride will take, in seconds, given only the pickup location, dropoff location, timestamp, and passenger count.

> **Why it matters:** Accurate trip duration prediction powers fare estimation, driver dispatch optimisation, and real-time ETA systems across millions of daily rides.

The pipeline covers the full ML lifecycle: data cleaning → feature engineering → feature selection → model training → evaluation → GUI deployment. Eleven engineering bugs in the original notebook were identified and corrected, yielding a **15+ point R² improvement** over the baseline.

---

## 📊 Results

<div align="center">

| Model | RMSE (seconds) | R² | Rank |
|:---:|:---:|:---:|:---:|
| 🥇 **XGBoost** | **241** | **0.871** | Best |
| 🥈 LightGBM | 248 | 0.867 | 2nd |
| 🥉 Random Forest | 312 | 0.821 | 3rd |
| ⚠️ Buggy Baseline | ~390 | 0.756 | — |

</div>

```
Improvement over baseline
─────────────────────────────────────────────────────
  R²        0.756  ──────────────────────►  0.871   (+15.2 pp)
  RMSE      ~390s  ──────────────────────►    241s  (−38%)
  Cardinal  100% "West" (bug)  ──────────►  4 balanced classes
  Clusters  5 coarse zones  ─────────────►  40 fine-grained zones
```

---

## 🔄 Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Data   │───►│   Cleaning  │───►│  Feature    │───►│  Feature    │
│  1.46M rows │    │  1.37M rows │    │  Engineering│    │  Selection  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                 │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│     GUI     │◄───│  Evaluation │◄───│   Model     │◄──────────┘
│  Deployment │    │  & Metrics  │    │  Training   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 📦 Dataset

- **Source:** [Kaggle — NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)
- **Records:** ~1.46 million trips (Jan – Jun 2016)
- **After cleaning:** ~1.37 million records
- **Target:** `trip_duration` (seconds) → log1p-transformed for training

| Column | Type | Description |
|--------|------|-------------|
| `pickup_datetime` | datetime | Trip start timestamp |
| `pickup_latitude/longitude` | float | Pickup GPS coordinates |
| `dropoff_latitude/longitude` | float | Dropoff GPS coordinates |
| `passenger_count` | int | Number of passengers |
| `OSM_distance` | float | Road-network distance (OpenStreetMap) |
| `trip_duration` | int | **Target** — trip length in seconds |

### Data Cleaning Steps

```python
# All filters applied in order:
1. Type corrections         → datetime parsing, coordinate coercion
2. Haversine distance       → great-circle distance_km computed
3. Duplicate removal        → exact duplicate rows dropped
4. Temporal filter          → 60s < duration < 10,800s (1 min – 3 hrs)
5. Geographic filter        → NYC bounding box (lat 40.47–40.92, lon -74.25 – -73.70)
6. Spatial filter           → 0.1 km < distance < 100 km
7. Speed filter             → 0.5 km/h < speed < 120 km/h
8. OSM NaN drop             → rows with missing OSM_distance removed
9. Passenger imputation     → mode-fill for missing passenger_count
```

---

## ⚙️ Features

### Temporal Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `hour_sin` / `hour_cos` | sin/cos(2π·h/24) | Cyclic encoding — hour 23 and 0 are adjacent |
| `day_of_week` | dt.dayofweek | Thu/Fri ~15% longer than Mon/Tue at same hour |
| `is_weekend` | dayofweek ≥ 5 | Weekends have inverted traffic peak patterns |
| `morning_rush` | Weekday & 07–10 | +20% avg duration vs off-peak (pivot-confirmed) |
| `afternoon_peak` | Weekday & 14–16 | True weekday congestion peak (~1000–1023 s avg) |
| `late_night` | hour ≥ 22 or == 0 | Faster traffic; distinct speed dynamics |
| `month` | dt.month | Seasonal variation (Jan vs June tourism/weather) |

### Geospatial Features

| Feature | Description |
|---------|-------------|
| `distance_km` | Haversine great-circle distance |
| `OSM_distance` | Road-network distance — follows actual driveable routes |
| `manhattan_distance_km` | L1 distance with **corrected** longitude factor: `111 × cos(40.7°) ≈ 84.7 km/°` |
| `cardinal_direction` | N/S/E/W mapping using **degree thresholds** (±45°, ±135°) |
| `pickup_cluster` / `dropoff_cluster` | KMeans (n=**40**) spatial clusters |
| `dist_from_midtown` | Haversine from pickup to Midtown Manhattan (40.7549°N, 73.984°W) |
| `distance_per_passenger` | distance_km / passenger_count — ride-share efficiency proxy |

### Target Transformation

```python
# Right-skewed target → log-normal via log1p
df["log_trip_duration"] = np.log1p(df["trip_duration"])

# All models trained on log scale; back-transformed for evaluation
y_pred_real = np.expm1(model.predict(X_test))
```

> **Why log1p?** Compresses the long tail of outlier trips, reducing disproportionate gradient updates from rare 3-hour rides while preserving the full dataset.

---

## 🔍 Feature Selection

Five complementary techniques applied **exclusively on training data** to prevent leakage:

```
┌─────────────────────────┬────────────────┬───────────────────────────────────────┐
│ Technique               │ Type           │ Key Finding                           │
├─────────────────────────┼────────────────┼───────────────────────────────────────┤
│ Correlation Analysis    │ Filter         │ distance_km ↔ OSM_distance |r| > 0.85 │
│ Mutual Information      │ Filter         │ hour_sin/cos > raw hour (non-linear)  │
│ Random Forest Importance│ Embedded       │ Distance features > 60% of total gain │
│ RFECV (Ridge, 5-fold)   │ Wrapper        │ 11 features optimal for linear model  │
│ Permutation Importance  │ Model-agnostic │ All 20 features contribute positively │
└─────────────────────────┴────────────────┴───────────────────────────────────────┘
```

Results are normalised to [0, 1] and averaged into a `final_score`. Three feature subsets — **core** (8), **extended** (12), **all** (20) — are each benchmarked with Random Forest; the lowest RMSE subset wins.

> ⚠️ `speed_kmh` is explicitly excluded before all selection — it is a direct function of the target (`distance / duration`) and its inclusion constitutes **data leakage**.

---

## 🤖 Models

### LightGBM

```python
lgb.LGBMRegressor(
    n_estimators      = 2000,
    learning_rate     = 0.02,
    num_leaves        = 127,      # was 31 — severely under-fit on 1M rows
    max_depth         = -1,       # leaf-wise growth; num_leaves controls complexity
    min_child_samples = 50,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.9,
    bagging_freq      = 1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    callbacks         = [lgb.early_stopping(100)],
)
# ✅ No StandardScaler — tree models are scale-invariant
```

### XGBoost

```python
xgb.XGBRegressor(
    n_estimators          = 2000,
    learning_rate         = 0.02,
    max_depth             = 8,
    min_child_weight      = 20,
    subsample             = 0.85,
    colsample_bytree      = 0.80,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    gamma                 = 0.05,   # min loss reduction to split
    early_stopping_rounds = 50,
    eval_metric           = "rmse",
)
# Feature importance type: "gain" (more reliable than split count)
```

### Random Forest

```python
RandomForestRegressor(
    n_estimators     = 300,
    max_depth        = 20,
    min_samples_leaf = 10,
    n_jobs           = -1,
)
```

### Imbalance Handling

```python
# Stratified 80/20 split preserves short/medium/long trip proportions
train_test_split(..., stratify=df["tripduration_cat"])

# Categories:  short < 600s  |  medium 600–1200s  |  long > 1200s
```

---

*\* Found in the final notebook version*

</details>

---

## 📁 Project Structure

```
nyc-taxi-trip-duration/
│
├── 📓 NYC_Taxi_Trip_Duration.ipynb   # Main notebook (corrected pipeline)
├── 📄 README.md                      # This file
├── 📊 NYC_Taxi_Report.docx           # Full technical report
│
├── data/
│   └── NYC.csv                       # Raw dataset (download from Kaggle)
│
├── models/
│   ├── lgb_model.pkl                 # Trained LightGBM model
│   ├── xgb_model.pkl                 # Trained XGBoost model
│   ├── kmeans_40.pkl                 # Fitted KMeans (40 clusters)
│   └── coord_scaler.pkl              # StandardScaler for coordinates
│
└── gui/
    └── app.py                        # Streamlit / Gradio prediction interface
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/nyc-taxi-trip-duration.git
cd nyc-taxi-trip-duration

pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Option A — Kaggle CLI
kaggle competitions download -c nyc-taxi-trip-duration
unzip nyc-taxi-trip-duration.zip -d data/

# Option B — opendatasets
pip install opendatasets
python -c "import opendatasets as od; od.download('https://www.kaggle.com/competitions/nyc-taxi-trip-duration')"
```

### 3. Run the notebook

```bash
jupyter notebook NYC_Taxi_Trip_Duration.ipynb
```

### 4. Launch the GUI

```bash
streamlit run gui/app.py
```

### Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter
streamlit          # for GUI
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**NYC Taxi Trip Duration** — Data Science Project 2025/2026

*Built with Python · LightGBM · XGBoost · scikit-learn · pandas · matplotlib*

</div>