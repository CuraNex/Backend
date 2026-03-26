"""
PharmaFlow AI — Central Configuration
All project-wide constants, paths, and hyperparameter defaults.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models_saved"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# Ensure directories exist
for d in [SYNTHETIC_DIR, MODEL_DIR, MLFLOW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Generation Parameters ─────────────────────────────────────────────────
N_RETAILERS = 200          # Reduced for MVP iteration speed (scale to 2300 for prod)
N_SKUS = 50                # Reduced for MVP (scale to 500 for prod)
N_WEEKS = 104              # 2 years of weekly data
START_DATE = "2024-01-01"  # Synthetic data start

# Retailer distribution
RETAILER_TYPES = ["hospital_attached", "standalone", "chain_outlet"]
RETAILER_TYPE_PROBS = [0.20, 0.60, 0.20]

RETAILER_TIERS = ["A", "B", "C"]
RETAILER_TIER_PROBS = [0.20, 0.50, 0.30]

SRI_LANKA_DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Mullaitivu", "Vavuniya", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle"
]

# SKU categories (ATC Level-2 simplified)
THERAPEUTIC_CATEGORIES = [
    "antibiotics", "cardiovascular", "analgesics", "antidiabetics",
    "respiratory", "gastrointestinal", "dermatological", "vitamins_supplements",
    "antipyretics", "antihistamines", "antihypertensives", "otc_general"
]

BRAND_TYPES = ["branded", "generic", "otc"]
BRAND_TYPE_PROBS = [0.40, 0.40, 0.20]

PRICE_BANDS = ["low", "medium", "high"]
PRICE_BAND_PROBS = [0.35, 0.45, 0.20]

# ── Feature Engineering ────────────────────────────────────────────────────────
LAG_WEEKS = [1, 2, 3, 4, 13, 52]
ROLLING_WINDOWS = [4, 13]
FORECAST_HORIZONS = [1, 2, 4, 8, 13]  # weeks ahead

# Train / Validation / Test split ratios (temporal)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Model Hyperparameters (Defaults — Optuna will override) ────────────────────

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

LIGHTGBM_QUANTILE_PARAMS = {
    "objective": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}

# TFT / N-BEATS (NeuralForecast)
TFT_PARAMS = {
    "input_size": 52,          # lookback window in weeks
    "h": 13,                   # max forecast horizon
    "hidden_size": 64,
    "n_head": 4,
    "learning_rate": 1e-3,
    "max_steps": 500,
    "batch_size": 64,
    "windows_batch_size": 256,
    "scaler_type": "robust",
    "random_seed": 42,
}

NBEATS_PARAMS = {
    "input_size": 52,
    "h": 13,
    "n_blocks": [3, 3],        # trend + seasonality stacks
    "mlp_units": [[256, 256], [256, 256]],
    "learning_rate": 1e-3,
    "max_steps": 500,
    "batch_size": 64,
    "windows_batch_size": 256,
    "scaler_type": "robust",
    "random_seed": 42,
}

# Ensemble
ENSEMBLE_META_LEARNER = "ridge"  # "ridge" or "lightgbm"
ENSEMBLE_RIDGE_ALPHA = 1.0

# Cold Start
COLD_START_THRESHOLD_WEEKS = 12
COLD_START_BLEND_WEEKS = 24
N_CLUSTERS = 15

# ── Optuna ─────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 30
OPTUNA_CV_FOLDS = 3  # time-series expanding window folds

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "CuraNex"
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_DIR.as_posix()}"

# ── Sri Lankan Calendar Events (ISO week approximations) ───────────────────────
SRI_LANKA_HOLIDAYS = {
    "sinhala_tamil_new_year": [15, 16],     # mid-April (weeks 15-16)
    "vesak": [20, 21],                       # May full moon (weeks 20-21)
    "poson": [24, 25],                       # June full moon
    "esala": [30, 31],                       # July/August
    "christmas_new_year": [52, 1],           # Dec-Jan
}

# Dengue season: peaks June-November (weeks 22-48)
DENGUE_PEAK_WEEKS = list(range(22, 49))
# Respiratory season: peaks Dec-Feb (weeks 48-52, 1-8)
RESPIRATORY_PEAK_WEEKS = list(range(48, 53)) + list(range(1, 9))

# ── Random Seed ────────────────────────────────────────────────────────────────
SEED = 42
