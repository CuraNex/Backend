"""
CuraNex AI — Central Configuration
All project-wide constants, paths, and hyperparameter defaults.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models_saved"
PLOTS_DIR = PROJECT_ROOT / "plots"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
EVAL_DIR = PROJECT_ROOT / "evaluation"

# Ensure directories exist
for d in [SYNTHETIC_DIR, MODEL_DIR, MLFLOW_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Source Paths ──────────────────────────────────────────────────────────
SKU_EXCEL_PATH = PROJECT_ROOT / "hemas_50_verified_complete.xlsx"
RETAILERS_CSV_PATH = SYNTHETIC_DIR / "retailers.csv"

# ── Data Generation Parameters ─────────────────────────────────────────────────
N_RETAILERS = 500          # Real SLMC-registered pharmacies
N_SKUS = 50                # Real NMRA-registered drugs
N_WEEKS = 104              # 2 years of weekly data
START_DATE = "2024-01-01"  # Synthetic data start

# Retailer distribution (used only for synthetic generation fallback)
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

# ── SKU Configuration (Real NMRA data) ────────────────────────────────────────

# 14 real therapeutic categories from hemas_50_verified_complete.xlsx
THERAPEUTIC_CATEGORIES = [
    "Antibiotic", "Antipyretic", "Antihistamine", "Antileukotriene",
    "Respiratory", "Cardiovascular", "Diabetes", "Analgesic/NSAID",
    "Gastrointestinal", "Antiemetic", "Corticosteroid", "Antifungal",
    "Neuropathic/Pain", "Immunomodulator",
]

# Drug groups — drives demand pattern behavior
SKU_GROUPS = {
    "A": "Spreading Disease (Infectious/Outbreak)",
    "B": "Non-Spreading (Chronic)",
    "C": "Mixed-Use (Both Contexts)",
}

PRICE_BANDS = ["Low", "Medium", "High"]

# Dosage form → group mapping (for retailer-type demand crosses)
DOSAGE_FORM_GROUPS = {
    # Oral solids — available everywhere
    "TABLETS": "oral_solid",
    "FILM COA TABLETS": "oral_solid",
    "FILM COATED TABLETS": "oral_solid",
    "TABLET": "oral_solid",
    # Capsules — available everywhere
    "HARD GELATINE CAPSULES": "capsule",
    "CAPSULES": "capsule",
    # Injectables — hospital/chain only
    "INJECTION": "injectable",
    "POWDER FOR INJECTION": "injectable",
    # Nebulizing/inhaled — hospital/chain primarily
    "NEBULIZING SOLUTION": "nebulizing",
    # Oral liquids — available everywhere
    "ORAL SOLUTION": "oral_liquid",
}

# Supplier lead time by source country (days)
COUNTRY_LEAD_TIMES = {
    "INDIA": 45,
    "BANGLADESH": 50,
    "PAKISTAN": 55,
    "INDONESIA": 60,
    "CHINA": 70,
    "SWEDEN": 90,
    "ITALY": 90,
    "FRANCE": 90,
    "GERMANY": 90,
    "UNITED STATES": 85,
    "BELGIUM": 90,
}

# ── Climate Zones (for district-level rainfall) ───────────────────────────────

# Sri Lanka climate zone classification
WET_ZONE_DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Galle", "Ratnapura", "Kegalle",
    "Nuwara Eliya",
]
DRY_ZONE_DISTRICTS = [
    "Jaffna", "Kilinochchi", "Mullaitivu", "Mannar", "Vavuniya",
    "Trincomalee", "Batticaloa", "Ampara", "Hambantota", "Polonnaruwa",
    "Anuradhapura", "Puttalam",
]
INTERMEDIATE_ZONE_DISTRICTS = [
    "Kandy", "Matale", "Kurunegala", "Badulla", "Monaragala", "Matara",
]

# Rainfall intensity per zone per season (0-100 scale)
# Seasons defined by ISO week ranges
RAINFALL_SEASONS = {
    # SW Monsoon: May-Sep (weeks 18-39)
    "sw_monsoon": {"weeks": list(range(18, 40)), "wet": 80, "dry": 18, "inter": 50},
    # NE Monsoon: Dec-Feb (weeks 49-52 + 1-8)
    "ne_monsoon": {"weeks": list(range(49, 53)) + list(range(1, 9)), "wet": 50, "dry": 70, "inter": 50},
    # Inter-monsoon 1: Mar-Apr (weeks 9-17)
    "inter_1": {"weeks": list(range(9, 18)), "wet": 50, "dry": 22, "inter": 40},
    # Inter-monsoon 2: Oct-Nov (weeks 40-48)
    "inter_2": {"weeks": list(range(40, 49)), "wet": 70, "dry": 50, "inter": 50},
}

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
