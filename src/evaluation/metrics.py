"""
PharmaFlow AI — Evaluation Framework
======================================
Comprehensive metrics for demand forecasting evaluation:
- Point metrics: MAPE, wMAPE, RMSE, Bias
- Stratified evaluation: by tier, category, order pattern, horizon
- Statistical significance: Diebold-Mariano test
- Business metrics: fill rate, inventory impact
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import setup_logger

logger = setup_logger("Evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# POINT METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (excludes zero actuals)."""
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error.
    
    Volume-weighted — higher-volume pairs contribute more to the metric.
    This is the primary business-aligned accuracy metric.
    """
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — penalizes large errors."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Forecast bias: Mean(Predicted - Actual) / Mean(Actual).
    
    Positive = over-forecasting    (leads to excess inventory)
    Negative = under-forecasting   (leads to stockouts)
    """
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        return 0.0
    return float(np.mean(y_pred - y_true) / mean_actual * 100)


def fill_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simulated fill rate: what % of demand would be fulfilled
    if we stocked exactly the predicted quantity.
    
    fill_rate = sum(min(pred, actual)) / sum(actual)
    """
    total_demand = np.sum(y_true)
    if total_demand == 0:
        return 100.0
    fulfilled = np.sum(np.minimum(y_pred, y_true))
    return float(fulfilled / total_demand * 100)


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "wMAPE": wmape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "Fill_Rate": fill_rate(y_true, y_pred),
        "N_samples": len(y_true),
    }


def stratified_evaluation(
    df: pd.DataFrame,
    y_pred_col: str = "prediction",
    y_true_col: str = "quantity_ordered",
    stratify_cols: list = None,
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy across different segments.
    
    Segments:
    - Retailer Tier (A/B/C)
    - Therapeutic Category (critical vs non-critical)
    - Order Pattern (regular / intermittent / sporadic)
    
    Returns a DataFrame with metrics per segment.
    """
    if stratify_cols is None:
        stratify_cols = ["tier", "therapeutic_category", "is_critical"]
    
    results = []
    
    for col in stratify_cols:
        if col not in df.columns:
            continue
        
        for segment_val in df[col].unique():
            mask = df[col] == segment_val
            segment_df = df[mask]
            
            if len(segment_df) == 0:
                continue
            
            y_true = segment_df[y_true_col].values
            y_pred = segment_df[y_pred_col].values
            
            metrics = compute_all_metrics(y_true, y_pred)
            metrics["Stratification"] = col
            metrics["Segment"] = str(segment_val)
            results.append(metrics)
    
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano_test(
    y_true: np.ndarray,
    preds_1: np.ndarray,
    preds_2: np.ndarray,
    loss_fn: str = "squared",
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test for comparing forecast accuracy between two models.
    
    H0: Both models have equal predictive accuracy
    H1: Model accuracies differ
    
    Args:
        y_true: Actual values
        preds_1: Predictions from model 1
        preds_2: Predictions from model 2
        loss_fn: "squared" or "absolute"
        h: Forecast horizon (for HAC variance)
    
    Returns:
        dict with DM statistic, p-value, and interpretation
    """
    y_true = np.asarray(y_true)
    preds_1 = np.asarray(preds_1)
    preds_2 = np.asarray(preds_2)
    
    if loss_fn == "squared":
        e1 = (y_true - preds_1) ** 2
        e2 = (y_true - preds_2) ** 2
    else:
        e1 = np.abs(y_true - preds_1)
        e2 = np.abs(y_true - preds_2)
    
    d = e1 - e2  # loss differential
    n = len(d)
    
    # Mean and variance of loss differential
    d_mean = np.mean(d)
    
    # HAC (Heteroskedasticity and Autocorrelation Consistent) variance estimate
    gamma_0 = np.var(d)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0
        gamma_sum += gamma_k
    
    var_d = (gamma_0 + 2 * gamma_sum) / n
    
    if var_d <= 0:
        return {"dm_statistic": 0, "p_value": 1.0, "significant": False, "better_model": "neither"}
    
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    
    better = "model_1" if dm_stat > 0 else "model_2" if dm_stat < 0 else "neither"
    
    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "better_model": better if p_value < 0.05 else "no_significant_difference",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON REPORT
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison_report(
    y_true: np.ndarray,
    model_predictions: dict,
) -> pd.DataFrame:
    """
    Compare all models side-by-side.
    
    Args:
        y_true: Actual values
        model_predictions: {model_name: predictions_array}
    
    Returns:
        DataFrame with metrics for each model, sorted by wMAPE
    """
    results = []
    
    for model_name, preds in model_predictions.items():
        metrics = compute_all_metrics(y_true, preds)
        metrics["Model"] = model_name
        results.append(metrics)
    
    report = pd.DataFrame(results)
    report = report.sort_values("wMAPE")
    
    # Add rank
    report["Rank"] = range(1, len(report) + 1)
    
    cols = ["Rank", "Model", "MAE", "RMSE", "MAPE", "wMAPE", "Bias", "Fill_Rate", "N_samples"]
    report = report[[c for c in cols if c in report.columns]]
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON REPORT")
    logger.info("=" * 80)
    logger.info("\n" + report.to_string(index=False))
    
    return report


def generate_evaluation_summary(
    y_true: np.ndarray,
    model_predictions: dict,
    df: pd.DataFrame = None,
) -> dict:
    """
    Generate a complete evaluation summary including:
    - Overall comparison
    - Stratified metrics (if df provided)
    - Pairwise DM tests between ensemble and each base model
    """
    summary = {}
    
    # Overall comparison
    summary["comparison"] = model_comparison_report(y_true, model_predictions)
    
    # DM tests: ensemble vs each base model
    if "ensemble" in model_predictions:
        dm_results = {}
        ensemble_preds = model_predictions["ensemble"]
        
        for name, preds in model_predictions.items():
            if name != "ensemble":
                dm = diebold_mariano_test(y_true, preds, ensemble_preds, loss_fn="squared")
                dm_results[name] = dm
                
                sig = "YES ✓" if dm["significant"] else "no"
                logger.info(f"  DM Test: ensemble vs {name} — "
                             f"p={dm['p_value']:.4f}, significant={sig}")
        
        summary["dm_tests"] = dm_results
    
    # Stratified evaluation (if data available)
    if df is not None and "ensemble" in model_predictions:
        df_eval = df.copy()
        df_eval["prediction"] = model_predictions["ensemble"]
        
        strat = stratified_evaluation(df_eval)
        summary["stratified"] = strat
    
    return summary
