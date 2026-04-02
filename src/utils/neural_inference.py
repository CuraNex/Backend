import numpy as np
import pandas as pd
from src.utils.helpers import setup_logger

logger = setup_logger("NeuralInference")

def extract_median(forecasts, model_name):
    """Extract the median/point prediction column from NeuralForecast output."""
    median_col = [c for c in forecasts.columns if "median" in c.lower() or c == model_name]
    if median_col:
        cols = ["unique_id", median_col[0]]
        if "ds" in forecasts.columns:
            cols = ["unique_id", "ds", median_col[0]]
        return forecasts[cols].rename(columns={median_col[0]: "pred"})
    return None

def align_to_split(neural_preds_df, split_df, n_expected):
    """
    Map neural predictions back to the full split index.
    
    Strategy:
    1. Try matching by (unique_id, ds) for exact date alignment
    2. Fall back to per-series mean if dates don't match
    
    Returns: np.array of length n_expected, with 0 for unmatched rows.
    """
    # Build lookup keys from the split DataFrame
    r_vals = split_df["retailer_id"].values[:n_expected]
    s_vals = split_df["sku_id"].values[:n_expected]
    split_uids = np.array([f"{r}_{s}" for r, s in zip(r_vals, s_vals)])
    split_dates = pd.to_datetime(split_df["date"]).values[:n_expected]
    
    result = np.zeros(n_expected)
    
    # Strategy 1: exact (unique_id, ds) match
    if "ds" in neural_preds_df.columns:
        neural_preds_df = neural_preds_df.copy()
        neural_preds_df["ds"] = pd.to_datetime(neural_preds_df["ds"])
        lookup = neural_preds_df.set_index(["unique_id", "ds"])["pred"]
        
        for i in range(n_expected):
            key = (split_uids[i], split_dates[i])
            if key in lookup.index:
                result[i] = lookup[key]
    
    n_date_matched = (result != 0).sum()
    
    # Strategy 2: for unmatched rows, use per-series mean
    if n_date_matched < n_expected:
        avg_by_id = neural_preds_df.groupby("unique_id")["pred"].mean()
        for i in range(n_expected):
            if result[i] == 0 and split_uids[i] in avg_by_id.index:
                result[i] = avg_by_id[split_uids[i]]
    
    n_total_matched = (result != 0).sum()
    logger.info(f"  Aligned {n_total_matched:,}/{n_expected:,} predictions "
                f"({n_date_matched:,} by date, {n_total_matched - n_date_matched:,} by series mean)")
    return result

def generate_rolling_predictions(model, df_history, df_target):
    """Generate full auto-regressive recursive predictions with feature engineering."""
    all_preds = []
    current_history = df_history.copy()
    
    df_target = df_target.copy()
    df_target["date"] = pd.to_datetime(df_target["date"])
    max_target_date = df_target["date"].max()
    
    while True:
        preds = model.predict(current_history)
        all_preds.append(preds)
        
        max_pred_date = pd.to_datetime(preds["ds"]).max()
        
        # The referee must wait until the SLOWEST runner crosses the finish line
        slowest_runner_date = pd.to_datetime(preds.groupby("unique_id")["ds"].max()).min()
        if slowest_runner_date >= max_target_date:
            break
            
        # 1. Borrow only the NEW environment/structure from df_target
        current_max_date = current_history["date"].max()
        history_addition = df_target[(df_target["date"] > current_max_date) & (df_target["date"] <= max_pred_date)].copy()
        
        if history_addition.empty:
            break
        
        # Use all concatenated predictions to guarantee complete override coverage
        master_preds = pd.concat(all_preds)
        
        # 2. Extract model predictions securely
        model_col = [c for c in master_preds.columns if "median" in c.lower() or model.name.lower() in c.lower()]
        pred_col = model_col[0] if model_col else master_preds.columns[-1]
        
        preds_mapped = master_preds[["unique_id", "ds", pred_col]].copy()
        preds_mapped["ds"] = pd.to_datetime(preds_mapped["ds"])
        
        # 3. Merge predictions into history_addition
        history_addition["unique_id"] = [f"{r}_{s}" for r, s in zip(history_addition["retailer_id"], history_addition["sku_id"])]
        
        history_addition = history_addition.merge(
            preds_mapped, 
            left_on=["unique_id", "date"], 
            right_on=["unique_id", "ds"],
            how="left"
        )
        
        # Overwrite quantity_ordered with the AI's prediction
        history_addition["quantity_ordered"] = history_addition[pred_col].fillna(history_addition["quantity_ordered"])
        
        # Clean up temporary merge columns
        history_addition = history_addition.drop(columns=[pred_col, "ds", "unique_id"], errors="ignore")
        
        # 4. Concatenate
        current_history = pd.concat([df_history, history_addition])
        
        # 5. Dynamically Re-engineer Temporal Features! 
        try:
            from src.features.feature_engineering import add_lag_features, add_rolling_features
            group_cols = ["retailer_id", "sku_id"]
            current_history = add_lag_features(current_history, group_cols)
            current_history = add_rolling_features(current_history, group_cols)
            
            features_to_fill = [c for c in current_history.columns if c.startswith("lag_") or c.startswith("rolling_")]
            current_history[features_to_fill] = current_history[features_to_fill].fillna(0)
        except Exception as e:
            logger.warning(f"Could not calculate dynamic features: {e}")
            
    final_preds = pd.concat(all_preds).drop_duplicates(subset=["unique_id", "ds"], keep="last")
    return final_preds
