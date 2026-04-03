import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from src.utils.helpers import setup_logger
from src.features.feature_engineering import build_features, add_lag_features, add_rolling_features

logger = setup_logger("ForecastPipeline")

def prepare_data(df):
    from src.features.feature_engineering import get_feature_columns
    cols = get_feature_columns()
    X = df[cols["all"]].copy()
    return X, None

def map_neural_to_split(neural_preds_df, target_uids, target_dates):
    neural_preds_df = neural_preds_df.copy()
    neural_preds_df["ds"] = pd.to_datetime(neural_preds_df["ds"])
    pred_col = [c for c in neural_preds_df.columns if "median" in c.lower() or c not in ["unique_id", "ds"]][-1]
    
    lookup = neural_preds_df.set_index(["unique_id", "ds"])[pred_col]
    
    results = np.zeros(len(target_dates))
    missing_idx = []
    for i in range(len(target_dates)):
        key = (target_uids[i], target_dates[i])
        if key in lookup.index:
            results[i] = lookup[key]
        else:
            missing_idx.append(i)
            
    if missing_idx:
        avg_lookup = neural_preds_df.groupby("unique_id")[pred_col].mean()
        for i in missing_idx:
            uid = target_uids[i]
            if uid in avg_lookup.index:
                results[i] = avg_lookup[uid]
                
    return results

def main():
    logger.info("Starting Forecasting Pipeline (Historical Fusion + Recursive Future Inference)")
    forecast_dir = cfg.DATA_DIR / "forecast"
    
    # 1. Combine Historical and Future Databases
    logger.info("Fusing databases to preserve historical lag calculations...")
    
    trans_hist = pd.read_csv(cfg.SYNTHETIC_DIR / "transactions.csv", parse_dates=["date"])
    cal_hist = pd.read_csv(cfg.SYNTHETIC_DIR / "calendar.csv", parse_dates=["date"])
    health_hist = pd.read_csv(cfg.SYNTHETIC_DIR / "health_signals.csv", parse_dates=["date"])
    retailers = pd.read_csv(cfg.SYNTHETIC_DIR / "retailers.csv")
    skus = pd.read_csv(cfg.SYNTHETIC_DIR / "skus.csv")
    
    trans_fut = pd.read_csv(forecast_dir / "transactions.csv", parse_dates=["date"])
    cal_fut = pd.read_csv(forecast_dir / "calendar.csv", parse_dates=["date"])
    health_fut = pd.read_csv(forecast_dir / "health_signals.csv", parse_dates=["date"])
    
    trans_combined = pd.concat([trans_hist, trans_fut], ignore_index=True)
    cal_combined = pd.concat([cal_hist, cal_fut], ignore_index=True)
    health_combined = pd.concat([health_hist, health_fut], ignore_index=True)
    
    max_hist_date = trans_hist["date"].max()
    future_dates = sorted(trans_fut["date"].unique())
    
    # 2. Rebuild Feature System Globally
    logger.info("Executing global feature engineering build-out...")
    features_df = build_features(
        transactions=trans_combined,
        retailers=retailers,
        skus=skus,
        calendar=cal_combined,
        health_signals=health_combined,
        drop_na_lags=False
    )
    
    features_df.loc[features_df["date"] > max_hist_date, "split"] = "future"
    
    # 2.5 Hard Memory Flush
    # Clear out all raw loaded databases to free up GBs of system RAM before Neural Networks load
    del trans_combined, cal_combined, health_combined, trans_hist, cal_hist, health_hist, trans_fut, cal_fut, health_fut
    import gc
    gc.collect()
    
    # 3. Load Models
    models = {}
    for mf in cfg.MODEL_DIR.glob("*.pkl"):
        try:
            with open(mf, "rb") as f:
                models[mf.stem] = pickle.load(f)
        except Exception:
            pass
            
    if "ensemble_model" not in models:
        logger.error("Ensemble not loaded. Cannot proceed.")
        return
        
    current_df = features_df # No deep copy needed, saves 1GB+
    current_df["date"] = pd.to_datetime(current_df["date"])
    
    all_ensemble_preds = []
    group_cols = ["retailer_id", "sku_id"]
    
    # 4. Step-by-Step Recursive Forecast Loop
    for i, week_dt in enumerate(future_dates):
        logger.info(f"--- Forecasting Week {i+1}/{len(future_dates)}: {week_dt.date()} ---")
        
        target_indices = current_df["date"] == week_dt
        target_slice = current_df[target_indices].copy()
        target_uids = [f"{r}_{s}" for r, s in zip(target_slice["retailer_id"], target_slice["sku_id"])]
        target_dts = target_slice["date"].values
        
        X_target, _ = prepare_data(target_slice)
        
        week_preds = {}
        for m_name in ["lgbm_model", "xgb_model"]:
            if m_name in models:
                week_preds[m_name.split("_")[0]] = models[m_name].predict(X_target)
                
        if "seasonal_naive_model" in models:
            week_preds["snaive"] = models["seasonal_naive_model"].predict(target_slice)
            
        # Cap history to 52 weeks to evade neural memory exhaustion
        neural_history_start = week_dt - pd.Timedelta(weeks=52)
        neural_history_mask = (current_df["date"] < week_dt) & (current_df["date"] >= neural_history_start)
        neural_history = current_df[neural_history_mask]
        for n_name, n_key in [("tft_model", "tft"), ("nbeats_model", "nbeats")]:
            if n_name in models:
                try:
                    preds = models[n_name].predict(neural_history)
                    week_preds[n_key] = map_neural_to_split(preds, target_uids, target_dts)
                except Exception as e:
                    fallback_mean = (week_preds.get("lgbm", np.zeros(len(target_slice))) + week_preds.get("xgb", np.zeros(len(target_slice)))) / 2
                    week_preds[n_key] = fallback_mean
                
        # The Ensemble Meta-Learner requires a dictionary of base predictions and segment features to predict
        segment_feats = target_slice[["tier_encoded", "therapeutic_category_encoded"]].reset_index(drop=True)
        # Filter out snaive since it isn't part of the core ensemble inputs
        base_preds_dict = {k: v for k, v in week_preds.items() if k != "snaive"}
        ensembled = models["ensemble_model"].predict(base_preds_dict, segment_feats)
        
        target_slice["pred_ensemble"] = ensembled
        base_names = ["lgbm", "xgb", "tft", "nbeats"]
        for b in base_names:
            if b in week_preds:
                target_slice[f"pred_{b}"] = week_preds[b]
        
        all_ensemble_preds.append(target_slice[["retailer_id", "sku_id", "date", "quantity_ordered", "pred_ensemble", "pred_lgbm", "pred_xgb", "pred_tft", "pred_nbeats"]])
        
        # Inject predictions back into system
        current_df.loc[current_df["date"] == week_dt, "quantity_ordered"] = ensembled
        
        # Dynamically recalculate context strictly to enforce auto-regressive feedback logic into the future!
        if week_dt != future_dates[-1]:
             # MEMORY PATCH: Truncate to a lightweight view window before recalculating!
             # This prevents Windows MemoryError on 500,000+ rows while still keeping enough history for lag_13w
             cutoff_dt = week_dt - pd.Timedelta(weeks=15)
             calc_df = current_df[current_df["date"] >= cutoff_dt].copy()
             
             import gc
             gc.collect()
             
             from src.features.feature_engineering import add_lag_features, add_rolling_features, add_order_pattern_features, add_seasonal_index
             calc_df = add_lag_features(calc_df, group_cols)
             calc_df = add_rolling_features(calc_df, group_cols)
             calc_df = add_order_pattern_features(calc_df, group_cols)
             calc_df = add_seasonal_index(calc_df, group_cols)
             
             # Map updated predictive lags back to the main un-truncated dataframe
             update_cols = [c for c in calc_df.columns if c.startswith("lag_") or c.startswith("rolling_") or c in ["days_since_last_order", "order_frequency_13w", "growth_rate_6m", "seasonal_index"]]
             
             # Overwrite ONLY the targeted newly predicted values into the overarching timeline
             mask = current_df["date"] >= cutoff_dt
             current_df = current_df.set_index(["retailer_id", "sku_id", "date"])
             calc_df = calc_df.set_index(["retailer_id", "sku_id", "date"])
             
             current_df.update(calc_df[update_cols])
             current_df = current_df.reset_index()
             
             # Protect regressions spanning missing older history values
             features_to_fill = [c for c in current_df.columns if c.startswith("lag_") or c.startswith("rolling_") or c in ["order_frequency_13w", "growth_rate_6m", "seasonal_index"]]
             current_df[features_to_fill] = current_df[features_to_fill].fillna(0)

    # 5. Extract Pure Future Predictions
    final_future_df = pd.concat(all_ensemble_preds, ignore_index=True)
    
    # Save purely predicted rows specifically to evaluation matrix directory as requested
    future_pred_path = cfg.EVAL_DIR / "forecasts.csv"
    final_future_df.to_csv(future_pred_path, index=False)
    
    # 6. Save the FULL Continuous Dataset exactly as requested
    full_dataset_path = forecast_dir / "features.csv"
    current_df.to_csv(full_dataset_path, index=False)
    
    logger.info(f"Success! Pure predictive array saved -> {future_pred_path}")
    logger.info(f"Success! Unbroken Full Feature Array  -> {full_dataset_path}")

if __name__ == "__main__":
    main()
