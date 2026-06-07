"""
CuraNex AI — End-to-End Training Pipeline
===============================================
Orchestrates the full workflow:
1. Data loading (or generation)
2. Feature engineering
3. Base model training (LightGBM, XGBoost, TFT, N-BEATS)
4. Out-of-fold prediction generation
5. Stacking meta-learner training
6. Cold-start cluster fitting
7. Evaluation and model comparison report
8. Model artifact saving

This is the single entry point for training the entire system.
"""

import sys
import time
import pickle
from pathlib import Path
import mlflow

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from src.utils.helpers import setup_logger, set_seed, load_csv, ensure_dir
from src.data_generation.generate_synthetic import generate_all
from src.features.feature_engineering import build_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.baselines import SeasonalNaive, ExponentialSmoothing
from src.models.ensemble import StackingEnsemble, simple_average_ensemble
from src.cold_start.clustering import ColdStartHandler
from src.evaluation.metrics import (
    compute_all_metrics,
    model_comparison_report,
    generate_evaluation_summary,
    stratified_evaluation,
)

logger = setup_logger("Pipeline")


def load_or_generate_data() -> dict:
    """Load existing synthetic data or generate fresh."""
    transactions_path = cfg.SYNTHETIC_DIR / "transactions.csv"
    
    if transactions_path.exists():
        logger.info("Loading existing synthetic data...")
        data = {
            "retailers": load_csv(cfg.SYNTHETIC_DIR / "retailers.csv"),
            "skus": load_csv(cfg.SYNTHETIC_DIR / "skus.csv"),
            "calendar": load_csv(cfg.SYNTHETIC_DIR / "calendar.csv", parse_dates=["date"]),
            "health_signals": load_csv(cfg.SYNTHETIC_DIR / "health_signals.csv", parse_dates=["date"]),
            "transactions": load_csv(cfg.SYNTHETIC_DIR / "transactions.csv", parse_dates=["date"]),
        }
        logger.info(f"Loaded {len(data['transactions']):,} transactions")
    else:
        logger.info("No data found. Generating synthetic data...")
        data = generate_all(save=True)
    
    return data


def prepare_splits(df: pd.DataFrame, feature_cols: list, target: str):
    """Extract train/val/test splits as numpy arrays."""
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    
    # Ensure columns exist
    available_cols = [c for c in feature_cols if c in df.columns]
    
    return {
        "X_train": train[available_cols],
        "y_train": train[target],
        "X_val": val[available_cols],
        "y_val": val[target],
        "X_test": test[available_cols],
        "y_test": test[target],
        "df_train": train,
        "df_val": val,
        "df_test": test,
        "feature_cols": available_cols,
    }


def train_gbm_models(splits: dict, cat_features: list, tuned_lgbm_params: dict | None = None, tuned_xgb_params: dict | None = None) -> dict:
    """Train LightGBM and XGBoost models."""
    models = {}
    predictions = {}
    
    # Filter categorical features to those that exist
    cat_cols = [c for c in cat_features if c in splits["X_train"].columns]
    
    # ── LightGBM (Global) ──
    logger.info("\n" + "━" * 60)
    logger.info("Training LightGBM (Global Model)")
    logger.info("━" * 60)
    
    lgbm = LightGBMForecaster(params=tuned_lgbm_params, name="lgbm_global")
    lgbm.train(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        categorical_features=cat_cols,
    )
    
    models["lgbm"] = lgbm
    predictions["lgbm_val"] = lgbm.predict(splits["X_val"])
    predictions["lgbm_test"] = lgbm.predict(splits["X_test"])
    
    # Train quantile models
    lgbm.train_quantile(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        quantiles=[0.1, 0.5, 0.9],
        categorical_features=cat_cols,
    )
    
    # ── XGBoost ──
    logger.info("\n" + "━" * 60)
    logger.info("Training XGBoost")
    logger.info("━" * 60)
    
    xgb_model = XGBoostForecaster(params=tuned_xgb_params, name="xgb_segment")
    xgb_model.train(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    
    models["xgb"] = xgb_model
    predictions["xgb_val"] = xgb_model.predict(splits["X_val"])
    predictions["xgb_test"] = xgb_model.predict(splits["X_test"])
    
    xgb_model.train_quantile(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        quantiles=[0.1, 0.5, 0.9],
    )
    
    return models, predictions


def train_neural_models(splits: dict) -> dict:
    """
    Train TFT and N-BEATS models.
    Falls back gracefully if NeuralForecast is not installed.
    """
    models = {}
    predictions = {}
    from src.utils.neural_inference import extract_median, align_to_split, generate_rolling_predictions
    
    # ── TFT ──
    try:
        from src.models.tft_model import TFTForecaster
        
        logger.info("\n" + "━" * 60)
        logger.info("Training Temporal Fusion Transformer")
        logger.info("━" * 60)
        
        tft = TFTForecaster(name="tft")
        tft.train(splits["df_train"], df_val=splits["df_val"], max_series=1000000000)
        
        # Val predictions
        tft_val = generate_rolling_predictions(tft, splits["df_train"], splits["df_val"])
        if tft_val is not None and len(tft_val) > 0:
            preds_df = extract_median(tft_val, "TFT")
            if preds_df is not None:
                models["tft"] = tft
                predictions["tft_val"] = align_to_split(
                    preds_df, splits["df_val"], len(splits["y_val"])
                )
                
                # Test predictions
                try:
                    df_history_for_test = pd.concat([splits["df_train"], splits["df_val"]])
                    tft_test = generate_rolling_predictions(tft, df_history_for_test, splits["df_test"])
                    if tft_test is not None and len(tft_test) > 0:
                        test_df = extract_median(tft_test, "TFT")
                        if test_df is not None:
                            predictions["tft_test"] = align_to_split(
                                test_df, splits["df_test"], len(splits["y_test"])
                            )
                except Exception as e:
                    logger.warning(f"TFT test prediction failed: {e}")
    
    except Exception as e:
        logger.warning(f"TFT training failed (will skip in ensemble): {e}")
    
    # ── N-BEATS ──
    try:
        from src.models.nbeats_model import NBEATSForecaster
        
        logger.info("\n" + "━" * 60)
        logger.info("Training N-BEATS")
        logger.info("━" * 60)
        
        nbeats = NBEATSForecaster(name="nbeats")
        nbeats.train(splits["df_train"], df_val=splits["df_val"], max_series=1000000000)
        
        # Val predictions
        nbeats_val = generate_rolling_predictions(nbeats, splits["df_train"], splits["df_val"])
        if nbeats_val is not None and len(nbeats_val) > 0:
            preds_df = extract_median(nbeats_val, "NBEATS")
            if preds_df is not None:
                models["nbeats"] = nbeats
                predictions["nbeats_val"] = align_to_split(
                    preds_df, splits["df_val"], len(splits["y_val"])
                )
                
                # Test predictions
                try:
                    df_history_for_test = pd.concat([splits["df_train"], splits["df_val"]])
                    nbeats_test = generate_rolling_predictions(nbeats, df_history_for_test, splits["df_test"])
                    if nbeats_test is not None and len(nbeats_test) > 0:
                        test_df = extract_median(nbeats_test, "NBEATS")
                        if test_df is not None:
                            predictions["nbeats_test"] = align_to_split(
                                test_df, splits["df_test"], len(splits["y_test"])
                            )
                except Exception as e:
                    logger.warning(f"N-BEATS test prediction failed: {e}")
    
    except Exception as e:
        logger.warning(f"N-BEATS training failed (will skip in ensemble): {e}")
    
    return models, predictions


def train_baselines(splits: dict) -> dict:
    """Train statistical baselines."""
    models = {}
    predictions = {}
    
    # ── Seasonal Naïve ──
    logger.info("\n" + "━" * 60)
    logger.info("Training Seasonal Naïve Baseline")
    logger.info("━" * 60)
    
    snaive = SeasonalNaive()
    snaive.fit(splits["df_train"])
    
    models["seasonal_naive"] = snaive
    predictions["snaive_val"] = snaive.predict(splits["df_val"])
    predictions["snaive_test"] = snaive.predict(splits["df_test"])
    
    return models, predictions


def train_ensemble(
    val_predictions: dict,
    y_val: np.ndarray,
    segment_features: pd.DataFrame = None,
) -> StackingEnsemble:
    """Train the stacking meta-learner on out-of-fold predictions."""
    logger.info("\n" + "━" * 60)
    logger.info("Training Stacking Meta-Learner")
    logger.info("━" * 60)
    
    # Build base prediction dict for meta-learner
    base_preds = {}
    min_len = len(y_val)
    
    for key, preds in val_predictions.items():
        if key.endswith("_val") and key != "snaive_val":
            model_name = key.replace("_val", "")
            preds_arr = np.array(preds)
            # Ensure all prediction arrays have same length
            if len(preds_arr) >= min_len:
                base_preds[model_name] = preds_arr[:min_len]
            else:
                # Pad with mean if too short
                padded = np.full(min_len, np.mean(preds_arr))
                padded[:len(preds_arr)] = preds_arr
                base_preds[model_name] = padded
    
    if len(base_preds) < 2:
        logger.warning("Not enough base models for stacking — using simple average")
        return None
    
    # Truncate segment features
    if segment_features is not None:
        segment_features = segment_features.iloc[:min_len]
    
    ensemble = StackingEnsemble()
    ensemble.train(base_preds, y_val.values[:min_len], segment_features)
    
    # Show model contributions
    contrib = ensemble.get_model_contributions()
    if len(contrib) > 0:
        logger.info("\nModel Contributions:")
        logger.info("\n" + contrib.to_string(index=False))
    
    return ensemble


def save_models(models: dict, ensemble: StackingEnsemble = None) -> None:
    """Save all trained models to disk."""
    model_dir = ensure_dir(cfg.MODEL_DIR)
    
    for name, model in models.items():
        path = model_dir / f"{name}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} → {path}")
    
    if ensemble is not None:
        path = model_dir / "ensemble_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(ensemble, f)
        logger.info(f"Saved ensemble → {path}")


def run_pipeline(
    tune_hyperparameters: bool = True,
    train_neural: bool = True,
    save: bool = True,
    generate_plots: bool = False,
) -> dict:
    """
    Run the full training pipeline.
    
    Args:
        tune_hyperparameters: Run Optuna tuning (slow but better results)
        train_neural: Train TFT/N-BEATS (requires neuralforecast + GPU recommended)
        save: Save models to disk
    """
    set_seed(cfg.SEED)
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("CuraNex AI — Full Training Pipeline")
    logger.info("=" * 80)

    mlflow.set_tracking_uri(str(cfg.MLFLOW_DIR))
    mlflow.set_experiment("curanex_ai_training")
    mlflow.start_run(run_name=f"training_run_{int(start_time)}")

    mlflow.log_params({
        "tune_hyperparameters": tune_hyperparameters,
        "train_neural": train_neural,
        "seed": cfg.SEED,
    })
    
    # ── 1. Load Data ──
    logger.info("\n[Phase 1] Loading data...")
    data = load_or_generate_data()
    
    # ── 2. Feature Engineering ──
    logger.info("\n[Phase 2] Feature engineering...")
    df = build_features(
        data["transactions"],
        data["retailers"],
        data["skus"],
        data["calendar"],
        data["health_signals"],
    )
    
    # Save featured data
    feature_path = cfg.SYNTHETIC_DIR / "features.csv"
    df.to_csv(feature_path, index=False)
    logger.info(f"Saved featured data → {feature_path}")
    
    # ── 3. Prepare splits ──
    feat_info = get_feature_columns()
    splits = prepare_splits(df, feat_info["all"], feat_info["target"])
    
    logger.info(f"\nSplits — Train: {len(splits['X_train']):,}, "
                 f"Val: {len(splits['X_val']):,}, "
                 f"Test: {len(splits['X_test']):,}")
    
    mlflow.log_metrics({
        "n_train_samples": len(splits["X_train"]),
        "n_val_samples": len(splits["X_val"]),
        "n_test_samples": len(splits["X_test"]),
        "n_features": len(splits["feature_cols"]),
    })
    
    
    all_models = {}
    all_predictions = {}
    
    # ── 4. Optuna Tuning ──
    tuned_lgbm_params = None
    tuned_xgb_params = None
    
    if tune_hyperparameters:
        logger.info("\n[Phase 3a] Hyperparameter tuning (LightGBM)...")
        lgbm_tuner = LightGBMForecaster(name="lgbm_tuner")
        tuned_lgbm_params = lgbm_tuner.tune_hyperparameters(
            splits["X_train"], splits["y_train"],
            categorical_features=feat_info["categorical"],
        )

        if tuned_lgbm_params:
            mlflow.log_params({
                f"lgbm_tuned_{k}": v for k, v in tuned_lgbm_params.items()
        })
        
        logger.info("\n[Phase 3b] Hyperparameter tuning (XGBoost)...")
        xgb_tuner = XGBoostForecaster(name="xgb_tuner")
        tuned_xgb_params = xgb_tuner.tune_hyperparameters(
            splits["X_train"], splits["y_train"],
        )
    
    # ── 5. Train GBM Models ──
    logger.info("\n[Phase 3] Training GBM models...")
    gbm_models, gbm_preds = train_gbm_models(splits, feat_info["categorical"], tuned_lgbm_params, tuned_xgb_params)
    all_models.update(gbm_models)
    all_predictions.update(gbm_preds)
    
    # ── 6. Train Neural Models ──
    if train_neural:
        logger.info("\n[Phase 4] Training neural models...")
        neural_models, neural_preds = train_neural_models(splits)
        all_models.update(neural_models)
        all_predictions.update(neural_preds)
    
    # ── 7. Train Baselines ──
    logger.info("\n[Phase 5] Training baselines...")
    baseline_models, baseline_preds = train_baselines(splits)
    all_models.update(baseline_models)
    all_predictions.update(baseline_preds)
    
    # ── 8. Train Ensemble ──
    logger.info("\n[Phase 6] Training ensemble meta-learner...")
    segment_feats = splits["df_val"][["tier_encoded", "therapeutic_category_encoded"]].reset_index(drop=True)
    ensemble = train_ensemble(all_predictions, splits["y_val"], segment_feats)
    
    # ── 9. Generate Ensemble Predictions ──
    ensemble_test_preds = None
    n_test = len(splits["y_test"])
    if ensemble is not None:
        test_base_preds = {}
        for key, preds in all_predictions.items():
            if key.endswith("_test") and key != "snaive_test":
                model_name = key.replace("_test", "")
                p = preds[:n_test]
                # Only include predictions that cover ALL test samples
                # (neural models may only cover a subset of series)
                if len(p) == n_test:
                    test_base_preds[model_name] = p
                else:
                    logger.info(f"Skipping {model_name} from ensemble test "
                                f"({len(p)} vs {n_test} expected)")
        
        if test_base_preds:
            segment_test = splits["df_test"][["tier_encoded", "therapeutic_category_encoded"]].reset_index(drop=True)
            ensemble_test_preds = ensemble.predict(test_base_preds, segment_test.iloc[:n_test])
    else:
        # Fallback to simple average
        test_preds = {k.replace("_test", ""): v for k, v in all_predictions.items() if k.endswith("_test")}
        if test_preds:
            ensemble_test_preds = simple_average_ensemble(test_preds)
    
    # ── 10. Evaluation ──
    logger.info("\n[Phase 7] Evaluation...")
    test_model_preds = {}
    
    for key, preds in all_predictions.items():
        if key.endswith("_test"):
            name = key.replace("_test", "")
            p = np.asarray(preds)[:n_test]
            if len(p) == n_test:
                test_model_preds[name] = p
    
    test_model_preds["seasonal_naive"] = all_predictions.get("snaive_test", np.zeros(n_test))[:n_test]
    
    if ensemble_test_preds is not None:
        test_model_preds["ensemble"] = ensemble_test_preds[:n_test]
    
    evaluation = generate_evaluation_summary(
        splits["y_test"].values,
        test_model_preds,
        splits["df_test"],
    )

    if "comparison" in evaluation:
        comp_df = evaluation["comparison"]
        for _, row in comp_df.iterrows():
            model_name = row["Model"].lower().replace(" ", "_")
            mlflow.log_metrics({
                f"{model_name}_mae": float(row.get("MAE", 0)),
                f"{model_name}_rmse": float(row.get("RMSE", 0)),
                f"{model_name}_wmape": float(row.get("wMAPE", 0)),
                f"{model_name}_fill_rate": float(row.get("Fill_Rate", 0)),
                f"{model_name}_bias": float(row.get("Bias", 0)),
            })
    
    # Collect ensemble predictions for val + test (used by plots)
    ensemble_preds_by_split = {}
    
    # Val: ensemble trained on val preds — use them directly
    n_val = len(splits["y_val"])
    if ensemble is not None:
        val_base_preds = {}
        for key, preds in all_predictions.items():
            if key.endswith("_val") and key != "snaive_val":
                model_name = key.replace("_val", "")
                p = np.asarray(preds)[:n_val]
                if len(p) == n_val:
                    val_base_preds[model_name] = p
        if val_base_preds:
            segment_val = splits["df_val"][["tier_encoded", "therapeutic_category_encoded"]].reset_index(drop=True)
            ensemble_val_preds = ensemble.predict(val_base_preds, segment_val.iloc[:n_val])
            ensemble_preds_by_split["val"] = ensemble_val_preds
    
    if ensemble_test_preds is not None:
        ensemble_preds_by_split["test"] = ensemble_test_preds
    
    # ── 11. Cold-Start Clustering ──
    logger.info("\n[Phase 8] Cold-start clustering...")
    cold_start = ColdStartHandler()
    cold_start.fit(data["transactions"], data["retailers"])
    all_models["cold_start"] = cold_start
    
    # ── 12. Save Models ──
    if save:
        logger.info("\n[Phase 9] Saving models...")
        save_models(all_models, ensemble)
        
        # Save evaluation results
        eval_path = cfg.EVAL_DIR / "evaluation_report.csv"
        if "comparison" in evaluation:
            evaluation["comparison"].to_csv(eval_path, index=False)
            logger.info(f"Saved evaluation report → {eval_path}")
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info("=" * 80)
    
    # ── 13. Generate Plots ──
    if generate_plots:
        logger.info("\n[Phase 10] Generating evaluation plots...")
        from src.evaluation.plots import generate_all_plots
        generate_all_plots(
            splits=splits,
            all_predictions=all_predictions,
            models=all_models,
            ensemble_preds=ensemble_preds_by_split,
        )

        mlflow.log_artifacts(cfg.PLOTS_DIR.as_posix(), artifact_path="evaluation_plots")
    
    return {
        "models": all_models,
        "ensemble": ensemble,
        "predictions": all_predictions,
        "evaluation": evaluation,
        "featured_data": df,
        "splits": splits,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CuraNex AI Training Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--neural", action="store_true", help="Train TFT and N-BEATS models")
    parser.add_argument("--no-save", action="store_true", help="Don't save models to disk")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    results = run_pipeline(
        tune_hyperparameters=args.tune,
        train_neural=args.neural,
        save=not args.no_save,
        generate_plots=args.plots,
    )
