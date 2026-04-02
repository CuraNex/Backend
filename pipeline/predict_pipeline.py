"""
CuraNex AI — Prediction / Inference Pipeline
===============================================
Load saved models from disk and run predictions + evaluation
without retraining.

Usage:
    python predict_pipeline.py                  # evaluate on test split
    python predict_pipeline.py --plots          # + generate all plots
    python predict_pipeline.py --csv output.csv # save predictions to CSV
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from src.utils.helpers import setup_logger, load_csv, ensure_dir
from src.features.feature_engineering import build_features, get_feature_columns
from src.evaluation.metrics import (
    compute_all_metrics,
    model_comparison_report,
    generate_evaluation_summary,
)

logger = setup_logger("Predict")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_dir: Path = None) -> dict:
    """Load all saved model artifacts from disk."""
    model_dir = model_dir or cfg.MODEL_DIR
    models = {}

    for pkl_path in sorted(model_dir.glob("*_model.pkl")):
        name = pkl_path.stem.replace("_model", "")
        try:
            with open(pkl_path, "rb") as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded {name} ← {pkl_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {pkl_path.name}: {e}")

    if not models:
        raise FileNotFoundError(
            f"No model files found in {model_dir}. Run the training pipeline first."
        )

    return models


def load_data_and_features() -> tuple:
    """Load data and rebuild features (fast — uses cached transactions)."""
    logger.info("Loading data...")
    data = {
        "retailers": load_csv(cfg.SYNTHETIC_DIR / "retailers.csv"),
        "skus": load_csv(cfg.SYNTHETIC_DIR / "skus.csv"),
        "calendar": load_csv(cfg.SYNTHETIC_DIR / "calendar.csv", parse_dates=["date"]),
        "health_signals": load_csv(cfg.SYNTHETIC_DIR / "health_signals.csv", parse_dates=["date"]),
        "transactions": load_csv(cfg.SYNTHETIC_DIR / "transactions.csv", parse_dates=["date"]),
    }

    # Try to load cached features first
    feature_path = cfg.SYNTHETIC_DIR / "features.csv"
    if feature_path.exists():
        logger.info("Loading cached feature data...")
        df = load_csv(feature_path, parse_dates=["date"])
        logger.info(f"Loaded {len(df):,} featured rows")
    else:
        logger.info("Building features from scratch...")
        df = build_features(
            data["transactions"], data["retailers"], data["skus"],
            data["calendar"], data["health_signals"],
        )

    return df, data


def prepare_splits(df: pd.DataFrame, feature_cols: list, target: str):
    """Same as train_pipeline — extract train/val/test."""
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

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


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_predictions(
    models: dict,
    splits: dict,
    split_name: str = "test",
) -> dict:
    """
    Generate predictions from all loaded models on the chosen split.

    Args:
        models: {"lgbm": model, "xgb": model, "ensemble": model, ...}
        splits: output of prepare_splits
        split_name: "train", "val", or "test"

    Returns:
        {"lgbm": preds, "xgb": preds, "ensemble": preds, ...}
    """
    X = splits[f"X_{split_name}"]
    y = splits[f"y_{split_name}"]
    df_split = splits[f"df_{split_name}"]

    predictions = {}

    # ── GBM models (LightGBM / XGBoost) ──
    for name in ["lgbm", "xgb"]:
        if name in models:
            try:
                preds = models[name].predict(X)
                predictions[name] = preds[:len(y)]
                logger.info(f"  {name}: predicted {len(preds):,} samples")
            except Exception as e:
                logger.warning(f"  {name} prediction failed: {e}")

    # ── Seasonal Naive ──
    if "seasonal_naive" in models:
        try:
            preds = models["seasonal_naive"].predict(df_split)
            predictions["snaive"] = preds[:len(y)]
            logger.info(f"  seasonal_naive: predicted {len(preds):,} samples")
        except Exception as e:
            logger.warning(f"  seasonal_naive prediction failed: {e}")

    # ── Neural models (TFT / N-BEATS) ──
    if split_name in ["val", "test"]:
        try:
            from src.utils.neural_inference import extract_median, align_to_split, generate_rolling_predictions
            
            if split_name == "val":
                df_history = splits["df_train"]
            else:  # test
                df_history = pd.concat([splits["df_train"], splits["df_val"]])
                
            for name in ["tft", "nbeats"]:
                if name in models:
                    try:
                        logger.info(f"  {name}: generating auto-regressive predictions for {split_name}...")
                        raw_preds = generate_rolling_predictions(models[name], df_history, df_split)
                        
                        if raw_preds is not None and len(raw_preds) > 0:
                            preds_df = extract_median(raw_preds, name.upper())
                            if preds_df is not None:
                                aligned = align_to_split(preds_df, df_split, len(y))
                                predictions[name] = aligned
                                logger.info(f"  {name}: predicted {len(aligned):,} samples")
                    except Exception as e:
                        logger.warning(f"  {name} auto-regressive prediction failed: {e}")
        except Exception as e:
            logger.warning(f"  Failed to run neural inference: {e}")

    # ── Ensemble meta-learner ──
    if "ensemble" in models and len(predictions) >= 2:
        try:
            base_preds = {k: v for k, v in predictions.items() if k != "snaive"}
            segment_feats = df_split[["tier_encoded", "therapeutic_category_encoded"]].reset_index(drop=True)
            segment_feats = segment_feats.iloc[:len(y)]
            ensemble_preds = models["ensemble"].predict(base_preds, segment_feats)
            predictions["ensemble"] = ensemble_preds[:len(y)]
            logger.info(f"  ensemble: predicted {len(ensemble_preds):,} samples")
        except Exception as e:
            logger.warning(f"  ensemble prediction failed: {e}")

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(
    generate_plots: bool = False,
    csv_output: str = None,
    split: str = "test",
) -> dict:
    """
    Full inference pipeline: load models → predict → evaluate → (plot).
    """
    start = time.time()

    logger.info("=" * 80)
    logger.info("CuraNex AI — Prediction / Evaluation Pipeline")
    logger.info("=" * 80)

    # 1. Load models
    logger.info("\n[1] Loading saved models...")
    models = load_models()
    logger.info(f"  Loaded {len(models)} models: {list(models.keys())}")

    # 2. Load data + features
    logger.info("\n[2] Loading data & features...")
    df, data = load_data_and_features()
    feat_info = get_feature_columns()
    splits = prepare_splits(df, feat_info["all"], feat_info["target"])

    logger.info(f"  Splits — Train: {len(splits['X_train']):,}, "
                f"Val: {len(splits['X_val']):,}, "
                f"Test: {len(splits['X_test']):,}")

    # 3. Generate predictions on all splits
    all_split_preds = {}
    ensemble_split_preds = {}

    for s in ["train", "val", "test"]:
        logger.info(f"\n[3] Predicting on {s}...")
        preds = generate_predictions(models, splits, split_name=s)
        all_split_preds[s] = preds
        if "ensemble" in preds:
            ensemble_split_preds[s] = preds["ensemble"]

    # 4. Evaluate
    logger.info(f"\n[4] Evaluating on {split} set...")
    y_eval = splits[f"y_{split}"].values
    split_preds = all_split_preds[split]

    evaluation = generate_evaluation_summary(
        y_eval,
        split_preds,
        splits[f"df_{split}"],
    )

    # Print split-level summary for all models
    logger.info("\n" + "=" * 80)
    logger.info("PER-SPLIT METRICS")
    logger.info("=" * 80)

    for s in ["train", "val", "test"]:
        y_s = splits[f"y_{s}"].values
        preds_s = all_split_preds[s]
        logger.info(f"\n── {s.upper()} ──")
        for model_name, p in preds_s.items():
            metrics = compute_all_metrics(y_s, np.asarray(p)[:len(y_s)])
            logger.info(f"  {model_name:>15s}  MAE={metrics['MAE']:.3f}  "
                        f"RMSE={metrics['RMSE']:.3f}  wMAPE={metrics['wMAPE']:.1f}%  "
                        f"Bias={metrics['Bias']:.1f}%")

    # 5. Plots
    if generate_plots:
        logger.info("\n[5] Generating plots...")
        from src.evaluation.plots import generate_all_plots

        # Build all_predictions dict in the format the plotter expects
        flat_preds = {}
        for s, preds in all_split_preds.items():
            for name, p in preds.items():
                flat_preds[f"{name}_{s}"] = p

        generate_all_plots(
            splits=splits,
            all_predictions=flat_preds,
            models=models,
            ensemble_preds=ensemble_split_preds,
        )

    # 6. CSV export
    if csv_output:
        logger.info(f"\n[6] Saving predictions → {csv_output}")
        export_df = splits[f"df_{split}"][["retailer_id", "sku_id", "date", "quantity_ordered"]].copy()
        for name, p in split_preds.items():
            export_df[f"pred_{name}"] = np.asarray(p)[:len(export_df)]
        export_df.to_csv(csv_output, index=False)
        logger.info(f"  Saved {len(export_df):,} rows to {csv_output}")

    elapsed = time.time() - start
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Prediction pipeline complete! ({elapsed:.1f}s)")
    logger.info("=" * 80)

    return {
        "predictions": all_split_preds,
        "evaluation": evaluation,
        "splits": splits,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CuraNex AI — Predict & Evaluate")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--csv", type=str, default=None, help="Save predictions to CSV")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test)")

    args = parser.parse_args()

    run_predict(
        generate_plots=args.plots,
        csv_output=args.csv,
        split=args.split,
    )
