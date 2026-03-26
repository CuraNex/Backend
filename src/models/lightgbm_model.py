"""
PharmaFlow AI — LightGBM Model
================================
Global and segment-specific LightGBM models for demand forecasting.
Supports point predictions (MAE/MSE) and quantile regression (P10/P50/P90).
Includes Optuna hyperparameter tuning with expanding-window time-series CV.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("LightGBM")

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMForecaster:
    """
    LightGBM-based demand forecaster with:
    - Global model across all retailer-SKU pairs
    - Optional segment-specific models for top-tier retailers
    - Quantile regression for probabilistic outputs (P10, P50, P90)
    - Optuna hyperparameter tuning
    """
    
    def __init__(self, params: dict = None, name: str = "lgbm_global"):
        self.params = params or cfg.LIGHTGBM_PARAMS.copy()
        self.name = name
        self.model = None
        self.quantile_models = {}  # {quantile: model}
        self.feature_importance = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        categorical_features: list = None,
    ) -> "LightGBMForecaster":
        """Train the point-prediction model (MAE objective)."""
        logger.info(f"[{self.name}] Training on {len(X_train):,} samples, {X_train.shape[1]} features")
        
        params = self.params.copy()
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]
        
        self.model = lgb.LGBMRegressor(**params)
        
        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "categorical_feature": categorical_features or "auto",
            "callbacks": callbacks,
        }
        
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        
        self.model.fit(**fit_kwargs)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        logger.info(f"[{self.name}] Training complete — "
                     f"Best iteration: {self.model.best_iteration_}")
        return self
    
    def train_quantile(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        quantiles: list = (0.1, 0.5, 0.9),
        categorical_features: list = None,
    ) -> "LightGBMForecaster":
        """Train separate quantile regression models for uncertainty estimation."""
        for q in quantiles:
            logger.info(f"[{self.name}] Training quantile={q}...")
            params = cfg.LIGHTGBM_QUANTILE_PARAMS.copy()
            params["alpha"] = q
            
            model = lgb.LGBMRegressor(**params)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ]
            
            fit_kwargs = {
                "X": X_train,
                "y": y_train,
                "categorical_feature": categorical_features or "auto",
                "callbacks": callbacks,
            }
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            
            model.fit(**fit_kwargs)
            self.quantile_models[q] = model
        
        logger.info(f"[{self.name}] Quantile training complete for {quantiles}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        preds = self.model.predict(X)
        return np.maximum(0, preds)  # demand can't be negative
    
    def predict_quantiles(self, X: pd.DataFrame) -> dict:
        """Quantile predictions → {quantile: array}."""
        results = {}
        for q, model in self.quantile_models.items():
            preds = model.predict(X)
            results[q] = np.maximum(0, preds)
        return results
    
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = cfg.OPTUNA_N_TRIALS,
        n_cv_folds: int = cfg.OPTUNA_CV_FOLDS,
        categorical_features: list = None,
    ) -> dict:
        """
        Bayesian hyperparameter optimization using Optuna with
        expanding-window time-series cross-validation.
        """
        logger.info(f"[{self.name}] Starting Optuna tuning ({n_trials} trials, {n_cv_folds} folds)")
        
        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": cfg.SEED,
                "verbose": -1,
                "n_jobs": -1,
            }
            
            tscv = TimeSeriesSplit(n_splits=n_cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_vl, y_vl)],
                    categorical_feature=categorical_features or "auto",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                    ],
                )
                
                preds = model.predict(X_vl)
                mae = np.mean(np.abs(y_vl - preds))
                scores.append(mae)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "random_state": cfg.SEED,
            "verbose": -1,
            "n_jobs": -1,
        })
        
        logger.info(f"[{self.name}] Best MAE: {study.best_value:.4f}")
        logger.info(f"[{self.name}] Best params: {best_params}")
        
        self.params = best_params
        return best_params
    
    def get_shap_values(self, X: pd.DataFrame):
        """Compute SHAP values for model interpretability."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return shap_values
        except ImportError:
            logger.warning("SHAP not installed — skipping explainability")
            return None
