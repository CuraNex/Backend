"""
CuraNex AI — XGBoost Model
===============================
XGBoost variant providing ensemble diversity through different tree-building
strategy and regularization approach vs LightGBM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("XGBoost")

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostForecaster:
    """
    XGBoost demand forecaster — complements LightGBM in the ensemble
    by using a different tree-building algorithm (depth-wise vs leaf-wise).
    
    Provides:
    - Point predictions (squared error)
    - Quantile regression for uncertainty
    - Optuna hyperparameter tuning
    """
    
    def __init__(self, params: dict = None, name: str = "xgb_segment"):
        self.params = params or cfg.XGBOOST_PARAMS.copy()
        self.name = name
        self.model = None
        self.quantile_models = {}
        self.feature_importance = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> "XGBoostForecaster":
        """Train XGBoost regressor with early stopping."""
        logger.info(f"[{self.name}] Training on {len(X_train):,} samples")
        
        params = self.params.copy()
        
        self.model = xgb.XGBRegressor(**params)
        
        fit_kwargs = {"X": X_train, "y": y_train, "verbose": False}
        
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        
        self.model.fit(**fit_kwargs)
        
        self.feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        logger.info(f"[{self.name}] Training complete")
        return self
    
    def train_quantile(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        quantiles: list = (0.1, 0.5, 0.9),
    ) -> "XGBoostForecaster":
        """Train quantile regression models."""
        for q in quantiles:
            logger.info(f"[{self.name}] Training quantile={q}...")
            params = self.params.copy()
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = q
            
            model = xgb.XGBRegressor(**params)
            
            fit_kwargs = {"X": X_train, "y": y_train, "verbose": False}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            
            model.fit(**fit_kwargs)
            self.quantile_models[q] = model
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction."""
        preds = self.model.predict(X)
        return np.maximum(0, preds)
    
    def predict_quantiles(self, X: pd.DataFrame) -> dict:
        """Quantile predictions."""
        return {q: np.maximum(0, m.predict(X)) for q, m in self.quantile_models.items()}
    
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = cfg.OPTUNA_N_TRIALS,
        n_cv_folds: int = cfg.OPTUNA_CV_FOLDS,
    ) -> dict:
        """Optuna hyperparameter tuning with time-series CV."""
        logger.info(f"[{self.name}] Starting Optuna tuning ({n_trials} trials)")
        
        def objective(trial):
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "gamma": trial.suggest_float("gamma", 0, 5.0),
                "random_state": cfg.SEED,
                "tree_method": "hist",
                "n_jobs": -1,
            }
            
            tscv = TimeSeriesSplit(n_splits=n_cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                
                preds = model.predict(X_vl)
                scores.append(np.mean(np.abs(y_vl - preds)))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best = study.best_params
        best.update({
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": cfg.SEED,
            "tree_method": "hist",
            "n_jobs": -1,
        })
        
        logger.info(f"[{self.name}] Best MAE: {study.best_value:.4f}")
        self.params = best
        return best
