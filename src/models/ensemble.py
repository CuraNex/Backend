"""
PharmaFlow AI — Stacking Meta-Learner Ensemble
================================================
Combines base model predictions (LightGBM, XGBoost, TFT, N-BEATS) through
stacked generalization — a second-level model that learns optimal combination
weights conditioned on retailer segment and therapeutic category.

Key design decisions:
1. Segment-aware: separate weight learning per retailer tier × therapeutic category
2. Out-of-fold predictions to avoid overfitting the meta-learner to base models
3. Quantile aggregation for probabilistic forecasts
4. Graceful degradation: if a base model is missing, ensemble adjusts automatically
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("Ensemble")


class StackingEnsemble:
    """
    Stacking meta-learner that combines base model predictions.
    
    Architecture:
    - Level 0: Base models produce out-of-fold predictions
    - Level 1: Meta-learner (Ridge/ElasticNet/LightGBM) learns optimal combination
    
    The meta-learner receives:
    - Predictions from each base model
    - Retailer tier (segment indicator)
    - Therapeutic category
    
    This allows it to learn that, e.g., LightGBM is best for high-volume regular
    retailers while TFT handles volatile retailers better.
    """
    
    def __init__(self, meta_learner_type: str = None, name: str = "ensemble"):
        self.meta_type = meta_learner_type or cfg.ENSEMBLE_META_LEARNER
        self.name = name
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.base_model_names = []
        self.quantile_meta_learners = {}  # {quantile: model}
    
    def _build_meta_features(
        self,
        base_predictions: dict,
        segment_features: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix for the meta-learner.
        
        Args:
            base_predictions: {model_name: np.array of predictions}
            segment_features: DataFrame with 'tier_encoded', 'therapeutic_category_encoded'
        """
        meta_df = pd.DataFrame(base_predictions)
        self.base_model_names = list(base_predictions.keys())
        
        if segment_features is not None:
            for col in ["tier_encoded", "therapeutic_category_encoded"]:
                if col in segment_features.columns:
                    meta_df[col] = segment_features[col].values
        
        return meta_df
    
    def _create_meta_learner(self):
        """Instantiate the meta-learner model."""
        if self.meta_type == "ridge":
            return Ridge(
                alpha=cfg.ENSEMBLE_RIDGE_ALPHA,
                fit_intercept=True,
                positive=True,  # Force non-negative weights
            )
        elif self.meta_type == "elasticnet":
            return ElasticNet(alpha=0.5, l1_ratio=0.5, positive=True)
        elif self.meta_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=15,
                learning_rate=0.05,
                reg_lambda=1.0,
                verbose=-1,
            )
        else:
            return Ridge(alpha=1.0, positive=True)
    
    def train(
        self,
        base_predictions: dict,
        y_true: np.ndarray,
        segment_features: pd.DataFrame = None,
    ) -> "StackingEnsemble":
        """
        Train the stacking meta-learner.
        
        IMPORTANT: base_predictions should be OUT-OF-FOLD predictions from
        the validation set — not in-sample predictions, to avoid overfitting.
        
        Args:
            base_predictions: {model_name: predictions_array}
            y_true: Actual target values
            segment_features: Optional segment indicators for the meta-learner
        """
        meta_X = self._build_meta_features(base_predictions, segment_features)
        
        # Scale features
        meta_X_scaled = pd.DataFrame(
            self.scaler.fit_transform(meta_X),
            columns=meta_X.columns,
        )
        
        # Train meta-learner
        self.meta_learner = self._create_meta_learner()
        self.meta_learner.fit(meta_X_scaled, y_true)
        
        # Log learned weights
        if hasattr(self.meta_learner, "coef_"):
            weights = dict(zip(meta_X.columns, self.meta_learner.coef_))
            logger.info(f"[{self.name}] Learned weights: {weights}")
        
        # Evaluate ensemble on training fold
        ensemble_pred = self.meta_learner.predict(meta_X_scaled)
        mae = np.mean(np.abs(y_true - ensemble_pred))
        logger.info(f"[{self.name}] Training MAE: {mae:.4f}")
        
        return self
    
    def train_quantile_ensemble(
        self,
        base_quantile_predictions: dict,
        y_true: np.ndarray,
        quantiles: list = (0.1, 0.5, 0.9),
    ) -> "StackingEnsemble":
        """
        Train separate meta-learners for each quantile level.
        
        Args:
            base_quantile_predictions: {quantile: {model_name: predictions}}
        """
        for q in quantiles:
            if q in base_quantile_predictions:
                preds = base_quantile_predictions[q]
                meta_X = pd.DataFrame(preds)
                
                # Use quantile-specific meta-learner
                model = Ridge(alpha=1.0, positive=True)
                model.fit(meta_X, y_true)
                self.quantile_meta_learners[q] = model
                
                logger.info(f"[{self.name}] Quantile {q} meta-learner trained")
        
        return self
    
    def predict(
        self,
        base_predictions: dict,
        segment_features: pd.DataFrame = None,
    ) -> np.ndarray:
        """
        Generate ensemble prediction by combining base model outputs.
        
        Handles missing models gracefully — if a base model's prediction
        is missing, only uses available models.
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train() first.")
        
        meta_X = self._build_meta_features(base_predictions, segment_features)
        
        # Handle missing model columns by filling with mean of available models
        for col in self.base_model_names:
            if col not in meta_X.columns:
                available = [c for c in self.base_model_names if c in meta_X.columns]
                meta_X[col] = meta_X[available].mean(axis=1)
        
        meta_X_scaled = pd.DataFrame(
            self.scaler.transform(meta_X),
            columns=meta_X.columns,
        )
        
        preds = self.meta_learner.predict(meta_X_scaled)
        return np.maximum(0, preds)
    
    def predict_quantiles(
        self,
        base_quantile_predictions: dict,
    ) -> dict:
        """Generate quantile ensemble predictions."""
        results = {}
        for q, model in self.quantile_meta_learners.items():
            if q in base_quantile_predictions:
                preds_dict = base_quantile_predictions[q]
                meta_X = pd.DataFrame(preds_dict)
                results[q] = np.maximum(0, model.predict(meta_X))
        return results
    
    def get_model_contributions(self) -> pd.DataFrame:
        """
        Show how much each base model contributes to the final ensemble.
        
        Higher weight = model is more important for the final prediction.
        """
        if not hasattr(self.meta_learner, "coef_"):
            return pd.DataFrame()
        
        contrib = pd.DataFrame({
            "model": self.base_model_names,
            "weight": self.meta_learner.coef_[:len(self.base_model_names)],
        })
        contrib["weight_pct"] = (contrib["weight"] / contrib["weight"].sum() * 100).round(1)
        return contrib.sort_values("weight_pct", ascending=False)


def simple_average_ensemble(base_predictions: dict) -> np.ndarray:
    """
    Simple average fallback — when meta-learner can't be trained
    (e.g., insufficient validation data).
    """
    preds = np.column_stack(list(base_predictions.values()))
    return np.maximum(0, preds.mean(axis=1))


def weighted_average_ensemble(base_predictions: dict, weights: dict) -> np.ndarray:
    """
    Manual weighted average with pre-specified weights.
    Useful for domain-expert-driven ensembling.
    """
    total_weight = sum(weights.get(k, 0) for k in base_predictions)
    result = np.zeros(len(next(iter(base_predictions.values()))))
    
    for model_name, preds in base_predictions.items():
        w = weights.get(model_name, 0) / total_weight
        result += w * preds
    
    return np.maximum(0, result)
