"""
CuraNex AI — N-BEATS Model
================================
Neural Basis Expansion Analysis for Time Series.
Pure univariate expert that decomposes demand into trend + seasonality
without requiring exogenous features.

Uses the interpretable variant with explicit trend and seasonality stacks.
Trained per retailer cluster for efficiency.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("NBEATS")


class NBEATSForecaster:
    
    """
    N-BEATS forecaster — the univariate expert in the ensemble.
    
    Strengths:
    - Explicit trend/seasonality decomposition
    - No feature engineering required (learns from raw series)
    - Strong on series with clear cyclical patterns
    
    Weakness:
    - Ignores exogenous variables entirely
    - Requires sufficient history per series
    """
    
    def __init__(self, params: dict = None, name: str = "nbeats"):
        self.params = params or cfg.NBEATS_PARAMS.copy()
        self.name = name
        self.nf = None
    
    def _prepare_data(self, df: pd.DataFrame, target: str = "quantity_ordered") -> pd.DataFrame:
        """Convert to NeuralForecast format (univariate — only unique_id, ds, y)."""
        nf_df = df.copy()
        nf_df["unique_id"] = [f"{r}_{s}" for r, s in zip(nf_df["retailer_id"], nf_df["sku_id"])]
        nf_df = nf_df.rename(columns={"date": "ds", target: "y"})
        nf_df["ds"] = pd.to_datetime(nf_df["ds"])
        nf_df = nf_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
        return nf_df
    
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame = None,
        target: str = "quantity_ordered",
        max_series: int = 100000000000,
    ) -> "NBEATSForecaster":
        """
        Train N-BEATS on the interpretable variant.
        
        Uses separate trend and seasonality stacks for decomposition.
        """
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATS
            from neuralforecast.losses.pytorch import MQLoss
        except ImportError:
            logger.error("NeuralForecast not installed.")
            raise
        
        nf_train = self._prepare_data(df_train, target)
        
        val_size = 0
        if df_val is not None:
            nf_val = self._prepare_data(df_val, target)
            val_size = nf_val["ds"].nunique()
            nf_train = pd.concat([nf_train, nf_val]).reset_index(drop=True)
            logger.info(f"[{self.name}] Enrolled validation set with val_size={val_size}. Early stopping enabled.")
        
        # Filter to viable series
        min_len = self.params.get("input_size", 52) + self.params.get("h", 13)
        lengths = nf_train.groupby("unique_id").size()
        valid_ids = lengths[lengths >= min_len].index
        
        if len(valid_ids) > max_series:
            # Take most active series
            top_ids = nf_train.groupby("unique_id")["y"].sum().nlargest(max_series).index
            valid_ids = top_ids
        
        nf_train = nf_train[nf_train["unique_id"].isin(valid_ids)]
        
        logger.info(f"[{self.name}] Training on {nf_train['unique_id'].nunique()} series")
        
        nbeats = NBEATS(
            h=self.params.get("h", 13),
            input_size=self.params.get("input_size", 52),
            stack_types=["trend", "seasonality"],
            n_blocks=self.params.get("n_blocks", [3, 3]),
            mlp_units=self.params.get("mlp_units", [[256, 256], [256, 256]]),
            learning_rate=self.params.get("learning_rate", 1e-3),
            max_steps=self.params.get("max_steps", 500),
            val_check_steps=self.params.get("val_check_steps", 50),
            early_stop_patience_steps=self.params.get("early_stop_patience", 10),
            batch_size=self.params.get("batch_size", 64),
            windows_batch_size=self.params.get("windows_batch_size", 256),
            scaler_type=self.params.get("scaler_type", "robust"),
            random_seed=self.params.get("random_seed", 42),
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
            enable_progress_bar=True,
        )
        
        self.nf = NeuralForecast(models=[nbeats], freq="W-MON")
        
        logger.info(f"[{self.name}] Starting N-BEATS training...")
        self.nf.fit(df=nf_train, val_size=val_size)
        logger.info(f"[{self.name}] N-BEATS training complete!")
        
        return self
        
    def predict(self, df: pd.DataFrame, target: str = "quantity_ordered") -> pd.DataFrame:
        """Generate forecasts."""
        if self.nf is None:
            raise ValueError("Model not trained.")
        
        nf_df = self._prepare_data(df, target)
        
        trained_ids = set(self.nf.uids) if hasattr(self.nf, "uids") else set()
        if trained_ids:
            nf_df = nf_df[nf_df["unique_id"].isin(trained_ids)]
        
        forecasts = self.nf.predict(df=nf_df)
        
        for col in forecasts.columns:
            if col not in ["unique_id", "ds"]:
                forecasts[col] = forecasts[col].clip(lower=0)
        
        return forecasts
