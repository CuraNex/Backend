"""
PharmaFlow AI — Temporal Fusion Transformer (TFT)
===================================================
Multi-horizon forecaster with attention-based interpretability.
Uses NeuralForecast (Nixtla) for the TFT implementation.

Key capabilities:
- Static covariates (retailer type, location, SKU category)
- Known future inputs (calendar, festivals)
- Observed historical inputs (past quantities, rolling stats)
- Quantile regression for probabilistic outputs
- Attention weights for interpretability
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("TFT")


class TFTForecaster:
    """
    Temporal Fusion Transformer wrapper using NeuralForecast.
    
    TFT excels at multi-horizon forecasting with heterogeneous inputs.
    Its variable selection network identifies the most relevant features,
    and the attention mechanism captures long-range temporal dependencies.
    """
    
    def __init__(self, params: dict = None, name: str = "tft"):
        self.params = params or cfg.TFT_PARAMS.copy()
        self.name = name
        self.model = None
        self.nf = None  # NeuralForecast object
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target: str = "quantity_ordered",
    ) -> pd.DataFrame:
        """
        Convert data to NeuralForecast format.
        
        NeuralForecast expects:
        - 'unique_id': series identifier (retailer_id + sku_id)
        - 'ds': datestamp
        - 'y': target
        - Additional columns as exogenous variables
        """
        nf_df = df.copy()
        nf_df["unique_id"] = nf_df["retailer_id"] + "_" + nf_df["sku_id"]
        nf_df = nf_df.rename(columns={"date": "ds", target: "y"})
        nf_df["ds"] = pd.to_datetime(nf_df["ds"])
        nf_df = nf_df.sort_values(["unique_id", "ds"])
        
        return nf_df
    
    def _get_exog_columns(self, df: pd.DataFrame) -> tuple:
        """Identify historic and future exogenous columns."""
        # Historic exogenous: observed values (lags, rolling stats)
        hist_exog = [c for c in df.columns if c.startswith(("lag_", "rolling_", "days_since", "order_freq", "growth_"))]
        
        # Future exogenous: known future values (calendar)
        futr_exog = [c for c in df.columns if c.startswith(("is_", "week_of_year", "dengue_", "respiratory_"))]
        
        # Static exogenous: time-invariant
        stat_exog = [c for c in df.columns if c.endswith("_encoded") or c in ["years_active", "is_critical",
                                                                                "shelf_life_months", "supplier_lead_time_days"]]
        
        return hist_exog, futr_exog, stat_exog
    
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame = None,
        target: str = "quantity_ordered",
        max_series: int = 500,
    ) -> "TFTForecaster":
        """
        Train TFT model.
        
        Args:
            df_train: Training DataFrame with all features
            df_val: Validation DataFrame (optional)
            target: Target column name
            max_series: Maximum number of time series to train on (for speed)
        """
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import TFT
            from neuralforecast.losses.pytorch import MQLoss
        except ImportError:
            logger.error("NeuralForecast not installed. Install with: pip install neuralforecast")
            raise
        
        logger.info(f"[{self.name}] Preparing data for TFT...")
        
        nf_train = self._prepare_data(df_train, target)
        
        # Limit series count for feasibility
        unique_ids = nf_train["unique_id"].unique()
        if len(unique_ids) > max_series:
            # Sample the most active series
            series_counts = nf_train.groupby("unique_id").size()
            top_ids = series_counts.nlargest(max_series).index
            nf_train = nf_train[nf_train["unique_id"].isin(top_ids)]
            logger.info(f"[{self.name}] Limited to top {max_series} series by volume")
        
        # Filter to series with enough history
        min_len = self.params.get("input_size", 52) + self.params.get("h", 13)
        lengths = nf_train.groupby("unique_id").size()
        valid_ids = lengths[lengths >= min_len].index
        nf_train = nf_train[nf_train["unique_id"].isin(valid_ids)]
        
        logger.info(f"[{self.name}] Training on {nf_train['unique_id'].nunique()} series, "
                     f"{len(nf_train):,} total rows")
        
        hist_exog, futr_exog, stat_exog = self._get_exog_columns(nf_train)
        
        # Keep only columns that exist and have no NaN
        keep_cols = ["unique_id", "ds", "y"]
        for col_list in [hist_exog, futr_exog, stat_exog]:
            for c in col_list:
                if c in nf_train.columns and nf_train[c].notna().all():
                    keep_cols.append(c)
        
        hist_exog = [c for c in hist_exog if c in keep_cols]
        futr_exog = [c for c in futr_exog if c in keep_cols]
        stat_exog = [c for c in stat_exog if c in keep_cols]
        
        nf_train = nf_train[keep_cols].copy()
        
        # Build TFT model
        tft = TFT(
            h=self.params.get("h", 13),
            input_size=self.params.get("input_size", 52),
            hidden_size=self.params.get("hidden_size", 64),
            n_head=self.params.get("n_head", 4),
            learning_rate=self.params.get("learning_rate", 1e-3),
            max_steps=self.params.get("max_steps", 500),
            batch_size=self.params.get("batch_size", 64),
            windows_batch_size=self.params.get("windows_batch_size", 256),
            scaler_type=self.params.get("scaler_type", "robust"),
            random_seed=self.params.get("random_seed", 42),
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
            hist_exog_list=hist_exog if hist_exog else None,
            futr_exog_list=futr_exog if futr_exog else None,
            stat_exog_list=stat_exog if stat_exog else None,
            enable_progress_bar=True,
        )
        
        self.nf = NeuralForecast(models=[tft], freq="W-MON")
        
        logger.info(f"[{self.name}] Starting TFT training...")
        self.nf.fit(df=nf_train)
        
        logger.info(f"[{self.name}] TFT training complete!")
        return self
    
    def predict(self, df: pd.DataFrame, target: str = "quantity_ordered") -> pd.DataFrame:
        """
        Generate multi-horizon forecasts.
        
        Returns DataFrame with columns:
        - unique_id, ds
        - TFT (median prediction)
        - TFT-lo-90 (P10), TFT-hi-90 (P90)
        """
        if self.nf is None:
            raise ValueError("Model not trained. Call train() first.")
        
        nf_df = self._prepare_data(df, target)
        
        # Only predict for series that were trained on
        trained_ids = set(self.nf.dataset.uids) if hasattr(self.nf, "dataset") else set()
        if trained_ids:
            nf_df = nf_df[nf_df["unique_id"].isin(trained_ids)]
        
        forecasts = self.nf.predict(df=nf_df)
        
        # Clamp to non-negative
        for col in forecasts.columns:
            if col not in ["unique_id", "ds"]:
                forecasts[col] = forecasts[col].clip(lower=0)
        
        return forecasts
