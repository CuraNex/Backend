"""
CuraNex AI — Temporal Fusion Transformer (TFT)
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
        # Static exogenous: time-invariant (identify first to exclude from others)
        stat_exog = [c for c in df.columns if c.endswith("_encoded") or c in ["years_active", "is_critical",
                                                                                "shelf_life_months", "supplier_lead_time_days"]]
        stat_set = set(stat_exog)
        
        # Historic exogenous: observed values (lags, rolling stats)
        hist_exog = [c for c in df.columns if c.startswith(("lag_", "rolling_", "days_since", "order_freq", "growth_"))
                     and c not in stat_set]
        
        # Future exogenous: known future values (calendar) — exclude static columns
        futr_exog = [c for c in df.columns if c.startswith(("is_", "week_of_year", "dengue_", "respiratory_"))
                     and c not in stat_set]
        
        return hist_exog, futr_exog, stat_exog
    
    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame = None,
        target: str = "quantity_ordered",
        max_series: int = 10000000000,
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
        
        # Keep only columns that exist and have no NaN (deduplicated)
        keep_cols = ["unique_id", "ds", "y"]
        seen = set(keep_cols)
        for col_list in [hist_exog, futr_exog, stat_exog]:
            for c in col_list:
                if c not in seen and c in nf_train.columns and nf_train[c].notna().all():
                    keep_cols.append(c)
                    seen.add(c)
        
        hist_exog = [c for c in hist_exog if c in seen]
        futr_exog = [c for c in futr_exog if c in seen]
        stat_exog = [c for c in stat_exog if c in seen]
        
        nf_train = nf_train[keep_cols].copy()
        
        # ── Separate static exogenous into a static_df ──
        # with one row per unique_id, passed via static_df= argument.
        static_df = None
        if stat_exog:
            static_df = (
                nf_train.groupby("unique_id")[stat_exog]
                .first()
                .reset_index()
            )
            # Remove static columns from the temporal DataFrame
            nf_train = nf_train.drop(columns=stat_exog)
        
        # Store for predict-time reuse
        self._stat_exog = stat_exog
        self._hist_exog = hist_exog
        self._futr_exog = futr_exog
        
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
        self.nf.fit(df=nf_train, static_df=static_df)
        
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
        trained_ids = set(self.nf.uids) if hasattr(self.nf, "uids") else set()
        if trained_ids:
            nf_df = nf_df[nf_df["unique_id"].isin(trained_ids)]
        
        # Build static_df for prediction
        static_df = None
        stat_exog = getattr(self, "_stat_exog", [])
        if stat_exog:
            available_stat = [c for c in stat_exog if c in nf_df.columns]
            if available_stat:
                static_df = (
                    nf_df.groupby("unique_id")[available_stat]
                    .first()
                    .reset_index()
                )
                nf_df = nf_df.drop(columns=available_stat)
        
        # Build futr_df — NF needs future exog values for the FORECAST HORIZON
        futr_df = None
        futr_exog = getattr(self, "_futr_exog", [])
        if futr_exog:
            # Get the expected future date grid from NeuralForecast
            futr_df = self.nf.make_future_dataframe(df=nf_df)
            
            # Populate calendar features from the future dates
            futr_df["ds"] = pd.to_datetime(futr_df["ds"])
            futr_df["week_of_year"] = futr_df["ds"].dt.isocalendar().week.astype(int)
            
            # Binary calendar flags based on week_of_year
            import config as cfg
            futr_df["is_public_holiday"] = 0
            futr_df["is_vesak_week"] = futr_df["week_of_year"].isin(cfg.SRI_LANKA_HOLIDAYS.get("vesak", [])).astype(int)
            futr_df["is_new_year_period"] = futr_df["week_of_year"].isin(cfg.SRI_LANKA_HOLIDAYS.get("sinhala_tamil_new_year", [])).astype(int)
            futr_df["is_month_end"] = 0  # approximate
            futr_df["is_school_term"] = 1  # default to in-term
            
            # Health signals — use seasonal averages as proxy for future
            futr_df["dengue_index"] = futr_df["week_of_year"].isin(cfg.DENGUE_PEAK_WEEKS).astype(float) * 0.7
            futr_df["respiratory_index"] = futr_df["week_of_year"].isin(cfg.RESPIRATORY_PEAK_WEEKS).astype(float) * 0.6
            
            # Keep only columns the model was trained with
            futr_keep = ["unique_id", "ds"] + [c for c in futr_exog if c in futr_df.columns]
            futr_df = futr_df[futr_keep]
        
        # Keep required temporal columns — include hist + futr exog (NF needs both in df)
        hist_exog = getattr(self, "_hist_exog", [])
        keep = ["unique_id", "ds", "y"]
        keep += [c for c in hist_exog if c in nf_df.columns]
        keep += [c for c in futr_exog if c in nf_df.columns]
        nf_df = nf_df[[c for c in keep if c in nf_df.columns]]
        
        forecasts = self.nf.predict(df=nf_df, static_df=static_df, futr_df=futr_df)
        
        # Clamp to non-negative
        for col in forecasts.columns:
            if col not in ["unique_id", "ds"]:
                forecasts[col] = forecasts[col].clip(lower=0)
        
        return forecasts
