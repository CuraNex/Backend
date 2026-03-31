"""
PharmaFlow AI — Statistical Baselines
=======================================
Baseline models for benchmarking ML performance:
- Seasonal Naïve (same-week-last-year)
- Exponential Smoothing (Holt-Winters with multiplicative seasonality)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import setup_logger

logger = setup_logger("Baselines")


class SeasonalNaive:
    """
    Seasonal Naïve baseline — predicts demand as the same-week-last-year value.
    
    This is the minimum bar that ML models must beat.
    If a model can't outperform seasonal naïve, it adds no value.
    """
    
    def __init__(self, seasonal_period: int = 52, name: str = "seasonal_naive"):
        self.seasonal_period = seasonal_period
        self.name = name
        self.history = {}  # {(retailer_id, sku_id): {week: qty}}
    
    def fit(self, df: pd.DataFrame) -> "SeasonalNaive":
        """Store historical demand by (retailer, sku, week_of_year)."""
        for _, row in df.iterrows():
            key = (str(row["retailer_id"]), str(row["sku_id"]))
            week = int(row["week_of_year"])
            
            if key not in self.history:
                self.history[key] = {}
            
            # Keep the most recent year's value for each week
            self.history[key][week] = row["quantity_ordered"]
        
        logger.info(f"[{self.name}] Fitted on {len(self.history)} retailer-SKU pairs")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using same-week-last-year demand."""
        predictions = []
        
        for _, row in df.iterrows():
            key = (str(row["retailer_id"]), str(row["sku_id"]))
            week = int(row["week_of_year"])
            
            if key in self.history and week in self.history[key]:
                pred = self.history[key][week]
            else:
                # Fallback: average of known weeks for this pair
                if key in self.history and self.history[key]:
                    pred = np.mean(list(self.history[key].values()))
                else:
                    pred = 0
            
            predictions.append(max(0, pred))
        
        return np.array(predictions)


class ExponentialSmoothing:
    """
    Holt-Winters Exponential Smoothing baseline.
    
    Captures level, trend, and multiplicative seasonality.
    Fitted per retailer-SKU pair on the aggregate weekly series.
    """
    
    def __init__(self, seasonal_period: int = 52, name: str = "ets"):
        self.seasonal_period = seasonal_period
        self.name = name
        self.models = {}  # {(retailer_id, sku_id): fitted model}
    
    def fit(self, df: pd.DataFrame, max_pairs: int = 500) -> "ExponentialSmoothing":
        """Fit ETS on each retailer-SKU pair's time series."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
        except ImportError:
            logger.error("statsmodels not installed")
            raise
        
        pairs = df.groupby(["retailer_id", "sku_id"]).size().nlargest(max_pairs).index
        
        for i, (rid, sid) in enumerate(pairs):
            series = df[(df["retailer_id"] == rid) & (df["sku_id"] == sid)].sort_values("date")
            y = series["quantity_ordered"].values.astype(float)
            
            if len(y) < self.seasonal_period + 2:
                # Not enough data for seasonal model — use simple exponential smoothing
                try:
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                    model = SimpleExpSmoothing(y, initialization_method="estimated").fit()
                    self.models[(rid, sid)] = model
                except Exception:
                    pass
                continue
            
            try:
                # Ensure positive values for multiplicative seasonality
                y = np.maximum(y, 0.1)
                model = HW(
                    y,
                    trend="add",
                    seasonal="mul",
                    seasonal_periods=min(self.seasonal_period, len(y) // 2),
                    initialization_method="estimated",
                ).fit(optimized=True)
                self.models[(rid, sid)] = model
            except Exception:
                # Fallback to additive if multiplicative fails
                try:
                    model = HW(
                        y,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=min(self.seasonal_period, len(y) // 2),
                        initialization_method="estimated",
                    ).fit(optimized=True)
                    self.models[(rid, sid)] = model
                except Exception:
                    pass
            
            if (i + 1) % 100 == 0:
                logger.info(f"[{self.name}] Fitted {i + 1}/{len(pairs)} pairs")
        
        logger.info(f"[{self.name}] Fitted {len(self.models)} models")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate forecasts from fitted ETS models."""
        predictions = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            key = (row["retailer_id"], row["sku_id"])
            if key in self.models:
                try:
                    # Forecast 1 step ahead
                    pred = self.models[key].forecast(1)[0]
                    predictions[i] = max(0, pred)
                except Exception:
                    predictions[i] = 0
        
        return predictions
