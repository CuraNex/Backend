"""
PharmaFlow AI — Feature Engineering Pipeline
=============================================
Transforms raw transaction data into ML-ready features:
- Temporal features: lags, rolling statistics, growth rates, seasonal indices
- Retailer profile features: RFM scores, type/tier encodings
- SKU metadata features: category, brand, price encodings
- External signals: holiday/festival flags, health indices
- Train/validation/test temporal split with zero data leakage
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("FeatureEngineering")


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, group_cols: list, target: str = "quantity_ordered") -> pd.DataFrame:
    """
    Create lag features for each retailer-SKU pair.
    
    Lags: 1, 2, 3, 4 weeks (short-term momentum), 13 weeks (quarterly),
    52 weeks (annual seasonality).
    
    IMPORTANT: We sort by group + date first, then shift within each group to
    prevent data leakage across retailer-SKU pairs.
    """
    df = df.sort_values(group_cols + ["date"]).copy()
    
    for lag in cfg.LAG_WEEKS:
        col_name = f"lag_{lag}w"
        df[col_name] = df.groupby(group_cols)[target].shift(lag)
        logger.info(f"  Created {col_name}")
    
    return df


def add_rolling_features(df: pd.DataFrame, group_cols: list, target: str = "quantity_ordered") -> pd.DataFrame:
    """
    Rolling statistics — computed from SHIFTED values to avoid leakage.
    
    For each window size, we compute mean and std from lagged data
    (shifted by 1 to exclude current week).
    """
    df = df.sort_values(group_cols + ["date"]).copy()
    
    for window in cfg.ROLLING_WINDOWS:
        shifted = df.groupby(group_cols)[target].shift(1)
        
        roll_mean = shifted.groupby(df[group_cols].apply(tuple, axis=1)).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        roll_std = shifted.groupby(df[group_cols].apply(tuple, axis=1)).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
        df[f"rolling_mean_{window}w"] = roll_mean
        df[f"rolling_std_{window}w"] = roll_std.fillna(0)
        
        logger.info(f"  Created rolling_mean_{window}w, rolling_std_{window}w")
    
    return df


def add_order_pattern_features(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Compute ordering pattern features:
    - days_since_last_order: reorder cycle signal
    - order_frequency_90d: ordering regularity
    - growth_rate_6m: long-term trajectory
    """
    df = df.sort_values(group_cols + ["date"]).copy()
    
    # Days since last order
    df["days_since_last_order"] = df.groupby(group_cols)["date"].diff().dt.days.fillna(7)
    
    # Order frequency in past ~13 weeks (≈90 days)
    def count_orders_past_n(group, n=13):
        return group.rolling(n, min_periods=1).count()
    
    df["order_frequency_13w"] = df.groupby(group_cols)["quantity_ordered"].transform(
        lambda x: x.shift(1).rolling(13, min_periods=1).count()
    ).fillna(0)
    
    # Growth rate: compare recent 13-week average to prior 13-week average
    shifted = df.groupby(group_cols)["quantity_ordered"].shift(1)
    group_keys = df[group_cols].apply(tuple, axis=1)
    
    recent_avg = shifted.groupby(group_keys).transform(
        lambda x: x.rolling(13, min_periods=1).mean()
    )
    prior_avg = shifted.groupby(group_keys).transform(
        lambda x: x.shift(13).rolling(13, min_periods=1).mean()
    )
    
    df["growth_rate_6m"] = ((recent_avg - prior_avg) / prior_avg.replace(0, np.nan)).fillna(0).clip(-2, 2)
    
    logger.info("  Created days_since_last_order, order_frequency_13w, growth_rate_6m")
    return df


def add_seasonal_index(df: pd.DataFrame, group_cols: list, target: str = "quantity_ordered") -> pd.DataFrame:
    """
    Compute a normalized seasonal index by ISO week.
    
    For each retailer-SKU pair, the seasonal index is the ratio of the
    average demand in that week-of-year to the overall average demand.
    This captures annual cyclicality.
    """
    # Compute average demand per group per week_of_year
    weekly_avg = df.groupby(group_cols + ["week_of_year"])[target].transform("mean")
    overall_avg = df.groupby(group_cols)[target].transform("mean")
    
    df["seasonal_index"] = (weekly_avg / overall_avg.replace(0, 1.0)).clip(0, 5)
    
    logger.info("  Created seasonal_index")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RETAILER PROFILE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_rfm_features(transactions: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features per retailer.
    
    Uses only data up to cutoff_date to prevent leakage.
    """
    df = transactions[transactions["date"] <= cutoff_date].copy()
    
    rfm = df.groupby("retailer_id").agg(
        rfm_recency=("date", lambda x: (cutoff_date - x.max()).days // 7),
        rfm_frequency=("date", "nunique"),
        rfm_monetary=("quantity_ordered", "sum"),
        sku_breadth=("sku_id", "nunique"),
    ).reset_index()
    
    logger.info(f"  Computed RFM features for {len(rfm)} retailers (cutoff: {cutoff_date.date()})")
    return rfm


def add_retailer_features(df: pd.DataFrame, retailers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge retailer profile features into the main dataset.
    Includes static attributes + computed RFM.
    """
    # Merge static retailer attributes
    df = df.merge(
        retailers[["retailer_id", "retailer_type", "district", "tier", "years_active"]],
        on="retailer_id",
        how="left",
    )
    
    # RFM — compute using the latest date in training data
    train_cutoff = df[df["split"] == "train"]["date"].max() if "split" in df.columns else df["date"].max()
    rfm = compute_rfm_features(transactions, train_cutoff)
    df = df.merge(rfm, on="retailer_id", how="left")
    
    # Encode categoricals as integers for GBM models
    for col in ["retailer_type", "district", "tier"]:
        df[f"{col}_encoded"] = df[col].astype("category").cat.codes
    
    logger.info("  Merged retailer features (static + RFM)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SKU METADATA FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_sku_features(df: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """Merge SKU metadata features."""
    df = df.merge(
        skus[["sku_id", "therapeutic_category", "brand_type", "price_band",
              "shelf_life_months", "supplier_lead_time_days", "is_critical"]],
        on="sku_id",
        how="left",
    )
    
    # Encode categoricals
    for col in ["therapeutic_category", "brand_type", "price_band"]:
        df[f"{col}_encoded"] = df[col].astype("category").cat.codes
    
    df["is_critical"] = df["is_critical"].astype(int)
    
    logger.info("  Merged SKU metadata features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL / CONTEXTUAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Merge calendar features (holidays, festivals, school terms)."""
    cal_cols = [
        "date", "week_of_year", "is_public_holiday", "is_vesak_week",
        "is_new_year_period", "is_month_end", "is_school_term",
    ]
    # Avoid duplication of week_of_year if it exists
    merge_cols = [c for c in cal_cols if c not in df.columns or c == "date"]
    
    df = df.merge(calendar[merge_cols], on="date", how="left")
    
    logger.info("  Merged calendar features")
    return df


def add_health_features(df: pd.DataFrame, health_signals: pd.DataFrame) -> pd.DataFrame:
    """Merge district-level health indices."""
    df = df.merge(
        health_signals[["date", "district", "dengue_index", "respiratory_index"]],
        on=["date", "district"],
        how="left",
    )
    
    # Fill missing health signals with 0
    df["dengue_index"] = df["dengue_index"].fillna(0)
    df["respiratory_index"] = df["respiratory_index"].fillna(0)
    
    logger.info("  Merged health signal features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split data temporally into train/validation/test sets.
    
    Uses the sorted date range to ensure:
    - Train: first 70% of weeks
    - Validation: next 15%
    - Test: final 15%
    
    This prevents any future data leaking into training.
    """
    dates_sorted = sorted(df["date"].unique())
    n = len(dates_sorted)
    
    train_end = dates_sorted[int(n * cfg.TRAIN_RATIO) - 1]
    val_end = dates_sorted[int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)) - 1]
    
    df["split"] = "test"
    df.loc[df["date"] <= train_end, "split"] = "train"
    df.loc[(df["date"] > train_end) & (df["date"] <= val_end), "split"] = "val"
    
    counts = df["split"].value_counts()
    logger.info(f"  Temporal split — Train: {counts.get('train', 0):,}, "
                f"Val: {counts.get('val', 0):,}, Test: {counts.get('test', 0):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns() -> dict:
    """Return column lists for ML models."""
    lag_cols = [f"lag_{l}w" for l in cfg.LAG_WEEKS]
    rolling_cols = []
    for w in cfg.ROLLING_WINDOWS:
        rolling_cols += [f"rolling_mean_{w}w", f"rolling_std_{w}w"]
    
    temporal_cols = lag_cols + rolling_cols + [
        "days_since_last_order", "order_frequency_13w", "growth_rate_6m", "seasonal_index",
    ]
    
    retailer_cols = [
        "retailer_type_encoded", "district_encoded", "tier_encoded",
        "years_active", "rfm_recency", "rfm_frequency", "rfm_monetary", "sku_breadth",
    ]
    
    sku_cols = [
        "therapeutic_category_encoded", "brand_type_encoded", "price_band_encoded",
        "shelf_life_months", "supplier_lead_time_days", "is_critical",
    ]
    
    calendar_cols = [
        "week_of_year", "is_public_holiday", "is_vesak_week",
        "is_new_year_period", "is_month_end", "is_school_term",
    ]
    
    health_cols = ["dengue_index", "respiratory_index"]
    
    all_features = temporal_cols + retailer_cols + sku_cols + calendar_cols + health_cols
    
    return {
        "temporal": temporal_cols,
        "retailer": retailer_cols,
        "sku": sku_cols,
        "calendar": calendar_cols,
        "health": health_cols,
        "all": all_features,
        "target": "quantity_ordered",
        "categorical": [
            "retailer_type_encoded", "district_encoded", "tier_encoded",
            "therapeutic_category_encoded", "brand_type_encoded", "price_band_encoded",
        ],
    }


def build_features(
    transactions: pd.DataFrame,
    retailers: pd.DataFrame,
    skus: pd.DataFrame,
    calendar: pd.DataFrame,
    health_signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Orchestrates all feature transformations in the correct order to
    avoid data leakage and ensure feature completeness.
    
    Returns a DataFrame ready for model training with a 'split' column.
    """
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline — Starting")
    logger.info("=" * 60)
    
    group_cols = ["retailer_id", "sku_id"]
    df = transactions.copy()
    
    # 1. Temporal split FIRST (so RFM and other features respect it)
    logger.info("[1/7] Temporal split...")
    df = temporal_split(df)
    
    # 2. Lag features
    logger.info("[2/7] Lag features...")
    df = add_lag_features(df, group_cols)
    
    # 3. Rolling statistics
    logger.info("[3/7] Rolling statistics...")
    df = add_rolling_features(df, group_cols)
    
    # 4. Order pattern features
    logger.info("[4/7] Order pattern features...")
    df = add_order_pattern_features(df, group_cols)
    
    # 5. Seasonal index
    logger.info("[5/7] Seasonal index...")
    df = add_seasonal_index(df, group_cols)
    
    # 6. Retailer, SKU, calendar, health features
    logger.info("[6/7] Retailer + SKU + calendar features...")
    df = add_retailer_features(df, retailers, transactions)
    df = add_sku_features(df, skus)
    df = add_calendar_features(df, calendar)
    
    logger.info("[7/7] Health signal features...")
    df = add_health_features(df, health_signals)
    
    # Drop rows with NaN in critical lag features (first few weeks per series)
    initial_len = len(df)
    feature_cols = get_feature_columns()["all"]
    df = df.dropna(subset=[f"lag_{cfg.LAG_WEEKS[0]}w"])  # At least lag_1w must exist
    
    # Fill remaining NaNs in features with 0
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    logger.info(f"Dropped {initial_len - len(df):,} rows with insufficient history")
    logger.info(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    logger.info("=" * 60)
    logger.info("Feature Engineering Complete!")
    logger.info("=" * 60)
    
    return df
