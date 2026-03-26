"""
PharmaFlow AI — Synthetic Data Generator
=========================================
Generates realistic pharmaceutical distribution data mirroring Hemas Pharmaceuticals'
network: retailers, SKUs, weekly transactions, calendar events, and health signals.

Key realism features:
- Heterogeneous demand profiles per retailer type (hospital vs standalone vs chain)
- Seasonal patterns: dengue (Jun-Nov), respiratory (Dec-Feb), festival spikes
- Intermittent demand for low-volume retailer-SKU pairs
- Bullwhip effect simulation in ordering patterns
- Trend components (growing / declining SKUs)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger, save_csv, set_seed

logger = setup_logger("DataGenerator")


# ─────────────────────────────────────────────────────────────────────────────
# 1. RETAILER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_retailers(n: int = cfg.N_RETAILERS, seed: int = cfg.SEED) -> pd.DataFrame:
    """
    Generate retailer master data with realistic Sri Lankan pharmacy profiles.
    
    Each retailer has:
    - A type (hospital_attached, standalone, chain_outlet) affecting order volume
    - District location affecting demand patterns
    - Tier (A/B/C) correlated with type and volume
    - Years active affecting data availability
    """
    rng = np.random.default_rng(seed)
    
    retailer_ids = [f"R{i:04d}" for i in range(1, n + 1)]
    
    # Type distribution — hospital pharmacies are higher-volume
    types = rng.choice(cfg.RETAILER_TYPES, size=n, p=cfg.RETAILER_TYPE_PROBS)
    
    # District — weighted toward Western Province (Colombo, Gampaha, Kalutara)
    district_weights = np.ones(len(cfg.SRI_LANKA_DISTRICTS))
    district_weights[0] = 8.0   # Colombo
    district_weights[1] = 5.0   # Gampaha
    district_weights[2] = 3.0   # Kalutara
    district_weights[3] = 3.0   # Kandy
    district_weights = district_weights / district_weights.sum()
    districts = rng.choice(cfg.SRI_LANKA_DISTRICTS, size=n, p=district_weights)
    
    # Tier — correlated with type (hospital_attached more likely to be A-tier)
    tiers = []
    for t in types:
        if t == "hospital_attached":
            tier = rng.choice(cfg.RETAILER_TIERS, p=[0.50, 0.35, 0.15])
        elif t == "chain_outlet":
            tier = rng.choice(cfg.RETAILER_TIERS, p=[0.30, 0.50, 0.20])
        else:  # standalone
            tier = rng.choice(cfg.RETAILER_TIERS, p=[0.10, 0.50, 0.40])
        tiers.append(tier)
    
    # Years active — newer pharmacies have less history
    years_active = rng.exponential(scale=5.0, size=n).clip(0.5, 20).round(1)
    
    df = pd.DataFrame({
        "retailer_id": retailer_ids,
        "retailer_type": types,
        "district": districts,
        "tier": tiers,
        "years_active": years_active,
    })
    
    logger.info(f"Generated {n} retailers — Types: {dict(zip(*np.unique(types, return_counts=True)))}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SKU GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_skus(n: int = cfg.N_SKUS, seed: int = cfg.SEED) -> pd.DataFrame:
    """
    Generate SKU master data representing pharmaceutical products.
    
    Products span 12 therapeutic categories with varied brand types,
    price bands, shelf lives, and supplier lead times.
    """
    rng = np.random.default_rng(seed)
    
    sku_ids = [f"SKU{i:04d}" for i in range(1, n + 1)]
    
    categories = rng.choice(cfg.THERAPEUTIC_CATEGORIES, size=n)
    brand_types = rng.choice(cfg.BRAND_TYPES, size=n, p=cfg.BRAND_TYPE_PROBS)
    price_bands = rng.choice(cfg.PRICE_BANDS, size=n, p=cfg.PRICE_BAND_PROBS)
    
    # Shelf life: 12-60 months, longer for tablets, shorter for liquids
    shelf_life = rng.integers(12, 61, size=n)
    
    # Supplier lead time: 14-180 days (imported products take longer)
    lead_times = rng.integers(14, 181, size=n)
    
    # Criticality flag — antibiotics, cardiovascular, antidiabetics are critical
    critical_categories = {"antibiotics", "cardiovascular", "antidiabetics", "antipyretics"}
    is_critical = [cat in critical_categories for cat in categories]
    
    df = pd.DataFrame({
        "sku_id": sku_ids,
        "therapeutic_category": categories,
        "brand_type": brand_types,
        "price_band": price_bands,
        "shelf_life_months": shelf_life,
        "supplier_lead_time_days": lead_times,
        "is_critical": is_critical,
    })
    
    logger.info(f"Generated {n} SKUs — Categories: {dict(zip(*np.unique(categories, return_counts=True)))}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALENDAR & EXTERNAL SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def generate_calendar(n_weeks: int = cfg.N_WEEKS, start_date: str = cfg.START_DATE) -> pd.DataFrame:
    """
    Generate Sri Lankan calendar with holidays, festivals, and seasonal indicators.
    """
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    
    weeks = dates.isocalendar().week.astype(int).values
    years = dates.year.values
    
    # Holiday flags
    is_public_holiday = np.zeros(n_weeks, dtype=int)
    is_vesak = np.zeros(n_weeks, dtype=int)
    is_new_year = np.zeros(n_weeks, dtype=int)
    is_month_end = np.zeros(n_weeks, dtype=int)
    is_school_term = np.zeros(n_weeks, dtype=int)
    
    for i, w in enumerate(weeks):
        if w in cfg.SRI_LANKA_HOLIDAYS.get("vesak", []):
            is_vesak[i] = 1
            is_public_holiday[i] = 1
        if w in cfg.SRI_LANKA_HOLIDAYS.get("sinhala_tamil_new_year", []):
            is_new_year[i] = 1
            is_public_holiday[i] = 1
        if w in cfg.SRI_LANKA_HOLIDAYS.get("christmas_new_year", []):
            is_public_holiday[i] = 1
        # Check all other holidays
        for holiday, wks in cfg.SRI_LANKA_HOLIDAYS.items():
            if w in wks:
                is_public_holiday[i] = 1
        
        # Month-end flag (last week of each month)
        if dates[i].day >= 25:
            is_month_end[i] = 1
        
        # School terms (roughly: Jan-April, May-Aug, Sep-Dec with breaks)
        if w not in [15, 16, 32, 33, 51, 52]:
            is_school_term[i] = 1
    
    df = pd.DataFrame({
        "date": dates,
        "year": years,
        "week_of_year": weeks,
        "is_public_holiday": is_public_holiday,
        "is_vesak_week": is_vesak,
        "is_new_year_period": is_new_year,
        "is_month_end": is_month_end,
        "is_school_term": is_school_term,
    })
    
    logger.info(f"Generated calendar: {n_weeks} weeks from {start_date}")
    return df


def generate_health_signals(
    n_weeks: int = cfg.N_WEEKS,
    districts: list = None,
    start_date: str = cfg.START_DATE,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """
    Generate district-level health surveillance indices (dengue, respiratory).
    
    Dengue peaks June-November; respiratory peaks December-February.
    Indices are 0-100 with realistic seasonal curves and district variation.
    """
    rng = np.random.default_rng(seed)
    if districts is None:
        districts = cfg.SRI_LANKA_DISTRICTS
    
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    weeks = dates.isocalendar().week.astype(int).values
    
    records = []
    for dist in districts:
        # Base intensity varies by district (Western Province higher dengue)
        dengue_base = 40 if dist in ["Colombo", "Gampaha", "Kalutara"] else 20
        resp_base = 30
        
        for i, (date, w) in enumerate(zip(dates, weeks)):
            # Dengue: sinusoidal peaking around week 35 (August)
            dengue_seasonal = 30 * np.sin(2 * np.pi * (w - 22) / 26) if w in cfg.DENGUE_PEAK_WEEKS else -10
            dengue_idx = max(0, min(100, dengue_base + dengue_seasonal + rng.normal(0, 8)))
            
            # Respiratory: peaks around week 1 (January)
            resp_seasonal = 25 * np.cos(2 * np.pi * (w - 1) / 52)
            resp_idx = max(0, min(100, resp_base + resp_seasonal + rng.normal(0, 6)))
            
            records.append({
                "date": date,
                "district": dist,
                "week_of_year": w,
                "dengue_index": round(dengue_idx, 1),
                "respiratory_index": round(resp_idx, 1),
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated health signals: {len(df)} rows ({len(districts)} districts × {n_weeks} weeks)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRANSACTION GENERATION (The core)
# ─────────────────────────────────────────────────────────────────────────────

def _retailer_base_demand(retailer_type: str, tier: str, rng) -> float:
    """Determine base weekly demand for a retailer based on type and tier."""
    base = {
        "hospital_attached": {"A": 150, "B": 80, "C": 40},
        "chain_outlet":      {"A": 100, "B": 50, "C": 25},
        "standalone":        {"A": 60,  "B": 30, "C": 12},
    }
    return base.get(retailer_type, {}).get(tier, 20) * rng.uniform(0.6, 1.4)


def _sku_demand_modifier(category: str, brand_type: str, rng) -> float:
    """SKU-level demand multiplier based on category and brand."""
    cat_mult = {
        "analgesics": 1.5, "antibiotics": 1.3, "antipyretics": 1.4,
        "cardiovascular": 1.0, "antidiabetics": 0.9, "respiratory": 1.1,
        "gastrointestinal": 1.0, "dermatological": 0.6, "antihistamines": 0.8,
        "vitamins_supplements": 0.7, "antihypertensives": 0.8, "otc_general": 1.0,
    }
    brand_mult = {"branded": 0.8, "generic": 1.2, "otc": 1.0}
    return cat_mult.get(category, 1.0) * brand_mult.get(brand_type, 1.0) * rng.uniform(0.7, 1.3)


def generate_transactions(
    retailers: pd.DataFrame,
    skus: pd.DataFrame,
    calendar: pd.DataFrame,
    health_signals: pd.DataFrame,
    n_weeks: int = cfg.N_WEEKS,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """
    Generate weekly transaction data for all retailer-SKU pairs.
    
    This is the heart of the synthetic data generator. For each retailer-SKU pair,
    we simulate a realistic demand time series with:
    
    1. Base demand level (function of retailer type/tier and SKU category)
    2. Trend component (some SKUs growing, some declining)
    3. Annual seasonality (week-of-year effect)
    4. Dengue/respiratory disease uplift for relevant categories
    5. Festival/holiday demand spikes
    6. Intermittency (some pairs only order occasionally)
    7. Noise + bullwhip amplification
    8. Fill rate simulation (quantity_fulfilled ≤ quantity_ordered)
    """
    rng = np.random.default_rng(seed)
    dates = calendar["date"].values
    weeks = calendar["week_of_year"].values
    
    # Pre-compute health signals lookup: district -> week_idx -> (dengue, resp)
    health_lookup = {}
    for _, row in health_signals.iterrows():
        key = (row["district"], row["date"])
        health_lookup[key] = (row["dengue_index"], row["respiratory_index"])
    
    # Categories affected by disease outbreaks
    dengue_affected = {"analgesics", "antipyretics", "antibiotics", "otc_general"}
    respiratory_affected = {"respiratory", "antibiotics", "antihistamines", "antipyretics"}
    
    all_records = []
    n_retailers = len(retailers)
    n_skus_total = len(skus)
    
    # Not every retailer orders every SKU — determine active SKU set per retailer
    for r_idx, retailer in retailers.iterrows():
        rid = retailer["retailer_id"]
        rtype = retailer["retailer_type"]
        tier = retailer["tier"]
        district = retailer["district"]
        
        # Number of actively ordered SKUs depends on retailer type
        if rtype == "hospital_attached":
            n_active = int(n_skus_total * rng.uniform(0.5, 0.85))
        elif rtype == "chain_outlet":
            n_active = int(n_skus_total * rng.uniform(0.3, 0.6))
        else:  # standalone
            n_active = int(n_skus_total * rng.uniform(0.15, 0.4))
        
        active_sku_indices = rng.choice(n_skus_total, size=n_active, replace=False)
        
        for s_idx in active_sku_indices:
            sku = skus.iloc[s_idx]
            sid = sku["sku_id"]
            category = sku["therapeutic_category"]
            brand = sku["brand_type"]
            
            # Base demand for this retailer-SKU pair
            base = _retailer_base_demand(rtype, tier, rng)
            sku_mod = _sku_demand_modifier(category, brand, rng)
            pair_base = base * sku_mod / n_skus_total * 10  # scale to per-SKU level
            
            # Intermittency: probability of ordering in any given week
            if tier == "A":
                order_prob = rng.uniform(0.7, 1.0)
            elif tier == "B":
                order_prob = rng.uniform(0.4, 0.8)
            else:
                order_prob = rng.uniform(0.15, 0.5)
            
            # Trend: slight growth or decline
            trend_slope = rng.normal(0, 0.002)  # per week
            
            # Generate weekly time series
            for w_idx in range(n_weeks):
                # Skip weeks where retailer doesn't order
                if rng.random() > order_prob:
                    continue
                
                date = dates[w_idx]
                week = weeks[w_idx]
                
                # 1. Base + trend
                qty = pair_base * (1 + trend_slope * w_idx)
                
                # 2. Annual seasonality (sinusoidal)
                seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * (week - 1) / 52)
                qty *= seasonality
                
                # 3. Disease outbreak uplift
                h_key = (district, date)
                if h_key in health_lookup:
                    dengue_idx, resp_idx = health_lookup[h_key]
                    if category in dengue_affected:
                        qty *= (1 + dengue_idx / 200)  # up to 50% uplift at peak
                    if category in respiratory_affected:
                        qty *= (1 + resp_idx / 250)
                
                # 4. Festival / holiday spikes
                if week in cfg.SRI_LANKA_HOLIDAYS.get("vesak", []):
                    if category in {"vitamins_supplements", "otc_general", "analgesics"}:
                        qty *= rng.uniform(1.2, 1.6)
                
                if week in cfg.SRI_LANKA_HOLIDAYS.get("sinhala_tamil_new_year", []):
                    qty *= rng.uniform(1.1, 1.4)
                
                # 5. Month-end institutional ordering
                if calendar.iloc[w_idx]["is_month_end"] and rtype == "hospital_attached":
                    qty *= rng.uniform(1.2, 1.5)
                
                # 6. Noise (multiplicative + additive)
                qty *= rng.lognormal(0, 0.15)
                qty += rng.normal(0, max(1, pair_base * 0.05))
                
                # 7. Floor at 0, round to integer
                qty = max(0, round(qty))
                
                if qty == 0:
                    continue
                
                # 8. Fill rate simulation
                if rng.random() < 0.08:  # 8% chance of partial fulfillment
                    fill_rate = rng.uniform(0.5, 0.95)
                else:
                    fill_rate = 1.0
                qty_fulfilled = max(1, round(qty * fill_rate))
                
                # Lead time (days)
                lead_time = int(rng.exponential(3) + 1)
                
                all_records.append({
                    "retailer_id": rid,
                    "sku_id": sid,
                    "date": date,
                    "week_of_year": int(week),
                    "quantity_ordered": qty,
                    "quantity_fulfilled": qty_fulfilled,
                    "lead_time_days": lead_time,
                })
        
        if (r_idx + 1) % 50 == 0:
            logger.info(f"  Processed {r_idx + 1}/{n_retailers} retailers...")
    
    df = pd.DataFrame(all_records)
    
    # Convert date column
    df["date"] = pd.to_datetime(df["date"])
    
    logger.info(
        f"Generated {len(df):,} transactions — "
        f"{df['retailer_id'].nunique()} retailers × {df['sku_id'].nunique()} SKUs"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(save: bool = True) -> dict:
    """Generate all synthetic datasets and optionally save to disk."""
    set_seed(cfg.SEED)
    
    logger.info("=" * 60)
    logger.info("PharmaFlow AI — Synthetic Data Generation")
    logger.info("=" * 60)
    
    # 1. Generate master data
    retailers = generate_retailers()
    skus = generate_skus()
    calendar = generate_calendar()
    
    # 2. Health signals (only for districts that have retailers)
    active_districts = retailers["district"].unique().tolist()
    health_signals = generate_health_signals(districts=active_districts)
    
    # 3. Generate transactions
    transactions = generate_transactions(retailers, skus, calendar, health_signals)
    
    data = {
        "retailers": retailers,
        "skus": skus,
        "calendar": calendar,
        "health_signals": health_signals,
        "transactions": transactions,
    }
    
    if save:
        for name, df in data.items():
            path = cfg.SYNTHETIC_DIR / f"{name}.csv"
            save_csv(df, path)
            logger.info(f"Saved {name}.csv — {len(df):,} rows")
    
    logger.info("=" * 60)
    logger.info("Data generation complete!")
    logger.info("=" * 60)
    
    return data


if __name__ == "__main__":
    generate_all(save=True)
