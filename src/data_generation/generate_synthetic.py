"""
CuraNex AI — Synthetic Data Generator (v2 — Real Data Grounded)
=================================================================
Generates realistic pharmaceutical distribution data grounded in:
- Real NMRA-registered drugs from hemas_50_verified_complete.xlsx
- Real SLMC-registered pharmacies from retailers.csv
- Real Sri Lankan monsoon / rainfall patterns per district

Key realism features:
- Dosage form × retailer type demand crosses (injectables → hospitals only)
- Group-based seasonality (Group A = outbreak-sensitive, B = chronic-steady, C = mixed)
- Rainfall → dengue/respiratory disease → drug demand correlation chain
- Schedule × tier ordering constraints
- Cold-start ramp-up for newer retailers (years_active)
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
# 1. RETAILER LOADING (from real CSV)
# ─────────────────────────────────────────────────────────────────────────────

def load_retailers() -> pd.DataFrame:
    """
    Load real retailer data from retailers.csv.

    Uses SLMC registration numbers as retailer_id and includes
    real pharmacy names, types, districts, tiers, and years active.
    """
    path = cfg.RETAILERS_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Retailers CSV not found at {path}")

    df = pd.read_csv(path)

    # Ensure retailer_id is string for consistent handling
    df["retailer_id"] = df["retailer_id"].astype(str)

    logger.info(
        f"Loaded {len(df)} retailers — "
        f"Types: {dict(df['retailer_type'].value_counts())}, "
        f"Districts: {df['district'].nunique()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SKU LOADING (from real Excel)
# ─────────────────────────────────────────────────────────────────────────────

def load_skus() -> pd.DataFrame:
    """
    Load real SKU data from hemas_50_verified_complete.xlsx.

    Maps NMRA registration numbers to sku_id and derives:
    - dosage_group from dosage_form (oral_solid, injectable, etc.)
    - supplier_lead_time_days from source country
    - is_critical from Yes/No to boolean
    """
    path = cfg.SKU_EXCEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"SKU Excel not found at {path}")

    raw = pd.read_excel(path)

    # Normalize column names (they have \n characters from Excel)
    col_map = {}
    for c in raw.columns:
        clean = c.replace("\n", " ").strip()
        col_map[c] = clean
    raw = raw.rename(columns=col_map)

    # Build clean SKU DataFrame
    df = pd.DataFrame()
    df["sku_id"] = raw["SKU_ID"].astype(str)
    df["generic_name"] = raw["Generic Name"]
    df["brand"] = raw["Brand"]
    df["dosage_form"] = raw["Dosage Form"]
    df["pack_size"] = raw["Pack Size"]
    df["pack_type"] = raw["Pack Type"]
    df["country"] = raw["Country"]
    df["reg_date"] = pd.to_datetime(raw["Reg. Date"])
    df["schedule"] = raw["Schedule"].str.strip().str.upper()
    df["validation"] = raw["Validation"]
    df["therapeutic_category"] = raw["Therapeutic Category"]
    df["group"] = raw["Group"]
    df["group_description"] = raw["Group Description"]
    df["price_band"] = raw["Price Band"]

    # Is Critical: "Yes"/"No" → boolean
    crit_col = [c for c in raw.columns if "Critical" in c][0]
    df["is_critical"] = raw[crit_col].str.strip().str.lower() == "yes"

    # Shelf life
    shelf_col = [c for c in raw.columns if "Shelf Life" in c][0]
    df["shelf_life_months"] = raw[shelf_col].astype(int)

    # Derive dosage_group from dosage_form
    df["dosage_group"] = df["dosage_form"].map(cfg.DOSAGE_FORM_GROUPS).fillna("oral_solid")

    # Derive supplier_lead_time_days from country
    df["supplier_lead_time_days"] = df["country"].map(cfg.COUNTRY_LEAD_TIMES).fillna(60).astype(int)

    # Normalize schedule values
    schedule_map = {
        "II B": "IIB", "II  B": "IIB", "11B": "IIB", "IIB": "IIB",
        "IIA": "IIA", "II A": "IIA", "II C": "IIC",
        "I": "I",
    }
    df["schedule"] = df["schedule"].map(schedule_map).fillna("IIB")

    logger.info(
        f"Loaded {len(df)} SKUs — "
        f"Categories: {dict(df['therapeutic_category'].value_counts())}"
    )
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
        for holiday, wks in cfg.SRI_LANKA_HOLIDAYS.items():
            if w in wks:
                is_public_holiday[i] = 1
        if dates[i].day >= 25:
            is_month_end[i] = 1
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


def _get_district_zone(district: str) -> str:
    """Return climate zone for a district."""
    if district in cfg.WET_ZONE_DISTRICTS:
        return "wet"
    elif district in cfg.DRY_ZONE_DISTRICTS:
        return "dry"
    else:
        return "inter"


def _get_rainfall_base(week: int, zone: str) -> float:
    """Get base rainfall intensity (0-100) for a given week and climate zone."""
    for season_name, season in cfg.RAINFALL_SEASONS.items():
        if week in season["weeks"]:
            return float(season[zone])
    # Fallback: inter-monsoon moderate
    return 30.0


def generate_health_signals(
    n_weeks: int = cfg.N_WEEKS,
    districts: list = None,
    start_date: str = cfg.START_DATE,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """
    Generate district-level health surveillance indices with rainfall.

    Realistic causal chain:
    - Rainfall drives dengue (with 2-week lag — stagnant water breeds mosquitoes)
    - Cool dry periods drive respiratory infections
    - District rainfall varies by climate zone (wet/dry/intermediate)
    """
    rng = np.random.default_rng(seed)
    if districts is None:
        districts = cfg.SRI_LANKA_DISTRICTS

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    weeks = dates.isocalendar().week.astype(int).values

    records = []
    for dist in districts:
        zone = _get_district_zone(dist)

        # Pre-compute rainfall for all weeks (needed for lagged dengue)
        rainfall_series = []
        for w in weeks:
            base = _get_rainfall_base(w, zone)
            rainfall = max(0, min(100, base + rng.normal(0, 10)))
            rainfall_series.append(round(rainfall, 1))

        for i, (date, w) in enumerate(zip(dates, weeks)):
            rainfall = rainfall_series[i]

            # Dengue: correlated with rainfall 2 weeks ago (mosquito breeding lag)
            lagged_rain = rainfall_series[max(0, i - 2)]
            dengue_base = 15 if dist in ["Colombo", "Gampaha", "Kalutara"] else 8
            dengue_seasonal = 0.0
            if w in cfg.DENGUE_PEAK_WEEKS:
                dengue_seasonal = 30 * np.sin(2 * np.pi * (w - 22) / 26)
            # Rainfall contribution: heavier rain → more stagnant water → more dengue
            dengue_rain_boost = lagged_rain * 0.3
            dengue_idx = max(0, min(100, dengue_base + dengue_seasonal + dengue_rain_boost + rng.normal(0, 6)))

            # Respiratory: peaks in cool/dry periods (inverse of rainfall)
            resp_base = 30
            resp_seasonal = 25 * np.cos(2 * np.pi * (w - 1) / 52)
            # Less rain → more respiratory (dry, cool air)
            resp_rain_effect = -(rainfall * 0.15)
            resp_idx = max(0, min(100, resp_base + resp_seasonal + resp_rain_effect + rng.normal(0, 5)))

            records.append({
                "date": date,
                "district": dist,
                "week_of_year": w,
                "dengue_index": round(dengue_idx, 1),
                "respiratory_index": round(resp_idx, 1),
                "rainfall_index": rainfall,
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated health signals: {len(df)} rows ({len(districts)} districts × {n_weeks} weeks) [with rainfall]")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRANSACTION GENERATION (The core — grounded in real data)
# ─────────────────────────────────────────────────────────────────────────────

def _retailer_base_demand(retailer_type: str, tier: str, rng) -> float:
    """Determine base weekly demand for a retailer based on type and tier."""
    base = {
        "hospital_attached": {"A": 150, "B": 80, "C": 40},
        "chain_outlet":      {"A": 100, "B": 50, "C": 25},
        "standalone":        {"A": 60,  "B": 30, "C": 12},
    }
    return base.get(retailer_type, {}).get(tier, 20) * rng.uniform(0.6, 1.4)


def _sku_demand_modifier(group: str, therapeutic_category: str, price_band: str, rng) -> float:
    """
    SKU-level demand multiplier based on group, category, and price.

    Group A (Spreading Disease): higher volume, outbreak-sensitive
    Group B (Chronic): moderate, very steady
    Group C (Mixed): moderate, some variability
    """
    group_mult = {"A": 1.3, "B": 0.9, "C": 1.0}

    cat_mult = {
        "Antibiotic": 1.4, "Antipyretic": 1.5, "Antihistamine": 1.1,
        "Antileukotriene": 0.8, "Respiratory": 1.0,
        "Cardiovascular": 0.9, "Diabetes": 0.85,
        "Analgesic/NSAID": 1.3, "Gastrointestinal": 1.0,
        "Antiemetic": 0.7, "Corticosteroid": 0.6,
        "Antifungal": 0.5, "Neuropathic/Pain": 0.6,
        "Immunomodulator": 0.4,
    }

    price_mult = {"Low": 1.3, "Medium": 1.0, "High": 0.6}

    return (
        group_mult.get(group, 1.0)
        * cat_mult.get(therapeutic_category, 1.0)
        * price_mult.get(price_band, 1.0)
        * rng.uniform(0.7, 1.3)
    )


def _dosage_retailer_modifier(dosage_group: str, retailer_type: str, tier: str) -> float:
    """
    Apply dosage form × retailer type cross.

    Injectables/nebulizing: almost exclusively hospital_attached or Tier A chains.
    Oral solids/liquids: high velocity everywhere.
    """
    if dosage_group in ("injectable", "nebulizing"):
        if retailer_type == "hospital_attached":
            return 1.0
        elif retailer_type == "chain_outlet" and tier == "A":
            return 0.4  # some large chains stock injectables
        elif retailer_type == "chain_outlet":
            return 0.05
        else:  # standalone
            if tier == "A":
                return 0.1
            else:
                return 0.0  # effectively zero demand
    elif dosage_group == "oral_liquid":
        return 0.9  # slightly lower than tablets
    else:
        return 1.0  # oral solids available everywhere


def _schedule_tier_modifier(schedule: str, tier: str) -> float:
    """
    Schedule × tier ordering constraint.

    Schedule I (OTC like Paracetamol): available everywhere, high volume.
    Schedule IIB (common Rx): normal demand everywhere.
    """
    if schedule == "I":
        # OTC — high demand everywhere
        return 1.3 if tier == "A" else 1.2 if tier == "B" else 1.1
    else:
        # Rx drugs — demand correlates with tier
        return 1.0 if tier == "A" else 0.9 if tier == "B" else 0.75


def generate_transactions(
    retailers: pd.DataFrame,
    skus: pd.DataFrame,
    calendar: pd.DataFrame,
    health_signals: pd.DataFrame,
    n_weeks: int = cfg.N_WEEKS,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """
    Generate weekly transaction data for retailer-SKU pairs.

    Demand is driven by:
    1. Base demand (retailer type/tier × SKU group/category/price)
    2. Dosage form × retailer type cross (injectables → hospitals only)
    3. Schedule × tier constraint
    4. Disease outbreak uplift (Group A drugs surge during outbreaks)
    5. Rainfall → disease → demand chain
    6. Festival/holiday spikes
    7. Trend + noise + intermittency + fill rate
    """
    rng = np.random.default_rng(seed)
    dates = calendar["date"].values
    weeks = calendar["week_of_year"].values

    # Pre-compute health signals lookup: (district, date) → (dengue, resp, rainfall)
    health_lookup = {}
    for _, row in health_signals.iterrows():
        key = (row["district"], row["date"])
        health_lookup[key] = (row["dengue_index"], row["respiratory_index"], row["rainfall_index"])

    # Categories affected by disease outbreaks (all Group A + some Group C)
    dengue_affected = {"Antibiotic", "Antipyretic", "Analgesic/NSAID", "Antihistamine"}
    respiratory_affected = {"Antibiotic", "Respiratory", "Antihistamine", "Antipyretic", "Antileukotriene"}

    all_records = []
    n_retailers = len(retailers)
    n_skus_total = len(skus)

    for r_idx, retailer in retailers.iterrows():
        rid = str(retailer["retailer_id"])
        rtype = retailer["retailer_type"]
        tier = retailer["tier"]
        district = retailer["district"]
        years_active = retailer["years_active"]

        # Number of actively ordered SKUs depends on retailer type
        if rtype == "hospital_attached":
            n_active = int(n_skus_total * rng.uniform(0.6, 0.9))
        elif rtype == "chain_outlet":
            n_active = int(n_skus_total * rng.uniform(0.35, 0.65))
        else:  # standalone
            n_active = int(n_skus_total * rng.uniform(0.15, 0.4))

        n_active = max(3, min(n_active, n_skus_total))
        active_sku_indices = rng.choice(n_skus_total, size=n_active, replace=False)

        # Years-active ramp: newer retailers start with lower demand
        maturity_factor = min(1.0, years_active / 5.0)  # full demand at 5+ years

        for s_idx in active_sku_indices:
            sku = skus.iloc[s_idx]
            sid = str(sku["sku_id"])
            category = sku["therapeutic_category"]
            group = sku["group"]
            price_band = sku["price_band"]
            dosage_group = sku["dosage_group"]
            schedule = sku["schedule"]

            # Skip this pair entirely if dosage form incompatible
            dosage_mod = _dosage_retailer_modifier(dosage_group, rtype, tier)
            if dosage_mod == 0.0:
                continue

            # Base demand for this retailer-SKU pair
            base = _retailer_base_demand(rtype, tier, rng)
            sku_mod = _sku_demand_modifier(group, category, price_band, rng)
            sched_mod = _schedule_tier_modifier(schedule, tier)
            pair_base = base * sku_mod * dosage_mod * sched_mod * maturity_factor / n_skus_total * 10

            # Intermittency: probability of ordering in any given week
            if tier == "A":
                order_prob = rng.uniform(0.7, 1.0)
            elif tier == "B":
                order_prob = rng.uniform(0.4, 0.8)
            else:
                order_prob = rng.uniform(0.15, 0.5)

            # Chronic drugs (Group B) are ordered more regularly
            if group == "B":
                order_prob = min(1.0, order_prob * 1.2)

            # Trend: slight growth or decline
            trend_slope = rng.normal(0, 0.002)

            # Generate weekly time series
            for w_idx in range(n_weeks):
                if rng.random() > order_prob:
                    continue

                date = dates[w_idx]
                week = weeks[w_idx]

                # 1. Base + trend
                qty = pair_base * (1 + trend_slope * w_idx)

                # 2. Annual seasonality
                seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * (week - 1) / 52)
                qty *= seasonality

                # 3. Disease outbreak uplift (stronger for Group A)
                h_key = (district, date)
                if h_key in health_lookup:
                    dengue_idx, resp_idx, rain_idx = health_lookup[h_key]

                    if category in dengue_affected:
                        # Group A gets stronger outbreak boost
                        boost_factor = 200 if group == "A" else 300 if group == "C" else 500
                        qty *= (1 + dengue_idx / boost_factor)

                    if category in respiratory_affected:
                        boost_factor = 200 if group == "A" else 300 if group == "C" else 500
                        qty *= (1 + resp_idx / boost_factor)

                    # Rainfall direct boost for antihistamines/respiratory (rainy = flu/cold)
                    if category in {"Antihistamine", "Respiratory", "Antipyretic"} and group == "A":
                        qty *= (1 + rain_idx / 400)

                # 4. Festival / holiday spikes
                if week in cfg.SRI_LANKA_HOLIDAYS.get("vesak", []):
                    if category in {"Analgesic/NSAID", "Gastrointestinal", "Antipyretic"}:
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
                if rng.random() < 0.08:
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
    """Generate all datasets using real SKU and retailer data."""
    set_seed(cfg.SEED)

    logger.info("=" * 60)
    logger.info("CuraNex AI — Data Generation (Real-Data Grounded)")
    logger.info("=" * 60)

    # 1. Load real master data
    retailers = load_retailers()
    skus = load_skus()
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
            if name == "retailers":
                # Retailers already exist as CSV — save updated version
                path = cfg.SYNTHETIC_DIR / f"{name}.csv"
                save_csv(df, path)
                logger.info(f"Saved {name}.csv — {len(df):,} rows")
            elif name == "skus":
                path = cfg.SYNTHETIC_DIR / f"{name}.csv"
                save_csv(df, path)
                logger.info(f"Saved {name}.csv — {len(df):,} rows")
            else:
                path = cfg.SYNTHETIC_DIR / f"{name}.csv"
                save_csv(df, path)
                logger.info(f"Saved {name}.csv — {len(df):,} rows")

    logger.info("=" * 60)
    logger.info("Data generation complete!")
    logger.info("=" * 60)

    return data


if __name__ == "__main__":
    generate_all(save=True)
