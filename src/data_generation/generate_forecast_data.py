import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("ForecastDataGen")

def main():
    forecast_dir = cfg.DATA_DIR / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load historical raw data to find boundaries
    trans_hist = pd.read_csv(cfg.SYNTHETIC_DIR / "transactions.csv", parse_dates=["date"])
    calendar_hist = pd.read_csv(cfg.SYNTHETIC_DIR / "calendar.csv", parse_dates=["date"])
    health_hist = pd.read_csv(cfg.SYNTHETIC_DIR / "health_signals.csv", parse_dates=["date"])
    retailers = pd.read_csv(cfg.SYNTHETIC_DIR / "retailers.csv")
    skus = pd.read_csv(cfg.SYNTHETIC_DIR / "skus.csv")
    
    max_date = trans_hist["date"].max()
    logger.info(f"Historical raw data ends at {max_date.date()}")
    
    horizon_weeks = 4
    future_dates = pd.date_range(start=max_date + pd.Timedelta(days=7), periods=horizon_weeks, freq="W-MON")
    logger.info(f"Generating 5 isolated raw datasets for pure future timeline up to {future_dates.max().date()}")
    
    # 2. Create Future Transactions (Target missing)
    unique_pairs = trans_hist[["retailer_id", "sku_id"]].drop_duplicates()
    n_pairs = len(unique_pairs)
    
    trans_future = pd.DataFrame({
        "retailer_id": np.repeat(unique_pairs["retailer_id"].values, horizon_weeks),
        "sku_id": np.repeat(unique_pairs["sku_id"].values, horizon_weeks),
        "date": np.tile(future_dates, n_pairs),
        "quantity_ordered": np.nan,
    })
    # Save pure future transactions
    trans_future.to_csv(forecast_dir / "transactions.csv", index=False)
    logger.info(f"  -> Saved transactions.csv (Future dates only: {len(trans_future)} rows)")
    
    # 3. Create Future Calendar
    cal_future = pd.DataFrame({"date": future_dates})
    cal_future["week_of_year"] = cal_future["date"].dt.isocalendar().week.astype(int)
    cal_future["is_public_holiday"] = 0
    cal_future["is_vesak_week"] = (cal_future["week_of_year"] == 19).astype(int) 
    cal_future["is_new_year_period"] = (cal_future["week_of_year"] == 15).astype(int)
    cal_future["is_month_end"] = cal_future["date"].dt.is_month_end.astype(int)
    cal_future["is_school_term"] = cal_future["week_of_year"].apply(
        lambda w: 1 if (2<=w<=13) or (20<=w<=32) or (36<=w<=48) else 0
    )
    cal_future.to_csv(forecast_dir / "calendar.csv", index=False)
    logger.info(f"  -> Saved calendar.csv (Future dates only: {len(cal_future)} rows)")
    
    # 4. Create Future Health Signals
    districts = retailers["district"].unique()
    health_future = pd.DataFrame({
        "date": np.tile(future_dates, len(districts)),
        "district": np.repeat(districts, horizon_weeks)
    })
    
    # Historical logic map
    health_hist["week_of_year"] = health_hist["date"].dt.isocalendar().week.astype(int)
    health_avgs = health_hist.groupby(["district", "week_of_year"])[["dengue_index", "respiratory_index", "rainfall_index"]].mean().reset_index()
    
    health_future["week_of_year"] = health_future["date"].dt.isocalendar().week.astype(int)
    health_future = pd.merge(health_future, health_avgs, on=["district", "week_of_year"], how="left").drop(columns=["week_of_year"])
    health_future.fillna(0, inplace=True)
    
    health_future.to_csv(forecast_dir / "health_signals.csv", index=False)
    logger.info(f"  -> Saved health_signals.csv (Future dates only: {len(health_future)} rows)")
    
    # 5. Copy Static files (Retailers, SKUs) since they don't have dates but user requested the 5 datasets
    shutil.copy(cfg.SYNTHETIC_DIR / "retailers.csv", forecast_dir / "retailers.csv")
    shutil.copy(cfg.SYNTHETIC_DIR / "skus.csv", forecast_dir / "skus.csv")
    logger.info("  -> Copied static retailers.csv and skus.csv")
    
    logger.info("=" * 60)
    logger.info("Pure Future Database Created Successfully.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
