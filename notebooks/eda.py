"""
PharmaFlow AI — Exploratory Data Analysis
============================================
Quick EDA script to validate synthetic data quality and explore patterns.
Run: python notebooks/eda.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from src.utils.helpers import setup_logger, load_csv

logger = setup_logger("EDA")


def run_eda():
    """Run exploratory data analysis on synthetic data."""
    
    logger.info("=" * 60)
    logger.info("PharmaFlow AI — Exploratory Data Analysis")
    logger.info("=" * 60)
    
    # Load data
    retailers = load_csv(cfg.SYNTHETIC_DIR / "retailers.csv")
    skus = load_csv(cfg.SYNTHETIC_DIR / "skus.csv")
    calendar = load_csv(cfg.SYNTHETIC_DIR / "calendar.csv", parse_dates=["date"])
    transactions = load_csv(cfg.SYNTHETIC_DIR / "transactions.csv", parse_dates=["date"])
    
    # ── Retailer Analysis ──
    logger.info("\n" + "─" * 40)
    logger.info("RETAILER ANALYSIS")
    logger.info("─" * 40)
    logger.info(f"Total retailers: {len(retailers)}")
    logger.info(f"\nType distribution:\n{retailers['retailer_type'].value_counts().to_string()}")
    logger.info(f"\nTier distribution:\n{retailers['tier'].value_counts().to_string()}")
    logger.info(f"\nTop districts:\n{retailers['district'].value_counts().head(10).to_string()}")
    logger.info(f"\nYears active: mean={retailers['years_active'].mean():.1f}, "
                 f"median={retailers['years_active'].median():.1f}")
    
    # ── SKU Analysis ──
    logger.info("\n" + "─" * 40)
    logger.info("SKU ANALYSIS")
    logger.info("─" * 40)
    logger.info(f"Total SKUs: {len(skus)}")
    logger.info(f"\nCategory distribution:\n{skus['therapeutic_category'].value_counts().to_string()}")
    logger.info(f"\nBrand type:\n{skus['brand_type'].value_counts().to_string()}")
    logger.info(f"\nCritical SKUs: {skus['is_critical'].sum()} / {len(skus)}")
    
    # ── Transaction Analysis ──
    logger.info("\n" + "─" * 40)
    logger.info("TRANSACTION ANALYSIS")
    logger.info("─" * 40)
    logger.info(f"Total transactions: {len(transactions):,}")
    logger.info(f"Date range: {transactions['date'].min()} to {transactions['date'].max()}")
    logger.info(f"Unique retailers with orders: {transactions['retailer_id'].nunique()}")
    logger.info(f"Unique SKUs ordered: {transactions['sku_id'].nunique()}")
    
    logger.info(f"\nQuantity ordered statistics:")
    logger.info(f"  Mean:   {transactions['quantity_ordered'].mean():.1f}")
    logger.info(f"  Median: {transactions['quantity_ordered'].median():.1f}")
    logger.info(f"  Std:    {transactions['quantity_ordered'].std():.1f}")
    logger.info(f"  Min:    {transactions['quantity_ordered'].min()}")
    logger.info(f"  Max:    {transactions['quantity_ordered'].max()}")
    
    # Orders per retailer
    orders_per_retailer = transactions.groupby("retailer_id").size()
    logger.info(f"\nOrders per retailer:")
    logger.info(f"  Mean:   {orders_per_retailer.mean():.0f}")
    logger.info(f"  Median: {orders_per_retailer.median():.0f}")
    logger.info(f"  Min:    {orders_per_retailer.min()}")
    logger.info(f"  Max:    {orders_per_retailer.max()}")
    
    # Active SKUs per retailer
    skus_per_retailer = transactions.groupby("retailer_id")["sku_id"].nunique()
    logger.info(f"\nActive SKUs per retailer:")
    logger.info(f"  Mean:   {skus_per_retailer.mean():.0f}")
    logger.info(f"  Median: {skus_per_retailer.median():.0f}")
    
    # Weekly demand pattern
    weekly_demand = transactions.groupby("date")["quantity_ordered"].sum()
    logger.info(f"\nWeekly total demand:")
    logger.info(f"  Mean:   {weekly_demand.mean():,.0f}")
    logger.info(f"  Std:    {weekly_demand.std():,.0f}")
    
    # Fill rate
    fill_rate = (transactions["quantity_fulfilled"] / transactions["quantity_ordered"]).mean()
    partial_fills = (transactions["quantity_fulfilled"] < transactions["quantity_ordered"]).mean()
    logger.info(f"\nFill rate: {fill_rate:.1%}")
    logger.info(f"Partial fulfillment rate: {partial_fills:.1%}")
    
    # Sparsity analysis
    logger.info("\n" + "─" * 40)
    logger.info("SPARSITY & INTERMITTENCY")
    logger.info("─" * 40)
    
    total_possible = len(retailers) * len(skus)
    active_pairs = transactions.groupby(["retailer_id", "sku_id"]).ngroups
    density = active_pairs / total_possible * 100
    
    logger.info(f"Possible retailer-SKU pairs: {total_possible:,}")
    logger.info(f"Active pairs: {active_pairs:,} ({density:.1f}%)")
    logger.info(f"Sparsity: {100 - density:.1f}%")
    
    # Order frequency distribution
    pair_orders = transactions.groupby(["retailer_id", "sku_id"]).size()
    logger.info(f"\nOrders per retailer-SKU pair:")
    logger.info(f"  Mean:   {pair_orders.mean():.1f}")
    logger.info(f"  Median: {pair_orders.median():.0f}")
    logger.info(f"  Pairs with <=10 orders: {(pair_orders <= 10).sum()} ({(pair_orders <= 10).mean():.1%})")
    
    logger.info("\n" + "=" * 60)
    logger.info("EDA Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_eda()
