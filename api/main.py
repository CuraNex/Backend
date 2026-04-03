"""
CuraNex AI — FastAPI Serving Layer
===================================
RESTful API endpoints for the React frontend dashboard.
All data processing mirrors the Streamlit dashboard logic.
"""

import sys
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CuraNex AI — Forecast API",
    description="Hybrid ensemble demand forecasting for pharmaceutical distribution",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (mirrors Streamlit load_data)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self._data = None
        self._loaded = False

    @property
    def data(self):
        if not self._loaded:
            self._load()
        return self._data

    def _load(self):
        self._data = {}

        # Load features
        features_path = cfg.SYNTHETIC_DIR / "features.csv"
        if features_path.exists():
            self._data["features"] = pd.read_csv(features_path, parse_dates=["date"])
        else:
            trans_path = cfg.SYNTHETIC_DIR / "transactions.csv"
            if trans_path.exists():
                self._data["features"] = pd.read_csv(trans_path, parse_dates=["date"])

        # Supporting data
        for name in ["retailers", "skus", "calendar"]:
            path = cfg.SYNTHETIC_DIR / f"{name}.csv"
            if path.exists():
                parse = ["date"] if name == "calendar" else None
                self._data[name] = pd.read_csv(path, parse_dates=parse)

        # Evaluation report
        eval_path = cfg.EVAL_DIR / "evaluation_report.csv"
        if eval_path.exists():
            self._data["evaluation"] = pd.read_csv(eval_path)

        # Test predictions (historical validation)
        pred_path = cfg.EVAL_DIR / "predictions.csv"
        if pred_path.exists() and "features" in self._data:
            try:
                preds = pd.read_csv(pred_path, parse_dates=["date"])
                if "pred_ensemble" in preds.columns:
                    preds_to_merge = preds.drop(columns=["quantity_ordered"], errors="ignore")
                    self._data["features"] = self._data["features"].merge(
                        preds_to_merge,
                        on=["retailer_id", "sku_id", "date"],
                        how="left",
                    )
            except Exception:
                pass

        # Future forecasts
        future_path = cfg.EVAL_DIR / "forecasts.csv"
        if future_path.exists():
            try:
                self._data["future_predictions"] = pd.read_csv(future_path, parse_dates=["date"])
            except Exception:
                pass

        # Load models for feature importance
        self._data["models"] = {}
        for model_file in cfg.MODEL_DIR.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    self._data["models"][model_file.stem] = pickle.load(f)
            except Exception:
                pass

        self._loaded = True


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_df():
    d = state.data
    if "features" not in d:
        raise HTTPException(503, "Data not loaded. Run the training pipeline first.")
    return d["features"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DASHBOARD OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/dashboard/overview", tags=["Dashboard"])
def dashboard_overview():
    d = state.data
    df = _get_df()
    retailers_df = d.get("retailers", pd.DataFrame())
    skus_df = d.get("skus", pd.DataFrame())
    eval_df = d.get("evaluation")

    # --- KPIs ---
    n_retailers = df["retailer_id"].nunique() if "retailer_id" in df.columns else 0
    n_skus = df["sku_id"].nunique() if "sku_id" in df.columns else 0
    total_orders = len(df)

    accuracy = 94.0
    stockout_risk = 3.8
    if eval_df is not None and "wMAPE" in eval_df.columns:
        ens = eval_df[eval_df["Model"].str.lower() == "ensemble"]
        if not ens.empty:
            accuracy = round(100.0 - float(ens.iloc[0].get("wMAPE", 10.0)), 1)
            fill = float(ens.iloc[0].get("Fill_Rate", 96.2))
            stockout_risk = round(100.0 - fill, 1)

    # --- Weekly Demand Trend (historical + future) ---
    weekly_trend = []
    if "date" in df.columns and "quantity_ordered" in df.columns:
        weekly = df.groupby("date")["quantity_ordered"].sum().reset_index().sort_values("date")
        for _, row in weekly.iterrows():
            weekly_trend.append({
                "date": row["date"].strftime("%b %Y"),
                "dateRaw": row["date"].isoformat(),
                "actual": int(row["quantity_ordered"]),
                "forecast": None,
            })

        future_preds = d.get("future_predictions")
        if future_preds is not None and not future_preds.empty:
            wk_future = future_preds.groupby("date")["pred_ensemble"].sum().reset_index().sort_values("date")
            # Bridge point: set forecast on the last historical entry so lines connect
            if len(weekly_trend) > 0:
                weekly_trend[-1]["forecast"] = weekly_trend[-1]["actual"]
            for _, row in wk_future.iterrows():
                weekly_trend.append({
                    "date": row["date"].strftime("%b %Y"),
                    "dateRaw": row["date"].isoformat(),
                    "actual": None,
                    "forecast": int(round(row["pred_ensemble"])),
                })

    # --- Top Demanded Drugs ---
    top_drugs = []
    if "sku_id" in df.columns and "quantity_ordered" in df.columns:
        drug_demand = df.groupby("sku_id")["quantity_ordered"].sum().nlargest(10).reset_index()
        if "skus" in d:
            drug_demand = drug_demand.merge(d["skus"][["sku_id", "generic_name", "therapeutic_category"]], on="sku_id", how="left")
        for _, row in drug_demand.iterrows():
            top_drugs.append({
                "name": row.get("generic_name", str(row["sku_id"])),
                "demand": int(row["quantity_ordered"]),
                "category": row.get("therapeutic_category", "Unknown"),
            })

    # --- Retailer Type Pie ---
    retailer_type_data = []
    if "retailer_type" in df.columns:
        type_demand = df.groupby("retailer_type")["quantity_ordered"].sum().reset_index()
        total = type_demand["quantity_ordered"].sum()
        for _, row in type_demand.iterrows():
            retailer_type_data.append({
                "name": row["retailer_type"],
                "value": round(float(row["quantity_ordered"]) / total * 100, 1),
            })

    # --- Demand by Category ---
    category_data = []
    if "therapeutic_category" in df.columns:
        cat = df.groupby("therapeutic_category")["quantity_ordered"].sum().sort_values(ascending=False).reset_index()
        for _, row in cat.iterrows():
            category_data.append({
                "category": row["therapeutic_category"],
                "demand": int(row["quantity_ordered"]),
            })

    return {
        "kpi": {
            "totalPharmacies": n_retailers,
            "totalSKUs": n_skus,
            "totalOrders": total_orders,
            "forecastAccuracy": accuracy,
            "stockoutRisk": stockout_risk,
        },
        "weeklyTrend": weekly_trend,
        "topDrugs": top_drugs,
        "retailerTypeData": retailer_type_data,
        "categoryData": category_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORECAST EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/forecast/filters", tags=["Forecast"])
def forecast_filters():
    """Return available filter options for forecast explorer."""
    d = state.data
    df = _get_df()
    retailers_df = d.get("retailers", pd.DataFrame())
    skus_df = d.get("skus", pd.DataFrame())

    # Retailer options
    retailer_options = []
    if "retailer_id" in df.columns:
        for rid in sorted(df["retailer_id"].unique()):
            label = str(rid)
            if len(retailers_df) > 0 and "retailer_name" in retailers_df.columns:
                match = retailers_df[retailers_df["retailer_id"] == rid]
                if len(match) > 0:
                    label = f"{match.iloc[0]['retailer_name']} ({rid})"
            retailer_options.append({"value": str(rid), "label": label})

    # SKU options
    sku_options = []
    if "sku_id" in df.columns:
        for sid in sorted(df["sku_id"].unique()):
            label = str(sid)
            if len(skus_df) > 0 and "generic_name" in skus_df.columns:
                match = skus_df[skus_df["sku_id"] == sid]
                if len(match) > 0:
                    label = f"{match.iloc[0]['generic_name']} ({sid})"
            sku_options.append({"value": str(sid), "label": label})

    # Category options
    categories = []
    if "therapeutic_category" in df.columns:
        categories = sorted(df["therapeutic_category"].unique().tolist())

    return {
        "retailers": retailer_options,
        "skus": sku_options,
        "categories": categories,
    }


@app.get("/api/forecast/timeseries", tags=["Forecast"])
def forecast_timeseries(
    retailer: str = Query("All"),
    sku: str = Query("All"),
    category: str = Query("All"),
):
    """Return time series data for the Demand Time Series and Future Extrapolation charts."""
    d = state.data
    df = _get_df().copy()

    # Apply filters
    if retailer != "All":
        df = df[df["retailer_id"].astype(str) == retailer]
    if sku != "All":
        df = df[df["sku_id"] == sku]
    if category != "All" and "therapeutic_category" in df.columns:
        df = df[df["therapeutic_category"] == category]

    records_count = len(df)

    has_preds = "pred_ensemble" in df.columns and df["pred_ensemble"].notna().any()
    has_snaive = "pred_snaive" in df.columns and df["pred_snaive"].notna().any()

    # Filter to test period if predictions exist
    if has_preds:
        test_start = df.dropna(subset=["pred_ensemble"])["date"].min()
        df = df[df["date"] >= test_start]

    # Aggregate per date
    agg_dict = {"quantity_ordered": ("quantity_ordered", "sum")}
    if has_preds:
        agg_dict["pred_ensemble"] = ("pred_ensemble", lambda x: x.sum(min_count=1))
    if has_snaive:
        agg_dict["pred_snaive"] = ("pred_snaive", lambda x: x.sum(min_count=1))
    ts = df.groupby("date", as_index=False).agg(**agg_dict).sort_values("date")

    # Build main chart data
    chart_data = []
    for _, row in ts.iterrows():
        point = {
            "week": row["date"].strftime("%b %d"),
            "dateRaw": row["date"].isoformat(),
            "actual": float(row["quantity_ordered"]) if pd.notna(row.get("quantity_ordered")) else None,
        }
        if has_snaive:
            point["naive"] = float(row["pred_snaive"]) if pd.notna(row.get("pred_snaive")) else None
        if has_preds:
            point["validationForecast"] = float(row["pred_ensemble"]) if pd.notna(row.get("pred_ensemble")) else None
        chart_data.append(point)

    # Future extrapolation
    future_data = []
    future_preds = d.get("future_predictions")
    if future_preds is not None and not future_preds.empty:
        fut = future_preds.copy()
        if retailer != "All":
            fut = fut[fut["retailer_id"].astype(str) == retailer]
        if sku != "All":
            fut = fut[fut["sku_id"] == sku]
        if category != "All" and "therapeutic_category" in fut.columns:
            fut = fut[fut["therapeutic_category"] == category]

        if len(fut) > 0:
            fut_ts = fut.groupby("date", as_index=False)["pred_ensemble"].sum().sort_values("date")

            # Connect to last historical point
            if len(chart_data) > 0:
                last = chart_data[-1]
                last_val = last.get("validationForecast") or last.get("actual") or 0
                future_data.append({
                    "week": last["week"],
                    "dateRaw": last["dateRaw"],
                    "futureExtrapolation": last_val,
                })

            for _, row in fut_ts.iterrows():
                future_data.append({
                    "week": row["date"].strftime("%b %d"),
                    "dateRaw": row["date"].isoformat(),
                    "futureExtrapolation": float(row["pred_ensemble"]),
                })

    # Summary statistics
    summary_stats = []
    if "quantity_ordered" in df.columns and len(df) > 0:
        qty = df["quantity_ordered"]
        summary_stats = [
            {"metric": "Mean", "value": f"{qty.mean():.1f}"},
            {"metric": "Median", "value": f"{qty.median():.1f}"},
            {"metric": "Std Dev", "value": f"{qty.std():.1f}"},
            {"metric": "Min", "value": f"{qty.min():.0f}"},
            {"metric": "Max", "value": f"{qty.max():.0f}"},
            {"metric": "Total", "value": f"{qty.sum():,.0f}"},
        ]

    # Order distribution
    order_dist = []
    if "quantity_ordered" in df.columns and len(df) > 0:
        hist, bin_edges = np.histogram(df["quantity_ordered"].dropna(), bins=30)
        for i in range(len(hist)):
            order_dist.append({
                "bin": f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}",
                "binMid": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "count": int(hist[i]),
            })

    return {
        "recordsCount": records_count,
        "chartData": chart_data,
        "futureData": future_data,
        "summaryStats": summary_stats,
        "orderDistribution": order_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHARMACY INSIGHTS (NEW — not in Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/pharmacy/top5", tags=["Pharmacy"])
def pharmacy_top5():
    d = state.data
    df = _get_df()
    retailers_df = d.get("retailers", pd.DataFrame())

    top = df.groupby("retailer_id")["quantity_ordered"].sum().nlargest(5).reset_index()
    if len(retailers_df) > 0:
        top = top.merge(retailers_df, on="retailer_id", how="left")

    pharmacies = []
    for _, row in top.iterrows():
        pharmacies.append({
            "id": str(row["retailer_id"]),
            "name": row.get("retailer_name", f"Retailer {row['retailer_id']}"),
            "district": row.get("district", "Unknown"),
            "type": row.get("retailer_type", "Unknown"),
            "tier": row.get("tier", "Unknown"),
        })
    return pharmacies


@app.get("/api/pharmacy/{retailer_id}/details", tags=["Pharmacy"])
def pharmacy_details(retailer_id: str):
    d = state.data
    df = _get_df()
    retailers_df = d.get("retailers", pd.DataFrame())

    rid = int(retailer_id) if retailer_id.isdigit() else retailer_id
    rdf = df[df["retailer_id"] == rid]
    if len(rdf) == 0:
        raise HTTPException(404, f"Retailer {retailer_id} not found")

    # Profile info
    info = {"type": "Unknown", "tier": "Unknown", "district": "Unknown", "name": f"Retailer {rid}"}
    if len(retailers_df) > 0:
        match = retailers_df[retailers_df["retailer_id"] == rid]
        if len(match) > 0:
            r = match.iloc[0]
            info = {
                "type": r.get("retailer_type", "Unknown"),
                "tier": r.get("tier", "Unknown"),
                "district": r.get("district", "Unknown"),
                "name": r.get("retailer_name", f"Retailer {rid}"),
            }

    # Order frequency
    dates = rdf["date"].sort_values()
    if len(dates) > 1:
        diffs = dates.diff().dropna().dt.days
        avg_gap = diffs.mean()
        if avg_gap <= 8:
            order_freq = "Weekly"
        elif avg_gap <= 16:
            order_freq = "Bi-weekly"
        else:
            order_freq = "Monthly"
    else:
        order_freq = "Unknown"

    # Demand trend (weekly sum of all SKUs)
    trend = rdf.groupby("date")["quantity_ordered"].sum().reset_index().sort_values("date")
    demand_trend = []
    for _, row in trend.iterrows():
        demand_trend.append({
            "date": row["date"].strftime("%b %d"),
            "demand": int(row["quantity_ordered"]),
            "forecast": None
        })

    # Add future predictions for this specific retailer
    future_preds = d.get("future_predictions")
    if future_preds is not None and not future_preds.empty and "retailer_id" in future_preds.columns:
        rf_preds = future_preds[future_preds["retailer_id"] == rid]
        if not rf_preds.empty:
            wk_future = rf_preds.groupby("date")["pred_ensemble"].sum().reset_index().sort_values("date")
            # Bridge point: set forecast on the last historical entry
            if len(demand_trend) > 0:
                demand_trend[-1]["forecast"] = demand_trend[-1]["demand"]
            for _, row in wk_future.iterrows():
                demand_trend.append({
                    "date": row["date"].strftime("%b %d"),
                    "demand": None,
                    "forecast": int(round(row["pred_ensemble"])),
                })

    # AI Insights — generated based on demand patterns
    insights = []
    total_demand = rdf["quantity_ordered"].sum()
    avg_demand = rdf["quantity_ordered"].mean()
    recent = rdf.sort_values("date").tail(13)
    older = rdf.sort_values("date").head(len(rdf) - 13) if len(rdf) > 13 else rdf
    recent_avg = recent["quantity_ordered"].mean()
    older_avg = older["quantity_ordered"].mean()

    if recent_avg > older_avg * 1.1:
        pct = round((recent_avg / older_avg - 1) * 100, 1)
        insights.append(f"Demand has increased by {pct}% in the last quarter compared to historical average.")
    elif recent_avg < older_avg * 0.9:
        pct = round((1 - recent_avg / older_avg) * 100, 1)
        insights.append(f"Demand has decreased by {pct}% in the last quarter — investigate possible causes.")
    else:
        insights.append("Demand has been stable over the last quarter with no significant changes.")

    if "therapeutic_category" in rdf.columns:
        top_cat = rdf.groupby("therapeutic_category")["quantity_ordered"].sum().idxmax()
        insights.append(f"Highest demand is in the {top_cat} category — consider prioritising stock for this segment.")

    n_skus = rdf["sku_id"].nunique()
    insights.append(f"This pharmacy orders {n_skus} unique SKUs with an average order quantity of {avg_demand:.1f} units.")

    if order_freq == "Weekly":
        insights.append("Consistent weekly ordering pattern detected — suitable candidate for automated reorder setup.")
    else:
        insights.append(f"Ordering frequency is {order_freq.lower()}. Consider increasing order frequency for better stock continuity.")

    return {
        "info": info,
        "orderFreq": order_freq,
        "demandTrend": demand_trend,
        "insights": insights,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SKU ANALYTICS (NEW — not in Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/sku/top5", tags=["SKU"])
def sku_top5():
    d = state.data
    df = _get_df()
    skus_df = d.get("skus", pd.DataFrame())

    # Calculate avg demand and suggested stock per SKU
    sku_stats = df.groupby("sku_id").agg(
        total_demand=("quantity_ordered", "sum"),
        avg_demand=("quantity_ordered", "mean"),
        std_demand=("quantity_ordered", "std"),
    ).reset_index()
    sku_stats["std_demand"] = sku_stats["std_demand"].fillna(0)
    sku_stats["suggested_stock"] = (sku_stats["avg_demand"] + 1.5 * sku_stats["std_demand"]).round(0).astype(int)

    top = sku_stats.nlargest(5, "total_demand")
    if len(skus_df) > 0:
        top = top.merge(skus_df, on="sku_id", how="left")

    result = []
    for _, row in top.iterrows():
        result.append({
            "sku": row["sku_id"],
            "name": row.get("generic_name", row["sku_id"]),
            "category": row.get("therapeutic_category", "Unknown"),
            "dosageForm": row.get("dosage_form", "Unknown"),
            "shelfLife": f"{int(row.get('shelf_life_months', 24))} months",
            "avgDemand": round(float(row["avg_demand"]), 1),
            "suggestedStock": int(row["suggested_stock"]),
        })
    return result


@app.get("/api/sku/{sku_id}/analytics", tags=["SKU"])
def sku_analytics(sku_id: str):
    d = state.data
    df = _get_df()
    sdf = df[df["sku_id"] == sku_id]
    if len(sdf) == 0:
        raise HTTPException(404, f"SKU {sku_id} not found")

    # Demand trend (weekly)
    trend = sdf.groupby("date")["quantity_ordered"].sum().reset_index().sort_values("date")
    demand_trend = []
    for _, row in trend.iterrows():
        demand_trend.append({
            "date": row["date"].strftime("%b %d"),
            "demand": int(row["quantity_ordered"]),
            "forecast": None
        })

    # Add future predictions for this specific SKU
    future_preds = d.get("future_predictions")
    if future_preds is not None and not future_preds.empty and "sku_id" in future_preds.columns:
        sf_preds = future_preds[future_preds["sku_id"] == sku_id]
        if not sf_preds.empty:
            wk_future = sf_preds.groupby("date")["pred_ensemble"].sum().reset_index().sort_values("date")
            # Bridge point: set forecast on the last historical entry
            if len(demand_trend) > 0:
                demand_trend[-1]["forecast"] = demand_trend[-1]["demand"]
            for _, row in wk_future.iterrows():
                demand_trend.append({
                    "date": row["date"].strftime("%b %d"),
                    "demand": None,
                    "forecast": int(round(row["pred_ensemble"])),
                })

    # Seasonality pattern (monthly aggregation)
    sdf_copy = sdf.copy()
    sdf_copy["month"] = sdf_copy["date"].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = sdf_copy.groupby("month")["quantity_ordered"].mean().reset_index()
    seasonality = []
    for _, row in monthly.iterrows():
        m = int(row["month"])
        seasonality.append({
            "month": month_names[m - 1],
            "demand": round(float(row["quantity_ordered"]), 1),
        })

    return {
        "demandTrend": demand_trend,
        "seasonality": seasonality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANOMALY ALERTS (mirrors Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/anomalies", tags=["Alerts"])
def get_anomalies():
    df = _get_df()

    pair_stats = df.groupby(["retailer_id", "sku_id"])["quantity_ordered"].agg(["mean", "std"]).reset_index()
    pair_stats.columns = ["retailer_id", "sku_id", "pair_mean", "pair_std"]
    pair_stats["pair_std"] = pair_stats["pair_std"].fillna(0)

    anomaly_df = df.merge(pair_stats, on=["retailer_id", "sku_id"])
    anomaly_df["z_score"] = np.where(
        anomaly_df["pair_std"] > 0,
        (anomaly_df["quantity_ordered"] - anomaly_df["pair_mean"]) / anomaly_df["pair_std"],
        0
    )
    anomaly_df["is_anomaly"] = anomaly_df["z_score"].abs() > 3
    anomalies = anomaly_df[anomaly_df["is_anomaly"]].sort_values("z_score", ascending=False)

    n_anomalies = len(anomalies)
    total_orders = len(df)
    anomaly_rate = round((n_anomalies / total_orders * 100), 2) if total_orders > 0 else 0

    # Table rows (top 50)
    table_rows = []
    for _, row in anomalies.head(50).iterrows():
        table_rows.append({
            "date": row["date"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row["date"]) else "",
            "retailer_id": int(row["retailer_id"]),
            "sku_id": str(row["sku_id"]),
            "qty": int(row["quantity_ordered"]),
            "pair_mean": round(float(row["pair_mean"]), 1),
            "z_score": round(float(row["z_score"]), 2),
        })

    # Timeline (monthly bins)
    anomalies_with_month = anomalies.copy()
    anomalies_with_month["month"] = anomalies_with_month["date"].dt.to_period("M")
    timeline_agg = anomalies_with_month.groupby("month").size().reset_index(name="count")
    timeline_agg["month"] = timeline_agg["month"].astype(str)
    timeline_agg = timeline_agg.sort_values("month")

    timeline = []
    for _, row in timeline_agg.iterrows():
        # Convert "2024-04" to "Apr 2024"
        try:
            dt = pd.to_datetime(row["month"])
            label = dt.strftime("%b %Y")
        except Exception:
            label = row["month"]
        timeline.append({"month": label, "count": int(row["count"])})

    return {
        "anomalyCount": n_anomalies,
        "totalOrders": total_orders,
        "anomalyRate": anomaly_rate,
        "tableRows": table_rows,
        "timeline": timeline,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. RECOMMENDATIONS (mirrors Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/recommendations", tags=["Recommendations"])
def get_recommendations(
    retailer: str = Query("All"),
    critical_only: bool = Query(False),
):
    d = state.data
    df = _get_df()
    retailers_df = d.get("retailers", pd.DataFrame())
    skus_df = d.get("skus", pd.DataFrame())

    # Generate recommendations based on recent demand - same as Streamlit
    recent = df.sort_values("date").groupby(["retailer_id", "sku_id"]).tail(4)
    recs = recent.groupby(["retailer_id", "sku_id"]).agg(
        avg_recent_demand=("quantity_ordered", "mean"),
        std_recent_demand=("quantity_ordered", "std"),
        last_order_qty=("quantity_ordered", "last"),
    ).reset_index()
    recs["std_recent_demand"] = recs["std_recent_demand"].fillna(0)
    recs["suggested_qty"] = (recs["avg_recent_demand"] + 1.5 * recs["std_recent_demand"]).round(0).astype(int)
    recs["safety_stock"] = (1.5 * recs["std_recent_demand"]).round(0).astype(int)

    # Merge SKU info
    if "skus" in d:
        recs = recs.merge(
            d["skus"][["sku_id", "therapeutic_category", "is_critical", "generic_name"]],
            on="sku_id", how="left",
        )

    # Merge retailer name
    if len(retailers_df) > 0 and "retailer_name" in retailers_df.columns:
        recs = recs.merge(
            retailers_df[["retailer_id", "retailer_name"]],
            on="retailer_id", how="left",
        )

    # Sort: critical first, then by suggested qty
    if "is_critical" in recs.columns:
        recs = recs.sort_values(["is_critical", "suggested_qty"], ascending=[False, False])
    else:
        recs = recs.sort_values("suggested_qty", ascending=False)

    # Build retailer options
    retailer_options = ["All"]
    for rid in sorted(recs["retailer_id"].unique()):
        label = str(rid)
        if "retailer_name" in recs.columns:
            match = recs[recs["retailer_id"] == rid]
            if len(match) > 0 and pd.notna(match.iloc[0].get("retailer_name")):
                label = f"{match.iloc[0]['retailer_name']} ({rid})"
        retailer_options.append(label)

    # Filter
    filtered = recs.copy()
    if retailer != "All":
        # Try matching by retailer_id
        filtered = filtered[filtered["retailer_id"].astype(str) == retailer]
        if len(filtered) == 0:
            # Try extracting ID from label
            try:
                rid_from_label = retailer.split("(")[-1].rstrip(")")
                filtered = recs[recs["retailer_id"].astype(str) == rid_from_label]
            except Exception:
                filtered = recs

    if critical_only and "is_critical" in filtered.columns:
        filtered = filtered[filtered["is_critical"] == True]

    # Build response items
    items = []
    for _, row in filtered.head(100).iterrows():
        items.append({
            "retailer_id": row.get("retailer_name", str(row["retailer_id"])),
            "sku_id": row.get("generic_name", str(row["sku_id"])),
            "category": row.get("therapeutic_category", "Unknown"),
            "avg_recent_demand": round(float(row["avg_recent_demand"]), 1),
            "last_order_qty": int(row["last_order_qty"]),
            "suggested_qty": int(row["suggested_qty"]),
            "safety_stock": int(row["safety_stock"]),
            "is_critical": bool(row.get("is_critical", False)),
        })

    # Category summary
    cat_summary = []
    if "therapeutic_category" in filtered.columns:
        cat_agg = filtered.groupby("therapeutic_category")["suggested_qty"].sum().sort_values(ascending=False).reset_index()
        for _, row in cat_agg.iterrows():
            cat_summary.append({
                "category": row["therapeutic_category"],
                "suggested_qty": int(row["suggested_qty"]),
            })

    return {
        "retailerOptions": retailer_options[:50],  # limit for performance
        "items": items,
        "totalItems": len(filtered),
        "categorySummary": cat_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL PERFORMANCE (mirrors Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models/evaluation", tags=["Models"])
def models_evaluation():
    d = state.data
    eval_df = d.get("evaluation")
    if eval_df is None or len(eval_df) == 0:
        raise HTTPException(404, "No evaluation data. Run training pipeline first.")

    # Model ranking table
    rankings = []
    for _, row in eval_df.iterrows():
        rankings.append({
            "rank": int(row["Rank"]),
            "model": row["Model"],
            "mae": round(float(row["MAE"]), 2),
            "rmse": round(float(row["RMSE"]), 2),
            "mape": f"{float(row['MAPE']):.1f}%",
            "wMAPE": f"{float(row['wMAPE']):.1f}%",
            "bias": f"{float(row['Bias']):.1f}%",
            "fillRate": f"{float(row['Fill_Rate']):.1f}%",
            "nSamples": int(row["N_samples"]),
        })

    # Determine best cells
    min_mae = eval_df["MAE"].min()
    min_rmse = eval_df["RMSE"].min()
    min_wmape = eval_df["wMAPE"].min()
    max_fill = eval_df["Fill_Rate"].max()
    for r in rankings:
        r["bestMAE"] = bool(abs(r["mae"] - float(min_mae)) < 0.01)
        r["bestRMSE"] = bool(abs(r["rmse"] - float(min_rmse)) < 0.01)
        r["bestWMAPE"] = bool(abs(float(r["wMAPE"].rstrip("%")) - float(min_wmape)) < 0.1)
        r["bestFill"] = bool(abs(float(r["fillRate"].rstrip("%")) - float(max_fill)) < 0.1)

    # wMAPE bar data
    wmape_bar = [{"model": r["model"], "wMAPE": float(r["wMAPE"].rstrip("%"))} for r in rankings]

    # Radar data (normalized)
    metrics = ["MAE", "RMSE", "wMAPE", "Fill_Rate"]
    mins = {m: eval_df[m].min() for m in metrics}
    maxs = {m: eval_df[m].max() for m in metrics}

    def norm(val, mi, mx, invert=False):
        if mx == mi:
            return 0.5
        n = (val - mi) / (mx - mi)
        return round(1 - n if invert else n, 3)

    radar_data = []
    for metric in metrics:
        point = {"metric": metric}
        for _, row in eval_df.iterrows():
            invert = metric != "Fill_Rate"
            point[row["Model"]] = norm(row[metric], mins[metric], maxs[metric], invert)
        radar_data.append(point)

    radar_models = [row["Model"] for _, row in eval_df.iterrows()]

    # KPI cards: best model, ensemble metrics
    best = rankings[0]
    ensemble_row = next((r for r in rankings if r["model"] == "ensemble"), rankings[0])

    return {
        "rankings": rankings,
        "wmapeBar": wmape_bar,
        "radarData": radar_data,
        "radarModels": radar_models,
        "kpi": {
            "bestModel": best["model"],
            "mape": ensemble_row["mape"],
            "rmse": str(ensemble_row["rmse"]),
            "fillRate": ensemble_row["fillRate"],
        },
    }


@app.get("/api/models/{model_name}/deep-dive", tags=["Models"])
def model_deep_dive(model_name: str):
    d = state.data
    eval_df = d.get("evaluation")

    # Model info
    model_info = None
    if eval_df is not None:
        match = eval_df[eval_df["Model"] == model_name]
        if len(match) > 0:
            row = match.iloc[0]
            model_info = {
                "mae": round(float(row["MAE"]), 2),
                "wMAPE": f"{float(row['wMAPE']):.1f}%",
                "fillRate": f"{float(row['Fill_Rate']):.1f}%",
            }

    # Feature importance (only for tree-based models)
    feature_importance = []
    models = d.get("models", {})
    # Models are stored as {name}_model in the pkl files
    model_obj = None
    for key in [model_name, f"{model_name}_model"]:
        candidate = models.get(key)
        if candidate is not None:
            model_obj = candidate
            break

    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        # Get feature names (different APIs for sklearn vs lightgbm vs xgboost)
        names = None
        if hasattr(model_obj, "feature_names_in_"):
            names = list(model_obj.feature_names_in_)
        elif hasattr(model_obj, "feature_name_") and callable(model_obj.feature_name_):
            names = model_obj.feature_name_()
        elif hasattr(model_obj, "get_booster") and callable(model_obj.get_booster):
            try:
                names = model_obj.get_booster().feature_names
            except Exception:
                pass
        if names is None:
            names = [f"feature_{i}" for i in range(len(importances))]
        fi = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:10]
        feature_importance = [{"feature": str(n), "importance": round(float(v), 4)} for n, v in fi]

    # Actual vs predicted scatter (from test predictions)
    scatter = []
    pred_col = f"pred_{model_name}"
    if "features" in d:
        df = d["features"]
        if pred_col in df.columns:
            valid = df[df[pred_col].notna() & df["quantity_ordered"].notna()].copy()
            if len(valid) > 500:
                valid = valid.sample(500, random_state=42)
            for _, row in valid.iterrows():
                scatter.append({
                    "x": round(float(row["quantity_ordered"]), 1),
                    "y": round(float(row[pred_col]), 1),
                })

    return {
        "modelInfo": model_info,
        "featureImportance": feature_importance,
        "scatter": scatter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROOT + HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {"service": "CuraNex AI — Forecast API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["System"])
def health():
    d = state.data
    return {
        "status": "healthy",
        "data_loaded": "features" in d,
        "total_retailers": d.get("retailers", pd.DataFrame()).shape[0],
        "total_skus": d.get("skus", pd.DataFrame()).shape[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
