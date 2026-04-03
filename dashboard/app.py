"""
CuraNex AI — Decision Intelligence Dashboard
=================================================
Streamlit-based interactive dashboard for pharmaceutical demand forecasting.

Pages:
1. Overview: KPI summary, forecast accuracy trends
2. Forecast Explorer: drill-down by retailer, SKU, region, category
3. Model Performance: comparison across all models
4. Anomaly Alerts: flagged unusual ordering patterns
5. Recommendations: pre-staging suggestions per retailer
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CuraNex AI — Demand Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0E1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #233554;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #8899A6;
        font-size: 0.85rem;
        margin-top: 5px;
        font-weight: 500;
    }
    .metric-delta {
        color: #00E396;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .metric-delta.negative {
        color: #FF4560;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 8px 16px;
        border: 1px solid #233554;
    }
    
    h1 { 
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .section-header {
        color: #E1E8ED;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #233554;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load featured data and model outputs."""
    data = {}
    
    # Load featured/transaction data
    features_path = cfg.SYNTHETIC_DIR / "features.csv"
    if features_path.exists():
        data["features"] = pd.read_csv(features_path, parse_dates=["date"])
    else:
        # Fall back to raw transactions
        trans_path = cfg.SYNTHETIC_DIR / "transactions.csv"
        if trans_path.exists():
            data["features"] = pd.read_csv(trans_path, parse_dates=["date"])
        else:
            st.error("No data found. Please run the training pipeline first.")
            return None
    
    # Load supporting data
    for name in ["retailers", "skus", "calendar"]:
        path = cfg.SYNTHETIC_DIR / f"{name}.csv"
        if path.exists():
            parse = ["date"] if name == "calendar" else None
            data[name] = pd.read_csv(path, parse_dates=parse)
    
    # Load evaluation report
    eval_path = cfg.EVAL_DIR / "evaluation_report.csv"
    if eval_path.exists():
        data["evaluation"] = pd.read_csv(eval_path)
    
    # Load test predictions and merge into features
    pred_path = cfg.EVAL_DIR / "predictions.csv"
    if pred_path.exists() and "features" in data:
        try:
            # We specifically parse dates to ensure merge works cleanly
            preds = pd.read_csv(pred_path, parse_dates=["date"])
            
            if "pred_ensemble" in preds.columns:
                # Merge the predictions onto the features dataframe
                # We drop quantity_ordered from preds to avoid duplicate _x, _y columns since features already has it
                preds_to_merge = preds.drop(columns=["quantity_ordered"], errors="ignore")
                data["features"] = data["features"].merge(
                    preds_to_merge, 
                    on=["retailer_id", "sku_id", "date"], 
                    how="left"
                )
        except Exception as e:
            st.warning(f"Could not load historical predictions: {e}")
            
    # Load true future predictions
    future_pred_path = cfg.EVAL_DIR / "forecasts.csv"
    if future_pred_path.exists():
        try:
            future_preds = pd.read_csv(future_pred_path, parse_dates=["date"])
            data["future_predictions"] = future_preds
        except Exception as e:
            st.warning(f"Could not load future forecasts: {e}")
            
    return data


def render_metric_card(label: str, value: str, delta: str = None, delta_type: str = "positive"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        cls = "negative" if delta_type == "negative" else ""
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 💊 CuraNex AI")
    st.markdown("*Demand Intelligence Platform*")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Forecast Explorer", "📈 Model Performance",
         "⚠️ Anomaly Alerts", "📋 Recommendations"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### Filters")


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW PAGE
# ─────────────────────────────────────────────────────────────────────────────

data = load_data()

if data is None:
    st.stop()

df = data.get("features", pd.DataFrame())

if page == "📊 Overview":
    st.markdown("# 📊 System Overview")
    st.markdown("*Real-time demand forecasting intelligence across the distribution network*")
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        n_retailers = df["retailer_id"].nunique() if "retailer_id" in df.columns else 0
        render_metric_card("Active Retailers", f"{n_retailers:,}", "")
    
    with col2:
        n_skus = df["sku_id"].nunique() if "sku_id" in df.columns else 0
        render_metric_card("SKUs Tracked", f"{n_skus:,}", "")
    
    with col3:
        total_orders = len(df)
        render_metric_card("Total Orders", f"{total_orders:,}", "")
    
    with col4:
        # wMAPE from evaluation
        eval_df = data.get("evaluation")
        if eval_df is not None and "wMAPE" in eval_df.columns:
            best_wmape = eval_df["wMAPE"].min()
            render_metric_card("Best wMAPE", f"{best_wmape:.1f}%", "↓ vs baseline", "positive")
        else:
            render_metric_card("Best wMAPE", "—")
    
    with col5:
        if "quantity_ordered" in df.columns:
            avg_daily = df["quantity_ordered"].mean()
            render_metric_card("Avg Order Qty", f"{avg_daily:.0f}", "")
    
    st.markdown("---")
    
    # Demand trends
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-header">📈 Weekly Demand Trend</div>', unsafe_allow_html=True)
        
        if "date" in df.columns and "quantity_ordered" in df.columns:
            weekly = df.groupby("date")["quantity_ordered"].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly["date"], y=weekly["quantity_ordered"],
                mode="lines",
                name="Historical Demand",
                fill="tozeroy",
                line=dict(color="#3a7bd5", width=2),
                fillcolor="rgba(58,123,213,0.2)"
            ))
            
            future_preds = data.get("future_predictions")
            if future_preds is not None and not future_preds.empty:
                wk_future = future_preds.groupby("date")["pred_ensemble"].sum().reset_index()
                
                # Prepend the last boundary point of historical actuals to seamlessly connect the traces
                last_hist_point = pd.DataFrame({
                    "date": [weekly["date"].iloc[-1]],
                    "pred_ensemble": [weekly["quantity_ordered"].iloc[-1]]
                })
                wk_future = pd.concat([last_hist_point, wk_future], ignore_index=True)
                
                fig.add_trace(go.Scatter(
                    x=wk_future["date"], y=wk_future["pred_ensemble"],
                    mode="lines",
                    name="Future Projection (4 Weeks)",
                    line=dict(color="#00E396", width=3, dash="solid"),
                ))
                
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="", yaxis_title="Total Quantity Ordered",
                height=350,
                margin=dict(l=20, r=20, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown('<div class="section-header">🏥 Demand by Retailer Type</div>', unsafe_allow_html=True)
        
        if "retailer_type" in df.columns:
            type_demand = df.groupby("retailer_type")["quantity_ordered"].sum().reset_index()
            fig = px.pie(
                type_demand, names="retailer_type", values="quantity_ordered",
                color_discrete_sequence=["#00d2ff", "#3a7bd5", "#7b2ff7"],
                hole=0.4,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Demand by therapeutic category
    st.markdown('<div class="section-header">💊 Demand by Therapeutic Category</div>', unsafe_allow_html=True)
    
    if "therapeutic_category" in df.columns:
        cat_demand = df.groupby("therapeutic_category")["quantity_ordered"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(
            cat_demand, x="quantity_ordered", y="therapeutic_category",
            orientation="h",
            color="quantity_ordered",
            color_continuous_scale=["#1a1a2e", "#3a7bd5", "#00d2ff"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Total Quantity", yaxis_title="",
            height=400,
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tier distribution
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-header">🏢 Demand by Retailer Tier</div>', unsafe_allow_html=True)
        if "tier" in df.columns:
            tier_data = df.groupby("tier")["quantity_ordered"].agg(["sum", "mean", "count"]).reset_index()
            tier_data.columns = ["Tier", "Total Demand", "Avg Order", "Order Count"]
            fig = px.bar(
                tier_data, x="Tier", y="Total Demand",
                color="Tier",
                color_discrete_map={"A": "#00d2ff", "B": "#3a7bd5", "C": "#7b2ff7"},
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown('<div class="section-header">📍 Top Districts by Demand</div>', unsafe_allow_html=True)
        if "district" in df.columns:
            district_data = df.groupby("district")["quantity_ordered"].sum().nlargest(10).reset_index()
            fig = px.bar(
                district_data, x="quantity_ordered", y="district",
                orientation="h",
                color_discrete_sequence=["#3a7bd5"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="Total Demand", yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Forecast Explorer":
    st.markdown("# 🔍 Forecast Explorer")
    st.markdown("*Drill down into demand forecasts by retailer, SKU, and region*")
    
    col1, col2, col3 = st.columns(3)
    
    # Build display label lookups from master data
    retailers_df = data.get("retailers", pd.DataFrame())
    skus_df = data.get("skus", pd.DataFrame())
    
    # Retailer: "Pharmacy Name (SLMC ID)" → retailer_id
    retailer_id_to_label = {}
    retailer_label_to_id = {}
    if "retailer_id" in df.columns:
        for rid in sorted(df["retailer_id"].unique()):
            rid_str = str(rid)
            if len(retailers_df) > 0 and "retailer_name" in retailers_df.columns:
                match = retailers_df[retailers_df["retailer_id"].astype(str) == rid_str]
                if len(match) > 0:
                    name = match.iloc[0]["retailer_name"]
                    label = f"{name} ({rid_str})"
                else:
                    label = rid_str
            else:
                label = rid_str
            retailer_id_to_label[rid] = label
            retailer_label_to_id[label] = rid
    
    # SKU: "Generic Name (NMRA ID)" → sku_id
    sku_id_to_label = {}
    sku_label_to_id = {}
    if "sku_id" in df.columns:
        for sid in sorted(df["sku_id"].unique()):
            sid_str = str(sid)
            if len(skus_df) > 0 and "generic_name" in skus_df.columns:
                match = skus_df[skus_df["sku_id"].astype(str) == sid_str]
                if len(match) > 0:
                    name = match.iloc[0]["generic_name"]
                    label = f"{name} ({sid_str})"
                else:
                    label = sid_str
            else:
                label = sid_str
            sku_id_to_label[sid] = label
            sku_label_to_id[label] = sid
    
    with col1:
        retailer_labels = sorted(retailer_id_to_label.values())
        selected_retailer_label = st.selectbox("Retailer", ["All"] + retailer_labels)
        selected_retailer = retailer_label_to_id.get(selected_retailer_label, "All") if selected_retailer_label != "All" else "All"
    
    with col2:
        sku_labels = sorted(sku_id_to_label.values())
        selected_sku_label = st.selectbox("SKU", ["All"] + sku_labels)
        selected_sku = sku_label_to_id.get(selected_sku_label, "All") if selected_sku_label != "All" else "All"
    
    with col3:
        categories = sorted(df["therapeutic_category"].unique()) if "therapeutic_category" in df.columns else []
        selected_cat = st.selectbox("Therapeutic Category", ["All"] + list(categories))
    
    # Filter data
    filtered = df.copy()
    if selected_retailer != "All":
        filtered = filtered[filtered["retailer_id"] == selected_retailer]
    if selected_sku != "All":
        filtered = filtered[filtered["sku_id"] == selected_sku]
    if selected_cat != "All":
        filtered = filtered[filtered["therapeutic_category"] == selected_cat]
    
    st.markdown(f"*Showing {len(filtered):,} records*")
    
    # Time series plot
    st.markdown('<div class="section-header">📈 Demand Time Series</div>', unsafe_allow_html=True)
    
    if "date" in filtered.columns and len(filtered) > 0:
        has_preds = "pred_ensemble" in filtered.columns and filtered["pred_ensemble"].notna().any()
        
        # If predictions exist, filter out training data to only show the test period boundary onwards
        if has_preds:
            test_start_date = filtered.dropna(subset=["pred_ensemble"])["date"].min()
            filtered = filtered[filtered["date"] >= test_start_date]
        
        has_snaive = "pred_snaive" in filtered.columns and filtered["pred_snaive"].notna().any()
        
        # Calculate sums per date. For predictions, if all values are NaN, return NaN instead of 0
        if has_preds or has_snaive:
            agg_dict = {"quantity_ordered": ("quantity_ordered", "sum")}
            if has_preds:
                agg_dict["pred_ensemble"] = ("pred_ensemble", lambda x: x.sum(min_count=1))
            if has_snaive:
                agg_dict["pred_snaive"] = ("pred_snaive", lambda x: x.sum(min_count=1))
            ts = filtered.groupby("date", as_index=False).agg(**agg_dict)
        else:
            ts = filtered.groupby("date", as_index=False).agg(
                quantity_ordered=("quantity_ordered", "sum")
            )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["quantity_ordered"],
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="#00d2ff", width=2),
            marker=dict(size=4),
        ))
        
        # Add SNaive Baseline if available
        if has_snaive:
            fig.add_trace(go.Scatter(
                x=ts["date"], y=ts["pred_snaive"],
                mode="lines",
                name="SNaïve Baseline",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
            ))
        
        # Add Historical AI Validation Forecast if available
        if has_preds:
            fig.add_trace(go.Scatter(
                x=ts["date"], y=ts["pred_ensemble"],
                mode="lines",
                name="Historical Validation Forecast",
                line=dict(color="#FFD700", width=3, dash="dot"),
            ))
            
        # Add True Future Exploration
        future_preds = data.get("future_predictions")
        if future_preds is not None and not future_preds.empty:
            fut_filtered = future_preds.copy()
            if selected_retailer != "All":
                fut_filtered = fut_filtered[fut_filtered["retailer_id"] == selected_retailer]
            if selected_sku != "All":
                fut_filtered = fut_filtered[fut_filtered["sku_id"] == selected_sku]
            
            # Note: Category is not inside future predictions trivially if we didn't join it back in yet.
            if selected_cat != "All":
                # Only filter if therapeutic category is present!
                if "therapeutic_category" in fut_filtered.columns:
                    fut_filtered = fut_filtered[fut_filtered["therapeutic_category"] == selected_cat]
            
            if len(fut_filtered) > 0:
                fut_ts = fut_filtered.groupby("date", as_index=False)["pred_ensemble"].sum()
                
                # Prepend the final Historical AI Validation point to physically connect the graphs
                if has_preds and len(ts) > 0:
                    last_val_point = pd.DataFrame({
                        "date": [ts["date"].iloc[-1]],
                        "pred_ensemble": [ts["pred_ensemble"].iloc[-1]]
                    })
                    fut_ts = pd.concat([last_val_point, fut_ts], ignore_index=True)
                elif len(ts) > 0:
                    # If AI wasn't run on validation, connect straight to actuals
                    last_val_point = pd.DataFrame({
                        "date": [ts["date"].iloc[-1]],
                        "pred_ensemble": [ts["quantity_ordered"].iloc[-1]]
                    })
                    fut_ts = pd.concat([last_val_point, fut_ts], ignore_index=True)

                fig.add_trace(go.Scatter(
                    x=fut_ts["date"], y=fut_ts["pred_ensemble"],
                    mode="lines",
                    name="Future Extrapolation (4 Weeks)",
                    line=dict(color="#00E396", width=4, dash="solid"),
                ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
        if "quantity_ordered" in filtered.columns and len(filtered) > 0:
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Total"],
                "Value": [
                    f"{filtered['quantity_ordered'].mean():.1f}",
                    f"{filtered['quantity_ordered'].median():.1f}",
                    f"{filtered['quantity_ordered'].std():.1f}",
                    f"{filtered['quantity_ordered'].min():.0f}",
                    f"{filtered['quantity_ordered'].max():.0f}",
                    f"{filtered['quantity_ordered'].sum():,.0f}",
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col_b:
        st.markdown('<div class="section-header">📦 Order Distribution</div>', unsafe_allow_html=True)
        if "quantity_ordered" in filtered.columns and len(filtered) > 0:
            fig = px.histogram(
                filtered, x="quantity_ordered",
                nbins=50,
                color_discrete_sequence=["#3a7bd5"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="Quantity Ordered",
                yaxis_title="Frequency",
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")
    st.markdown("*Compare forecasting accuracy across all model families*")
    
    eval_df = data.get("evaluation")
    
    if eval_df is not None and len(eval_df) > 0:
        # Model comparison table
        st.markdown('<div class="section-header">🏆 Model Ranking</div>', unsafe_allow_html=True)
        st.dataframe(
            eval_df.style.highlight_min(subset=["wMAPE", "MAE", "RMSE"], color="#1a472a")
                         .highlight_max(subset=["Fill_Rate"], color="#1a472a")
                         .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.1f}%", 
                                  "wMAPE": "{:.1f}%", "Bias": "{:.1f}%", "Fill_Rate": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        
        # Metric comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">wMAPE by Model</div>', unsafe_allow_html=True)
            fig = px.bar(
                eval_df.sort_values("wMAPE"),
                x="Model", y="wMAPE",
                color="wMAPE",
                color_continuous_scale=["#00d2ff", "#ff4560"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=10, b=20),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Fill Rate by Model</div>', unsafe_allow_html=True)
            fig = px.bar(
                eval_df.sort_values("Fill_Rate", ascending=False),
                x="Model", y="Fill_Rate",
                color="Fill_Rate",
                color_continuous_scale=["#ff4560", "#00d2ff"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=10, b=20),
                coloraxis_showscale=False,
                yaxis_range=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart
        st.markdown('<div class="section-header">🎯 Multi-Metric Radar</div>', unsafe_allow_html=True)
        
        metrics_for_radar = ["MAE", "RMSE", "wMAPE", "Fill_Rate"]
        available_metrics = [m for m in metrics_for_radar if m in eval_df.columns]
        
        if available_metrics:
            fig = go.Figure()
            colors = ["#00d2ff", "#3a7bd5", "#7b2ff7", "#FF6B6B", "#00E396"]
            
            for i, (_, row) in enumerate(eval_df.iterrows()):
                vals = [row[m] for m in available_metrics]
                # Normalize to 0-1 range for radar
                max_vals = eval_df[available_metrics].max()
                norm_vals = [v / mx if mx > 0 else 0 for v, mx in zip(vals, max_vals)]
                norm_vals.append(norm_vals[0])  # close the loop
                
                fig.add_trace(go.Scatterpolar(
                    r=norm_vals,
                    theta=available_metrics + [available_metrics[0]],
                    name=row["Model"],
                    line=dict(color=colors[i % len(colors)], width=2),
                    fill="toself",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)},{int(colors[i % len(colors)][3:5], 16)},{int(colors[i % len(colors)][5:7], 16)},0.1)",
                ))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=450,
                margin=dict(l=60, r=60, t=30, b=30),
                polar=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No evaluation data available. Run the training pipeline first:\n```\npython pipeline/train_pipeline.py\n```")


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY ALERTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚠️ Anomaly Alerts":
    st.markdown("# ⚠️ Anomaly Detection & Alerts")
    st.markdown("*Flag unusual ordering patterns for investigation*")
    
    if "quantity_ordered" in df.columns and "retailer_id" in df.columns:
        # Compute anomalies using IQR method per retailer-SKU pair
        st.markdown('<div class="section-header">🔍 Anomalous Orders Detected</div>', unsafe_allow_html=True)
        
        # Simple anomaly: orders > 3 std from the pair mean
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
        
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            n_anomalies = len(anomalies)
            render_metric_card("Anomalies Detected", str(n_anomalies), f"out of {len(df):,} orders")
        
        with col_b:
            pct = (n_anomalies / len(df) * 100) if len(df) > 0 else 0
            render_metric_card("Anomaly Rate", f"{pct:.2f}%", "")
        
        if len(anomalies) > 0:
            show_cols = ["date", "retailer_id", "sku_id", "quantity_ordered", "pair_mean", "z_score"]
            available = [c for c in show_cols if c in anomalies.columns]
            
            st.dataframe(
                anomalies[available].head(50).style.format({
                    "pair_mean": "{:.1f}",
                    "z_score": "{:.2f}",
                    "quantity_ordered": "{:.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            
            # Timeline of anomalies
            st.markdown('<div class="section-header">📅 Anomaly Timeline</div>', unsafe_allow_html=True)
            anom_timeline = anomalies.groupby("date").size().reset_index(name="count")
            fig = px.bar(
                anom_timeline, x="date", y="count",
                color_discrete_sequence=["#FF4560"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="", yaxis_title="Anomaly Count",
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📋 Recommendations":
    st.markdown("# 📋 Pre-Staging Recommendations")
    st.markdown("*AI-driven reorder suggestions for proactive inventory management*")
    
    if "retailer_id" in df.columns and "quantity_ordered" in df.columns:
        # Generate recommendations based on recent demand patterns
        recent = df.sort_values("date").groupby(["retailer_id", "sku_id"]).tail(4)
        
        recommendations = recent.groupby(["retailer_id", "sku_id"]).agg(
            avg_recent_demand=("quantity_ordered", "mean"),
            std_recent_demand=("quantity_ordered", "std"),
            last_order_qty=("quantity_ordered", "last"),
        ).reset_index()
        
        recommendations["std_recent_demand"] = recommendations["std_recent_demand"].fillna(0)
        
        # Suggested order = mean + 1.5*std (safety stock buffer)
        recommendations["suggested_qty"] = (
            recommendations["avg_recent_demand"] + 1.5 * recommendations["std_recent_demand"]
        ).round(0).astype(int)
        
        recommendations["safety_stock"] = (1.5 * recommendations["std_recent_demand"]).round(0).astype(int)
        
        # Add SKU info if available
        if "skus" in data:
            recommendations = recommendations.merge(
                data["skus"][["sku_id", "therapeutic_category", "is_critical"]],
                on="sku_id",
                how="left",
            )
        
        # Priority: critical SKUs first, then by suggested quantity
        if "is_critical" in recommendations.columns:
            recommendations = recommendations.sort_values(
                ["is_critical", "suggested_qty"],
                ascending=[False, False],
            )
        
        # Filters
        col1, col2 = st.columns(2)
        
        # Build retailer label lookup for this page
        retailers_df = data.get("retailers", pd.DataFrame())
        rec_retailer_labels = {}
        rec_label_to_id = {}
        for rid in sorted(recommendations["retailer_id"].unique()):
            rid_str = str(rid)
            if len(retailers_df) > 0 and "retailer_name" in retailers_df.columns:
                match = retailers_df[retailers_df["retailer_id"].astype(str) == rid_str]
                if len(match) > 0:
                    label = f"{match.iloc[0]['retailer_name']} ({rid_str})"
                else:
                    label = rid_str
            else:
                label = rid_str
            rec_retailer_labels[rid] = label
            rec_label_to_id[label] = rid
        
        with col1:
            retailer_labels_list = sorted(rec_retailer_labels.values())
            retailer_filter_label = st.selectbox(
                "Filter by Retailer",
                ["All"] + retailer_labels_list,
            )
            retailer_filter = rec_label_to_id.get(retailer_filter_label, "All") if retailer_filter_label != "All" else "All"
        with col2:
            critical_only = st.checkbox("Critical SKUs Only", value=False)
        
        filtered_rec = recommendations.copy()
        if retailer_filter != "All":
            filtered_rec = filtered_rec[filtered_rec["retailer_id"] == retailer_filter]
        if critical_only and "is_critical" in filtered_rec.columns:
            filtered_rec = filtered_rec[filtered_rec["is_critical"] == True]
        
        st.markdown(f'<div class="section-header">📦 Top Reorder Suggestions ({len(filtered_rec):,} items)</div>',
                     unsafe_allow_html=True)
        
        show_cols = ["retailer_id", "sku_id", "avg_recent_demand", "last_order_qty",
                     "suggested_qty", "safety_stock"]
        if "therapeutic_category" in filtered_rec.columns:
            show_cols.insert(2, "therapeutic_category")
        if "is_critical" in filtered_rec.columns:
            show_cols.insert(3, "is_critical")
        
        available = [c for c in show_cols if c in filtered_rec.columns]
        
        st.dataframe(
            filtered_rec[available].head(100).style.format({
                "avg_recent_demand": "{:.1f}",
                "last_order_qty": "{:.0f}",
                "suggested_qty": "{:.0f}",
                "safety_stock": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        # Summary by category
        if "therapeutic_category" in filtered_rec.columns:
            st.markdown('<div class="section-header">📊 Recommendations by Category</div>', unsafe_allow_html=True)
            cat_summary = filtered_rec.groupby("therapeutic_category")["suggested_qty"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(
                cat_summary, x="therapeutic_category", y="suggested_qty",
                color="suggested_qty",
                color_continuous_scale=["#1a1a2e", "#3a7bd5", "#00d2ff"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="", yaxis_title="Total Suggested Quantity",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #455A64; font-size: 0.8rem;">'
    'CuraNex AI — Hybrid Ensemble Demand Forecasting | AITHON 2026 | Hemas Pharmaceuticals'
    '</div>',
    unsafe_allow_html=True,
)
