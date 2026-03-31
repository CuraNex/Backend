"""
CuraNex AI — Evaluation Visualisations
========================================
Generates publication-quality plots for model evaluation:
- Model comparison bar charts (MAE, RMSE, wMAPE, Bias)
- Actual vs Predicted scatter and line plots
- Residual distributions and Q-Q plots
- Per-segment heatmaps (tier × category)
- Train / Val / Test metric comparison
- Feature importance charts

All plots are saved to  models_saved/plots/  as PNG files.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger, ensure_dir
from src.evaluation.metrics import compute_all_metrics

logger = setup_logger("Plots")

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = [
    "#6366f1", "#ec4899", "#14b8a6", "#f59e0b", "#8b5cf6",
    "#ef4444", "#22c55e", "#3b82f6", "#f97316", "#06b6d4",
]
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 150,
})

PLOT_DIR = ensure_dir(cfg.PLOTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    y_true: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    split_name: str = "Test",
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing MAE, RMSE, wMAPE, Bias across all models.
    """
    metrics_list = []
    for name, preds in model_preds.items():
        m = compute_all_metrics(y_true, preds)
        m["Model"] = name
        metrics_list.append(m)

    df = pd.DataFrame(metrics_list).set_index("Model")
    metric_cols = ["MAE", "RMSE", "wMAPE"]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        vals = df[metric].sort_values()
        colours = [PALETTE[i % len(PALETTE)] for i in range(len(vals))]
        bars = ax.barh(vals.index, vals.values, color=colours, edgecolor="none", height=0.55)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel(metric)
        for bar, v in zip(bars, vals.values):
            ax.text(v + vals.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=9, color="#e2e8f0")

    fig.suptitle(f"Model Comparison — {split_name} Set", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"model_comparison_{split_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. ACTUAL vs PREDICTED
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Ensemble",
    split_name: str = "Test",
    max_points: int = 5_000,
    save: bool = True,
) -> plt.Figure:
    """Scatter plot of actual vs predicted with a 45° reference line."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Subsample for readability
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, c=PALETTE[0], edgecolors="none")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#f59e0b", linewidth=1.2, label="Perfect forecast")
    ax.set_xlabel("Actual Quantity")
    ax.set_ylabel("Predicted Quantity")
    ax.set_title(f"{model_name} — Actual vs Predicted ({split_name})", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"actual_vs_pred_{model_name.lower()}_{split_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESIDUAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Ensemble",
    split_name: str = "Test",
    save: bool = True,
) -> plt.Figure:
    """Histogram of residuals (actual − predicted) with mean/std annotations."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(residuals, bins=80, color=PALETTE[0], edgecolor="none", alpha=0.85)
    ax.axvline(0, color="#f59e0b", linestyle="--", linewidth=1)
    ax.axvline(residuals.mean(), color=PALETTE[1], linestyle="-", linewidth=1.2, label=f"Mean = {residuals.mean():.2f}")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    # Residual vs Predicted
    ax2 = axes[1]
    idx = np.random.choice(len(y_pred), min(3000, len(y_pred)), replace=False)
    ax2.scatter(np.asarray(y_pred)[idx], residuals[idx], alpha=0.2, s=6, c=PALETTE[2], edgecolors="none")
    ax2.axhline(0, color="#f59e0b", linestyle="--", linewidth=1)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual vs Predicted", fontweight="bold")

    fig.suptitle(f"{model_name} — Residual Analysis ({split_name})", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"residuals_{model_name.lower()}_{split_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / VAL / TEST METRIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_split_comparison(
    split_metrics: Dict[str, Dict[str, float]],
    model_name: str = "Ensemble",
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart showing MAE / RMSE / wMAPE across Train, Val, Test splits.

    Args:
        split_metrics: {"Train": {"MAE": .., "RMSE": .., ...}, "Val": {...}, "Test": {...}}
    """
    df = pd.DataFrame(split_metrics).T
    metric_cols = [c for c in ["MAE", "RMSE", "wMAPE"] if c in df.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.22
    for i, metric in enumerate(metric_cols):
        bars = ax.bar(x + i * width, df[metric], width, label=metric,
                      color=PALETTE[i], edgecolor="none")
        for bar, v in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + df[metric].max() * 0.01,
                    f"{v:.2f}", ha="center", fontsize=8, color="#e2e8f0")

    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{model_name} — Train / Val / Test Metrics", fontweight="bold")
    ax.legend(ncol=len(metric_cols))
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"split_comparison_{model_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEGMENT HEATMAP  (Tier × Category → wMAPE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_segment_heatmap(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target: str = "quantity_ordered",
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Heatmap of wMAPE by retailer tier × therapeutic category.
    """
    if "tier" not in df.columns or "therapeutic_category" not in df.columns:
        logger.warning("Segment columns missing — skipping heatmap")
        return None

    eval_df = df.copy()
    eval_df["pred"] = y_pred[:len(eval_df)]

    from src.evaluation.metrics import wmape as _wmape

    pivot = eval_df.groupby(["tier", "therapeutic_category"]).apply(
        lambda g: _wmape(g[target].values, g["pred"].values), include_groups=False
    ).unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.9), max(4, len(pivot) * 1.2)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Therapeutic Category")
    ax.set_ylabel("Retailer Tier")
    ax.set_title("wMAPE by Segment (lower = better)", fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            colour = "#0f172a" if v < pivot.values.mean() else "#e2e8f0"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=colour)

    fig.colorbar(im, ax=ax, label="wMAPE %", shrink=0.8)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / "segment_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "LightGBM",
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of top‐N feature importances."""
    top = importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top["feature"], top["importance"], color=PALETTE[0], edgecolor="none", height=0.6)
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Top {top_n} Features", fontweight="bold")
    fig.tight_layout()

    if save:
        path = PLOT_DIR / f"feature_importance_{model_name.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. FORECAST TIMELINE  (sample series)
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_timeline(
    df: pd.DataFrame,
    preds: np.ndarray,
    n_series: int = 4,
    target: str = "quantity_ordered",
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Line chart showing actual vs predicted for a few sampled retailer-SKU pairs.
    Uses real retailer_name and brand for human-readable labels.
    """
    if "retailer_id" not in df.columns or "sku_id" not in df.columns:
        return None

    eval_df = df.copy()
    eval_df["pred"] = preds[:len(eval_df)]
    eval_df["pair"] = eval_df["retailer_id"].astype(str) + "_" + eval_df["sku_id"].astype(str)

    # Build display label from retailer_name and brand (if available)
    if "retailer_name" in eval_df.columns and "generic_name" in eval_df.columns:
        eval_df["pair_label"] = eval_df["retailer_name"].fillna(eval_df["retailer_id"].astype(str)) + " × " + eval_df["generic_name"].fillna(eval_df["sku_id"].astype(str))
    else:
        eval_df["pair_label"] = eval_df["pair"]

    # Pick most active pairs
    top_pairs = eval_df.groupby("pair")[target].sum().nlargest(n_series).index

    fig, axes = plt.subplots(n_series, 1, figsize=(12, 3.5 * n_series), sharex=True)
    if n_series == 1:
        axes = [axes]

    for ax, pair in zip(axes, top_pairs):
        sub = eval_df[eval_df["pair"] == pair].sort_values("date")
        display_label = sub["pair_label"].iloc[0] if len(sub) > 0 else pair
        ax.plot(sub["date"], sub[target], label="Actual", color=PALETTE[0], linewidth=1.5)
        ax.plot(sub["date"], sub["pred"], label="Predicted", color=PALETTE[1], linewidth=1.5, linestyle="--")
        ax.set_ylabel("Qty")
        ax.set_title(display_label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Forecast Timeline — Top Series", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / "forecast_timeline.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION — generate all plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    splits: dict,
    all_predictions: dict,
    models: dict,
    ensemble_preds: Dict[str, np.ndarray] = None,
) -> None:
    """
    Generate all evaluation plots after training.

    Args:
        splits: dict from prepare_splits (X_train, y_train, df_train, etc.)
        all_predictions: {model_key: preds} e.g. {"lgbm_val": ..., "lgbm_test": ...}
        models: {name: model_object}
        ensemble_preds: {"val": ..., "test": ..., optionally "train": ...}
    """
    logger.info("\n" + "━" * 60)
    logger.info("Generating Evaluation Plots")
    logger.info("━" * 60)

    ensure_dir(PLOT_DIR)
    plt.close("all")

    # ── 1. Model comparison on Test set ──
    test_preds = {}
    for key, preds in all_predictions.items():
        if key.endswith("_test"):
            name = key.replace("_test", "")
            test_preds[name] = np.asarray(preds)[:len(splits["y_test"])]

    if ensemble_preds and "test" in ensemble_preds:
        test_preds["ensemble"] = np.asarray(ensemble_preds["test"])[:len(splits["y_test"])]

    if test_preds:
        y_test = splits["y_test"].values
        plot_model_comparison(y_test, test_preds, split_name="Test")

        # ── 2. Actual vs Predicted + Residuals (best model or ensemble) ──
        best_name = "ensemble" if "ensemble" in test_preds else list(test_preds.keys())[0]
        best_preds = test_preds[best_name]

        plot_actual_vs_predicted(y_test, best_preds, model_name=best_name, split_name="Test")
        plot_residuals(y_test, best_preds, model_name=best_name, split_name="Test")

        # ── 3. Segment heatmap ──
        plot_segment_heatmap(splits["df_test"], best_preds)

        # ── 4. Forecast timeline ──
        plot_forecast_timeline(splits["df_test"], best_preds)

    # ── 5. Train / Val / Test split metrics for each model ──
    for model_name in ["lgbm", "xgb"]:
        split_m = {}
        val_key, test_key = f"{model_name}_val", f"{model_name}_test"

        if val_key in all_predictions:
            y_val = splits["y_val"].values
            split_m["Val"] = compute_all_metrics(y_val, np.asarray(all_predictions[val_key])[:len(y_val)])

        if test_key in all_predictions:
            y_test = splits["y_test"].values
            split_m["Test"] = compute_all_metrics(y_test, np.asarray(all_predictions[test_key])[:len(y_test)])

        if split_m:
            plot_split_comparison(split_m, model_name=model_name)

    # Ensemble split comparison
    if ensemble_preds:
        split_m = {}
        if "val" in ensemble_preds:
            split_m["Val"] = compute_all_metrics(
                splits["y_val"].values,
                np.asarray(ensemble_preds["val"])[:len(splits["y_val"])],
            )
        if "test" in ensemble_preds:
            split_m["Test"] = compute_all_metrics(
                splits["y_test"].values,
                np.asarray(ensemble_preds["test"])[:len(splits["y_test"])],
            )
        if split_m:
            plot_split_comparison(split_m, model_name="Ensemble")

    # ── 6. Feature importance ──
    for name, model in models.items():
        if hasattr(model, "feature_importance") and model.feature_importance is not None:
            plot_feature_importance(model.feature_importance, model_name=name)

    plt.close("all")
    logger.info(f"All plots saved to {PLOT_DIR}")
