"""
PharmaFlow AI — Cold Start Clustering Module
===============================================
Handles new retailers and SKUs with insufficient order history (<12 weeks)
using DTW-based similarity clustering and progressive personalization.

Strategy:
1. Cluster existing retailers by ordering patterns (DTW) + static attributes (k-means)
2. New retailers inherit their cluster's model predictions
3. As data accumulates, blend cluster and individual predictions:
   forecast = α × individual + (1-α) × cluster, where α = min(1, weeks / 24)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
from src.utils.helpers import setup_logger

logger = setup_logger("ColdStart")


class ColdStartHandler:
    """
    Cold-start module for retailers with insufficient history.
    
    Combines:
    - DTW (Dynamic Time Warping) distance for time-series similarity
    - k-means on static attributes (type, location, tier)
    - Progressive blending for transition to individual models
    """
    
    def __init__(self, n_clusters: int = cfg.N_CLUSTERS, name: str = "cold_start"):
        self.n_clusters = n_clusters
        self.name = name
        self.cluster_model = None
        self.scaler = StandardScaler()
        self.cluster_profiles = {}  # cluster_id -> average demand profile
        self.retailer_clusters = {}  # retailer_id -> cluster_id
    
    def _compute_dtw_features(self, df: pd.DataFrame, retailer_id: str) -> np.ndarray:
        """
        Extract demand profile features for DTW computation.
        
        For each retailer, compute a 52-week demand profile:
        average demand per week-of-year across all their SKUs.
        """
        r_data = df[df["retailer_id"] == retailer_id]
        
        if len(r_data) == 0:
            return np.zeros(52)
        
        # Average demand by week-of-year
        weekly_profile = r_data.groupby("week_of_year")["quantity_ordered"].mean()
        
        # Create full 52-week profile, filling gaps
        profile = np.zeros(52)
        for w, val in weekly_profile.items():
            if 1 <= w <= 52:
                profile[int(w) - 1] = val
        
        return profile
    
    def _compute_static_features(self, retailers: pd.DataFrame) -> np.ndarray:
        """
        Build static feature matrix for clustering.
        Encodes retailer type, tier, and district.
        """
        features = pd.get_dummies(
            retailers[["retailer_type", "tier", "district"]],
            drop_first=True,
        )
        return self.scaler.fit_transform(features.values)
    
    def fit(
        self,
        transactions: pd.DataFrame,
        retailers: pd.DataFrame,
    ) -> "ColdStartHandler":
        """
        Fit clustering model on existing retailers with sufficient history.
        
        Only retailers with >= COLD_START_THRESHOLD_WEEKS of data are used
        for building the clusters.
        """
        logger.info(f"[{self.name}] Fitting cold-start clusters...")
        
        # Identify retailers with sufficient history
        retailer_weeks = transactions.groupby("retailer_id")["date"].nunique()
        established = retailer_weeks[retailer_weeks >= cfg.COLD_START_THRESHOLD_WEEKS].index
        
        logger.info(f"[{self.name}] {len(established)} established retailers "
                     f"(>= {cfg.COLD_START_THRESHOLD_WEEKS} weeks of history)")
        
        # Compute demand profiles for DTW feature extraction
        profiles = []
        profile_retailers = []
        for rid in established:
            profile = self._compute_dtw_features(transactions, rid)
            if profile.sum() > 0:
                profiles.append(profile)
                profile_retailers.append(rid)
        
        profiles = np.array(profiles)
        
        # Combine with static features
        static_retailers = retailers[retailers["retailer_id"].isin(profile_retailers)].copy()
        static_retailers = static_retailers.set_index("retailer_id").loc[profile_retailers].reset_index()
        static_features = self._compute_static_features(static_retailers)
        
        # Normalize profiles
        profile_scaler = StandardScaler()
        profiles_scaled = profile_scaler.fit_transform(profiles)
        
        # Combine profile + static features (weighted)
        combined = np.hstack([profiles_scaled * 0.6, static_features * 0.4])
        
        # K-means clustering
        n_clusters = min(self.n_clusters, len(combined))
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=cfg.SEED,
            n_init=10,
        )
        cluster_labels = self.cluster_model.fit_predict(combined)
        
        # Store cluster assignments and average profiles
        for rid, label in zip(profile_retailers, cluster_labels):
            self.retailer_clusters[rid] = int(label)
        
        # Compute cluster profiles (average demand profile per cluster)
        for c in range(n_clusters):
            mask = cluster_labels == c
            self.cluster_profiles[c] = profiles[mask].mean(axis=0)
        
        logger.info(f"[{self.name}] Created {n_clusters} clusters")
        for c in range(n_clusters):
            count = (cluster_labels == c).sum()
            logger.info(f"  Cluster {c}: {count} retailers")
        
        return self
    
    def assign_cluster(self, retailer_id: str) -> int:
        """Get cluster assignment for a retailer."""
        return self.retailer_clusters.get(retailer_id, 0)
    
    def get_cluster_forecast(self, cluster_id: int, week_of_year: int) -> float:
        """Get the average demand for a cluster at a given week."""
        if cluster_id in self.cluster_profiles:
            idx = (week_of_year - 1) % 52
            return float(self.cluster_profiles[cluster_id][idx])
        return 0.0
    
    @staticmethod
    def blend_forecasts(
        individual_pred: float,
        cluster_pred: float,
        weeks_of_data: int,
    ) -> float:
        """
        Progressively blend individual and cluster forecasts.
        
        α = min(1, weeks_of_data / COLD_START_BLEND_WEEKS)
        
        - New retailer (0 weeks): α=0 → pure cluster prediction
        - 12 weeks: α=0.5 → equal blend
        - 24+ weeks: α=1 → pure individual prediction
        """
        alpha = min(1.0, weeks_of_data / cfg.COLD_START_BLEND_WEEKS)
        return alpha * individual_pred + (1 - alpha) * cluster_pred
    
    def is_cold_start(self, retailer_id: str, transactions: pd.DataFrame) -> bool:
        """Check if a retailer has insufficient history."""
        weeks = transactions[transactions["retailer_id"] == retailer_id]["date"].nunique()
        return weeks < cfg.COLD_START_THRESHOLD_WEEKS
    
    def get_weeks_of_data(self, retailer_id: str, transactions: pd.DataFrame) -> int:
        """Get number of weeks of order history for a retailer."""
        return transactions[transactions["retailer_id"] == retailer_id]["date"].nunique()
