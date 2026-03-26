"""
PharmaFlow AI — Shared Utilities
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save DataFrame to CSV with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    return pd.read_csv(path, **kwargs)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
