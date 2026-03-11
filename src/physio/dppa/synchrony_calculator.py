"""
Synchrony Calculator for DPPA Analysis.

Computes dynamic synchrony metrics between dyad members by correlating
Poincare feature time series across epochs. Unlike ICD (static similarity),
these metrics capture whether physiological signals change together over time.

Authors: Guillaume Dumas
Date: March 2026
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_centroid_correlation(
    centroids1: pd.DataFrame,
    centroids2: pd.DataFrame,
    feature: str = "centroid_x",
    min_valid: int = 30,
) -> dict[str, float]:
    """
    Pearson correlation of a feature time series between two participants.

    Args:
        centroids1: First participant's centroids (must have epoch_id + feature).
        centroids2: Second participant's centroids.
        feature: Column to correlate (default: centroid_x = mean heart rate).
        min_valid: Minimum overlapping non-NaN epochs required.

    Returns:
        Dict with correlation, p_value, n_valid_epochs.
    """
    merged = _merge_on_epoch(centroids1, centroids2, [feature])
    x = merged[f"{feature}1"]
    y = merged[f"{feature}2"]

    valid = x.notna() & y.notna()
    n_valid = int(valid.sum())

    if n_valid < min_valid:
        return {"correlation": np.nan, "p_value": np.nan, "n_valid_epochs": n_valid}

    r, p = stats.pearsonr(x[valid], y[valid])
    return {"correlation": float(r), "p_value": float(p), "n_valid_epochs": n_valid}


def compute_lagged_cross_correlation(
    centroids1: pd.DataFrame,
    centroids2: pd.DataFrame,
    feature: str = "centroid_x",
    max_lag: int = 5,
    min_valid: int = 30,
) -> dict[str, float]:
    """
    Cross-correlation at lags -max_lag..+max_lag epochs.

    Positive peak_lag means participant2 leads (participant1 follows).

    Args:
        centroids1: First participant's centroids.
        centroids2: Second participant's centroids.
        feature: Column to correlate.
        max_lag: Maximum lag in epochs (both directions).
        min_valid: Minimum overlapping non-NaN epochs per lag.

    Returns:
        Dict with peak_lag, peak_correlation, peak_p_value, zero_lag_correlation.
    """
    merged = _merge_on_epoch(centroids1, centroids2, [feature])
    x = merged[f"{feature}1"].values
    y = merged[f"{feature}2"].values

    nan_result = {
        "peak_lag": np.nan,
        "peak_correlation": np.nan,
        "peak_p_value": np.nan,
        "zero_lag_correlation": np.nan,
    }

    best_abs, best_r, best_p, best_lag = -1.0, np.nan, np.nan, 0
    zero_r = np.nan

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_seg = x[lag:]
            y_seg = y[: len(x) - lag]
        else:
            x_seg = x[: len(x) + lag]
            y_seg = y[-lag:]

        valid = ~(np.isnan(x_seg) | np.isnan(y_seg))
        if valid.sum() < min_valid:
            continue

        r, p = stats.pearsonr(x_seg[valid], y_seg[valid])

        if lag == 0:
            zero_r = float(r)

        if abs(r) > best_abs:
            best_abs = abs(r)
            best_r, best_p, best_lag = float(r), float(p), lag

    if best_abs < 0:
        return nan_result

    return {
        "peak_lag": best_lag,
        "peak_correlation": best_r,
        "peak_p_value": best_p,
        "zero_lag_correlation": zero_r,
    }


def compute_feature_concordance(
    centroids1: pd.DataFrame,
    centroids2: pd.DataFrame,
    features: list[str] | None = None,
    min_valid: int = 30,
) -> dict[str, dict[str, float]]:
    """
    Correlation of SD1/SD2/sd_ratio time series between dyad members.

    Tests whether autonomic variability co-fluctuates (distinct from
    centroid correlation which tests mean heart rate coupling).

    Args:
        centroids1: First participant's centroids.
        centroids2: Second participant's centroids.
        features: Features to correlate (default: sd1, sd2, sd_ratio).
        min_valid: Minimum overlapping non-NaN epochs required.

    Returns:
        Dict mapping feature name to {correlation, p_value, mean_abs_difference}.
    """
    if features is None:
        features = ["sd1", "sd2", "sd_ratio"]

    merged = _merge_on_epoch(centroids1, centroids2, features)
    results: dict[str, dict[str, float]] = {}

    for feat in features:
        col1, col2 = f"{feat}1", f"{feat}2"
        valid = merged[col1].notna() & merged[col2].notna()
        n_valid = int(valid.sum())

        if n_valid < min_valid:
            results[feat] = {
                "correlation": np.nan,
                "p_value": np.nan,
                "mean_abs_difference": np.nan,
            }
            continue

        x, y = merged.loc[valid, col1], merged.loc[valid, col2]
        r, p = stats.pearsonr(x, y)
        mad = float(np.mean(np.abs(x.values - y.values)))

        results[feat] = {
            "correlation": float(r),
            "p_value": float(p),
            "mean_abs_difference": mad,
        }

    return results


def _merge_on_epoch(
    centroids1: pd.DataFrame,
    centroids2: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Merge two centroid DataFrames on epoch_id, keeping only requested features."""
    cols = ["epoch_id", *features]
    merged = pd.merge(
        centroids1[cols],
        centroids2[cols],
        on="epoch_id",
        suffixes=("1", "2"),
        how="inner",
    ).sort_values("epoch_id")
    return merged
