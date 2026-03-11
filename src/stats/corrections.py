"""
Statistical correction utilities for multiple comparisons and effect sizes.

Provides reusable functions for p-value correction, effect size computation,
and intraclass correlation coefficients.

Authors: Remy Ramadour
Date: March 2026
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def correct_pvalues(
    p_values: np.ndarray | list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: Array of uncorrected p-values.
        method: Correction method ('bonferroni', 'holm', 'fdr_bh').
        alpha: Significance level.

    Returns:
        Tuple of (rejected, p_adjusted) arrays.
    """
    p_arr = np.asarray(p_values, dtype=float)
    if len(p_arr) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    rejected, p_adjusted, _, _ = multipletests(p_arr, alpha=alpha, method=method)
    return rejected, p_adjusted


def epsilon_squared(h_statistic: float, n_total: int, k_groups: int) -> float:
    """
    Compute epsilon-squared effect size for Kruskal-Wallis H-test.

    epsilon² = (H - k + 1) / (n - k)

    Args:
        h_statistic: Kruskal-Wallis H statistic.
        n_total: Total number of observations.
        k_groups: Number of groups.

    Returns:
        Epsilon-squared value in [0, 1].
    """
    denominator = n_total - k_groups
    if denominator <= 0:
        return np.nan
    return max(0.0, (h_statistic - k_groups + 1) / denominator)


def compute_icc(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> float:
    """
    Compute ICC(1) via one-way ANOVA decomposition.

    ICC(1) = (MSb - MSw) / (MSb + (k-1)*MSw)
    where k = mean group size.

    Args:
        data: DataFrame with observations.
        group_col: Column identifying groups (e.g., 'dyad', 'family').
        value_col: Column with the measurement values.

    Returns:
        ICC(1) value. Returns NaN if computation is not possible.
    """
    groups = data.groupby(group_col)[value_col]
    n_groups = groups.ngroups

    if n_groups < 2:
        return np.nan

    grand_mean = data[value_col].mean()
    group_means = groups.mean()
    group_sizes = groups.count()

    n_total = group_sizes.sum()
    k_mean = n_total / n_groups

    # Between-group sum of squares
    ss_between = sum(
        group_sizes[g] * (group_means[g] - grand_mean) ** 2 for g in group_means.index
    )
    # Within-group sum of squares
    ss_within = sum(
        ((data.loc[data[group_col] == g, value_col] - group_means[g]) ** 2).sum()
        for g in group_means.index
    )

    df_between = n_groups - 1
    df_within = n_total - n_groups

    if df_between == 0 or df_within == 0:
        return np.nan

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    denominator = ms_between + (k_mean - 1) * ms_within
    if denominator == 0:
        return np.nan

    return (ms_between - ms_within) / denominator
