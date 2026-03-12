"""
Transfer Entropy on RR Intervals.

Binned histogram estimation of transfer entropy to measure
directional information flow between two RR interval time series.

Authors: Guillaume Dumas
Date: March 2026
"""

import numpy as np

from src.physio.synchrony.rr_loader import RRTimeSeries, resample_rr_to_uniform


def _equal_frequency_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin signal into n_bins using equal-frequency (quantile) binning."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentiles)
    edges[-1] += 1e-10  # Include the max value
    return np.digitize(x, edges[1:-1])


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy from counts array."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _joint_counts(a: np.ndarray, b: np.ndarray, n_bins: int) -> np.ndarray:
    """2-D histogram counts."""
    return np.histogram2d(a, b, bins=n_bins, range=[[0, n_bins], [0, n_bins]])[0]


def _joint_counts_3d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, n_bins: int
) -> np.ndarray:
    """3-D histogram counts."""
    return np.histogramdd(
        np.column_stack([a, b, c]),
        bins=n_bins,
        range=[[0, n_bins], [0, n_bins], [0, n_bins]],
    )[0]


def _transfer_entropy_xy(x: np.ndarray, y: np.ndarray, lag: int, n_bins: int) -> float:
    """TE(X→Y) = H(Y_future, Y_past) + H(Y_past, X_past) - H(Y_past) - H(Y_future, Y_past, X_past).

    This is the conditional mutual information I(Y_future; X_past | Y_past).
    """
    y_future = y[lag:]
    y_past = y[:-lag]
    x_past = x[:-lag]

    n = len(y_future)
    if n < 50:
        return np.nan

    # Bin all series
    yf = _equal_frequency_bins(y_future, n_bins)
    yp = _equal_frequency_bins(y_past, n_bins)
    xp = _equal_frequency_bins(x_past, n_bins)

    # Joint entropies
    h_yf_yp = _entropy(_joint_counts(yf, yp, n_bins).ravel())
    h_yp_xp = _entropy(_joint_counts(yp, xp, n_bins).ravel())
    h_yp = _entropy(np.bincount(yp, minlength=n_bins))
    h_yf_yp_xp = _entropy(_joint_counts_3d(yf, yp, xp, n_bins).ravel())

    te = h_yf_yp + h_yp_xp - h_yp - h_yf_yp_xp
    return max(0.0, float(te))  # TE is non-negative


def compute_transfer_entropy(
    rr1: RRTimeSeries,
    rr2: RRTimeSeries,
    fs: float = 4.0,
    lag: int = 1,
    n_bins: int = 6,
) -> dict[str, float]:
    """Compute transfer entropy between two RR time series.

    Args:
        rr1: First participant's RR time series.
        rr2: Second participant's RR time series.
        fs: Resampling frequency in Hz.
        lag: Number of samples lag for TE computation.
        n_bins: Number of bins for histogram estimation.

    Returns:
        Dict with te_1_to_2, te_2_to_1, net_te, asymmetry_index.
    """
    _, v1 = resample_rr_to_uniform(rr1, fs)
    _, v2 = resample_rr_to_uniform(rr2, fs)

    # Truncate to same length
    n = min(len(v1), len(v2))
    v1, v2 = v1[:n], v2[:n]

    te_12 = _transfer_entropy_xy(v1, v2, lag, n_bins)
    te_21 = _transfer_entropy_xy(v2, v1, lag, n_bins)

    if np.isnan(te_12) or np.isnan(te_21):
        return {
            "te_1_to_2": np.nan,
            "te_2_to_1": np.nan,
            "net_te": np.nan,
            "asymmetry_index": np.nan,
        }

    net = te_12 - te_21
    total = te_12 + te_21
    asymmetry = net / total if total > 0 else 0.0

    return {
        "te_1_to_2": float(te_12),
        "te_2_to_1": float(te_21),
        "net_te": float(net),
        "asymmetry_index": float(asymmetry),
    }
