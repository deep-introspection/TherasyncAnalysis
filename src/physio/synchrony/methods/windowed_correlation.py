"""
Windowed Pearson Correlation on RR Intervals.

Sliding-window correlation captures intermittent coupling: real dyads
may have more high-synchrony bursts than pseudo dyads even if
session-level correlation is near zero.

Authors: Guillaume Dumas
Date: March 2026
"""

import numpy as np
from scipy.stats import pearsonr

from src.physio.synchrony.rr_loader import RRTimeSeries, resample_rr_to_uniform


def compute_windowed_correlation(
    rr1: RRTimeSeries,
    rr2: RRTimeSeries,
    fs: float = 4.0,
    window_s: float = 60.0,
    step_s: float = 30.0,
) -> dict[str, float]:
    """Sliding-window Pearson r on uniformly resampled RR intervals.

    Args:
        rr1: First participant's RR time series.
        rr2: Second participant's RR time series.
        fs: Resampling frequency in Hz.
        window_s: Window duration in seconds.
        step_s: Step size in seconds.

    Returns:
        Dict with mean_r, median_r, max_r, frac_significant, n_windows.
    """
    t1, v1 = resample_rr_to_uniform(rr1, fs)
    t2, v2 = resample_rr_to_uniform(rr2, fs)

    # Align to overlapping time range
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])

    if t_end - t_start < window_s:
        return {
            "mean_r": np.nan,
            "median_r": np.nan,
            "max_r": np.nan,
            "frac_significant": np.nan,
            "n_windows": 0,
        }

    n_samples = int((t_end - t_start) * fs) + 1
    t_common = np.linspace(t_start, t_end, n_samples)

    from scipy.interpolate import interp1d

    s1 = interp1d(t1, v1, kind="linear", fill_value="extrapolate")(t_common)
    s2 = interp1d(t2, v2, kind="linear", fill_value="extrapolate")(t_common)

    win_samples = int(window_s * fs)
    step_samples = int(step_s * fs)

    correlations: list[float] = []
    n_sig = 0
    pos = 0

    while pos + win_samples <= len(s1):
        w1 = s1[pos : pos + win_samples]
        w2 = s2[pos : pos + win_samples]

        if np.std(w1) > 1e-10 and np.std(w2) > 1e-10:
            # Detrend within window to remove shared drift
            t_win = np.arange(win_samples)
            w1d = w1 - np.polyval(np.polyfit(t_win, w1, 1), t_win)
            w2d = w2 - np.polyval(np.polyfit(t_win, w2, 1), t_win)
            if np.std(w1d) > 1e-10 and np.std(w2d) > 1e-10:
                r, p = pearsonr(w1d, w2d)
                correlations.append(r)
                if p < 0.05:
                    n_sig += 1

        pos += step_samples

    if not correlations:
        return {
            "mean_r": np.nan,
            "median_r": np.nan,
            "max_r": np.nan,
            "frac_significant": np.nan,
            "n_windows": 0,
        }

    r_arr = np.array(correlations)
    return {
        "mean_r": float(np.mean(r_arr)),
        "median_r": float(np.median(r_arr)),
        "max_r": float(np.max(np.abs(r_arr))),
        "frac_significant": float(n_sig / len(correlations)),
        "n_windows": len(correlations),
    }
