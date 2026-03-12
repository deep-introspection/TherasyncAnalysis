"""
Phase Locking Value (PLV) on RR Intervals.

Bandpass-filters RR intervals into LF or HF band, extracts
instantaneous phase via Hilbert transform, and computes PLV
as the consistency of the inter-participant phase difference.

Authors: Guillaume Dumas
Date: March 2026
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from src.physio.synchrony.rr_loader import RRTimeSeries, resample_rr_to_uniform


def compute_phase_locking_value(
    rr1: RRTimeSeries,
    rr2: RRTimeSeries,
    fs: float = 4.0,
    band: tuple[float, float] = (0.04, 0.15),
) -> dict[str, float]:
    """Compute Phase Locking Value between two RR time series.

    Args:
        rr1: First participant's RR time series.
        rr2: Second participant's RR time series.
        fs: Resampling frequency in Hz.
        band: Frequency band (low, high) in Hz.
              LF: (0.04, 0.15), HF: (0.15, 0.4).

    Returns:
        Dict with plv, mean_phase_diff, circular_std.
    """
    t1, v1 = resample_rr_to_uniform(rr1, fs)
    t2, v2 = resample_rr_to_uniform(rr2, fs)

    # Align to overlapping time range
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    min_duration = max(3.0 / band[0], 30.0)  # Need at least 3 cycles

    if t_end - t_start < min_duration:
        return {"plv": np.nan, "mean_phase_diff": np.nan, "circular_std": np.nan}

    n_samples = int((t_end - t_start) * fs) + 1
    t_common = np.linspace(t_start, t_end, n_samples)

    from scipy.interpolate import interp1d

    s1 = interp1d(t1, v1, kind="linear", fill_value="extrapolate")(t_common)
    s2 = interp1d(t2, v2, kind="linear", fill_value="extrapolate")(t_common)

    # Bandpass filter
    nyq = fs / 2.0
    low, high = band[0] / nyq, band[1] / nyq

    if high >= 1.0:
        high = 0.99

    b, a = butter(4, [low, high], btype="band")
    f1 = filtfilt(b, a, s1 - np.mean(s1))
    f2 = filtfilt(b, a, s2 - np.mean(s2))

    # Instantaneous phase via Hilbert transform
    phase1 = np.angle(hilbert(f1))
    phase2 = np.angle(hilbert(f2))

    # Phase difference
    dphi = phase1 - phase2

    # PLV = |mean(exp(i * dphi))|
    plv = float(np.abs(np.mean(np.exp(1j * dphi))))

    # Circular mean and std of phase difference
    mean_phase = float(np.angle(np.mean(np.exp(1j * dphi))))
    circular_std = float(np.sqrt(-2.0 * np.log(plv))) if plv > 0 else float("inf")

    return {
        "plv": plv,
        "mean_phase_diff": mean_phase,
        "circular_std": circular_std,
    }
