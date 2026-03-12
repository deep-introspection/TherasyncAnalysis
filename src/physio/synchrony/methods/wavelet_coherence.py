"""
Spectral Coherence on RR Intervals.

Uses Welch-based magnitude-squared coherence (scipy.signal.coherence)
averaged within LF and HF bands.

Authors: Guillaume Dumas
Date: March 2026
"""

import numpy as np
from scipy.signal import coherence as sp_coherence

from src.physio.synchrony.rr_loader import RRTimeSeries, resample_rr_to_uniform


def compute_spectral_coherence(
    rr1: RRTimeSeries,
    rr2: RRTimeSeries,
    fs: float = 4.0,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Welch-based magnitude-squared coherence between RR time series.

    Args:
        rr1: First participant's RR time series.
        rr2: Second participant's RR time series.
        fs: Resampling frequency in Hz.
        freq_bands: Frequency bands to average over.
                    Defaults to LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz).

    Returns:
        Dict with lf_coherence, hf_coherence, mean_coherence.
    """
    if freq_bands is None:
        freq_bands = {"lf": (0.04, 0.15), "hf": (0.15, 0.4)}

    t1, v1 = resample_rr_to_uniform(rr1, fs)
    t2, v2 = resample_rr_to_uniform(rr2, fs)

    # Align to overlapping time range
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])

    if t_end - t_start < 60.0:
        return {f"{name}_coherence": np.nan for name in freq_bands} | {
            "mean_coherence": np.nan
        }

    n_samples = int((t_end - t_start) * fs) + 1
    t_common = np.linspace(t_start, t_end, n_samples)

    from scipy.interpolate import interp1d

    s1 = interp1d(t1, v1, kind="linear", fill_value="extrapolate")(t_common)
    s2 = interp1d(t2, v2, kind="linear", fill_value="extrapolate")(t_common)

    # Welch coherence with ~60s segments
    nperseg = min(int(60 * fs), len(s1) // 2)
    if nperseg < int(20 * fs):
        return {f"{name}_coherence": np.nan for name in freq_bands} | {
            "mean_coherence": np.nan
        }

    freqs, coh = sp_coherence(s1, s2, fs=fs, nperseg=nperseg)

    result: dict[str, float] = {}
    all_band_values: list[float] = []

    for name, (flo, fhi) in freq_bands.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.any():
            val = float(np.mean(coh[mask]))
        else:
            val = np.nan
        result[f"{name}_coherence"] = val
        if not np.isnan(val):
            all_band_values.append(val)

    result["mean_coherence"] = (
        float(np.mean(all_band_values)) if all_band_values else np.nan
    )
    return result
