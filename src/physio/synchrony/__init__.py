"""
Raw RR Interval Synchrony Analysis.

Computes physiological coupling between dyads using raw RR interval
time series (windowed correlation, phase locking, spectral coherence,
cross-recurrence, transfer entropy) and event-aligned analysis.

Authors: Guillaume Dumas
Date: March 2026
"""

from .rr_loader import RRLoader, RRTimeSeries, resample_rr_to_uniform
from .rr_synchrony_stats import compute_rr_synchrony_for_all_dyads

__all__ = [
    "RRLoader",
    "RRTimeSeries",
    "resample_rr_to_uniform",
    "compute_rr_synchrony_for_all_dyads",
]
