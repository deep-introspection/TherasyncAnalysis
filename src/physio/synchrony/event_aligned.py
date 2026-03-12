"""
Event-Aligned Synchrony Analysis.

Slices RR time series by MOI annotation windows (positive / negative /
neutral alliance states) and applies any synchrony metric function to
each state's data.

Authors: Guillaume Dumas
Date: March 2026
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.physio.synchrony.rr_loader import RRTimeSeries

logger = logging.getLogger(__name__)


def _slice_rr(rr: RRTimeSeries, t_start: float, t_end: float) -> RRTimeSeries | None:
    """Extract a time window from an RRTimeSeries."""
    mask = (rr.times >= t_start) & (rr.times <= t_end)
    if mask.sum() < 10:
        return None

    return RRTimeSeries(
        times=rr.times[mask],
        rr_ms=rr.rr_ms[mask],
        subject=rr.subject,
        session=rr.session,
        task=rr.task,
        duration_s=float(t_end - t_start),
    )


def compute_event_aligned_synchrony(
    rr1: RRTimeSeries,
    rr2: RRTimeSeries,
    annotations: pd.DataFrame,
    metric_fn: Callable[[RRTimeSeries, RRTimeSeries], dict[str, float]],
    value_key: str,
    state_col: str = "state",
    start_col: str = "start_seconds",
    end_col: str = "end_seconds",
) -> dict[str, dict[str, float]]:
    """Compute synchrony metric within each annotated alliance state.

    Args:
        rr1: First participant's RR time series.
        rr2: Second participant's RR time series.
        annotations: DataFrame with state labels and time windows.
        metric_fn: Synchrony metric function(rr1, rr2) -> dict.
        value_key: Key in metric result for the primary scalar value.
        state_col: Column name for the state label.
        start_col: Column name for window start time (seconds).
        end_col: Column name for window end time (seconds).

    Returns:
        {state_label: metric_result_dict} for each state with enough data.
    """
    if state_col not in annotations.columns:
        logger.warning(f"State column '{state_col}' not in annotations")
        return {}

    if start_col not in annotations.columns or end_col not in annotations.columns:
        logger.warning(f"Time columns '{start_col}'/'{end_col}' not in annotations")
        return {}

    results: dict[str, dict[str, float]] = {}

    for state, group in annotations.groupby(state_col):
        state_str = str(state)
        segment_results: list[dict[str, float]] = []

        for _, row in group.iterrows():
            t_start = float(row[start_col])
            t_end = float(row[end_col])

            seg1 = _slice_rr(rr1, t_start, t_end)
            seg2 = _slice_rr(rr2, t_start, t_end)

            if seg1 is None or seg2 is None:
                continue

            result = metric_fn(seg1, seg2)
            val = result.get(value_key, np.nan)
            if not np.isnan(val):
                segment_results.append(result)

        if not segment_results:
            continue

        # Average across segments for this state
        all_keys = segment_results[0].keys()
        averaged: dict[str, float] = {}
        for k in all_keys:
            vals = [r[k] for r in segment_results if not np.isnan(r.get(k, np.nan))]
            averaged[k] = float(np.mean(vals)) if vals else np.nan

        averaged["n_segments"] = float(len(segment_results))
        results[state_str] = averaged

    return results
