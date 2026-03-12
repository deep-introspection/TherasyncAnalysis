"""Tests for event-aligned synchrony analysis."""

import numpy as np
import pandas as pd

from src.physio.synchrony.rr_loader import RRTimeSeries
from src.physio.synchrony.event_aligned import compute_event_aligned_synchrony


def _make_long_rr(seed: int = 0, duration: float = 600.0) -> RRTimeSeries:
    """Create a long RR time series spanning ~duration seconds."""
    rng = np.random.default_rng(seed)
    n = int(duration / 0.85)
    times = np.cumsum(rng.uniform(0.7, 1.0, n))
    times = times[times <= duration]
    rr_ms = 800.0 + rng.normal(0, 30, len(times))
    return RRTimeSeries(
        times=times,
        rr_ms=rr_ms,
        subject=f"s{seed}",
        session="ses-01",
        task="therapy",
        duration_s=float(times[-1]),
    )


def _simple_metric(rr1: RRTimeSeries, rr2: RRTimeSeries) -> dict[str, float]:
    """Simple test metric: mean RR difference."""
    return {"mean_diff": float(np.mean(rr1.rr_ms) - np.mean(rr2.rr_ms))}


class TestEventAlignedSynchrony:
    def test_splits_by_state(self) -> None:
        ts1 = _make_long_rr(seed=0)
        ts2 = _make_long_rr(seed=1)

        annotations = pd.DataFrame(
            {
                "state": ["positive", "negative", "positive"],
                "start_seconds": [10.0, 100.0, 200.0],
                "end_seconds": [90.0, 190.0, 350.0],
            }
        )

        results = compute_event_aligned_synchrony(
            ts1,
            ts2,
            annotations,
            _simple_metric,
            "mean_diff",
        )

        assert "positive" in results
        assert "negative" in results
        assert results["positive"]["n_segments"] == 2.0
        assert results["negative"]["n_segments"] == 1.0

    def test_skips_short_segments(self) -> None:
        ts1 = _make_long_rr(seed=0)
        ts2 = _make_long_rr(seed=1)

        # Very short window — won't have 10+ RR intervals
        annotations = pd.DataFrame(
            {
                "state": ["positive"],
                "start_seconds": [10.0],
                "end_seconds": [12.0],
            }
        )

        results = compute_event_aligned_synchrony(
            ts1,
            ts2,
            annotations,
            _simple_metric,
            "mean_diff",
        )

        assert "positive" not in results

    def test_missing_state_column_returns_empty(self) -> None:
        ts1 = _make_long_rr(seed=0)
        ts2 = _make_long_rr(seed=1)

        annotations = pd.DataFrame(
            {
                "label": ["positive"],
                "start_seconds": [10.0],
                "end_seconds": [90.0],
            }
        )

        results = compute_event_aligned_synchrony(
            ts1,
            ts2,
            annotations,
            _simple_metric,
            "mean_diff",
        )
        assert results == {}

    def test_missing_time_columns_returns_empty(self) -> None:
        ts1 = _make_long_rr(seed=0)
        ts2 = _make_long_rr(seed=1)

        annotations = pd.DataFrame(
            {
                "state": ["positive"],
                "start": [10.0],
                "end": [90.0],
            }
        )

        results = compute_event_aligned_synchrony(
            ts1,
            ts2,
            annotations,
            _simple_metric,
            "mean_diff",
        )
        assert results == {}

    def test_averages_across_segments(self) -> None:
        ts1 = _make_long_rr(seed=0, duration=400.0)
        ts2 = _make_long_rr(seed=1, duration=400.0)

        annotations = pd.DataFrame(
            {
                "state": ["A", "A", "A"],
                "start_seconds": [10.0, 100.0, 200.0],
                "end_seconds": [80.0, 180.0, 350.0],
            }
        )

        results = compute_event_aligned_synchrony(
            ts1,
            ts2,
            annotations,
            _simple_metric,
            "mean_diff",
        )

        assert "A" in results
        assert results["A"]["n_segments"] == 3.0
        # The averaged metric should be a float
        assert isinstance(results["A"]["mean_diff"], float)

    def test_custom_column_names(self) -> None:
        ts1 = _make_long_rr(seed=0)
        ts2 = _make_long_rr(seed=1)

        annotations = pd.DataFrame(
            {
                "alliance": ["pos", "neg"],
                "onset": [10.0, 200.0],
                "offset": [100.0, 350.0],
            }
        )

        results = compute_event_aligned_synchrony(
            ts1,
            ts2,
            annotations,
            _simple_metric,
            "mean_diff",
            state_col="alliance",
            start_col="onset",
            end_col="offset",
        )

        assert "pos" in results
        assert "neg" in results
