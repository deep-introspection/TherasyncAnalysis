"""
Tests for BVP frequency-domain duration validation.

Verifies that frequency-domain HRV metrics (LF, HF, LFHF, TP) are set to NaN
when epoch duration is below the minimum threshold (default 120s),
while time-domain metrics remain computed for all durations.
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.physio.preprocessing.bvp_metrics import BVPMetricsExtractor


@pytest.fixture
def extractor():
    """Create a BVPMetricsExtractor with mocked config."""
    with patch.object(
        BVPMetricsExtractor, "__init__", lambda self, config_path=None: None
    ):
        ext = BVPMetricsExtractor()
        ext.bvp_config = {
            "rr_intervals": {"min_valid_ms": 300, "max_valid_ms": 2000},
        }
        ext.metrics_config = {}
        ext.time_domain_metrics = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"]
        ext.frequency_domain_metrics = ["HRV_LF", "HRV_HF", "HRV_LFHF"]
        ext.nonlinear_metrics = []
        ext.min_duration_freq = 120  # seconds
        return ext


def _make_peaks(n_beats: int, sampling_rate: int = 64, rr_ms: float = 800.0):
    """Generate synthetic peak indices for n_beats."""
    rr_samples = rr_ms / 1000.0 * sampling_rate
    return np.arange(0, n_beats * rr_samples, rr_samples).astype(int)


class TestFrequencyDomainGating:
    """Tests for frequency-domain duration validation."""

    def test_short_epoch_frequency_metrics_nan(self, extractor):
        """Frequency metrics should be NaN for epochs shorter than threshold."""
        # 30 beats at 800ms RR = 24s total duration (< 120s)
        peaks = _make_peaks(30, sampling_rate=64, rr_ms=800)
        metrics = extractor._extract_hrv_metrics(peaks, 64, "short_epoch")

        # Frequency metrics should all be NaN
        assert np.isnan(metrics["HRV_LF"])
        assert np.isnan(metrics["HRV_HF"])
        assert np.isnan(metrics["HRV_LFHF"])

    def test_short_epoch_time_domain_computed(self, extractor):
        """Time-domain metrics should still be computed for short epochs."""
        peaks = _make_peaks(30, sampling_rate=64, rr_ms=800)
        metrics = extractor._extract_hrv_metrics(peaks, 64, "short_epoch")

        # Time-domain metrics should be valid numbers
        assert not np.isnan(metrics["HRV_MeanNN"])
        assert not np.isnan(metrics["HRV_SDNN"])
        assert not np.isnan(metrics["HRV_RMSSD"])

    def test_long_epoch_attempts_frequency_computation(self, extractor, caplog):
        """Long epochs should attempt frequency computation (not skip it)."""
        import logging

        # 200 beats at 800ms RR = 160s total (> 120s)
        peaks = _make_peaks(200, sampling_rate=64, rr_ms=800)
        with caplog.at_level(logging.WARNING):
            extractor._extract_hrv_metrics(peaks, 64, "long_epoch")

        # Should NOT contain the "below minimum" skip message
        skip_msgs = [
            r for r in caplog.records if "minimum for frequency-domain" in r.message
        ]
        assert len(skip_msgs) == 0

    def test_boundary_duration_not_skipped(self, extractor, caplog):
        """Epoch at threshold boundary should attempt frequency computation."""
        import logging

        # 151 beats at 800ms ≈ 120s
        peaks = _make_peaks(151, sampling_rate=64, rr_ms=800)
        with caplog.at_level(logging.WARNING):
            extractor._extract_hrv_metrics(peaks, 64, "boundary_epoch")

        skip_msgs = [
            r for r in caplog.records if "minimum for frequency-domain" in r.message
        ]
        assert len(skip_msgs) == 0

    def test_custom_threshold(self, extractor, caplog):
        """Test with a custom minimum duration threshold."""
        import logging

        extractor.min_duration_freq = 30  # Lower threshold

        # 50 beats at 800ms = 40s (> 30s threshold)
        peaks = _make_peaks(50, sampling_rate=64, rr_ms=800)
        with caplog.at_level(logging.WARNING):
            extractor._extract_hrv_metrics(peaks, 64, "custom_threshold")

        skip_msgs = [
            r for r in caplog.records if "minimum for frequency-domain" in r.message
        ]
        assert len(skip_msgs) == 0

    def test_insufficient_peaks_returns_all_nan(self, extractor):
        """With too few peaks, all metrics should be NaN."""
        peaks = _make_peaks(5, sampling_rate=64, rr_ms=800)
        metrics = extractor._extract_hrv_metrics(peaks, 64, "too_few")

        for metric in (
            extractor.time_domain_metrics + extractor.frequency_domain_metrics
        ):
            assert np.isnan(metrics[metric])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
