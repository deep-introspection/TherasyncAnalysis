"""Tests for RR interval synchrony methods."""

import numpy as np

from src.physio.synchrony.rr_loader import RRTimeSeries
from src.physio.synchrony.methods.windowed_correlation import (
    compute_windowed_correlation,
)
from src.physio.synchrony.methods.phase_locking import compute_phase_locking_value
from src.physio.synchrony.methods.wavelet_coherence import compute_spectral_coherence
from src.physio.synchrony.methods.cross_recurrence import compute_crqa
from src.physio.synchrony.methods.transfer_entropy import compute_transfer_entropy


def _make_rr(
    n: int = 1000,
    rr_mean: float = 800.0,
    seed: int = 0,
    freq: float = 0.1,
    amplitude: float = 30.0,
) -> RRTimeSeries:
    """Create synthetic RR time series with optional sinusoidal modulation."""
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.7, 1.0, n))
    rr_ms = (
        rr_mean + amplitude * np.sin(2 * np.pi * freq * times) + rng.normal(0, 10, n)
    )
    return RRTimeSeries(
        times=times,
        rr_ms=rr_ms,
        subject=f"s{seed}",
        session="ses-01",
        task="therapy",
        duration_s=float(times[-1] - times[0]),
    )


def _make_correlated_pair(
    n: int = 1000, correlation: float = 0.8, seed: int = 42
) -> tuple[RRTimeSeries, RRTimeSeries]:
    """Create two RR time series with controlled correlation."""
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.7, 1.0, n))

    # Shared oscillation
    shared = 30.0 * np.sin(2 * np.pi * 0.1 * times)
    noise1 = rng.normal(0, 10, n)
    noise2 = rng.normal(0, 10, n)

    rr1 = 800.0 + correlation * shared + (1 - correlation) * noise1
    rr2 = 800.0 + correlation * shared + (1 - correlation) * noise2

    ts1 = RRTimeSeries(
        times=times,
        rr_ms=rr1,
        subject="s1",
        session="ses-01",
        task="therapy",
        duration_s=float(times[-1] - times[0]),
    )
    ts2 = RRTimeSeries(
        times=times,
        rr_ms=rr2,
        subject="s2",
        session="ses-01",
        task="therapy",
        duration_s=float(times[-1] - times[0]),
    )
    return ts1, ts2


class TestWindowedCorrelation:
    def test_correlated_signals_high_frac_significant(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=2000, correlation=0.9)
        result = compute_windowed_correlation(ts1, ts2)
        assert result["n_windows"] > 0
        assert result["frac_significant"] > 0.3
        assert -1.0 <= result["mean_r"] <= 1.0

    def test_independent_signals_lower_frac_than_correlated(self) -> None:
        # Independent (noise-only) should have lower frac_significant than correlated
        ts_indep1 = _make_rr(n=2000, seed=0, amplitude=0.0)
        ts_indep2 = _make_rr(n=2000, seed=99, amplitude=0.0)
        result_indep = compute_windowed_correlation(ts_indep1, ts_indep2)

        ts_corr1, ts_corr2 = _make_correlated_pair(n=2000, correlation=0.9)
        result_corr = compute_windowed_correlation(ts_corr1, ts_corr2)

        assert result_indep["frac_significant"] < result_corr["frac_significant"]

    def test_short_signal_returns_nan(self) -> None:
        ts1 = _make_rr(n=20, seed=0)
        ts2 = _make_rr(n=20, seed=1)
        result = compute_windowed_correlation(ts1, ts2, window_s=60.0)
        assert np.isnan(result["mean_r"])

    def test_output_keys(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=1000)
        result = compute_windowed_correlation(ts1, ts2)
        assert set(result.keys()) == {
            "mean_r",
            "median_r",
            "max_r",
            "frac_significant",
            "n_windows",
        }


class TestPhaseLocking:
    def test_same_frequency_signals_high_plv(self) -> None:
        # Two signals with same 0.1 Hz modulation
        ts1, ts2 = _make_correlated_pair(n=3000, correlation=0.95)
        result = compute_phase_locking_value(ts1, ts2, band=(0.04, 0.15))
        assert result["plv"] > 0.3
        assert 0.0 <= result["plv"] <= 1.0

    def test_independent_signals_low_plv(self) -> None:
        ts1 = _make_rr(n=3000, seed=0, freq=0.08)
        ts2 = _make_rr(n=3000, seed=99, freq=0.12)
        result = compute_phase_locking_value(ts1, ts2, band=(0.04, 0.15))
        assert result["plv"] < 0.5

    def test_short_signal_returns_nan(self) -> None:
        ts1 = _make_rr(n=10, seed=0)
        ts2 = _make_rr(n=10, seed=1)
        result = compute_phase_locking_value(ts1, ts2, band=(0.04, 0.15))
        assert np.isnan(result["plv"])

    def test_output_keys(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=2000)
        result = compute_phase_locking_value(ts1, ts2)
        assert set(result.keys()) == {"plv", "mean_phase_diff", "circular_std"}


class TestSpectralCoherence:
    def test_correlated_signals_high_coherence(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=3000, correlation=0.95)
        result = compute_spectral_coherence(ts1, ts2)
        assert result["mean_coherence"] > 0.1
        assert 0.0 <= result["lf_coherence"] <= 1.0

    def test_independent_signals_low_coherence(self) -> None:
        ts1 = _make_rr(n=3000, seed=0)
        ts2 = _make_rr(n=3000, seed=99)
        result = compute_spectral_coherence(ts1, ts2)
        assert result["mean_coherence"] < 0.5

    def test_short_signal_returns_nan(self) -> None:
        ts1 = _make_rr(n=20, seed=0)
        ts2 = _make_rr(n=20, seed=1)
        result = compute_spectral_coherence(ts1, ts2)
        assert np.isnan(result["mean_coherence"])

    def test_output_keys(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=2000)
        result = compute_spectral_coherence(ts1, ts2)
        assert "lf_coherence" in result
        assert "hf_coherence" in result
        assert "mean_coherence" in result


class TestCRQA:
    def test_correlated_signals_higher_determinism(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=1000, correlation=0.9)
        result_corr = compute_crqa(ts1, ts2)

        ts3 = _make_rr(n=1000, seed=0)
        ts4 = _make_rr(n=1000, seed=99)
        result_indep = compute_crqa(ts3, ts4)

        assert result_corr["determinism"] > result_indep["determinism"]

    def test_output_keys(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=500)
        result = compute_crqa(ts1, ts2)
        assert set(result.keys()) == {
            "recurrence_rate",
            "determinism",
            "mean_diagonal_length",
        }

    def test_recurrence_rate_bounded(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=500)
        result = compute_crqa(ts1, ts2)
        assert 0.0 <= result["recurrence_rate"] <= 1.0

    def test_short_signal_returns_nan(self) -> None:
        # Only 2 points — too few for resampling + embedding
        ts = RRTimeSeries(
            times=np.array([0.0, 0.3]),
            rr_ms=np.array([800.0, 810.0]),
            subject="s1",
            session="ses-01",
            task="therapy",
            duration_s=0.3,
        )
        result = compute_crqa(ts, ts, max_points=500)
        assert np.isnan(result["recurrence_rate"])


class TestTransferEntropy:
    def test_asymmetric_coupling_detected(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        times = np.cumsum(rng.uniform(0.7, 1.0, n))

        # X drives Y with a lag
        x = 800.0 + 30.0 * np.sin(2 * np.pi * 0.1 * times) + rng.normal(0, 5, n)
        y = np.zeros(n)
        y[0] = 800.0
        for i in range(1, n):
            y[i] = 0.7 * y[i - 1] + 0.3 * x[i - 1] + rng.normal(0, 5)

        ts1 = RRTimeSeries(
            times=times,
            rr_ms=x,
            subject="s1",
            session="ses-01",
            task="therapy",
            duration_s=float(times[-1]),
        )
        ts2 = RRTimeSeries(
            times=times,
            rr_ms=y,
            subject="s2",
            session="ses-01",
            task="therapy",
            duration_s=float(times[-1]),
        )

        result = compute_transfer_entropy(ts1, ts2)
        # X→Y should be > Y→X
        assert result["te_1_to_2"] > result["te_2_to_1"]
        assert result["asymmetry_index"] > 0

    def test_symmetric_signals(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=2000, correlation=0.5)
        result = compute_transfer_entropy(ts1, ts2)
        # Both directions should be similar
        assert abs(result["asymmetry_index"]) < 0.5

    def test_output_keys(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=1000)
        result = compute_transfer_entropy(ts1, ts2)
        assert set(result.keys()) == {
            "te_1_to_2",
            "te_2_to_1",
            "net_te",
            "asymmetry_index",
        }

    def test_non_negative(self) -> None:
        ts1, ts2 = _make_correlated_pair(n=1000)
        result = compute_transfer_entropy(ts1, ts2)
        assert result["te_1_to_2"] >= 0
        assert result["te_2_to_1"] >= 0
