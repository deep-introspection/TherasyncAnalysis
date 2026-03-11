"""Tests for dynamic synchrony metrics."""

import numpy as np
import pandas as pd
from src.physio.dppa.synchrony_calculator import (
    compute_centroid_correlation,
    compute_feature_concordance,
    compute_lagged_cross_correlation,
)
from src.physio.dppa.synchrony_stats import compare_real_vs_pseudo_synchrony


def _make_centroids(
    n_epochs: int = 120,
    seed: int = 0,
    centroid_x: np.ndarray | None = None,
) -> pd.DataFrame:
    """Helper: build a centroid DataFrame with optional custom centroid_x."""
    rng = np.random.default_rng(seed)
    if centroid_x is None:
        centroid_x = rng.normal(800, 50, n_epochs)
    return pd.DataFrame(
        {
            "epoch_id": np.arange(1, n_epochs + 1),
            "centroid_x": centroid_x,
            "centroid_y": centroid_x + rng.normal(0, 5, n_epochs),
            "sd1": rng.normal(20, 5, n_epochs),
            "sd2": rng.normal(40, 10, n_epochs),
            "sd_ratio": rng.normal(0.5, 0.1, n_epochs),
            "n_intervals": rng.integers(50, 200, n_epochs),
        }
    )


class TestCentroidCorrelation:
    def test_perfect_correlation_detected(self) -> None:
        signal = np.sin(np.linspace(0, 4 * np.pi, 120)) * 50 + 800
        c1 = _make_centroids(centroid_x=signal, seed=1)
        c2 = _make_centroids(centroid_x=signal, seed=2)

        result = compute_centroid_correlation(c1, c2)
        assert result["correlation"] > 0.99
        assert result["p_value"] < 1e-10
        assert result["n_valid_epochs"] == 120

    def test_independent_series_near_zero(self) -> None:
        c1 = _make_centroids(seed=42)
        c2 = _make_centroids(seed=99)

        result = compute_centroid_correlation(c1, c2)
        assert abs(result["correlation"]) < 0.3

    def test_nan_epochs_excluded(self) -> None:
        c1 = _make_centroids(seed=1)
        c2 = _make_centroids(seed=2)
        c1.loc[0:9, "centroid_x"] = np.nan
        c2.loc[50:59, "centroid_x"] = np.nan

        result = compute_centroid_correlation(c1, c2)
        assert result["n_valid_epochs"] == 100

    def test_insufficient_epochs_returns_nan(self) -> None:
        c1 = _make_centroids(n_epochs=10, seed=1)
        c2 = _make_centroids(n_epochs=10, seed=2)

        result = compute_centroid_correlation(c1, c2, min_valid=30)
        assert np.isnan(result["correlation"])
        assert np.isnan(result["p_value"])


class TestLaggedCrossCorrelation:
    def test_lagged_signal_detected_at_correct_lag(self) -> None:
        n = 120
        signal = np.sin(np.linspace(0, 6 * np.pi, n + 2)) * 50 + 800
        c1 = _make_centroids(centroid_x=signal[:n], seed=1)
        c2 = _make_centroids(centroid_x=signal[2 : n + 2], seed=2)

        result = compute_lagged_cross_correlation(c1, c2, max_lag=5)
        assert result["peak_lag"] == 2
        assert result["peak_correlation"] > 0.95

    def test_zero_lag_matches_centroid_correlation(self) -> None:
        signal = np.sin(np.linspace(0, 4 * np.pi, 120)) * 50 + 800
        c1 = _make_centroids(centroid_x=signal, seed=1)
        c2 = _make_centroids(centroid_x=signal, seed=2)

        lagged = compute_lagged_cross_correlation(c1, c2)
        direct = compute_centroid_correlation(c1, c2)
        assert abs(lagged["zero_lag_correlation"] - direct["correlation"]) < 0.01


class TestFeatureConcordance:
    def test_sd_concordance_with_known_correlation(self) -> None:
        rng = np.random.default_rng(42)
        n = 120
        shared = rng.normal(20, 5, n)
        noise1 = rng.normal(0, 1, n)
        noise2 = rng.normal(0, 1, n)

        c1 = _make_centroids(seed=1)
        c2 = _make_centroids(seed=2)
        c1["sd1"] = shared + noise1
        c2["sd1"] = shared + noise2

        result = compute_feature_concordance(c1, c2, features=["sd1"])
        assert result["sd1"]["correlation"] > 0.8
        assert result["sd1"]["p_value"] < 0.001

    def test_independent_sd_features(self) -> None:
        c1 = _make_centroids(seed=10)
        c2 = _make_centroids(seed=20)

        result = compute_feature_concordance(c1, c2, features=["sd1", "sd2"])
        for feat in ("sd1", "sd2"):
            assert abs(result[feat]["correlation"]) < 0.3


class TestRealVsPseudoSynchrony:
    def test_planted_signal_detected(self) -> None:
        """Real dyads have r~0.5, pseudo have r~0, aggregated test detects it."""
        rng = np.random.default_rng(123)
        n_real, n_pseudo = 20, 80
        n_participants = 10

        records = []
        for i in range(n_real):
            p1 = f"p{i % n_participants:02d}"
            p2 = f"p{(i + 1) % n_participants:02d}"
            records.append(
                {
                    "dyad_pair": f"{p1}_ses-01_vs_{p2}_ses-01",
                    "is_real": True,
                    "participant1": p1,
                    "session1": "ses-01",
                    "participant2": p2,
                    "session2": "ses-01",
                    "metric_value": rng.normal(0.5, 0.15),
                }
            )
        for i in range(n_pseudo):
            p1 = f"p{i % n_participants:02d}"
            p2 = f"p{(i + 3) % n_participants:02d}"
            records.append(
                {
                    "dyad_pair": f"{p1}_ses-01_vs_{p2}_ses-02",
                    "is_real": False,
                    "participant1": p1,
                    "session1": "ses-01",
                    "participant2": p2,
                    "session2": "ses-02",
                    "metric_value": rng.normal(0.0, 0.15),
                }
            )

        df = pd.DataFrame(records)
        results = compare_real_vs_pseudo_synchrony(df)

        # Naive should easily detect
        assert results["naive"]["p"] < 0.001
        assert results["naive"]["d"] > 1.0

        # Aggregated should also detect
        assert results["aggregated"]["p"] < 0.05
