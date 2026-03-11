"""
Tests for statistical correction utilities.

Tests cover:
- correct_pvalues: multiple comparison correction (bonferroni, holm, fdr_bh)
- epsilon_squared: effect size for Kruskal-Wallis
- compute_icc: intraclass correlation coefficient
"""

import numpy as np
import pandas as pd
import pytest

from src.stats.corrections import compute_icc, correct_pvalues, epsilon_squared


class TestCorrectPvalues:
    """Tests for correct_pvalues function."""

    def test_fdr_bh_correction(self):
        """Test FDR (Benjamini-Hochberg) correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]
        rejected, p_adjusted = correct_pvalues(p_values, method="fdr_bh")

        assert len(rejected) == 4
        assert len(p_adjusted) == 4
        # Adjusted p-values should be >= original
        assert all(p_adjusted[i] >= p_values[i] for i in range(4))

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]
        rejected, p_adjusted = correct_pvalues(p_values, method="bonferroni")

        # Bonferroni: p_adj = p * n_tests
        assert p_adjusted[0] == pytest.approx(0.04)
        assert p_adjusted[3] == pytest.approx(0.80)

    def test_holm_correction(self):
        """Test Holm correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]
        rejected, p_adjusted = correct_pvalues(p_values, method="holm")

        assert len(rejected) == 4
        # Holm is less conservative than Bonferroni
        _, p_bonf = correct_pvalues(p_values, method="bonferroni")
        assert all(p_adjusted[i] <= p_bonf[i] for i in range(4))

    def test_empty_input(self):
        """Test with empty p-values array."""
        rejected, p_adjusted = correct_pvalues([])

        assert len(rejected) == 0
        assert len(p_adjusted) == 0

    def test_single_pvalue(self):
        """Test with single p-value."""
        rejected, p_adjusted = correct_pvalues([0.03])

        assert len(rejected) == 1
        assert p_adjusted[0] == pytest.approx(0.03)

    def test_all_significant(self):
        """Test when all p-values are significant."""
        p_values = [0.001, 0.002, 0.005]
        rejected, _ = correct_pvalues(p_values, method="fdr_bh")

        assert all(rejected)

    def test_none_significant(self):
        """Test when no p-values are significant."""
        p_values = [0.5, 0.6, 0.7]
        rejected, _ = correct_pvalues(p_values, method="bonferroni")

        assert not any(rejected)


class TestEpsilonSquared:
    """Tests for epsilon_squared function."""

    def test_basic_computation(self):
        """Test basic epsilon-squared computation."""
        # H=10, n=100, k=3 → (10-3+1)/(100-3) = 8/97 ≈ 0.0825
        result = epsilon_squared(10.0, 100, 3)
        assert result == pytest.approx(8 / 97, rel=1e-4)

    def test_zero_effect(self):
        """Test epsilon-squared with H near k-1 (no effect)."""
        # H=k-1 → epsilon² = 0
        result = epsilon_squared(2.0, 100, 3)
        assert result == pytest.approx(0.0)

    def test_clamps_to_zero(self):
        """Test that negative epsilon-squared clamps to 0."""
        # H < k-1 → raw value negative, should clamp
        result = epsilon_squared(1.0, 100, 3)
        assert result == 0.0

    def test_degenerate_denominator(self):
        """Test with n <= k (degenerate case)."""
        result = epsilon_squared(5.0, 3, 3)
        assert np.isnan(result)

    def test_large_effect(self):
        """Test with large effect size."""
        result = epsilon_squared(50.0, 60, 3)
        assert 0 < result <= 1.0


class TestComputeICC:
    """Tests for compute_icc function."""

    def test_high_icc(self):
        """Test ICC with high between-group variance."""
        np.random.seed(42)
        data = []
        for group in range(5):
            group_mean = group * 100
            for _ in range(20):
                data.append(
                    {"group": f"g{group}", "value": group_mean + np.random.normal(0, 1)}
                )
        df = pd.DataFrame(data)

        icc = compute_icc(df, "group", "value")
        assert icc > 0.9

    def test_low_icc(self):
        """Test ICC with low between-group variance."""
        np.random.seed(42)
        data = []
        for group in range(5):
            for _ in range(20):
                data.append({"group": f"g{group}", "value": np.random.normal(0, 10)})
        df = pd.DataFrame(data)

        icc = compute_icc(df, "group", "value")
        assert icc < 0.2

    def test_single_group(self):
        """Test ICC with single group returns NaN."""
        df = pd.DataFrame({"group": ["a"] * 10, "value": range(10)})
        icc = compute_icc(df, "group", "value")
        assert np.isnan(icc)

    def test_single_observation_per_group(self):
        """Test ICC with one observation per group returns NaN."""
        df = pd.DataFrame({"group": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        icc = compute_icc(df, "group", "value")
        # df_within = n_total - n_groups = 3-3 = 0, so NaN
        assert np.isnan(icc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
