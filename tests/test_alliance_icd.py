"""
Tests for Alliance-ICD Analysis modules.

Tests cover:
- AllianceICDLoader: data loading and merging
- AllianceICDAnalyzer: statistical computations
- AllianceICDStatsPlotter: visualization generation

Authors: Remy Ramadour
Date: November 2025
"""

import warnings

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alliance.alliance_icd_loader import AllianceICDLoader
from src.alliance.alliance_icd_analyzer import AllianceICDAnalyzer


class TestAllianceICDLoader:
    """Tests for AllianceICDLoader class."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration."""
        return {
            "paths": {"derivatives": str(tmp_path / "derivatives")},
            "output": {
                "alliance_dir": "alliance",
                "alliance_subdirs": {"annotations": "annotations"},
            },
        }

    @pytest.fixture
    def loader(self, mock_config):
        """Create loader with mocked config."""
        with patch.object(
            AllianceICDLoader, "__init__", lambda self, config_path=None: None
        ):
            loader = AllianceICDLoader()
            loader.config = mock_config
            loader.derivatives_path = Path(mock_config["paths"]["derivatives"])
            loader.alliance_dir = mock_config["output"]["alliance_dir"]
            loader._alliance_cache = {}
            loader._icd_cache = {}
            return loader

    def test_alliance_states_mapping(self):
        """Test that alliance states are correctly defined."""
        assert AllianceICDLoader.ALLIANCE_STATES[0] == "neutral"
        assert AllianceICDLoader.ALLIANCE_STATES[1] == "positive"
        assert AllianceICDLoader.ALLIANCE_STATES[-1] == "negative"
        assert AllianceICDLoader.ALLIANCE_STATES[2] == "split"

    def test_compute_alliance_states_empty(self, loader):
        """Test computing alliance states with empty data."""
        df = pd.DataFrame({"epoch_nsplit": [[], []], "alliance": [np.nan, np.nan]})

        states = loader._compute_alliance_states(df, "epoch_nsplit")
        assert states == {}

    def test_compute_alliance_states_positive(self, loader):
        """Test computing positive alliance state."""
        df = pd.DataFrame({"epoch_nsplit": [[0, 1], [1, 2]], "alliance": [1, 1]})

        states = loader._compute_alliance_states(df, "epoch_nsplit")

        assert states[0] == 1  # positive
        assert states[1] == 1  # positive
        assert states[2] == 1  # positive

    def test_compute_alliance_states_negative(self, loader):
        """Test computing negative alliance state."""
        df = pd.DataFrame({"epoch_nsplit": [[0, 1]], "alliance": [-1]})

        states = loader._compute_alliance_states(df, "epoch_nsplit")

        assert states[0] == -1  # negative
        assert states[1] == -1  # negative

    def test_compute_alliance_states_split(self, loader):
        """Test computing split (mixed) alliance state."""
        df = pd.DataFrame({"epoch_nsplit": [[0], [0]], "alliance": [1, -1]})

        states = loader._compute_alliance_states(df, "epoch_nsplit")

        assert states[0] == 2  # split (both positive and negative)

    def test_compute_alliance_states_neutral(self, loader):
        """Test computing neutral state (no annotations)."""
        df = pd.DataFrame(
            {
                "epoch_nsplit": [[1, 2]],  # Only epochs 1 and 2 have annotations
                "alliance": [1],
            }
        )

        states = loader._compute_alliance_states(df, "epoch_nsplit")

        # Epoch 0 should be neutral (no annotations)
        assert states[0] == 0
        assert states[1] == 1
        assert states[2] == 1


class TestAllianceICDAnalyzer:
    """Tests for AllianceICDAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample merged data with participant-identifiable dyad names."""
        np.random.seed(42)

        # Participants: g01p01..g01p06 (6 participants, various pairings)
        real_dyads = [
            ("g01p01", "g01p02"),
            ("g01p03", "g01p04"),
            ("g01p05", "g01p06"),
        ]
        pseudo_dyads = [
            ("g01p01", "g01p04"),
            ("g01p02", "g01p05"),
            ("g01p03", "g01p06"),
            ("g01p01", "g01p06"),
            ("g01p02", "g01p03"),
            ("g01p04", "g01p05"),
        ]

        data = []
        for state, label in [
            (0, "neutral"),
            (1, "positive"),
            (-1, "negative"),
            (2, "split"),
        ]:
            base_icd = {0: 0.5, 1: 0.3, -1: 0.7, 2: 0.6}[state]

            for dtype, dyad_pairs in [("real", real_dyads), ("pseudo", pseudo_dyads)]:
                n_epochs = 15 if dtype == "real" else 15
                for p1, p2 in dyad_pairs:
                    dyad_name = f"{p1}_ses-01_vs_{p2}_ses-01"
                    family = p1[:3]
                    for epoch in range(n_epochs):
                        data.append(
                            {
                                "epoch_id": epoch,
                                "icd": base_icd + np.random.normal(0, 0.1),
                                "alliance_state": state,
                                "alliance_label": label,
                                "dyad": dyad_name,
                                "dyad_type": dtype,
                                "family": family,
                                "session": "ses-01",
                            }
                        )

        return pd.DataFrame(data)

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked loader."""
        with patch.object(
            AllianceICDAnalyzer, "__init__", lambda self, config_path=None: None
        ):
            analyzer = AllianceICDAnalyzer()
            analyzer.loader = Mock()
            analyzer._data = None
            analyzer.ALLIANCE_LABELS = {
                0: "Neutral",
                1: "Positive",
                -1: "Negative",
                2: "Split",
            }
            return analyzer

    def test_compute_descriptive_stats(self, analyzer, sample_data):
        """Test computing descriptive statistics."""
        stats = analyzer.compute_descriptive_stats(sample_data)

        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert "median" in stats.columns
        assert "n" in stats.columns

        # Check that we have stats for all alliance types
        assert len(stats) == 4

    def test_compute_descriptive_stats_empty(self, analyzer):
        """Test descriptive stats with empty data."""
        stats = analyzer.compute_descriptive_stats(pd.DataFrame())
        assert stats.empty

    def test_test_alliance_effect_naive(self, analyzer, sample_data):
        """Test naive Kruskal-Wallis test."""
        result = analyzer.test_alliance_effect_naive(sample_data)

        assert "test" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "groups" in result
        assert result["test"] == "Kruskal-Wallis H-test"

    def test_test_alliance_effect_deprecation(self, analyzer, sample_data):
        """Test that old method emits deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.test_alliance_effect(sample_data)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_test_alliance_effect_empty(self, analyzer):
        """Test with empty data."""
        result = analyzer.test_alliance_effect_naive(pd.DataFrame())
        assert "error" in result

    def test_pairwise_comparisons_fdr(self, analyzer, sample_data):
        """Test pairwise Mann-Whitney U tests with FDR correction."""
        pairwise = analyzer.pairwise_comparisons(sample_data)

        assert "group1" in pairwise.columns
        assert "group2" in pairwise.columns
        assert "p_value" in pairwise.columns
        assert "p_adjusted" in pairwise.columns
        assert "significant" in pairwise.columns
        assert "correction_method" in pairwise.columns
        assert (pairwise["correction_method"] == "fdr_bh").all()

        # Should have C(4,2) = 6 pairwise comparisons
        assert len(pairwise) == 6

    def test_pairwise_comparisons_bonferroni(self, analyzer, sample_data):
        """Test pairwise comparisons with Bonferroni correction."""
        pairwise = analyzer.pairwise_comparisons(sample_data, correction="bonferroni")

        assert (pairwise["correction_method"] == "bonferroni").all()
        assert len(pairwise) == 6
        # All adjusted p-values should be >= raw p-values
        assert (pairwise["p_adjusted"] >= pairwise["p_value"]).all()

    def test_compute_effect_sizes(self, analyzer, sample_data):
        """Test effect size computation."""
        effects = analyzer.compute_effect_sizes(sample_data)

        assert "epsilon_squared" in effects
        assert "h_statistic" in effects
        assert "n_total" in effects
        assert "k_groups" in effects
        assert 0 <= effects["epsilon_squared"] <= 1

    def test_compute_icc_structure(self, analyzer, sample_data):
        """Test ICC structure computation."""
        icc = analyzer.compute_icc_structure(sample_data)

        assert "icc_family" in icc
        assert "icc_dyad" in icc
        assert "n_families" in icc
        assert "n_dyads" in icc
        assert isinstance(icc["icc_family"], float)
        assert isinstance(icc["icc_dyad"], float)

    def test_compare_real_vs_pseudo_naive_overall(self, analyzer, sample_data):
        """Test naive Mann-Whitney comparison of real vs pseudo dyads."""
        result = analyzer.compare_real_vs_pseudo_naive(sample_data, by_alliance=False)

        assert "overall" in result
        assert "n_real" in result["overall"]
        assert "n_pseudo" in result["overall"]
        assert "mean_real" in result["overall"]
        assert "mean_pseudo" in result["overall"]
        assert "p_value" in result["overall"]

    def test_compare_real_vs_pseudo_naive_by_alliance(self, analyzer, sample_data):
        """Test naive comparison by alliance state."""
        result = analyzer.compare_real_vs_pseudo_naive(sample_data, by_alliance=True)

        assert "Neutral" in result
        assert "Positive" in result
        assert "Negative" in result
        assert "Split" in result

    def test_compare_real_vs_pseudo_deprecation(self, analyzer, sample_data):
        """Test that old method emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer.compare_real_vs_pseudo(sample_data, by_alliance=False)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "compare_real_vs_pseudo" in str(w[0].message)

    def test_compare_real_vs_pseudo_mixed(self, analyzer, sample_data):
        """Test mixed-model comparison of real vs pseudo dyads."""
        result = analyzer.compare_real_vs_pseudo_mixed(sample_data, by_alliance=False)

        assert "error" not in result
        assert result["test"] == "Linear Mixed-Effects Model"
        assert "coefficient" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "icc_participant" in result
        assert result["n_participants"] == 6
        assert result["ci_lower"] < result["coefficient"] < result["ci_upper"]

    def test_compare_real_vs_pseudo_mixed_by_alliance(self, analyzer, sample_data):
        """Test mixed-model comparison per alliance state."""
        result = analyzer.compare_real_vs_pseudo_mixed(sample_data, by_alliance=True)

        for label in ["Neutral", "Positive", "Negative", "Split"]:
            assert label in result
            assert "coefficient" in result[label] or "error" in result[label]

    def test_compare_real_vs_pseudo_aggregated(self, analyzer, sample_data):
        """Test participant-aggregated Wilcoxon comparison."""
        result = analyzer.compare_real_vs_pseudo_aggregated(sample_data)

        assert "error" not in result
        assert result["test"] == "Wilcoxon signed-rank (participant-aggregated)"
        assert result["n_participants"] == 6
        assert "mean_real" in result
        assert "mean_pseudo" in result
        assert "W_statistic" in result
        assert "p_value" in result
        assert "cohens_d_paired" in result

    def test_extract_participants_from_dyad(self):
        """Test dyad name parsing into participant IDs."""
        p1, p2 = AllianceICDAnalyzer._extract_participants_from_dyad(
            "g01p02_ses-01_vs_g01p01_ses-01"
        )
        assert p1 == "g01p02"
        assert p2 == "g01p01"

    def test_compute_stats_by_dyad_type(self, analyzer, sample_data):
        """Test computing stats separately by dyad type."""
        results = analyzer.compute_stats_by_dyad_type(sample_data)

        assert "real" in results
        assert "pseudo" in results

        # Each should have stats for all alliance types
        assert len(results["real"]) == 4
        assert len(results["pseudo"]) == 4


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_alliance_state_consistency(self):
        """Test that alliance states are consistent across modules."""
        loader_states = AllianceICDLoader.ALLIANCE_STATES
        analyzer_labels = AllianceICDAnalyzer.ALLIANCE_LABELS

        # All states in loader should have corresponding labels in analyzer
        for state, name in loader_states.items():
            assert state in analyzer_labels

    def test_data_flow(self):
        """Test data can flow from loader format to analyzer."""
        # Create minimal test data in loader format
        data = pd.DataFrame(
            {
                "epoch_id": [0, 1, 2],
                "icd": [0.5, 0.6, 0.4],
                "alliance_state": [0, 1, -1],
                "alliance_label": ["neutral", "positive", "negative"],
                "dyad": ["d1", "d1", "d1"],
                "dyad_type": ["real", "real", "real"],
                "family": ["g01", "g01", "g01"],
                "session": "ses-01",
            }
        )

        # Verify it can be processed by analyzer methods
        with patch.object(
            AllianceICDAnalyzer, "__init__", lambda self, config_path=None: None
        ):
            analyzer = AllianceICDAnalyzer()
            analyzer.ALLIANCE_LABELS = {
                0: "Neutral",
                1: "Positive",
                -1: "Negative",
                2: "Split",
            }

            stats = analyzer.compute_descriptive_stats(data)

            assert not stats.empty
            assert "n" in stats.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
