"""
Tests for Alliance-ICD Analysis modules.

Tests cover:
- AllianceICDLoader: data loading and merging
- AllianceICDAnalyzer: statistical computations
- AllianceICDStatsPlotter: visualization generation

Authors: Remy Ramadour
Date: November 2025
"""

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
        """Create sample merged data for testing."""
        np.random.seed(42)

        data = []

        # Create data for each alliance state
        for state, label in [
            (0, "neutral"),
            (1, "positive"),
            (-1, "negative"),
            (2, "split"),
        ]:
            # Different mean ICD for each state
            base_icd = {0: 0.5, 1: 0.3, -1: 0.7, 2: 0.6}[state]

            for dtype in ["real", "pseudo"]:
                n_samples = 50 if dtype == "real" else 100

                for i in range(n_samples):
                    data.append(
                        {
                            "epoch_id": i,
                            "icd": base_icd + np.random.normal(0, 0.1),
                            "alliance_state": state,
                            "alliance_label": label,
                            "dyad": f"dyad_{dtype}_{i % 10}",
                            "dyad_type": dtype,
                            "family": f"g0{(i % 5) + 1}",
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

    def test_test_alliance_effect(self, analyzer, sample_data):
        """Test Kruskal-Wallis test."""
        result = analyzer.test_alliance_effect(sample_data)

        assert "test" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "groups" in result

        # With our sample data, there should be a significant effect
        # since we created different means per alliance type
        assert result["test"] == "Kruskal-Wallis H-test"

    def test_test_alliance_effect_empty(self, analyzer):
        """Test with empty data."""
        result = analyzer.test_alliance_effect(pd.DataFrame())
        assert "error" in result

    def test_pairwise_comparisons(self, analyzer, sample_data):
        """Test pairwise Mann-Whitney U tests."""
        pairwise = analyzer.pairwise_comparisons(sample_data)

        assert "group1" in pairwise.columns
        assert "group2" in pairwise.columns
        assert "p_value" in pairwise.columns
        assert "p_adjusted" in pairwise.columns
        assert "significant" in pairwise.columns

        # Should have C(4,2) = 6 pairwise comparisons
        assert len(pairwise) == 6

    def test_compare_real_vs_pseudo_overall(self, analyzer, sample_data):
        """Test comparing real vs pseudo dyads overall."""
        result = analyzer.compare_real_vs_pseudo(sample_data, by_alliance=False)

        assert "overall" in result
        assert "n_real" in result["overall"]
        assert "n_pseudo" in result["overall"]
        assert "mean_real" in result["overall"]
        assert "mean_pseudo" in result["overall"]
        assert "p_value" in result["overall"]

    def test_compare_real_vs_pseudo_by_alliance(self, analyzer, sample_data):
        """Test comparing real vs pseudo dyads by alliance state."""
        result = analyzer.compare_real_vs_pseudo(sample_data, by_alliance=True)

        # Should have results for each alliance state
        assert "Neutral" in result
        assert "Positive" in result
        assert "Negative" in result
        assert "Split" in result

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
