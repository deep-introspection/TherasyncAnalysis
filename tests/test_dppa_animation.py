"""
Unit tests for DPPA animation modules.

Tests epoch-by-epoch Poincaré plot generation, ellipse calculation,
and frame preparation.

Author: Lena Adel, Remy Ramadour
Date: 2025-11-12
"""

import numpy as np
import pandas as pd
import pytest

from src.physio.dppa.epoch_animator import EpochAnimator
from src.physio.dppa.poincare_plotter import PoincarePlotter


class TestEpochAnimator:
    """Test EpochAnimator class for loading and processing RR intervals."""

    @pytest.fixture
    def animator(self):
        """Create EpochAnimator instance."""
        return EpochAnimator()

    @pytest.fixture
    def sample_rr_df(self):
        """Create sample RR intervals DataFrame."""
        return pd.DataFrame(
            {
                "time_peak_start": [1.0, 2.0, 3.0, 4.0, 5.0],
                "time_peak_end": [2.0, 3.0, 4.0, 5.0, 6.0],
                "rr_interval_ms": [750.0, 760.0, 755.0, 765.0, 750.0],
                "is_valid": [1, 1, 1, 1, 1],
                "epoch_sliding_duration30s_step5s": [
                    "[0]",
                    "[0, 1]",
                    "[1]",
                    "[1, 2]",
                    "[2]",
                ],
            }
        )

    def test_init(self, animator):
        """Test EpochAnimator initialization."""
        assert animator.config is not None
        assert animator.data_root is not None

    def test_load_rr_intervals_real_data(self, animator):
        """Test loading RR intervals from real file."""
        rr_df = animator.load_rr_intervals(
            "g01p01", "ses-01", "therapy", "sliding_duration30s_step5s"
        )

        assert isinstance(rr_df, pd.DataFrame)
        assert len(rr_df) > 0
        assert "rr_interval_ms" in rr_df.columns
        assert "epoch_sliding_duration30s_step5s" in rr_df.columns

    def test_load_rr_intervals_session_prefix(self, animator):
        """Test session prefix auto-addition."""
        # Should work with or without 'ses-' prefix
        rr_df1 = animator.load_rr_intervals(
            "g01p01", "ses-01", "therapy", "sliding_duration30s_step5s"
        )
        rr_df2 = animator.load_rr_intervals(
            "g01p01", "01", "therapy", "sliding_duration30s_step5s"
        )

        assert len(rr_df1) == len(rr_df2)

    def test_load_rr_intervals_file_not_found(self, animator):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            animator.load_rr_intervals(
                "invalid_sub", "ses-99", "therapy", "sliding_duration30s_step5s"
            )

    def test_get_rr_for_epoch(self, animator, sample_rr_df):
        """Test extracting RR intervals for specific epoch."""
        # Epoch 0: intervals at indices 0, 1 (JSON: [0] and [0, 1])
        rr_epoch_0 = animator.get_rr_for_epoch(
            sample_rr_df, 0, "sliding_duration30s_step5s"
        )
        assert len(rr_epoch_0) == 2
        np.testing.assert_array_equal(rr_epoch_0, [750.0, 760.0])

        # Epoch 1: intervals at indices 1, 2, 3
        rr_epoch_1 = animator.get_rr_for_epoch(
            sample_rr_df, 1, "sliding_duration30s_step5s"
        )
        assert len(rr_epoch_1) == 3
        np.testing.assert_array_equal(rr_epoch_1, [760.0, 755.0, 765.0])

        # Epoch 2: intervals at indices 3, 4
        rr_epoch_2 = animator.get_rr_for_epoch(
            sample_rr_df, 2, "sliding_duration30s_step5s"
        )
        assert len(rr_epoch_2) == 2
        np.testing.assert_array_equal(rr_epoch_2, [765.0, 750.0])

    def test_get_rr_for_epoch_invalid_method(self, animator, sample_rr_df):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Column.*not found"):
            animator.get_rr_for_epoch(sample_rr_df, 0, "invalid_method")

    def test_compute_poincare_points(self, animator):
        """Test Poincaré point computation."""
        rr_intervals = np.array([750.0, 760.0, 755.0, 765.0])
        rr_n, rr_n_plus_1 = animator.compute_poincare_points(rr_intervals)

        # Should return N-1 pairs
        assert len(rr_n) == 3
        assert len(rr_n_plus_1) == 3

        # Check pairing
        np.testing.assert_array_equal(rr_n, [750.0, 760.0, 755.0])
        np.testing.assert_array_equal(rr_n_plus_1, [760.0, 755.0, 765.0])

    def test_compute_poincare_points_insufficient_data(self, animator):
        """Test Poincaré computation with <2 intervals."""
        rr_intervals = np.array([750.0])
        rr_n, rr_n_plus_1 = animator.compute_poincare_points(rr_intervals)

        assert len(rr_n) == 0
        assert len(rr_n_plus_1) == 0

    def test_compute_poincare_for_epoch(self, animator, sample_rr_df):
        """Test full Poincaré computation for epoch."""
        result = animator.compute_poincare_for_epoch(
            sample_rr_df, 1, "sliding_duration30s_step5s"
        )

        assert isinstance(result, dict)
        assert "rr_n" in result
        assert "rr_n_plus_1" in result
        assert "n_points" in result

        # Epoch 1 has 3 intervals → 2 Poincaré points
        assert result["n_points"] == 2
        assert len(result["rr_n"]) == 2
        assert len(result["rr_n_plus_1"]) == 2

    def test_compute_poincare_for_epoch_real_data(self, animator):
        """Test Poincaré computation with real data."""
        rr_df = animator.load_rr_intervals(
            "g01p01", "ses-01", "therapy", "sliding_duration30s_step5s"
        )

        result = animator.compute_poincare_for_epoch(
            rr_df, 0, "sliding_duration30s_step5s"
        )

        assert result["n_points"] > 0
        assert len(result["rr_n"]) == result["n_points"]
        assert len(result["rr_n_plus_1"]) == result["n_points"]

        # Verify all values are positive
        assert np.all(result["rr_n"] > 0)
        assert np.all(result["rr_n_plus_1"] > 0)


class TestPoincarePlotter:
    """Test PoincarePlotter class for visualizations."""

    @pytest.fixture
    def plotter(self):
        """Create PoincarePlotter instance."""
        return PoincarePlotter()

    @pytest.fixture
    def sample_poincare_data(self):
        """Create sample Poincaré points."""
        return {
            "rr_n": np.array([750.0, 760.0, 755.0, 765.0]),
            "rr_n_plus_1": np.array([760.0, 755.0, 765.0, 750.0]),
        }

    def test_init(self, plotter):
        """Test PoincarePlotter initialization."""
        assert plotter.config is not None
        assert plotter.colors is not None
        assert plotter.styles is not None

    def test_calculate_ellipse_parameters(self, plotter, sample_poincare_data):
        """Test ellipse parameter calculation."""
        params = plotter.calculate_ellipse_parameters(
            sample_poincare_data["rr_n"], sample_poincare_data["rr_n_plus_1"]
        )

        assert isinstance(params, dict)
        assert "centroid_x" in params
        assert "centroid_y" in params
        assert "sd1" in params
        assert "sd2" in params
        assert "angle" in params

        # Verify centroid is mean of points
        assert params["centroid_x"] == pytest.approx(
            np.mean(sample_poincare_data["rr_n"])
        )
        assert params["centroid_y"] == pytest.approx(
            np.mean(sample_poincare_data["rr_n_plus_1"])
        )

        # Verify angle is 45°
        assert params["angle"] == 45.0

        # Verify SD1 and SD2 are positive
        assert params["sd1"] > 0
        assert params["sd2"] > 0

    def test_calculate_ellipse_parameters_empty_data(self, plotter):
        """Test ellipse calculation with empty data."""
        params = plotter.calculate_ellipse_parameters(np.array([]), np.array([]))

        assert np.isnan(params["centroid_x"])
        assert np.isnan(params["centroid_y"])
        assert np.isnan(params["sd1"])
        assert np.isnan(params["sd2"])

    def test_calculate_ellipse_parameters_single_point(self, plotter):
        """Test ellipse calculation with single point."""
        params = plotter.calculate_ellipse_parameters(
            np.array([750.0]), np.array([760.0])
        )

        # Centroid should be the single point
        assert params["centroid_x"] == 750.0
        assert params["centroid_y"] == 760.0

        # SD1 and SD2 should be 0 (no variability)
        assert params["sd1"] == 0.0
        assert params["sd2"] == 0.0

    def test_draw_ellipse(self, plotter):
        """Test ellipse drawing."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ellipse = plotter.draw_ellipse(
            ax, 750.0, 760.0, 50.0, 100.0, 45.0, "blue", "Test"
        )

        assert ellipse is not None
        assert ellipse.center == (750.0, 760.0)
        assert ellipse.width == 100.0  # 2 * SD1
        assert ellipse.height == 200.0  # 2 * SD2
        assert ellipse.angle == 45.0

        plt.close(fig)

    def test_draw_icd_line(self, plotter):
        """Test ICD line drawing."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        line = plotter.draw_icd_line(ax, 750.0, 760.0, 800.0, 810.0)

        assert line is not None
        assert len(line.get_xdata()) == 2
        assert len(line.get_ydata()) == 2

        plt.close(fig)

    def test_annotate_metrics(self, plotter):
        """Test metric annotation."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plotter.annotate_metrics(ax, 123.4, 45.6, 78.9, 52.1, 81.3, "S1", "S2")

        # Verify text box was added
        texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
        assert len(texts) > 0

        plt.close(fig)
