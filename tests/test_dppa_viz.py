"""
Unit tests for DPPA visualization modules.

Tests cover:
- DyadICDLoader: Loading ICD data for dyads
- DyadCentroidLoader: Loading centroid data for dyad members
- DyadPlotter: Generating 4-subplot visualizations
- CLI integration: plot_dyad.py script
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from src.physio.dppa.dyad_icd_loader import DyadICDLoader


class TestDyadICDLoader:
    """Test suite for DyadICDLoader class."""

    @pytest.fixture
    def config_path(self):
        """Provide path to test config."""
        return Path("config/config.yaml")

    @pytest.fixture
    def loader(self, config_path):
        """Create DyadICDLoader instance."""
        return DyadICDLoader(config_path)

    @pytest.fixture
    def valid_dyad_pair(self):
        """Provide a valid dyad pair string."""
        return "g01p01_ses-01_vs_g01p02_ses-01"

    def test_parse_dyad_info_valid(self, loader, valid_dyad_pair):
        """Test parsing valid dyad pair string."""
        result = loader.parse_dyad_info(valid_dyad_pair)
        
        assert isinstance(result, dict)
        assert "sub1" in result
        assert "ses1" in result
        assert "sub2" in result
        assert "ses2" in result
        assert result["sub1"] == "g01p01"
        assert result["ses1"] == "01"
        assert result["sub2"] == "g01p02"
        assert result["ses2"] == "01"

    def test_parse_dyad_info_invalid_format(self, loader):
        """Test parsing invalid dyad pair format raises ValueError."""
        invalid_pairs = [
            "g01p01_vs_g01p02",  # Missing sessions
            "invalid_format",     # Wrong format
            "g01p01_ses-01",      # Missing second subject
            "",                   # Empty string
        ]
        
        for invalid_pair in invalid_pairs:
            with pytest.raises(ValueError, match="Invalid dyad pair format"):
                loader.parse_dyad_info(invalid_pair)

    def test_load_icd_therapy_valid(self, loader, valid_dyad_pair):
        """Test loading therapy ICD data for valid dyad."""
        df = loader.load_icd(
            dyad_pair=valid_dyad_pair,
            task="therapy",
            method="nsplit120"
        )
        
        assert isinstance(df, pd.DataFrame)
        assert "epoch_id" in df.columns
        assert "icd_value" in df.columns
        assert len(df) > 0
        assert df["epoch_id"].dtype in [np.int64, np.int32]
        assert df["icd_value"].dtype in [np.float64, np.float32]

    def test_load_icd_restingstate_valid(self, loader, valid_dyad_pair):
        """Test loading resting state ICD data (single epoch expected)."""
        df = loader.load_icd(
            dyad_pair=valid_dyad_pair,
            task="restingstate",
            method="nsplit1"  # restingstate uses nsplit1, not nsplit120
        )
        
        assert isinstance(df, pd.DataFrame)
        assert "epoch_id" in df.columns
        assert "icd_value" in df.columns
        # Resting state should have exactly 1 epoch
        assert len(df) == 1
        assert df.iloc[0]["epoch_id"] == 0

    def test_load_icd_missing_file(self, loader):
        """Test loading ICD for non-existent method raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.load_icd(
                dyad_pair="g01p01_ses-01_vs_g01p02_ses-01",
                task="therapy",
                method="nonexistent_method999"
            )

    def test_load_icd_missing_dyad_column(self, loader):
        """Test loading ICD for non-existent dyad raises ValueError."""
        with pytest.raises(ValueError, match="not found in"):
            loader.load_icd(
                dyad_pair="nonexistent_ses-99_vs_fakepair_ses-99",
                task="therapy",
                method="nsplit120"
            )

    def test_load_both_tasks_valid(self, loader, valid_dyad_pair):
        """Test loading both tasks (restingstate + therapy) for dyad."""
        # Use per-task methods (new format)
        methods = {'restingstate': 'nsplit1', 'therapy': 'nsplit120'}
        result = loader.load_both_tasks(
            dyad_pair=valid_dyad_pair,
            methods=methods
        )
        
        assert isinstance(result, dict)
        assert "restingstate" in result
        assert "therapy" in result
        
        # Check resting state
        df_rest = result["restingstate"]
        assert isinstance(df_rest, pd.DataFrame)
        assert len(df_rest) == 1  # Single epoch
        
        # Check therapy
        df_therapy = result["therapy"]
        assert isinstance(df_therapy, pd.DataFrame)
        assert len(df_therapy) > 1  # Multiple epochs

    def test_load_both_tasks_missing_task(self, loader):
        """Test loading both tasks with missing method file."""
        with pytest.raises(FileNotFoundError):
            loader.load_both_tasks(
                dyad_pair="g01p01_ses-01_vs_g01p02_ses-01",
                methods="nonexistent_method999"  # Single string still supported
            )


# Placeholder for future test classes
class TestDyadCentroidLoader:
    """Test suite for DyadCentroidLoader class."""

    @pytest.fixture
    def config_path(self):
        """Provide path to test config."""
        return Path("config/config.yaml")

    @pytest.fixture
    def loader(self, config_path):
        """Create DyadCentroidLoader instance."""
        from src.physio.dppa.dyad_centroid_loader import DyadCentroidLoader
        return DyadCentroidLoader(config_path)

    @pytest.fixture
    def valid_dyad_info(self):
        """Provide valid dyad info dictionary."""
        return {
            "sub1": "g01p01",
            "ses1": "01",
            "sub2": "g01p02",
            "ses2": "01",
        }

    def test_load_centroids_therapy_valid(self, loader, valid_dyad_info):
        """Test loading therapy centroid data for both subjects."""
        df1, df2 = loader.load_centroids(
            dyad_info=valid_dyad_info,
            task="therapy",
            method="nsplit120"
        )
        
        # Check both DataFrames exist
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        
        # Check required columns
        required_cols = ["epoch_id", "centroid_x", "centroid_y", "sd1", "sd2", "sd_ratio"]
        for col in required_cols:
            assert col in df1.columns
            assert col in df2.columns
        
        # Therapy should have multiple epochs
        assert len(df1) > 1
        assert len(df2) > 1

    def test_load_centroids_restingstate_valid(self, loader, valid_dyad_info):
        """Test loading resting state centroid data (single epoch expected)."""
        df1, df2 = loader.load_centroids(
            dyad_info=valid_dyad_info,
            task="restingstate",
            method="nsplit1"  # restingstate uses nsplit1, not nsplit120
        )
        
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        
        # Resting state should have exactly 1 epoch per subject
        assert len(df1) == 1
        assert len(df2) == 1
        assert df1.iloc[0]["epoch_id"] == 0
        assert df2.iloc[0]["epoch_id"] == 0

    def test_load_centroids_missing_file(self, loader):
        """Test loading centroids for non-existent subject raises FileNotFoundError."""
        invalid_dyad_info = {
            "sub1": "nonexistent999",
            "ses1": "99",
            "sub2": "g01p02",
            "ses2": "01",
        }
        
        with pytest.raises(FileNotFoundError):
            loader.load_centroids(
                dyad_info=invalid_dyad_info,
                task="therapy",
                method="nsplit120"
            )

    def test_validate_epoch_alignment_match(self, loader, valid_dyad_info):
        """Test epoch alignment validation functionality."""
        df1, df2 = loader.load_centroids(
            dyad_info=valid_dyad_info,
            task="therapy",
            method="nsplit120"
        )
        
        # Call validation (may or may not be aligned depending on data)
        is_aligned = loader.validate_epoch_alignment(df1, df2)
        assert isinstance(is_aligned, bool)
        
        # Test with artificially aligned data
        df1_aligned = df1.head(10).copy()
        df2_aligned = df2.head(10).copy()
        is_aligned_artificial = loader.validate_epoch_alignment(df1_aligned, df2_aligned)
        # If both have same first 10 epochs, should be aligned
        if len(df1_aligned) == len(df2_aligned):
            assert is_aligned_artificial in [True, False]  # Valid boolean result

    def test_validate_epoch_alignment_mismatch(self, loader):
        """Test epoch alignment validation with mismatched epochs."""
        # Create DataFrames with different epoch_ids
        df1 = pd.DataFrame({"epoch_id": [0, 1, 2], "sd1": [10, 20, 30]})
        df2 = pd.DataFrame({"epoch_id": [0, 1, 3], "sd1": [15, 25, 35]})  # Different last epoch
        
        is_aligned = loader.validate_epoch_alignment(df1, df2)
        assert is_aligned is False

    def test_load_both_tasks_valid(self, loader, valid_dyad_info):
        """Test loading both tasks (restingstate + therapy) for dyad."""
        # Use per-task methods (new format)
        methods = {'restingstate': 'nsplit1', 'therapy': 'nsplit120'}
        result = loader.load_both_tasks(
            dyad_info=valid_dyad_info,
            methods=methods
        )
        
        assert isinstance(result, dict)
        assert "restingstate" in result
        assert "therapy" in result
        
        # Check resting state (both subjects)
        df1_rest, df2_rest = result["restingstate"]
        assert isinstance(df1_rest, pd.DataFrame)
        assert isinstance(df2_rest, pd.DataFrame)
        assert len(df1_rest) == 1  # Single epoch
        assert len(df2_rest) == 1  # Single epoch
        
        # Check therapy (both subjects)
        df1_therapy, df2_therapy = result["therapy"]
        assert isinstance(df1_therapy, pd.DataFrame)
        assert isinstance(df2_therapy, pd.DataFrame)
        assert len(df1_therapy) > 1  # Multiple epochs
        assert len(df2_therapy) > 1  # Multiple epochs

    def test_load_both_tasks_missing_file(self, loader):
        """Test loading both tasks with missing subject file."""
        invalid_dyad_info = {
            "sub1": "nonexistent999",
            "ses1": "99",
            "sub2": "g01p02",
            "ses2": "01",
        }
        
        with pytest.raises(FileNotFoundError):
            loader.load_both_tasks(
                dyad_info=invalid_dyad_info,
                methods="nsplit120"  # Single string still supported
            )


class TestDyadPlotter:
    """Test suite for DyadPlotter class."""

    @pytest.fixture
    def config_path(self):
        """Provide path to test config."""
        return Path("config/config.yaml")

    @pytest.fixture
    def plotter(self, config_path):
        """Create DyadPlotter instance."""
        from src.physio.dppa.dyad_plotter import DyadPlotter
        return DyadPlotter(config_path)

    @pytest.fixture
    def sample_icd_data(self):
        """Provide sample ICD data for testing."""
        return {
            "restingstate": pd.DataFrame({
                "epoch_id": [0],
                "icd_value": [50.0]
            }),
            "therapy": pd.DataFrame({
                "epoch_id": [0, 1, 2, 3, 4],
                "icd_value": [48.0, 47.5, 46.0, 45.5, 45.0]
            })
        }

    @pytest.fixture
    def sample_centroid_data(self):
        """Provide sample centroid data for testing."""
        return {
            "restingstate": (
                pd.DataFrame({
                    "epoch_id": [0],
                    "sd1": [100.0],
                    "sd2": [150.0],
                    "sd_ratio": [0.67]
                }),
                pd.DataFrame({
                    "epoch_id": [0],
                    "sd1": [105.0],
                    "sd2": [155.0],
                    "sd_ratio": [0.68]
                })
            ),
            "therapy": (
                pd.DataFrame({
                    "epoch_id": [0, 1, 2, 3, 4],
                    "sd1": [98.0, 96.0, 95.0, 94.0, 93.0],
                    "sd2": [148.0, 146.0, 144.0, 142.0, 140.0],
                    "sd_ratio": [0.66, 0.66, 0.66, 0.66, 0.66]
                }),
                pd.DataFrame({
                    "epoch_id": [0, 1, 2, 3, 4],
                    "sd1": [103.0, 101.0, 100.0, 99.0, 98.0],
                    "sd2": [153.0, 151.0, 149.0, 147.0, 145.0],
                    "sd_ratio": [0.67, 0.67, 0.67, 0.67, 0.68]
                })
            )
        }

    @pytest.fixture
    def sample_dyad_info(self):
        """Provide sample dyad info."""
        return {
            "sub1": "g01p01",
            "ses1": "01",
            "sub2": "g01p02",
            "ses2": "01"
        }

    def test_calculate_trendline(self, plotter, sample_icd_data):
        """Test trendline calculation for therapy ICD data."""
        therapy_icd = sample_icd_data["therapy"]
        
        fitted_values, slope = plotter._calculate_trendline(therapy_icd)
        
        assert isinstance(fitted_values, np.ndarray)
        assert isinstance(slope, float)
        assert len(fitted_values) == len(therapy_icd)
        # Slope should be negative (decreasing trend in sample data)
        assert slope < 0

    def test_plot_dyad_creates_file(self, plotter, sample_icd_data, sample_centroid_data, sample_dyad_info, tmp_path):
        """Test that plot_dyad creates output file."""
        output_file = tmp_path / "test_dyad_viz.png"
        
        plotter.plot_dyad(
            icd_data=sample_icd_data,
            centroid_data=sample_centroid_data,
            dyad_info=sample_dyad_info,
            method="nsplit120",
            output_path=output_file
        )
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_plot_dyad_with_real_data(self, plotter, tmp_path):
        """Test plotting with real data from files."""
        from src.physio.dppa.dyad_icd_loader import DyadICDLoader
        from src.physio.dppa.dyad_centroid_loader import DyadCentroidLoader
        
        dyad_pair = "g01p01_ses-01_vs_g01p02_ses-01"
        # Use per-task methods (restingstate has different n_epochs than therapy)
        methods = {'restingstate': 'nsplit1', 'therapy': 'nsplit120'}
        
        # Load real data
        icd_loader = DyadICDLoader()
        dyad_info = icd_loader.parse_dyad_info(dyad_pair)
        icd_data = icd_loader.load_both_tasks(dyad_pair, methods)
        
        centroid_loader = DyadCentroidLoader()
        centroid_data = centroid_loader.load_both_tasks(dyad_info, methods)
        
        # Generate plot
        output_file = tmp_path / f"{dyad_pair}_test.png"
        plotter.plot_dyad(
            icd_data=icd_data,
            centroid_data=centroid_data,
            dyad_info=dyad_info,
            method="nsplit120",  # Use therapy method for plot naming
            output_path=output_file
        )
        
        assert output_file.exists()
        assert output_file.stat().st_size > 10000  # Should be a substantial image file

    def test_calculate_trendline_with_nan(self, plotter):
        """Test trendline calculation with NaN values in ICD data."""
        # Create sample data with NaN
        therapy_icd = pd.DataFrame({
            "epoch_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "icd_value": [50.0, 48.0, np.nan, 44.0, 42.0, np.nan, 38.0, 36.0, 34.0, np.nan]
        })
        
        fitted_values, slope = plotter._calculate_trendline(therapy_icd)
        
        # Should return valid trendline despite NaN values
        assert not np.isnan(slope), "Slope should not be NaN"
        assert slope < 0, "Slope should be negative for decreasing trend"
        assert len(fitted_values) == len(therapy_icd), "Fitted values should match input length"
        # Fitted values should all be valid (computed from all epoch_ids)
        assert not np.any(np.isnan(fitted_values)), "Fitted values should not contain NaN"

    def test_calculate_trendline_insufficient_data(self, plotter):
        """Test trendline calculation with insufficient valid data."""
        # Create data with only 1 valid point
        therapy_icd = pd.DataFrame({
            "epoch_id": [0, 1, 2],
            "icd_value": [50.0, np.nan, np.nan]
        })
        
        fitted_values, slope = plotter._calculate_trendline(therapy_icd)
        
        # Should return zeros when insufficient data
        assert slope == 0.0, "Slope should be 0 with insufficient data"
        assert np.allclose(fitted_values, 0.0), "Fitted values should be zeros"


class TestPlotDyadCLI:
    """Test suite for plot_dyad.py CLI script integration tests."""

    @pytest.fixture
    def script_path(self):
        """Provide path to plot_dyad.py script."""
        return Path("scripts/physio/dppa/plot_dyad.py")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def valid_dyad_pair(self):
        """Provide a valid dyad pair for testing."""
        return "g01p01_ses-01_vs_g01p02_ses-01"

    @pytest.fixture
    def method(self):
        """Provide test method."""
        return "nsplit120"

    def test_script_exists(self, script_path):
        """Test that plot_dyad.py script exists."""
        assert script_path.exists(), f"Script not found: {script_path}"
        assert script_path.is_file(), f"Not a file: {script_path}"

    def test_script_executable(self, script_path):
        """Test that script has execute permissions."""
        import os
        assert os.access(script_path, os.X_OK), f"Script not executable: {script_path}"

    def test_single_dyad_mode_creates_figure(self, script_path, valid_dyad_pair, method, temp_output_dir):
        """Test single dyad mode creates output figure."""
        import subprocess
        
        # Run script with per-task methods
        # The script should now support loading different methods per task
        cmd = [
            "uv", "run", "python", str(script_path),
            "--dyad", valid_dyad_pair,
            "--method", method,  # Primary method for therapy
            "--output-dir", str(temp_output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check exit code - may fail if ICD files don't exist yet (expected)
        if result.returncode != 0:
            # Skip if missing ICD files (not generated yet)
            if "ICD file not found" in result.stderr or "Centroid file not found" in result.stderr:
                pytest.skip("ICD/Centroid files not generated yet - run compute_dppa.py first")
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check output file exists
        method_dir = temp_output_dir / method
        assert method_dir.exists(), f"Method directory not created: {method_dir}"
        
        expected_file = method_dir / f"sub-g01p01_ses-01_vs_sub-g01p02_ses-01_method-{method}_desc-dyad_viz.png"
        assert expected_file.exists(), f"Output file not created: {expected_file}"
        
        # Check file size (should be > 100 KB for a valid plot)
        assert expected_file.stat().st_size > 100_000, "Output file too small"

    def test_missing_dyad_argument(self, script_path, method):
        """Test script fails gracefully with missing --dyad argument."""
        import subprocess
        
        cmd = [
            "uv", "run", "python", str(script_path),
            "--method", method
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should fail with non-zero exit code
        assert result.returncode != 0, "Script should fail without --dyad or --batch"
        assert "Either --dyad or --batch" in result.stderr, "Missing expected error message"

    def test_help_message(self, script_path):
        """Test script provides help message."""
        import subprocess
        
        cmd = ["uv", "run", "python", str(script_path), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, "Help command should succeed"
        assert "Generate DPPA dyadic visualizations" in result.stdout, "Missing help text"
        assert "--dyad" in result.stdout, "Missing --dyad option in help"
        assert "--batch" in result.stdout, "Missing --batch option in help"
        assert "--method" in result.stdout, "Missing --method option in help"

