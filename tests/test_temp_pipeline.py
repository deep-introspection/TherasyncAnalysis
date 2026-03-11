"""
Unit tests for Temperature (TEMP) processing pipeline components.

This module provides comprehensive testing for all temperature processing components
including data loading, cleaning (outlier removal, artifact detection),
metrics extraction, and BIDS output.

Authors: Lena Adel, Remy Ramadour
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from src.physio.preprocessing.temp_loader import TEMPLoader
from src.physio.preprocessing.temp_cleaner import TEMPCleaner
from src.physio.preprocessing.temp_metrics import TEMPMetricsExtractor
from src.physio.preprocessing.temp_bids_writer import TEMPBIDSWriter


class TestTEMPLoader(unittest.TestCase):
    """Test Temperature data loading functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test data structure
        self.test_subject = "sub-g01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"

        # Create test directories
        physio_dir = (
            self.temp_path
            / "sourcedata"
            / self.test_subject
            / self.test_session
            / "physio"
        )
        physio_dir.mkdir(parents=True)

        # Create test temperature files
        self._create_test_temp_files(physio_dir)

        # Create test config
        self.test_config = {
            "paths": {"rawdata": str(self.temp_path / "sourcedata")},
            "physio": {"temp": {"sampling_rate": 4}},
            "moments": [{"name": self.test_moment}],
        }

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def _create_test_temp_files(self, physio_dir: Path):
        """Create test temperature data files."""
        # Create test TSV data (60 seconds at 4 Hz = 240 samples)
        duration = 60  # seconds
        sampling_rate = 4  # Hz
        time_values = np.arange(0, duration, 1 / sampling_rate)

        # Simulate realistic peripheral skin temperature:
        # - Start around 32°C (typical peripheral temp)
        # - Slow drift (temperature changes slowly)
        # - Small noise
        base_temp = 32.0
        drift = 0.5 * np.sin(2 * np.pi * 0.001 * time_values)  # Very slow drift
        noise = np.random.normal(0, 0.05, len(time_values))
        temp_values = base_temp + drift + noise

        test_data = pd.DataFrame({"time": time_values, "temp": temp_values})

        # Save TSV file
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-temp"
        tsv_file = physio_dir / f"{base_filename}.tsv"
        test_data.to_csv(tsv_file, sep="\t", index=False)

        # Save JSON metadata
        json_file = physio_dir / f"{base_filename}.json"
        metadata = {
            "SamplingFrequency": 4.0,
            "StartTime": 0,
            "Columns": ["time", "temp"],
            "Units": ["s", "°C"],
            "TaskName": self.test_moment,
            "RecordingType": "Temperature",
            "FamilyID": "g01",
            "DeviceManufacturer": "Empatica",
            "DeviceModel": "E4",
        }

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)

    @patch("src.physio.preprocessing.temp_loader.ConfigLoader")
    def test_loader_basic_functionality(self, mock_config):
        """Test basic temperature loader functionality."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.sampling_rate": 4,
            "paths.rawdata": str(self.temp_path / "sourcedata"),
        }.get(key, default)

        loader = TEMPLoader()
        self.assertIsInstance(loader, TEMPLoader)
        self.assertEqual(loader.sampling_rate, 4)

    def test_load_subject_session_success(self):
        """Test successful loading of subject/session data."""
        with patch("src.physio.preprocessing.temp_loader.ConfigLoader") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "physio.temp.sampling_rate": 4,
            }.get(key, default)

            loader = TEMPLoader()
            data, metadata = loader.load_subject_session(
                self.test_subject, self.test_session, self.test_moment
            )

            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn("time", data.columns)
            self.assertIn("temp", data.columns)
            self.assertEqual(len(data), 240)  # 60 seconds * 4 Hz

            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata["SamplingFrequency"], 4.0)
            self.assertEqual(metadata["TaskName"], self.test_moment)

            # Verify temperature values are reasonable (25-40°C)
            self.assertTrue((data["temp"] >= 20).all())
            self.assertTrue((data["temp"] <= 45).all())

    def test_load_subject_session_missing_file(self):
        """Test loading with missing files."""
        with patch("src.physio.preprocessing.temp_loader.ConfigLoader") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "physio.temp.sampling_rate": 4,
            }.get(key, default)

            loader = TEMPLoader()
            with self.assertRaises(FileNotFoundError):
                loader.load_subject_session(
                    self.test_subject, self.test_session, "nonexistent"
                )

    def test_data_validation(self):
        """Test temperature data validation functionality."""
        with patch("src.physio.preprocessing.temp_loader.ConfigLoader") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "physio.temp.sampling_rate": 4,
            }.get(key, default)

            loader = TEMPLoader()

            # Test with valid data
            valid_data = pd.DataFrame(
                {"time": [0, 0.25, 0.5, 0.75], "temp": [32.0, 32.1, 32.05, 32.2]}
            )
            valid_metadata = {"SamplingFrequency": 4, "Columns": ["time", "temp"]}

            # Should not raise exception
            loader._validate_temp_data(valid_data, valid_metadata, Path("test.tsv"))

            # Test with invalid data (missing columns)
            invalid_data = pd.DataFrame({"time": [0, 0.25, 0.5]})

            with self.assertRaises((ValueError, KeyError)):
                loader._validate_temp_data(
                    invalid_data, valid_metadata, Path("test.tsv")
                )


class TestTEMPCleaner(unittest.TestCase):
    """Test Temperature signal cleaning functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            "physio": {
                "temp": {
                    "sampling_rate": 4,
                    "processing": {
                        "outlier_threshold": [25, 40],
                        "jump_threshold": 2.0,
                        "interpolation_max_gap": 10,
                        "quality_min_samples": 30,
                    },
                }
            }
        }

        # Create test signal (60 seconds at 4 Hz)
        self.test_signal = self._create_test_signal()

    def _create_test_signal(self):
        """Create synthetic temperature signal for testing."""
        duration = 60
        sampling_rate = 4
        t = np.arange(0, duration, 1 / sampling_rate)

        # Realistic temperature with slow drift
        temp = (
            32.0
            + 0.3 * np.sin(2 * np.pi * 0.002 * t)
            + np.random.normal(0, 0.05, len(t))
        )

        return temp

    @patch("src.physio.preprocessing.temp_cleaner.ConfigLoader")
    def test_cleaner_initialization(self, mock_config):
        """Test temperature cleaner initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp": self.test_config["physio"]["temp"],
            "physio.temp.processing": self.test_config["physio"]["temp"]["processing"],
            "physio.temp.processing.outlier_threshold": [25, 40],
            "physio.temp.processing.jump_threshold": 2.0,
            "physio.temp.processing.interpolation_max_gap": 10,
            "physio.temp.processing.quality_min_samples": 30,
            "physio.temp.sampling_rate": 4,
        }.get(key, default)

        cleaner = TEMPCleaner()
        self.assertEqual(cleaner.outlier_threshold, [25, 40])
        self.assertEqual(cleaner.jump_threshold, 2.0)
        self.assertEqual(cleaner.interpolation_max_gap, 10)

    @patch("src.physio.preprocessing.temp_cleaner.ConfigLoader")
    def test_clean_signal_success(self, mock_config):
        """Test successful temperature signal cleaning."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp": self.test_config["physio"]["temp"],
            "physio.temp.processing": self.test_config["physio"]["temp"]["processing"],
            "physio.temp.sampling_rate": 4,
        }.get(key, default)

        # Create input DataFrame
        test_df = pd.DataFrame(
            {"time": np.arange(0, 60, 0.25), "temp": self.test_signal}
        )

        cleaner = TEMPCleaner()
        processed_signals, metadata = cleaner.clean_signal(test_df, moment="test")

        self.assertIsInstance(processed_signals, pd.DataFrame)
        self.assertIn("TEMP_Clean", processed_signals.columns)
        self.assertIn("TEMP_Raw", processed_signals.columns)
        self.assertIn("TEMP_Quality", processed_signals.columns)
        self.assertIn("TEMP_Outliers", processed_signals.columns)
        self.assertIn("TEMP_Interpolated", processed_signals.columns)

        # Check metadata
        self.assertIn("quality_score", metadata)
        self.assertIn("valid_samples", metadata)
        self.assertIn("outlier_samples", metadata)

    @patch("src.physio.preprocessing.temp_cleaner.ConfigLoader")
    def test_outlier_detection(self, mock_config):
        """Test outlier detection in temperature data."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp": self.test_config["physio"]["temp"],
            "physio.temp.processing": self.test_config["physio"]["temp"]["processing"],
            "physio.temp.sampling_rate": 4,
        }.get(key, default)

        # Create signal with outliers
        test_signal = self.test_signal.copy()
        test_signal[10] = 20.0  # Below threshold (25°C)
        test_signal[50] = 45.0  # Above threshold (40°C)

        test_df = pd.DataFrame({"time": np.arange(0, 60, 0.25), "temp": test_signal})

        cleaner = TEMPCleaner()
        processed_signals, metadata = cleaner.clean_signal(test_df, moment="test")

        # Check that outliers were detected
        self.assertGreater(metadata["outlier_samples"], 0)

        # Check that outliers are marked
        self.assertTrue(processed_signals["TEMP_Outliers"].any())

    @patch("src.physio.preprocessing.temp_cleaner.ConfigLoader")
    def test_artifact_detection(self, mock_config):
        """Test artifact detection (sudden jumps) in temperature data."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp": self.test_config["physio"]["temp"],
            "physio.temp.processing": self.test_config["physio"]["temp"]["processing"],
            "physio.temp.sampling_rate": 4,
        }.get(key, default)

        # Create signal with sudden jump (artifact)
        test_signal = self.test_signal.copy()
        test_signal[100] = test_signal[99] + 5.0  # Sudden 5°C jump (> 2°C threshold)

        test_df = pd.DataFrame({"time": np.arange(0, 60, 0.25), "temp": test_signal})

        cleaner = TEMPCleaner()
        processed_signals, metadata = cleaner.clean_signal(test_df, moment="test")

        # Artifact should be detected as outlier
        self.assertGreater(metadata["outlier_samples"], 0)

    @patch("src.physio.preprocessing.temp_cleaner.ConfigLoader")
    def test_input_validation(self, mock_config):
        """Test input signal validation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp": self.test_config["physio"]["temp"],
            "physio.temp.processing": self.test_config["physio"]["temp"]["processing"],
            "physio.temp.sampling_rate": 4,
        }.get(key, default)

        cleaner = TEMPCleaner()

        # Test with empty DataFrame
        with self.assertRaises(ValueError):
            cleaner.clean_signal(pd.DataFrame(), moment="test")


class TestTEMPMetricsExtractor(unittest.TestCase):
    """Test temperature metrics extraction functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = {"physio": {"temp": {"metrics": {"baseline_window": 60}}}}

        # Create test processed signals
        self.test_signals = self._create_test_data()

    def _create_test_data(self):
        """Create test processed temperature data."""
        n_samples = 240  # 60 seconds at 4 Hz

        # Simulate temperature that increases slightly over time (relaxation)
        time_values = np.arange(0, 60, 0.25)
        temp_values = (
            31.5 + 0.5 * (time_values / 60) + np.random.normal(0, 0.03, n_samples)
        )

        signals = pd.DataFrame(
            {
                "time": time_values,
                "TEMP_Clean": temp_values,
                "TEMP_Raw": temp_values + np.random.normal(0, 0.01, n_samples),
                "TEMP_Quality": np.ones(n_samples),
            }
        )

        return signals

    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    def test_extractor_initialization(self, mock_config):
        """Test metrics extractor initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.metrics": self.test_config["physio"]["temp"]["metrics"]
        }.get(key, default)

        extractor = TEMPMetricsExtractor()
        self.assertIsInstance(extractor, TEMPMetricsExtractor)

    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    def test_extract_metrics_success(self, mock_config):
        """Test successful metrics extraction."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.metrics": self.test_config["physio"]["temp"]["metrics"]
        }.get(key, default)

        extractor = TEMPMetricsExtractor()
        metrics = extractor.extract_metrics(self.test_signals, moment="test")

        # Check structure
        self.assertIsInstance(metrics, dict)
        self.assertIn("moment", metrics)
        self.assertIn("descriptive", metrics)
        self.assertIn("trend", metrics)
        self.assertIn("stability", metrics)
        self.assertIn("contextual", metrics)
        self.assertIn("summary", metrics)

        # Check descriptive metrics
        self.assertIn("temp_mean", metrics["descriptive"])
        self.assertIn("temp_std", metrics["descriptive"])
        self.assertIn("temp_min", metrics["descriptive"])
        self.assertIn("temp_max", metrics["descriptive"])
        self.assertIn("temp_range", metrics["descriptive"])
        self.assertIn("temp_median", metrics["descriptive"])
        self.assertIn("temp_iqr", metrics["descriptive"])

        # Check trend metrics
        self.assertIn("temp_slope", metrics["trend"])
        self.assertIn("temp_initial", metrics["trend"])
        self.assertIn("temp_final", metrics["trend"])
        self.assertIn("temp_change", metrics["trend"])

        # Check stability metrics
        self.assertIn("temp_stability", metrics["stability"])
        self.assertIn("temp_cv", metrics["stability"])

    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    def test_trend_detection(self, mock_config):
        """Test temperature trend detection."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.metrics": self.test_config["physio"]["temp"]["metrics"]
        }.get(key, default)

        # Create signal with clear increasing trend (relaxation)
        n_samples = 240
        time_values = np.arange(0, 60, 0.25)
        temp_values = 31.0 + 1.0 * (time_values / 60)  # +1°C over 60 seconds

        signals = pd.DataFrame(
            {
                "time": time_values,
                "TEMP_Clean": temp_values,
                "TEMP_Quality": np.ones(n_samples),
            }
        )

        extractor = TEMPMetricsExtractor()
        metrics = extractor.extract_metrics(signals, moment="test")

        # Check positive trend
        self.assertGreater(metrics["trend"]["temp_slope"], 0)
        self.assertGreater(metrics["trend"]["temp_change"], 0)
        self.assertGreater(
            metrics["trend"]["temp_final"], metrics["trend"]["temp_initial"]
        )

    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    def test_stability_metrics(self, mock_config):
        """Test stability metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.metrics": self.test_config["physio"]["temp"]["metrics"]
        }.get(key, default)

        # Create very stable signal
        n_samples = 240
        time_values = np.arange(0, 60, 0.25)
        temp_values = 32.0 + np.random.normal(
            0, 0.01, n_samples
        )  # Very low variability

        signals = pd.DataFrame(
            {
                "time": time_values,
                "TEMP_Clean": temp_values,
                "TEMP_Quality": np.ones(n_samples),
            }
        )

        extractor = TEMPMetricsExtractor()
        metrics = extractor.extract_metrics(signals, moment="test")

        # High stability expected
        self.assertGreater(metrics["stability"]["temp_stability"], 0.95)
        self.assertLess(metrics["stability"]["temp_cv"], 0.01)

    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    def test_session_metrics_extraction(self, mock_config):
        """Test session metrics extraction for multiple moments."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.metrics": self.test_config["physio"]["temp"]["metrics"]
        }.get(key, default)

        extractor = TEMPMetricsExtractor()

        # Create multiple moment data
        processed_results = {
            "restingstate": self.test_signals,
            "therapy": self.test_signals.copy(),
        }

        session_metrics = extractor.extract_session_metrics(processed_results)

        self.assertIn("restingstate", session_metrics)
        self.assertIn("therapy", session_metrics)

        # Check flat metrics structure
        self.assertIn("temp_mean", session_metrics["restingstate"])
        self.assertIn("temp_slope", session_metrics["restingstate"])

    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    def test_metrics_dataframe_extraction(self, mock_config):
        """Test metrics extraction as DataFrame."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.temp.metrics": self.test_config["physio"]["temp"]["metrics"]
        }.get(key, default)

        extractor = TEMPMetricsExtractor()

        processed_results = {
            "restingstate": self.test_signals,
            "therapy": self.test_signals.copy(),
        }

        metrics_df = extractor.extract_metrics_dataframe(processed_results)

        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertEqual(len(metrics_df), 2)  # Two moments
        self.assertIn("moment", metrics_df.columns)
        self.assertIn("temp_mean", metrics_df.columns)


class TestTEMPBIDSWriter(unittest.TestCase):
    """Test temperature BIDS output writer functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.test_subject = "sub-g01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"

        # Create test data
        n_samples = 240  # 60 seconds at 4 Hz
        self.test_signals = pd.DataFrame(
            {
                "time": np.arange(0, 60, 0.25),
                "TEMP_Raw": np.random.uniform(31.5, 32.5, n_samples),
                "TEMP_Clean": np.random.uniform(31.5, 32.5, n_samples),
                "TEMP_Quality": np.ones(n_samples),
                "TEMP_Outliers": np.zeros(n_samples, dtype=bool),
                "TEMP_Interpolated": np.zeros(n_samples, dtype=bool),
            }
        )

        self.test_metadata = {
            "SamplingFrequency": 4.0,
            "TaskName": self.test_moment,
            "quality_score": 0.95,
            "valid_samples": 240,
            "total_samples": 240,
        }

        self.test_config = {
            "paths": {"derivatives": str(self.temp_path / "derivatives")}
        }

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_writer_initialization(self, mock_config):
        """Test BIDS writer initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "paths.derivatives": str(self.temp_path / "derivatives")
        }.get(key, default)

        writer = TEMPBIDSWriter(config_path=None)
        self.assertIsInstance(writer, TEMPBIDSWriter)
        self.assertEqual(writer.modality, "temp")

    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_save_processed_data_basic(self, mock_config):
        """Test basic functionality of save_processed_data."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "paths.derivatives": str(self.temp_path / "derivatives")
        }.get(key, default)

        writer = TEMPBIDSWriter(config_path=None)

        # Prepare data in expected format
        processed_results = {self.test_moment: self.test_signals}

        output_files = writer.save_processed_data(
            subject_id=self.test_subject,
            session_id=self.test_session,
            processed_results=processed_results,
            session_metrics=None,
            processing_metadata={self.test_moment: self.test_metadata},
        )

        # Verify output files dict
        self.assertIsInstance(output_files, dict)

        # Verify file types
        self.assertIn("physio", output_files)
        self.assertIn("physio_json", output_files)
        self.assertIn("metrics", output_files)
        self.assertIn("metrics_json", output_files)
        self.assertIn("summary", output_files)

        # Verify at least some files were created
        total_files = sum(len(files) for files in output_files.values())
        self.assertGreater(total_files, 0)

    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_bids_directory_structure(self, mock_config):
        """Test BIDS-compliant directory structure creation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "paths.derivatives": str(self.temp_path / "derivatives"),
            "output.preprocessing_dir": "preprocessing",
            "output.modality_subdirs.temp": "temp",
        }.get(key, default)

        writer = TEMPBIDSWriter(config_path=None)

        processed_results = {self.test_moment: self.test_signals}

        writer.save_processed_data(
            subject_id=self.test_subject,
            session_id=self.test_session,
            processed_results=processed_results,
            session_metrics=None,
            processing_metadata={self.test_moment: self.test_metadata},
        )

        # Verify preprocessing directory exists
        preprocessing_dir = self.temp_path / "derivatives" / "preprocessing"
        self.assertTrue(preprocessing_dir.exists())

        # Verify subject/session/modality structure
        temp_dir = preprocessing_dir / self.test_subject / self.test_session / "temp"
        self.assertTrue(temp_dir.exists())

    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_output_file_naming(self, mock_config):
        """Test BIDS-compliant file naming."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "paths.derivatives": str(self.temp_path / "derivatives"),
            "output.preprocessing_dir": "preprocessing",
            "output.modality_subdirs.temp": "temp",
        }.get(key, default)

        writer = TEMPBIDSWriter(config_path=None)

        processed_results = {self.test_moment: self.test_signals}

        output_files = writer.save_processed_data(
            subject_id=self.test_subject,
            session_id=self.test_session,
            processed_results=processed_results,
            session_metrics=None,
            processing_metadata={self.test_moment: self.test_metadata},
        )

        # Check physio file naming
        if output_files["physio"]:
            physio_file = output_files["physio"][0]
            expected_pattern = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_desc-processed_recording-temp.tsv"
            self.assertEqual(physio_file.name, expected_pattern)


class TestTEMPPipelineIntegration(unittest.TestCase):
    """Integration tests for complete temperature pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.test_subject = "sub-g01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"

        # Create full test data structure
        self._create_test_data_structure()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def _create_test_data_structure(self):
        """Create complete test data structure."""
        # Create sourcedata directory
        physio_dir = (
            self.temp_path
            / "sourcedata"
            / self.test_subject
            / self.test_session
            / "physio"
        )
        physio_dir.mkdir(parents=True)

        # Create realistic temperature data
        duration = 60
        sampling_rate = 4
        t = np.arange(0, duration, 1 / sampling_rate)

        # Temperature with slight increasing trend (relaxation)
        temp = 31.5 + 0.5 * (t / duration) + np.random.normal(0, 0.05, len(t))

        # Save TSV
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-temp"
        data = pd.DataFrame({"time": t, "temp": temp})
        data.to_csv(physio_dir / f"{base_filename}.tsv", sep="\t", index=False)

        # Save JSON
        metadata = {
            "SamplingFrequency": 4.0,
            "StartTime": 0,
            "Columns": ["time", "temp"],
            "Units": ["s", "°C"],
            "TaskName": self.test_moment,
            "RecordingType": "Temperature",
            "FamilyID": "g01",
        }

        with open(physio_dir / f"{base_filename}.json", "w") as f:
            json.dump(metadata, f)

    @patch("src.physio.preprocessing.temp_loader.ConfigLoader")
    @patch("src.physio.preprocessing.temp_cleaner.ConfigLoader")
    @patch("src.physio.preprocessing.temp_metrics.ConfigLoader")
    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_full_pipeline_execution(
        self,
        mock_writer_config,
        mock_metrics_config,
        mock_cleaner_config,
        mock_loader_config,
    ):
        """Test complete temperature pipeline from load to write."""

        # Mock all configs
        def get_config(key, default=None):
            config_map = {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "paths.derivatives": str(self.temp_path / "derivatives"),
                "physio.temp.sampling_rate": 4,
                "physio.temp.processing": {
                    "outlier_threshold": [25, 40],
                    "jump_threshold": 2.0,
                    "interpolation_max_gap": 10,
                    "quality_min_samples": 30,
                },
                "physio.temp.metrics": {"baseline_window": 60},
            }
            return config_map.get(key, default)

        for mock_cfg in [
            mock_loader_config,
            mock_cleaner_config,
            mock_metrics_config,
            mock_writer_config,
        ]:
            mock_cfg.return_value.get.side_effect = get_config

        # Step 1: Load data
        loader = TEMPLoader()
        data, metadata = loader.load_subject_session(
            self.test_subject, self.test_session, self.test_moment
        )

        self.assertGreater(len(data), 0)
        self.assertIn("temp", data.columns)

        # Step 2: Clean data
        cleaner = TEMPCleaner()
        cleaned_data, cleaning_metadata = cleaner.clean_signal(
            data, moment=self.test_moment
        )

        self.assertIn("TEMP_Clean", cleaned_data.columns)
        self.assertIn("quality_score", cleaning_metadata)

        # Step 3: Extract metrics
        extractor = TEMPMetricsExtractor()
        metrics = extractor.extract_metrics(cleaned_data, moment=self.test_moment)

        self.assertIn("descriptive", metrics)
        self.assertIn("trend", metrics)

        # Step 4: Write BIDS output
        writer = TEMPBIDSWriter()

        processed_results = {self.test_moment: cleaned_data}
        output_files = writer.save_processed_data(
            subject_id=self.test_subject,
            session_id=self.test_session,
            processed_results=processed_results,
            session_metrics=None,
            processing_metadata={self.test_moment: cleaning_metadata},
        )

        # Verify output was created
        total_files = sum(len(files) for files in output_files.values())
        self.assertGreater(total_files, 0)


if __name__ == "__main__":
    unittest.main()
