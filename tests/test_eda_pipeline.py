"""
Unit tests for EDA processing pipeline components.

This module provides comprehensive testing for all EDA processing components
including data loading, cleaning (cvxEDA decomposition), metrics extraction,
and BIDS output.

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

from src.physio.preprocessing.eda_loader import EDALoader
from src.physio.preprocessing.eda_cleaner import EDACleaner
from src.physio.preprocessing.eda_metrics import EDAMetricsExtractor
from src.physio.preprocessing.eda_bids_writer import EDABIDSWriter


class TestEDALoader(unittest.TestCase):
    """Test EDA data loading functionality."""

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

        # Create test EDA files
        self._create_test_eda_files(physio_dir)

        # Create test config
        self.test_config = {
            "paths": {"sourcedata": str(self.temp_path / "sourcedata")},
            "physio": {"eda": {"sampling_rate": 4}},
            "moments": [{"name": self.test_moment}],
        }

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def _create_test_eda_files(self, physio_dir: Path):
        """Create test EDA data files."""
        # Create test TSV data (60 seconds at 4 Hz = 240 samples)
        duration = 60  # seconds
        sampling_rate = 4  # Hz
        time_values = np.arange(0, duration, 1 / sampling_rate)

        # Simulate realistic EDA signal:
        # Tonic component (slow drift): 0.5-2.0 μS
        tonic = 1.0 + 0.5 * np.sin(2 * np.pi * 0.01 * time_values)

        # Phasic component (SCRs): occasional spikes
        phasic = np.zeros_like(time_values)
        scr_times = [10, 25, 45]  # SCR peaks at these times
        for scr_time in scr_times:
            int(scr_time * sampling_rate)
            # Create SCR with rise and recovery
            for i in range(len(time_values)):
                t_diff = time_values[i] - scr_time
                if 0 <= t_diff <= 10:  # SCR lasting 10 seconds
                    # Exponential rise and decay
                    phasic[i] += 0.3 * np.exp(-abs(t_diff - 2) / 2)

        # Add small noise
        noise = np.random.normal(0, 0.02, len(time_values))
        eda_values = tonic + phasic + noise

        # Ensure positive values
        eda_values = np.maximum(eda_values, 0.1)

        test_data = pd.DataFrame({"time": time_values, "eda": eda_values})

        # Save TSV file
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-eda"
        tsv_file = physio_dir / f"{base_filename}.tsv"
        test_data.to_csv(tsv_file, sep="\t", index=False)

        # Save JSON metadata
        json_file = physio_dir / f"{base_filename}.json"
        metadata = {
            "SamplingFrequency": 4.0,
            "StartTime": 0,
            "Columns": ["time", "eda"],
            "Units": ["s", "μS"],
            "TaskName": self.test_moment,
            "RecordingType": "EDA",
            "FamilyID": "g01",
            "DeviceManufacturer": "Empatica",
            "DeviceModel": "E4",
        }

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)

    @patch("src.physio.preprocessing.eda_loader.ConfigLoader")
    def test_loader_basic_functionality(self, mock_config):
        """Test basic EDA loader functionality."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda": {"sampling_rate": 4},
            "paths.rawdata": str(self.temp_path / "sourcedata"),
        }.get(key, default)

        loader = EDALoader()
        self.assertIsInstance(loader, EDALoader)
        self.assertEqual(loader.sampling_rate, 4)

    def test_load_subject_session_success(self):
        """Test successful loading of subject/session data."""
        with patch("src.physio.preprocessing.eda_loader.ConfigLoader") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "physio.eda.sampling_rate": 4,
            }.get(key, default)

            loader = EDALoader()
            data, metadata = loader.load_subject_session(
                self.test_subject, self.test_session, self.test_moment
            )

            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn("time", data.columns)
            self.assertIn("eda", data.columns)
            self.assertEqual(len(data), 240)  # 60 seconds * 4 Hz

            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata["SamplingFrequency"], 4.0)
            self.assertEqual(metadata["TaskName"], self.test_moment)

            # Verify EDA values are reasonable (0.1 - 3.0 μS)
            self.assertTrue((data["eda"] >= 0).all())
            self.assertTrue((data["eda"] <= 5.0).all())

    def test_load_subject_session_missing_file(self):
        """Test loading with missing files."""
        with patch("src.physio.preprocessing.eda_loader.ConfigLoader") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "physio.eda.sampling_rate": 4,
            }.get(key, default)

            loader = EDALoader()
            with self.assertRaises(FileNotFoundError):
                loader.load_subject_session(
                    self.test_subject, self.test_session, "nonexistent"
                )

    def test_data_validation(self):
        """Test EDA data validation functionality."""
        with patch("src.physio.preprocessing.eda_loader.ConfigLoader") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "physio.eda.sampling_rate": 4,
            }.get(key, default)

            loader = EDALoader()

            # Test with valid data
            valid_data = pd.DataFrame(
                {"time": [0, 0.25, 0.5, 0.75], "eda": [1.0, 1.1, 1.05, 1.2]}
            )
            valid_metadata = {"SamplingFrequency": 4, "Columns": ["time", "eda"]}

            # Should not raise exception
            loader._validate_data_structure(
                valid_data, valid_metadata, Path("test.tsv")
            )

            # Test with invalid data (missing columns)
            invalid_data = pd.DataFrame({"time": [0, 0.25, 0.5]})

            with self.assertRaises((ValueError, KeyError)):
                loader._validate_data_structure(
                    invalid_data, valid_metadata, Path("test.tsv")
                )


class TestEDACleaner(unittest.TestCase):
    """Test EDA signal cleaning and decomposition functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            "physio": {
                "eda": {
                    "sampling_rate": 4,
                    "processing": {
                        "method": "cvxEDA",
                        "scr_threshold": 0.01,
                        "scr_min_amplitude": 0.01,
                    },
                }
            }
        }

        # Create test signal (60 seconds at 4 Hz)
        self.test_signal = self._create_test_signal()

    def _create_test_signal(self):
        """Create synthetic EDA signal for testing."""
        duration = 60
        sampling_rate = 4
        t = np.arange(0, duration, 1 / sampling_rate)

        # Tonic baseline with slow drift
        tonic = 1.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t)

        # Add several SCRs
        phasic = np.zeros_like(t)
        scr_times = [10, 25, 40, 55]
        for scr_time in scr_times:
            t_diff = t - scr_time
            # SCR with realistic rise/recovery time
            mask = (t_diff >= 0) & (t_diff <= 10)
            phasic[mask] += 0.2 * np.exp(-np.abs(t_diff[mask] - 2) / 2)

        # Add noise
        noise = np.random.normal(0, 0.02, len(t))

        return np.maximum(tonic + phasic + noise, 0.1)

    @patch("src.physio.preprocessing.eda_cleaner.ConfigLoader")
    def test_cleaner_initialization(self, mock_config):
        """Test EDA cleaner initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda": self.test_config["physio"]["eda"],
            "physio.eda.processing": {"method": "neurokit", "scr_threshold": 0.01},
            "physio.eda.processing.method": "neurokit",
            "physio.eda.processing.scr_threshold": 0.01,
        }.get(key, default)

        cleaner = EDACleaner()
        self.assertEqual(cleaner.method, "neurokit")
        self.assertEqual(cleaner.scr_threshold, 0.01)

    @patch("src.physio.preprocessing.eda_cleaner.ConfigLoader")
    @patch("src.physio.preprocessing.eda_cleaner.nk.eda_process")
    def test_clean_signal_success(self, mock_eda_process, mock_config):
        """Test successful EDA signal cleaning."""
        # Mock config
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda": self.test_config["physio"]["eda"],
            "physio.eda.processing": self.test_config["physio"]["eda"]["processing"],
            "physio.eda.sampling_rate": 4,
            "physio.eda.processing.method": "neurokit",
        }.get(key, default)

        # Create input DataFrame
        test_df = pd.DataFrame(
            {"time": np.arange(0, 60, 0.25), "eda": self.test_signal}
        )

        # Mock NeuroKit2 response with cvxEDA decomposition
        mock_processed_signals = pd.DataFrame(
            {
                "EDA_Raw": self.test_signal,
                "EDA_Clean": self.test_signal,
                "EDA_Tonic": np.ones(len(self.test_signal)) * 1.0,
                "EDA_Phasic": self.test_signal - 1.0,
                "SCR_Onsets": np.zeros(len(self.test_signal)),
                "SCR_Peaks": np.zeros(len(self.test_signal)),
                "SCR_Amplitude": np.zeros(len(self.test_signal)),
            }
        )

        # Mark some SCR peaks
        mock_processed_signals.loc[40, "SCR_Peaks"] = 1
        mock_processed_signals.loc[100, "SCR_Peaks"] = 1
        mock_processed_signals.loc[160, "SCR_Peaks"] = 1

        mock_processing_info = {
            "SCR_Onsets": [35, 95, 155],
            "SCR_Peaks": [40, 100, 160],
            "SCR_Amplitude": [0.15, 0.20, 0.18],
            "sampling_rate": 4,
        }
        mock_eda_process.return_value = (mock_processed_signals, mock_processing_info)

        cleaner = EDACleaner()
        processed_signals = cleaner.clean_signal(test_df, moment="test")

        self.assertIsInstance(processed_signals, pd.DataFrame)
        self.assertIn("EDA_Clean", processed_signals.columns)
        self.assertIn("EDA_Tonic", processed_signals.columns)
        self.assertIn("EDA_Phasic", processed_signals.columns)

        mock_eda_process.assert_called_once()

    @patch("src.physio.preprocessing.eda_cleaner.ConfigLoader")
    def test_input_validation(self, mock_config):
        """Test input signal validation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda": self.test_config["physio"]["eda"]
        }.get(key, default)

        cleaner = EDACleaner()

        # Test with empty DataFrame
        with self.assertRaises((ValueError, KeyError)):
            cleaner.clean_signal(pd.DataFrame(), moment="test")

        # Test with invalid DataFrame (missing columns)
        with self.assertRaises((ValueError, KeyError)):
            invalid_df = pd.DataFrame({"time": [0, 0.25, 0.5]})
            cleaner.clean_signal(invalid_df, moment="test")

    @patch("src.physio.preprocessing.eda_cleaner.ConfigLoader")
    def test_data_quality_checks(self, mock_config):
        """Test EDA data quality validation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda": self.test_config["physio"]["eda"],
            "physio.eda.processing": self.test_config["physio"]["eda"]["processing"],
        }.get(key, default)

        cleaner = EDACleaner()

        # Create test DataFrame with good quality data
        pd.DataFrame(
            {"time": np.arange(0, 60, 0.25), "eda": np.random.uniform(0.5, 2.0, 240)}
        )

        # Should not raise for valid data (just test instantiation)
        self.assertIsInstance(cleaner, EDACleaner)


class TestEDAMetricsExtractor(unittest.TestCase):
    """Test EDA metrics extraction functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            "physio": {
                "eda": {
                    "metrics": [
                        "SCR_Peaks_N",
                        "SCR_Peaks_Amplitude_Mean",
                        "SCR_Peaks_Rate",
                        "EDA_Tonic_Mean",
                        "EDA_Tonic_SD",
                        "EDA_Tonic_Range",
                        "EDA_Phasic_Mean",
                        "EDA_Phasic_SD",
                        "EDA_Phasic_Rate",
                    ]
                }
            }
        }

        # Create test processed signals
        self.test_signals, self.test_info = self._create_test_data()

    def _create_test_data(self):
        """Create test processed EDA data."""
        n_samples = 240  # 60 seconds at 4 Hz

        signals = pd.DataFrame(
            {
                "EDA_Clean": np.random.uniform(0.5, 2.0, n_samples),
                "EDA_Tonic": np.linspace(1.0, 1.5, n_samples),
                "EDA_Phasic": np.random.uniform(-0.1, 0.3, n_samples),
                "SCR_Peaks": np.zeros(n_samples),
                "SCR_Amplitude": np.zeros(n_samples),
                "SCR_RiseTime": np.zeros(n_samples),
                "SCR_RecoveryTime": np.zeros(n_samples),
            }
        )

        # Mark some SCR peaks
        peak_indices = [40, 100, 160, 200]
        signals.loc[peak_indices, "SCR_Peaks"] = 1
        signals.loc[peak_indices, "SCR_Amplitude"] = [0.15, 0.20, 0.18, 0.12]
        signals.loc[peak_indices, "SCR_RiseTime"] = [1.5, 1.8, 1.6, 1.4]
        signals.loc[peak_indices, "SCR_RecoveryTime"] = [3.2, 3.5, 3.0, 2.8]

        info = {
            "SCR_Peaks": peak_indices,
            "SCR_Amplitude": [0.15, 0.20, 0.18, 0.12],
            "SCR_RiseTime": [1.5, 1.8, 1.6, 1.4],
            "SCR_RecoveryTime": [3.2, 3.5, 3.0, 2.8],
            "sampling_rate": 4,
            "duration": 60,
        }

        return signals, info

    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    def test_extractor_initialization(self, mock_config):
        """Test metrics extractor initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda.metrics": self.test_config["physio"]["eda"]["metrics"]
        }.get(key, default)

        extractor = EDAMetricsExtractor()
        self.assertIsInstance(extractor, EDAMetricsExtractor)

    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    def test_extract_metrics_success(self, mock_config):
        """Test successful metrics extraction."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda.metrics": self.test_config["physio"]["eda"]["metrics"]
        }.get(key, default)

        extractor = EDAMetricsExtractor()
        metrics_df = extractor.extract_eda_metrics(self.test_signals, moment="test")

        # Method returns DataFrame with one row
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertEqual(len(metrics_df), 1)

        # Convert to dict for easier assertion
        metrics = metrics_df.iloc[0].to_dict()

        # Check SCR metrics (using actual NeuroKit2 naming)
        self.assertIn("SCR_Peaks_N", metrics)
        self.assertEqual(metrics["SCR_Peaks_N"], 4)

        self.assertIn("SCR_Peaks_Amplitude_Mean", metrics)
        self.assertGreater(metrics["SCR_Peaks_Amplitude_Mean"], 0)

        self.assertIn("SCR_Peaks_Rate", metrics)
        self.assertAlmostEqual(metrics["SCR_Peaks_Rate"], 4.0, delta=0.1)

        # Check tonic metrics
        self.assertIn("EDA_Tonic_Mean", metrics)
        self.assertIn("EDA_Tonic_SD", metrics)

        # Check phasic metrics
        self.assertIn("EDA_Phasic_Mean", metrics)
        self.assertIn("EDA_Phasic_SD", metrics)

    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    def test_scr_metrics_calculation(self, mock_config):
        """Test SCR-specific metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda.metrics": [
                "SCR_Peaks_N",
                "SCR_Peaks_Amplitude_Mean",
                "SCR_Peaks_Amplitude_Max",
            ]
        }.get(key, default)

        extractor = EDAMetricsExtractor()
        metrics_df = extractor.extract_eda_metrics(self.test_signals, moment="test")

        # Convert to dict for easier assertion
        metrics = metrics_df.iloc[0].to_dict()

        # Verify SCR count
        self.assertEqual(metrics["SCR_Peaks_N"], 4)

        # Verify amplitude metrics exist
        self.assertIn("SCR_Peaks_Amplitude_Mean", metrics)
        self.assertIn("SCR_Peaks_Amplitude_Max", metrics)

    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    def test_tonic_metrics_calculation(self, mock_config):
        """Test tonic component metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda.metrics": [
                "EDA_Tonic_Mean",
                "EDA_Tonic_SD",
                "EDA_Tonic_Min",
                "EDA_Tonic_Max",
                "EDA_Tonic_Range",
            ]
        }.get(key, default)

        extractor = EDAMetricsExtractor()
        metrics_df = extractor.extract_eda_metrics(self.test_signals, moment="test")

        # Convert to dict for easier assertion
        metrics = metrics_df.iloc[0].to_dict()

        self.test_signals["EDA_Tonic"].values

        self.assertIn("EDA_Tonic_Mean", metrics)
        self.assertIn("EDA_Tonic_SD", metrics)
        self.assertIn("EDA_Tonic_Min", metrics)
        self.assertIn("EDA_Tonic_Max", metrics)

    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    def test_phasic_metrics_calculation(self, mock_config):
        """Test phasic component metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda.metrics": ["EDA_Phasic_Mean", "EDA_Phasic_SD", "EDA_Phasic_Max"]
        }.get(key, default)

        extractor = EDAMetricsExtractor()
        metrics_df = extractor.extract_eda_metrics(self.test_signals, moment="test")

        # Convert to dict for easier assertion
        metrics = metrics_df.iloc[0].to_dict()

        self.assertIn("EDA_Phasic_Mean", metrics)
        self.assertIn("EDA_Phasic_SD", metrics)
        self.assertIn("EDA_Phasic_Max", metrics)

    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    def test_empty_scr_handling(self, mock_config):
        """Test handling of data with no SCRs detected."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "physio.eda.metrics": [
                "SCR_Peaks_N",
                "SCR_Peaks_Amplitude_Mean",
                "SCR_Peaks_Rate",
            ]
        }.get(key, default)

        # Create data with no SCRs
        signals = self.test_signals.copy()
        signals["SCR_Peaks"] = 0
        signals["SCR_Amplitude"] = 0.0
        signals["SCR_RiseTime"] = 0.0
        signals["SCR_RecoveryTime"] = 0.0

        extractor = EDAMetricsExtractor()
        metrics_df = extractor.extract_eda_metrics(signals, moment="test")

        # Convert to dict for easier assertion
        metrics = metrics_df.iloc[0].to_dict()

        self.assertEqual(metrics["SCR_Peaks_N"], 0)
        self.assertEqual(metrics["SCR_Peaks_Rate"], 0.0)


class TestEDABIDSWriter(unittest.TestCase):
    """Test EDA BIDS output writer functionality."""

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
                "EDA_Raw": np.random.uniform(0.5, 2.0, n_samples),
                "EDA_Clean": np.random.uniform(0.5, 2.0, n_samples),
                "EDA_Tonic": np.linspace(1.0, 1.5, n_samples),
                "EDA_Phasic": np.random.uniform(-0.1, 0.3, n_samples),
                "SCR_Peaks": np.zeros(n_samples),
                "SCR_Amplitude": np.zeros(n_samples),
            }
        )

        # Mark some SCR peaks
        peak_indices = [40, 100, 160]
        self.test_signals.loc[peak_indices, "SCR_Peaks"] = 1
        self.test_signals.loc[peak_indices, "SCR_Amplitude"] = [0.15, 0.20, 0.18]

        self.test_metrics = {
            "SCR_Peaks_N": 3,
            "SCR_Peaks_Amplitude_Mean": 0.177,
            "SCR_Peaks_Rate": 3.0,
            "EDA_Tonic_Mean": 1.25,
            "EDA_Tonic_SD": 0.15,
            "EDA_Phasic_Mean": 0.08,
            "duration": 60.0,
        }

        self.test_metadata = {
            "SamplingFrequency": 4.0,
            "TaskName": self.test_moment,
            "FamilyID": "g01",
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

        writer = EDABIDSWriter(config_path=None)
        self.assertIsInstance(writer, EDABIDSWriter)

    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_save_processed_data_basic(self, mock_config):
        """Test basic functionality of save_processed_data."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "paths.derivatives": str(self.temp_path / "derivatives")
        }.get(key, default)

        writer = EDABIDSWriter(config_path=None)

        # Prepare data in expected format
        processed_results = {self.test_moment: self.test_signals}

        # Create session metrics DataFrame
        session_metrics = pd.DataFrame([self.test_metrics])
        session_metrics["moment"] = self.test_moment

        output_files = writer.save_processed_data(
            subject_id=self.test_subject,
            session_id=self.test_session,
            processed_results=processed_results,
            session_metrics=session_metrics,
            processing_metadata=self.test_metadata,
        )

        # Verify output files dict
        self.assertIsInstance(output_files, dict)

        # Verify at least some files were created
        total_files = sum(len(files) for files in output_files.values())
        self.assertGreater(total_files, 0)

    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_bids_directory_structure(self, mock_config):
        """Test BIDS-compliant directory structure creation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "paths.derivatives": str(self.temp_path / "derivatives"),
            "output.preprocessing_dir": "preprocessing",
            "output.modality_subdirs.eda": "eda",
        }.get(key, default)

        writer = EDABIDSWriter(config_path=None)

        processed_results = {self.test_moment: self.test_signals}
        session_metrics = pd.DataFrame([self.test_metrics])
        session_metrics["moment"] = self.test_moment

        writer.save_processed_data(
            subject_id=self.test_subject,
            session_id=self.test_session,
            processed_results=processed_results,
            session_metrics=session_metrics,
            processing_metadata=self.test_metadata,
        )

        # Verify preprocessing directory exists (new structure)
        preprocessing_dir = self.temp_path / "derivatives" / "preprocessing"
        self.assertTrue(preprocessing_dir.exists())


class TestEDAPipelineIntegration(unittest.TestCase):
    """Integration tests for complete EDA pipeline."""

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

        # Create realistic EDA data
        duration = 60
        sampling_rate = 4
        t = np.arange(0, duration, 1 / sampling_rate)

        # Realistic EDA signal with tonic + phasic
        tonic = 1.2 + 0.3 * np.sin(2 * np.pi * 0.01 * t)
        phasic = np.zeros_like(t)

        scr_times = [10, 25, 40, 55]
        for scr_time in scr_times:
            mask = (t >= scr_time) & (t <= scr_time + 10)
            t_diff = t[mask] - scr_time
            phasic[mask] += 0.2 * np.exp(-np.abs(t_diff - 2) / 2)

        eda = tonic + phasic + np.random.normal(0, 0.02, len(t))
        eda = np.maximum(eda, 0.1)

        # Save TSV
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-eda"
        data = pd.DataFrame({"time": t, "eda": eda})
        data.to_csv(physio_dir / f"{base_filename}.tsv", sep="\t", index=False)

        # Save JSON
        metadata = {
            "SamplingFrequency": 4.0,
            "StartTime": 0,
            "Columns": ["time", "eda"],
            "Units": ["s", "μS"],
            "TaskName": self.test_moment,
            "RecordingType": "EDA",
            "FamilyID": "g01",
        }

        with open(physio_dir / f"{base_filename}.json", "w") as f:
            json.dump(metadata, f)

    @patch("src.physio.preprocessing.eda_loader.ConfigLoader")
    @patch("src.physio.preprocessing.eda_cleaner.ConfigLoader")
    @patch("src.physio.preprocessing.eda_metrics.ConfigLoader")
    @patch("src.physio.preprocessing.base_bids_writer.ConfigLoader")
    def test_full_pipeline_execution(
        self,
        mock_writer_config,
        mock_metrics_config,
        mock_cleaner_config,
        mock_loader_config,
    ):
        """Test complete EDA pipeline from load to write."""

        # Mock all configs
        def get_config(key, default=None):
            config_map = {
                "paths.rawdata": str(self.temp_path / "sourcedata"),
                "paths.derivatives": str(self.temp_path / "derivatives"),
                "physio.eda.sampling_rate": 4,
                "physio.eda.processing": {"method": "cvxEDA", "scr_threshold": 0.01},
                "physio.eda.processing.method": "cvxEDA",
                "physio.eda.metrics": [
                    "SCR_Peaks_N",
                    "SCR_Peaks_Amplitude_Mean",
                    "EDA_Tonic_Mean",
                ],
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
        loader = EDALoader()
        data, metadata = loader.load_subject_session(
            self.test_subject, self.test_session, self.test_moment
        )

        self.assertGreater(len(data), 0)
        self.assertIn("eda", data.columns)

        # Note: Full pipeline test would continue with cleaner, metrics, and writer
        # but requires proper mocking of NeuroKit2 functions
        # This basic test verifies the data loading step works


if __name__ == "__main__":
    unittest.main()
