"""
Unit tests for the Alliance (MOI) module.

Tests MOILoader, MOIEpocher, MOIWriter, and MOIVisualizer.

Authors: Remy Ramadour
Date: November 2025
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alliance.moi_loader import MOILoader
from src.alliance.moi_epocher import MOIEpocher
from src.alliance.moi_writer import MOIWriter
from src.alliance.moi_visualizer import MOIVisualizer


class TestMOILoader(unittest.TestCase):
    """Tests for MOILoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "paths": {
                "rawdata": "data/raw",
                "derivatives": "data/derivatives",
                "logs": "log",
            }
        }

    @patch("src.alliance.moi_loader.ConfigLoader")
    def test_loader_initialization(self, mock_config):
        """Test MOILoader initializes correctly."""
        mock_config.return_value.config = self.test_config

        loader = MOILoader()

        self.assertIsNotNone(loader.rawdata_path)
        self.assertIsNotNone(loader.sourcedata_path)

    @patch("src.alliance.moi_loader.ConfigLoader")
    def test_timestamp_to_seconds_valid(self, mock_config):
        """Test timestamp conversion with valid inputs."""
        mock_config.return_value.config = self.test_config

        loader = MOILoader()

        # Test basic conversions
        self.assertEqual(loader._timestamp_to_seconds("00:00:00"), 0.0)
        self.assertEqual(loader._timestamp_to_seconds("00:01:00"), 60.0)
        self.assertEqual(loader._timestamp_to_seconds("00:02:07"), 127.0)
        self.assertEqual(loader._timestamp_to_seconds("01:00:00"), 3600.0)
        self.assertEqual(loader._timestamp_to_seconds("01:08:30"), 4110.0)

    @patch("src.alliance.moi_loader.ConfigLoader")
    def test_timestamp_to_seconds_invalid(self, mock_config):
        """Test timestamp conversion with invalid inputs."""
        mock_config.return_value.config = self.test_config

        loader = MOILoader()

        # Test invalid formats
        self.assertEqual(loader._timestamp_to_seconds(""), 0.0)
        self.assertEqual(loader._timestamp_to_seconds("invalid"), 0.0)
        self.assertEqual(loader._timestamp_to_seconds("00:00"), 0.0)  # Missing seconds
        self.assertEqual(loader._timestamp_to_seconds(None), 0.0)

    @patch("src.alliance.moi_loader.ConfigLoader")
    def test_convert_timestamps_to_seconds(self, mock_config):
        """Test DataFrame timestamp conversion."""
        mock_config.return_value.config = self.test_config

        loader = MOILoader()

        df = pd.DataFrame(
            {
                "start": ["00:00:21", "00:01:11", "00:05:50"],
                "end": ["00:00:28", "00:03:58", "00:06:46"],
                "alliance": [1, -1, 0],
            }
        )

        result = loader._convert_timestamps_to_seconds(df)

        self.assertIn("start_seconds", result.columns)
        self.assertIn("end_seconds", result.columns)
        self.assertEqual(result["start_seconds"].iloc[0], 21.0)
        self.assertEqual(result["end_seconds"].iloc[0], 28.0)
        self.assertEqual(result["start_seconds"].iloc[1], 71.0)
        self.assertEqual(result["end_seconds"].iloc[1], 238.0)


class TestMOIEpocher(unittest.TestCase):
    """Tests for MOIEpocher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "paths": {
                "rawdata": "data/raw",
                "derivatives": "data/derivatives",
                "logs": "log",
            },
            "epoching": {
                "methods": {
                    "fixed": {
                        "enabled": True,
                        "therapy": {"duration": 30, "overlap": 5},
                    },
                    "nsplit": {"enabled": True, "therapy": {"n_epochs": 120}},
                    "sliding": {
                        "enabled": True,
                        "therapy": {"duration": 30, "step": 5},
                    },
                }
            },
        }

    @patch("src.alliance.moi_epocher.ConfigLoader")
    @patch("src.alliance.moi_epocher.EpochAssigner")
    def test_epocher_initialization(self, mock_assigner, mock_config):
        """Test MOIEpocher initializes correctly."""
        mock_config.return_value.config = self.test_config

        epocher = MOIEpocher()

        self.assertIsNotNone(epocher.config)

    @patch("src.alliance.moi_epocher.ConfigLoader")
    @patch("src.alliance.moi_epocher.EpochAssigner")
    def test_assign_fixed_epochs_single_interval(self, mock_assigner, mock_config):
        """Test fixed epoch assignment for a single interval."""
        mock_config.return_value.config = self.test_config

        epocher = MOIEpocher()

        # Single 10-second interval starting at t=0
        start_times = np.array([0.0])
        end_times = np.array([10.0])
        duration = 100.0
        config = {"duration": 30, "overlap": 5}

        result = epocher._assign_fixed_epochs(start_times, end_times, duration, config)

        # With duration=30, overlap=5, step=25
        # Epoch 0: [0, 30], Epoch 1: [25, 55], ...
        # Interval [0, 10] should only intersect epoch 0
        self.assertEqual(len(result), 1)
        self.assertIn(0, result[0])

    @patch("src.alliance.moi_epocher.ConfigLoader")
    @patch("src.alliance.moi_epocher.EpochAssigner")
    def test_assign_fixed_epochs_spanning_multiple(self, mock_assigner, mock_config):
        """Test fixed epoch assignment for interval spanning multiple epochs."""
        mock_config.return_value.config = self.test_config

        epocher = MOIEpocher()

        # Interval from 20s to 80s
        start_times = np.array([20.0])
        end_times = np.array([80.0])
        duration = 100.0
        config = {"duration": 30, "overlap": 5}

        result = epocher._assign_fixed_epochs(start_times, end_times, duration, config)

        # With step=25, epochs are:
        # 0: [0,30], 1: [25,55], 2: [50,80], 3: [75,105]
        # [20,80] should intersect all of these
        self.assertEqual(len(result), 1)
        self.assertTrue(len(result[0]) >= 3)  # Should span at least 3 epochs

    @patch("src.alliance.moi_epocher.ConfigLoader")
    @patch("src.alliance.moi_epocher.EpochAssigner")
    def test_assign_nsplit_epochs(self, mock_assigner, mock_config):
        """Test n-split epoch assignment."""
        mock_config.return_value.config = self.test_config

        epocher = MOIEpocher()

        # Interval covering first 10% of session
        start_times = np.array([0.0])
        end_times = np.array([100.0])
        duration = 1000.0
        config = {"n_epochs": 10}

        result = epocher._assign_nsplit_epochs(start_times, end_times, duration, config)

        # Each epoch is 100s, [0,100] should intersect epoch 0 only
        self.assertEqual(len(result), 1)
        self.assertIn(0, result[0])

    @patch("src.alliance.moi_epocher.ConfigLoader")
    @patch("src.alliance.moi_epocher.EpochAssigner")
    def test_assign_sliding_epochs(self, mock_assigner, mock_config):
        """Test sliding window epoch assignment."""
        mock_config.return_value.config = self.test_config

        epocher = MOIEpocher()

        # 15-second interval
        start_times = np.array([10.0])
        end_times = np.array([25.0])
        duration = 100.0
        config = {"duration": 30, "step": 5}

        result = epocher._assign_sliding_epochs(
            start_times, end_times, duration, config
        )

        # With step=5, duration=30:
        # Epoch 0: [0,30], 1: [5,35], 2: [10,40], 3: [15,45], 4: [20,50], 5: [25,55]
        # [10,25] should intersect epochs 0,1,2,3,4
        self.assertEqual(len(result), 1)
        self.assertTrue(len(result[0]) >= 4)


class TestMOIWriter(unittest.TestCase):
    """Tests for MOIWriter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "paths": {
                "rawdata": "data/raw",
                "derivatives": "data/derivatives",
                "logs": "log",
            },
            "output": {
                "alliance_dir": "alliance",
                "alliance_subdirs": {
                    "annotations": "annotations",
                    "figures": "figures",
                },
            },
        }

    @patch("src.alliance.moi_writer.ConfigLoader")
    def test_writer_initialization(self, mock_config):
        """Test MOIWriter initializes correctly."""
        mock_config.return_value.config = self.test_config

        writer = MOIWriter()

        self.assertIsNotNone(writer.derivatives_path)
        self.assertEqual(writer.alliance_dir, "alliance")

    @patch("src.alliance.moi_writer.ConfigLoader")
    def test_save_epoched_moi(self, mock_config):
        """Test saving epoched MOI data."""
        mock_config.return_value.config = self.test_config

        with tempfile.TemporaryDirectory() as tmpdir:
            # Override derivatives path
            test_config = self.test_config.copy()
            test_config["paths"]["derivatives"] = tmpdir
            mock_config.return_value.config = test_config

            writer = MOIWriter()

            # Create test data
            df = pd.DataFrame(
                {
                    "start": ["00:00:21"],
                    "end": ["00:00:28"],
                    "start_seconds": [21.0],
                    "end_seconds": [28.0],
                    "source": ["Fille"],
                    "target": ["Thérapeute"],
                    "alliance": [1],
                    "emotion": [-1],
                    "epoch_fixed": [[0]],
                    "epoch_nsplit": [[0]],
                    "epoch_sliding": [[0, 1, 2]],
                }
            )

            metadata = {"Duration": 3600, "GroupID": "g01", "SessionID": "01"}

            output_file = writer.save_epoched_moi(df, metadata, "g01", "01")

            # Verify file was created
            self.assertTrue(output_file.exists())

            # Verify JSON sidecar was created
            json_file = output_file.with_suffix(".json")
            # The actual suffix is different, adjust
            json_file = output_file.parent / output_file.name.replace(".tsv", ".json")
            self.assertTrue(json_file.exists())

            # Verify content
            loaded_df = pd.read_csv(output_file, sep="\t")
            self.assertEqual(len(loaded_df), 1)


class TestMOIVisualizer(unittest.TestCase):
    """Tests for MOIVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "paths": {
                "rawdata": "data/raw",
                "derivatives": "data/derivatives",
                "logs": "log",
            },
            "output": {
                "alliance_dir": "alliance",
                "alliance_subdirs": {
                    "annotations": "annotations",
                    "figures": "figures",
                },
            },
        }

    @patch("src.alliance.moi_visualizer.ConfigLoader")
    def test_visualizer_initialization(self, mock_config):
        """Test MOIVisualizer initializes correctly."""
        mock_config.return_value.config = self.test_config

        visualizer = MOIVisualizer()

        self.assertIsNotNone(visualizer.colors)
        self.assertIn("positive", visualizer.colors)
        self.assertIn("negative", visualizer.colors)

    @patch("src.alliance.moi_visualizer.ConfigLoader")
    def test_compute_epoch_states(self, mock_config):
        """Test epoch state computation."""
        mock_config.return_value.config = self.test_config

        visualizer = MOIVisualizer()

        # Create test data with known epoch assignments
        df = pd.DataFrame(
            {
                "epoch_fixed": [[0], [0, 1], [2]],
                "alliance": [1, -1, 0],
                "emotion": [-1, 1, 0],
            }
        )

        alliance_states, emotion_states = visualizer.compute_epoch_states(df, "fixed")

        # Epoch 0 has both positive (1) and negative (-1) alliance -> mixed (2)
        self.assertEqual(alliance_states[0], 2)

        # Epoch 1 has only negative alliance
        self.assertEqual(alliance_states[1], -1)

        # Epoch 2 has neutral alliance
        self.assertEqual(alliance_states[2], 0)

    @patch("src.alliance.moi_visualizer.ConfigLoader")
    def test_count_states(self, mock_config):
        """Test state counting."""
        mock_config.return_value.config = self.test_config

        visualizer = MOIVisualizer()

        states = {0: 1, 1: -1, 2: 0, 3: 2, 4: 1, 5: 0}

        counts = visualizer._count_states(states)

        self.assertEqual(counts["positive"], 2)  # epochs 0, 4
        self.assertEqual(counts["negative"], 1)  # epoch 1
        self.assertEqual(counts["none"], 2)  # epochs 2, 5
        self.assertEqual(counts["mixed"], 1)  # epoch 3


class TestMOIIntegration(unittest.TestCase):
    """Integration tests for MOI pipeline."""

    @patch("src.alliance.moi_loader.ConfigLoader")
    @patch("src.alliance.moi_epocher.ConfigLoader")
    @patch("src.alliance.moi_epocher.EpochAssigner")
    def test_loader_to_epocher_flow(
        self, mock_assigner, mock_epocher_config, mock_loader_config
    ):
        """Test data flow from loader to epocher."""
        test_config = {
            "paths": {
                "rawdata": "data/raw",
                "derivatives": "data/derivatives",
                "logs": "log",
            },
            "epoching": {
                "methods": {
                    "fixed": {
                        "enabled": True,
                        "therapy": {"duration": 30, "overlap": 5},
                    },
                    "nsplit": {"enabled": True, "therapy": {"n_epochs": 120}},
                    "sliding": {
                        "enabled": True,
                        "therapy": {"duration": 30, "step": 5},
                    },
                }
            },
        }

        mock_loader_config.return_value.config = test_config
        mock_epocher_config.return_value.config = test_config

        # Create sample data as if from loader
        df = pd.DataFrame(
            {
                "start": ["00:00:21", "00:01:11"],
                "end": ["00:00:28", "00:03:58"],
                "start_seconds": [21.0, 71.0],
                "end_seconds": [28.0, 238.0],
                "source": ["Fille", "Fille"],
                "target": ["Thérapeute", "Mère"],
                "alliance": [1, -1],
                "emotion": [-1, -1],
            }
        )

        metadata = {"Duration": 3600}

        epocher = MOIEpocher()
        result = epocher.add_epoch_columns(df, metadata)

        # Verify epoch columns were added
        self.assertIn("epoch_fixed", result.columns)
        self.assertIn("epoch_nsplit", result.columns)
        self.assertIn("epoch_sliding", result.columns)

        # Verify epochs are lists
        self.assertIsInstance(result["epoch_fixed"].iloc[0], list)


if __name__ == "__main__":
    unittest.main()
