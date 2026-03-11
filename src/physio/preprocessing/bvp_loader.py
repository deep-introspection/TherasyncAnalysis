"""
BVP Data Loader for TherasyncPipeline.

This module provides functionality to load Blood Volume Pulse (BVP) data files
from Empatica devices in BIDS format and prepare them for preprocessing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader
from src.core.bids_utils import BIDSUtils


logger = logging.getLogger(__name__)


class BVPLoader:
    """
    Load and validate BVP data files from BIDS-formatted Empatica recordings.

    This class handles loading BVP data with associated metadata, validates data
    integrity, and segments data according to configured moments (e.g., resting_state, therapy).
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BVP loader with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        self.bids_utils = BIDSUtils()

        # Get BVP-specific configuration
        self.bvp_config = self.config.get("physio.bvp", {})
        self.moments_config = self.config.get("moments", [])

        logger.info("BVP Loader initialized")

    def load_subject_session_data(
        self, subject_id: str, session_id: str, moments: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Load all BVP data for a subject and session across specified moments.

        Args:
            subject_id: Subject identifier (e.g., 'sub-g01p01')
            session_id: Session identifier (e.g., 'ses-01')
            moments: List of moment names to load. If None, loads all configured moments.

        Returns:
            Dictionary with moment names as keys and loaded data as values.
            Each moment contains: {'data': DataFrame, 'metadata': Dict}

        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data validation fails
        """
        if moments is None:
            moments = [moment["name"] for moment in self.moments_config]

        logger.info(
            f"Loading BVP data for {subject_id}, {session_id}, moments: {moments}"
        )

        loaded_data = {}

        for moment in moments:
            try:
                data, metadata = self.load_moment_data(subject_id, session_id, moment)
                loaded_data[moment] = {"data": data, "metadata": metadata}
                logger.debug(f"Successfully loaded {moment} data: {len(data)} samples")

            except FileNotFoundError as e:
                logger.warning(
                    f"Data not found for {subject_id}/{session_id}/{moment}: {e}"
                )
                continue
            except Exception as e:
                logger.error(f"Error loading {subject_id}/{session_id}/{moment}: {e}")
                raise

        if not loaded_data:
            raise FileNotFoundError(f"No BVP data found for {subject_id}/{session_id}")

        return loaded_data

    def load_moment_data(
        self, subject_id: str, session_id: str, moment: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load BVP data for a specific moment (task).

        Args:
            subject_id: Subject identifier (e.g., 'sub-g01p01')
            session_id: Session identifier (e.g., 'ses-01')
            moment: Moment name (e.g., 'restingstate', 'therapy')

        Returns:
            Tuple of (data_dataframe, metadata_dict)

        Raises:
            FileNotFoundError: If TSV or JSON files are not found
            ValueError: If data validation fails
        """
        # Construct file paths using BIDS conventions
        # Ensure subject and session have BIDS prefixes
        subject_dir = (
            subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
        )
        session_dir = (
            session_id if session_id.startswith("ses-") else f"ses-{session_id}"
        )
        data_dir = (
            Path(self.config.get("paths.rawdata"))
            / subject_dir
            / session_dir
            / "physio"
        )

        # BIDS filename pattern: sub-{subject}_ses-{session}_task-{task}_recording-bvp.{ext}
        base_filename = f"{subject_dir}_{session_dir}_task-{moment}_recording-bvp"
        tsv_file = data_dir / f"{base_filename}.tsv"
        json_file = data_dir / f"{base_filename}.json"

        logger.debug(f"Loading BVP files: {tsv_file}, {json_file}")

        # Check if files exist
        if not tsv_file.exists():
            raise FileNotFoundError(f"BVP data file not found: {tsv_file}")
        if not json_file.exists():
            raise FileNotFoundError(f"BVP metadata file not found: {json_file}")

        # Load metadata
        with open(json_file, "r") as f:
            metadata = json.load(f)

        # Load data
        data = pd.read_csv(tsv_file, sep="\t")

        # Validate data structure
        self._validate_data_structure(data, metadata, tsv_file)

        # Validate sampling frequency
        self._validate_sampling_frequency(data, metadata)

        # Add computed fields
        metadata["duration_seconds"] = len(data) / metadata["SamplingFrequency"]
        metadata["num_samples"] = len(data)
        metadata["file_paths"] = {"data": str(tsv_file), "metadata": str(json_file)}

        logger.info(
            f"Loaded BVP data: {len(data)} samples, "
            f"{metadata['duration_seconds']:.1f}s @ {metadata['SamplingFrequency']}Hz"
        )

        return data, metadata

    def _validate_data_structure(
        self, data: pd.DataFrame, metadata: Dict, file_path: Path
    ) -> None:
        """
        Validate the structure and content of loaded BVP data.

        Args:
            data: Loaded DataFrame
            metadata: Associated metadata dictionary
            file_path: Path to the data file (for error reporting)

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_columns = ["time", "bvp"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file_path}: {missing_columns}"
            )

        # Check for empty data
        if len(data) == 0:
            raise ValueError(f"Empty data file: {file_path}")

        # Check for NaN values in critical columns
        if data["time"].isna().any():
            raise ValueError(f"NaN values found in time column: {file_path}")

        if data["bvp"].isna().any():
            logger.warning(f"NaN values found in BVP data: {file_path}")

        # Check time column monotonicity
        if not data["time"].is_monotonic_increasing:
            raise ValueError(
                f"Time column is not monotonically increasing: {file_path}"
            )

        # Check metadata consistency with data columns
        expected_columns = metadata.get("Columns", [])
        if expected_columns and expected_columns != list(data.columns):
            logger.warning(
                f"Column mismatch in {file_path}: "
                f"expected {expected_columns}, found {list(data.columns)}"
            )

    def _validate_sampling_frequency(self, data: pd.DataFrame, metadata: Dict) -> None:
        """
        Validate the sampling frequency against the actual data.

        Args:
            data: Loaded DataFrame with time column
            metadata: Metadata dictionary with SamplingFrequency

        Raises:
            ValueError: If sampling frequency validation fails
        """
        if len(data) < 2:
            return  # Cannot validate with less than 2 samples

        # Calculate actual sampling frequency from time differences
        time_diffs = np.diff(data["time"].values)
        median_interval = np.median(time_diffs)
        actual_freq = 1.0 / median_interval if median_interval > 0 else 0

        expected_freq = metadata.get("SamplingFrequency", 0)

        # Allow 5% tolerance
        tolerance = 0.05
        if abs(actual_freq - expected_freq) / expected_freq > tolerance:
            logger.warning(
                f"Sampling frequency mismatch: "
                f"expected {expected_freq}Hz, actual {actual_freq:.2f}Hz"
            )

    def get_available_data(
        self, subject_pattern: str = "*"
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan for available BVP data files in the sourcedata directory.

        Args:
            subject_pattern: Pattern to match subject directories (default: all subjects)

        Returns:
            Nested dictionary: {subject: {session: [moments]}}
        """
        rawdata_path = Path(self.config.get("paths.rawdata"))
        available_data = {}

        for subject_dir in rawdata_path.glob(subject_pattern):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name
            available_data[subject_id] = {}

            for session_dir in subject_dir.glob("ses-*"):
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name
                physio_dir = session_dir / "physio"

                if not physio_dir.exists():
                    continue

                # Find all BVP task files
                bvp_files = list(physio_dir.glob("*_recording-bvp.tsv"))
                moments = []

                for bvp_file in bvp_files:
                    # Extract task name from filename
                    # Pattern: sub-{subject}_ses-{session}_task-{task}_recording-bvp.tsv
                    parts = bvp_file.stem.split("_")
                    task_part = [part for part in parts if part.startswith("task-")]
                    if task_part:
                        moment = task_part[0].replace("task-", "")
                        moments.append(moment)

                if moments:
                    available_data[subject_id][session_id] = sorted(moments)

        logger.info(f"Found BVP data for {len(available_data)} subjects")
        return available_data
