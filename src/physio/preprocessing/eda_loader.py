"""
EDA Data Loader for TherasyncPipeline.

This module provides functionality to load Electrodermal Activity (EDA) data files
from Empatica devices in BIDS format and prepare them for preprocessing.

Authors: Lena Adel, Remy Ramadour
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class EDALoader:
    """
    Load and validate EDA data files from BIDS-formatted Empatica recordings.

    This class handles loading EDA (skin conductance) data with associated metadata,
    validates data integrity, and segments data according to configured moments
    (e.g., restingstate, therapy).

    EDA data from Empatica E4:
    - Sampling rate: 4 Hz
    - Unit: microsiemens (μS)
    - Measures skin conductance response (SCR)
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the EDA loader with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)

        # Get paths from config
        self.rawdata_path = Path(self.config.get("paths.rawdata", "data/raw"))

        # Get EDA-specific configuration
        self.sampling_rate = self.config.get(
            "physio.eda.sampling_rate", 4
        )  # Default 4 Hz for E4

        logger.info(f"EDA Loader initialized (sampling rate: {self.sampling_rate} Hz)")

    def load_subject_session(
        self, subject: str, session: str, moment: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load EDA data for a specific subject/session, optionally filtered by moment.

        Args:
            subject: Subject ID (e.g., 'sub-g01p01')
            session: Session ID (e.g., 'ses-01')
            moment: Optional moment/task name (e.g., 'restingstate', 'therapy').
                   If None, loads and concatenates all moments.

        Returns:
            Tuple of:
                - DataFrame with columns ['time', 'eda']
                - Dictionary with combined metadata from JSON sidecars

        Raises:
            FileNotFoundError: If no EDA files found for subject/session
            ValueError: If data validation fails

        Example:
            >>> loader = EDALoader()
            >>> data, metadata = loader.load_subject_session('sub-g01p01', 'ses-01', moment='restingstate')
            >>> print(f"Loaded {len(data)} samples at {metadata['SamplingFrequency']} Hz")
        """
        logger.info(
            f"Loading EDA data: {subject}/{session}" + (f"/{moment}" if moment else "")
        )

        # Find all EDA files for this subject/session
        file_pairs = self.find_eda_files(subject, session)

        if not file_pairs:
            raise FileNotFoundError(
                f"No EDA files found for {subject}/{session} in {self.rawdata_path}"
            )

        # Filter by moment if specified
        if moment:
            file_pairs = [
                (tsv, json_file)
                for tsv, json_file in file_pairs
                if f"_task-{moment}_" in tsv.name
            ]

            if not file_pairs:
                raise FileNotFoundError(
                    f"No EDA files found for {subject}/{session}/task-{moment}"
                )

        # Load all matching files
        all_data = []
        all_metadata = {}

        for tsv_file, json_file in file_pairs:
            data, metadata = self._load_single_recording(tsv_file, json_file)
            all_data.append(data)

            # Merge metadata (first file's metadata takes precedence for common keys)
            if not all_metadata:
                all_metadata = metadata.copy()
            else:
                # Update with non-conflicting keys
                for key, value in metadata.items():
                    if key not in all_metadata:
                        all_metadata[key] = value

        # Concatenate all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Reset time to start from 0
        if len(combined_data) > 0:
            combined_data["time"] = (
                combined_data["time"] - combined_data["time"].iloc[0]
            )

        logger.info(
            f"Loaded {len(combined_data)} EDA samples "
            f"({len(combined_data) / self.sampling_rate:.1f}s) from {len(file_pairs)} file(s)"
        )

        return combined_data, all_metadata

    def find_eda_files(self, subject: str, session: str) -> List[Tuple[Path, Path]]:
        """
        Find all EDA TSV/JSON file pairs for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-g01p01')
            session: Session ID (e.g., 'ses-01')

        Returns:
            List of tuples (tsv_path, json_path) for each EDA recording

        Raises:
            FileNotFoundError: If physio directory doesn't exist

        Example:
            >>> loader = EDALoader()
            >>> files = loader.find_eda_files('sub-g01p01', 'ses-01')
            >>> print(f"Found {len(files)} EDA recordings")
        """
        # Construct path to physio directory
        # Ensure subject and session have BIDS prefixes
        subject_dir = subject if subject.startswith("sub-") else f"sub-{subject}"
        session_dir = session if session.startswith("ses-") else f"ses-{session}"
        physio_dir = self.rawdata_path / subject_dir / session_dir / "physio"

        if not physio_dir.exists():
            raise FileNotFoundError(f"Physio directory not found: {physio_dir}")

        # Find all EDA TSV files
        eda_tsv_files = list(
            physio_dir.glob(f"{subject_dir}_{session_dir}_*_recording-eda.tsv")
        )

        if not eda_tsv_files:
            logger.warning(f"No EDA files found in {physio_dir}")
            return []

        # Pair each TSV with its JSON sidecar
        file_pairs = []
        for tsv_file in eda_tsv_files:
            json_file = tsv_file.with_suffix(".json")

            if not json_file.exists():
                logger.warning(f"JSON sidecar not found for {tsv_file.name}, skipping")
                continue

            file_pairs.append((tsv_file, json_file))

        logger.debug(f"Found {len(file_pairs)} EDA file pairs in {physio_dir}")

        return file_pairs

    def _load_single_recording(
        self, tsv_file: Path, json_file: Path
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load a single EDA recording from TSV and JSON files.

        Args:
            tsv_file: Path to TSV data file
            json_file: Path to JSON metadata file

        Returns:
            Tuple of (DataFrame, metadata dict)

        Raises:
            ValueError: If data structure is invalid
        """
        logger.debug(f"Loading {tsv_file.name}")

        # Load metadata
        with open(json_file, "r") as f:
            metadata = json.load(f)

        # Load data
        data = pd.read_csv(tsv_file, sep="\t")

        # Validate data structure
        self._validate_data_structure(data, metadata, tsv_file)

        # Validate sampling frequency
        self._validate_sampling_frequency(data, metadata)

        return data, metadata

    def _validate_data_structure(
        self, data: pd.DataFrame, metadata: dict, file_path: Path
    ) -> None:
        """
        Validate that EDA data has correct structure.

        Args:
            data: DataFrame to validate
            metadata: Associated metadata
            file_path: Path to file (for error messages)

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_columns = ["time", "eda"]
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file_path.name}: {missing_columns}. "
                f"Found columns: {list(data.columns)}"
            )

        # Check for empty data
        if len(data) == 0:
            raise ValueError(f"No data found in {file_path.name}")

        # Check for NaN values
        if data["eda"].isna().any():
            n_nan = data["eda"].isna().sum()
            logger.warning(
                f"{file_path.name} contains {n_nan} NaN values in EDA column "
                f"({n_nan / len(data) * 100:.1f}%)"
            )

        # Check metadata required fields
        required_metadata = ["SamplingFrequency", "Columns"]
        missing_metadata = set(required_metadata) - set(metadata.keys())

        if missing_metadata:
            logger.warning(
                f"Missing metadata fields in {file_path.with_suffix('.json').name}: "
                f"{missing_metadata}"
            )

    def _validate_sampling_frequency(self, data: pd.DataFrame, metadata: dict) -> None:
        """
        Validate that actual sampling frequency matches expected.

        Args:
            data: DataFrame with time column
            metadata: Metadata dict with SamplingFrequency

        Raises:
            ValueError: If sampling frequency mismatch is significant
        """
        if len(data) < 2:
            return  # Cannot validate with <2 samples

        # Calculate actual sampling rate from time differences
        time_diffs = data["time"].diff().dropna()
        median_diff = time_diffs.median()
        actual_rate = 1 / median_diff

        # Get expected rate
        expected_rate = metadata.get("SamplingFrequency", self.sampling_rate)

        # Allow 5% tolerance
        tolerance = 0.05
        if abs(actual_rate - expected_rate) / expected_rate > tolerance:
            logger.warning(
                f"Sampling rate mismatch: expected {expected_rate} Hz, "
                f"got {actual_rate:.2f} Hz (difference: {abs(actual_rate - expected_rate):.2f} Hz)"
            )

    def get_moment_duration(self, data: pd.DataFrame) -> float:
        """
        Calculate duration of EDA recording in seconds.

        Args:
            data: DataFrame with time column

        Returns:
            Duration in seconds

        Example:
            >>> duration = loader.get_moment_duration(data)
            >>> print(f"Recording duration: {duration:.1f} seconds")
        """
        if len(data) == 0:
            return 0.0

        return data["time"].max() - data["time"].min()

    def get_data_info(self, data: pd.DataFrame, metadata: dict) -> dict:
        """
        Get summary information about loaded EDA data.

        Args:
            data: Loaded EDA DataFrame
            metadata: Associated metadata

        Returns:
            Dictionary with data summary information

        Example:
            >>> info = loader.get_data_info(data, metadata)
            >>> print(f"Samples: {info['num_samples']}, Duration: {info['duration_seconds']:.1f}s")
        """
        duration = self.get_moment_duration(data)

        return {
            "num_samples": len(data),
            "duration_seconds": duration,
            "sampling_rate": metadata.get("SamplingFrequency", self.sampling_rate),
            "eda_mean": data["eda"].mean() if len(data) > 0 else 0,
            "eda_std": data["eda"].std() if len(data) > 0 else 0,
            "eda_min": data["eda"].min() if len(data) > 0 else 0,
            "eda_max": data["eda"].max() if len(data) > 0 else 0,
            "nan_count": data["eda"].isna().sum(),
            "nan_percentage": (data["eda"].isna().sum() / len(data) * 100)
            if len(data) > 0
            else 0,
        }
