"""
Temperature Data Loader for TherasyncPipeline.

This module provides functionality to load peripheral skin temperature data files
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


class TEMPLoader:
    """
    Load and validate temperature data files from BIDS-formatted Empatica recordings.

    This class handles loading peripheral skin temperature data with associated metadata,
    validates data integrity, and segments data according to configured moments
    (e.g., restingstate, therapy).

    Temperature data from Empatica E4:
    - Sampling rate: 4 Hz
    - Unit: degrees Celsius (°C)
    - Measures peripheral skin temperature
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the temperature loader with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)

        # Get paths from config
        self.rawdata_path = Path(self.config.get("paths.rawdata", "data/raw"))

        # Get temperature-specific configuration
        self.sampling_rate = self.config.get(
            "physio.temp.sampling_rate", 4
        )  # Default 4 Hz for E4

        logger.info(
            f"Temperature Loader initialized (sampling rate: {self.sampling_rate} Hz)"
        )

    def load_subject_session(
        self, subject: str, session: str, moment: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load temperature data for a specific subject/session, optionally filtered by moment.

        Args:
            subject: Subject ID (e.g., 'sub-g01p01')
            session: Session ID (e.g., 'ses-01')
            moment: Optional moment/task name (e.g., 'restingstate', 'therapy').
                   If None, loads and concatenates all moments.

        Returns:
            Tuple of:
                - DataFrame with columns ['time', 'temp']
                - Dictionary with combined metadata from JSON sidecars

        Raises:
            FileNotFoundError: If no temperature files found for subject/session
            ValueError: If data validation fails

        Example:
            >>> loader = TEMPLoader()
            >>> data, metadata = loader.load_subject_session('sub-g01p01', 'ses-01', moment='restingstate')
            >>> print(f"Loaded {len(data)} samples at {metadata['SamplingFrequency']} Hz")
        """
        logger.info(
            f"Loading temperature data: {subject}/{session}"
            + (f"/{moment}" if moment else "")
        )

        # Find all temperature files for this subject/session
        file_pairs = self.find_temp_files(subject, session)

        if not file_pairs:
            raise FileNotFoundError(
                f"No temperature files found for {subject}/{session} in {self.rawdata_path}"
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
                    f"No temperature files found for {subject}/{session}/{moment}"
                )

        # Load and combine data from all matching files
        combined_data = []
        combined_metadata = {}

        for tsv_file, json_file in file_pairs:
            # Load TSV data
            data = pd.read_csv(tsv_file, sep="\t")

            # Load JSON metadata
            with open(json_file, "r") as f:
                metadata = json.load(f)

            # Validate data structure
            self._validate_temp_data(data, metadata, tsv_file)

            # Add to combined data
            combined_data.append(data)

            # Merge metadata (first file takes precedence for conflicting keys)
            if not combined_metadata:
                combined_metadata = metadata.copy()
            else:
                # Add task-specific info
                task_name = metadata.get("TaskName", "unknown")
                combined_metadata["tasks_included"] = combined_metadata.get(
                    "tasks_included", []
                ) + [task_name]

        # Concatenate all data
        if len(combined_data) == 1:
            final_data = combined_data[0].copy()
        else:
            # For multiple files, concatenate with time offset
            final_data = self._concatenate_temp_data(combined_data, file_pairs)

        # Log summary
        duration = final_data["time"].max() - final_data["time"].min()
        temp_range = (final_data["temp"].min(), final_data["temp"].max())
        logger.info(
            f"Loaded {len(final_data)} temperature samples ({duration:.1f}s) from {len(file_pairs)} file(s)"
        )
        logger.info(f"Temperature range: {temp_range[0]:.2f} - {temp_range[1]:.2f} °C")

        # Add summary to metadata
        combined_metadata.update(
            {
                "LoadedSamples": len(final_data),
                "LoadedDuration": float(duration),
                "LoadedFiles": len(file_pairs),
                "TEMP_Range": list(temp_range),
                "TEMP_Mean": float(final_data["temp"].mean()),
            }
        )

        return final_data, combined_metadata

    def find_temp_files(self, subject: str, session: str) -> List[Tuple[Path, Path]]:
        """
        Find all temperature TSV and JSON file pairs for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-g01p01')
            session: Session ID (e.g., 'ses-01')

        Returns:
            List of tuples, each containing (tsv_path, json_path)
        """
        # Ensure subject and session have BIDS prefixes
        subject_dir = subject if subject.startswith("sub-") else f"sub-{subject}"
        session_dir = session if session.startswith("ses-") else f"ses-{session}"
        physio_dir = self.rawdata_path / subject_dir / session_dir / "physio"

        if not physio_dir.exists():
            return []

        # Find all temperature TSV files
        tsv_pattern = f"{subject_dir}_{session_dir}_*_recording-temp.tsv"
        tsv_files = list(physio_dir.glob(tsv_pattern))

        file_pairs = []
        for tsv_file in tsv_files:
            # Find corresponding JSON file
            json_file = tsv_file.with_suffix(".json")

            if json_file.exists():
                file_pairs.append((tsv_file, json_file))
            else:
                logger.warning(f"Missing JSON metadata for {tsv_file}")

        return sorted(file_pairs)

    def _validate_temp_data(
        self, data: pd.DataFrame, metadata: dict, filepath: Path
    ) -> None:
        """
        Validate temperature data structure and content.

        Args:
            data: Temperature DataFrame to validate
            metadata: Associated metadata
            filepath: Path to data file (for error messages)

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_columns = ["time", "temp"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filepath}: {missing_columns}"
            )

        # Check for empty data
        if len(data) == 0:
            raise ValueError(f"Empty temperature data in {filepath}")

        # Check for NaN values
        if data["temp"].isna().any():
            nan_count = data["temp"].isna().sum()
            logger.warning(
                f"Found {nan_count} NaN values in temperature data: {filepath}"
            )

        # Check temperature range (physiological plausibility)
        temp_min, temp_max = data["temp"].min(), data["temp"].max()
        if temp_min < 20 or temp_max > 42:
            logger.warning(
                f"Temperature values outside expected range (20-42°C): "
                f"{temp_min:.2f} - {temp_max:.2f} in {filepath}"
            )

        # Check time consistency
        if not data["time"].is_monotonic_increasing:
            raise ValueError(f"Time values not monotonically increasing in {filepath}")

        # Check sampling rate consistency
        if len(data) > 1:
            time_diffs = data["time"].diff().dropna()
            expected_interval = 1.0 / self.sampling_rate
            median_interval = time_diffs.median()

            if abs(median_interval - expected_interval) > 0.1:
                logger.warning(
                    f"Sampling rate mismatch in {filepath}: "
                    f"expected {expected_interval:.2f}s, got {median_interval:.2f}s"
                )

        logger.debug(f"Temperature data validation passed for {filepath}")

    def _concatenate_temp_data(
        self, data_list: List[pd.DataFrame], file_pairs: List[Tuple[Path, Path]]
    ) -> pd.DataFrame:
        """
        Concatenate multiple temperature data files with proper time offsets.

        Args:
            data_list: List of temperature DataFrames to concatenate
            file_pairs: Corresponding file pairs for metadata

        Returns:
            Concatenated DataFrame with continuous time axis
        """
        if len(data_list) == 1:
            return data_list[0].copy()

        concatenated_frames = []
        cumulative_time = 0.0

        for i, (data, (tsv_file, _)) in enumerate(zip(data_list, file_pairs)):
            data_copy = data.copy()

            if i > 0:
                # Offset time to continue from previous segment
                data_copy["time"] = data_copy["time"] + cumulative_time

            concatenated_frames.append(data_copy)

            # Update cumulative time for next segment
            cumulative_time = data_copy["time"].max() + (1.0 / self.sampling_rate)

            logger.debug(
                f"Added temperature segment from {tsv_file.name}: {len(data)} samples"
            )

        # Combine all frames
        final_data = pd.concat(concatenated_frames, ignore_index=True)

        logger.info(
            f"Concatenated {len(data_list)} temperature segments into {len(final_data)} total samples"
        )

        return final_data

    def get_available_moments(self, subject: str, session: str) -> List[str]:
        """
        Get list of available moments/tasks for a subject/session.

        Args:
            subject: Subject ID
            session: Session ID

        Returns:
            List of moment names (e.g., ['restingstate', 'therapy'])
        """
        file_pairs = self.find_temp_files(subject, session)

        moments = []
        for tsv_file, _ in file_pairs:
            # Extract task name from filename
            # Format: sub-g01p01_ses-01_task-restingstate_recording-temp.tsv
            filename = tsv_file.name
            if "_task-" in filename and "_recording-temp" in filename:
                task_part = filename.split("_task-")[1].split("_recording-temp")[0]
                moments.append(task_part)

        return sorted(list(set(moments)))

    def load_single_moment(
        self, subject: str, session: str, moment: str
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load temperature data for a single moment/task.

        Args:
            subject: Subject ID
            session: Session ID
            moment: Moment/task name

        Returns:
            Tuple of (DataFrame, metadata) for the specified moment

        Raises:
            FileNotFoundError: If moment not found
        """
        return self.load_subject_session(subject, session, moment=moment)
