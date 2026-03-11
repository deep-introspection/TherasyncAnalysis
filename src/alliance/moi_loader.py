"""
MOI (Moments of Interest) Loader.

Loads alliance and emotion annotation files from sourcedata.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class MOILoader:
    """Loads MOI annotation files and their metadata."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize MOI loader.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.rawdata_path = Path(self.config["paths"]["rawdata"])
        self.sourcedata_path = self.rawdata_path / "sourcedata"

    def load_moi_file(self, group_id: str, session_id: str) -> Dict:
        """
        Load MOI annotation file and its metadata.

        Args:
            group_id: Group ID (e.g., 'g01')
            session_id: Session ID (e.g., '01')

        Returns:
            Dictionary containing:
                - annotations: DataFrame with annotation data
                - metadata: Dictionary with metadata from JSON sidecar
                - group_id: Group identifier
                - session_id: Session identifier

        Raises:
            FileNotFoundError: If TSV or JSON file not found
        """
        # Construct paths
        subject_dir = f"sub-{group_id}shared"
        session_dir = f"ses-{session_id}"

        tsv_filename = f"sub-{group_id}_ses-{session_id}_desc-alliance_annotations.tsv"
        json_filename = (
            f"sub-{group_id}_ses-{session_id}_desc-alliance_annotations.json"
        )

        moi_dir = self.sourcedata_path / subject_dir / session_dir / "moi_tables"
        tsv_file = moi_dir / tsv_filename
        json_file = moi_dir / json_filename

        # Check files exist
        if not tsv_file.exists():
            raise FileNotFoundError(f"MOI TSV file not found: {tsv_file}")
        if not json_file.exists():
            raise FileNotFoundError(f"MOI JSON sidecar not found: {json_file}")

        logger.info(f"Loading MOI annotations: {group_id}/ses-{session_id}")

        # Load TSV
        df = pd.read_csv(tsv_file, sep="\t")
        logger.debug(f"Loaded {len(df)} annotations from {tsv_file.name}")

        # Load JSON metadata
        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Convert timestamps to seconds
        df = self._convert_timestamps_to_seconds(df)

        return {
            "annotations": df,
            "metadata": metadata,
            "group_id": group_id,
            "session_id": session_id,
            "tsv_file": tsv_file,
            "json_file": json_file,
        }

    def _convert_timestamps_to_seconds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert HH:MM:SS timestamps to seconds.

        Args:
            df: DataFrame with 'start' and 'end' columns in HH:MM:SS format

        Returns:
            DataFrame with additional 'start_seconds' and 'end_seconds' columns
        """
        df = df.copy()

        for col in ["start", "end"]:
            if col in df.columns:
                # Convert HH:MM:SS to seconds
                seconds = df[col].apply(self._timestamp_to_seconds)
                df[f"{col}_seconds"] = seconds

        return df

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """
        Convert HH:MM:SS timestamp to seconds.

        Args:
            timestamp: String in HH:MM:SS format

        Returns:
            Total seconds as float

        Examples:
            '00:02:07' -> 127.0
            '01:08:30' -> 4110.0
        """
        if pd.isna(timestamp) or timestamp == "":
            return 0.0

        parts = timestamp.strip().split(":")
        if len(parts) != 3:
            logger.warning(f"Invalid timestamp format: {timestamp}")
            return 0.0

        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing timestamp {timestamp}: {e}")
            return 0.0

    def get_available_sessions(self) -> list:
        """
        Get list of available MOI annotation sessions.

        Returns:
            List of tuples (group_id, session_id)
        """
        sessions = []

        if not self.sourcedata_path.exists():
            logger.warning(f"Sourcedata path not found: {self.sourcedata_path}")
            return sessions

        # Find all TSV files
        tsv_files = list(
            self.sourcedata_path.glob("*/*/moi_tables/*_desc-alliance_annotations.tsv")
        )

        for tsv_file in tsv_files:
            # Parse group and session from filename
            # Format: sub-g01_ses-01_desc-alliance_annotations.tsv
            filename = tsv_file.stem  # Without extension
            parts = filename.split("_")

            if len(parts) >= 2:
                group_part = parts[0].replace("sub-", "")  # g01
                session_part = parts[1].replace("ses-", "")  # 01
                sessions.append((group_part, session_part))

        return sorted(sessions)
