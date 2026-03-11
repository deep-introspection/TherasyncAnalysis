"""
BIDS utilities for Therasync Pipeline.

This module provides utilities for working with BIDS (Brain Imaging Data Structure)
format, adapted for physiological data from the Therasync project.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime


logger = logging.getLogger(__name__)


class BIDSError(Exception):
    """Custom exception for BIDS-related errors."""

    pass


class BIDSUtils:
    """
    Utility class for BIDS-compliant file operations.

    Provides methods for creating BIDS-compliant file names, directory structures,
    and metadata files for physiological data.
    """

    def __init__(self, derivatives_root: str = "data/derivatives/therasync-physio"):
        """
        Initialize BIDS utilities.

        Args:
            derivatives_root: Root path for BIDS derivatives.
        """
        self.derivatives_root = Path(derivatives_root)

    def parse_subject_id(self, subject_id: str) -> Dict[str, str]:
        """
        Parse subject ID to extract family and participant information.

        Args:
            subject_id: Subject ID in format 'sub-fXXpYY' or 'fXXpYY'.

        Returns:
            Dictionary with 'family' and 'participant' keys.

        Raises:
            BIDSError: If subject ID format is invalid.
        """
        # Remove 'sub-' prefix if present
        if subject_id.startswith("sub-"):
            subject_id = subject_id[4:]

        # Expected format: fXXpYY (family XX, participant YY)
        if not subject_id.startswith("f") or "p" not in subject_id:
            raise BIDSError(
                f"Invalid subject ID format: {subject_id}. Expected format: fXXpYY"
            )

        try:
            family_part, participant_part = subject_id.split("p")
            family_id = family_part[1:]  # Remove 'f' prefix
            participant_id = participant_part

            return {
                "family": family_id,
                "participant": participant_id,
                "full_id": subject_id,
            }
        except (ValueError, IndexError):
            raise BIDSError(f"Invalid subject ID format: {subject_id}")

    def create_bids_filename(
        self,
        subject: str,
        session: str,
        task: str,
        datatype: str,
        suffix: str,
        extension: str = ".tsv",
    ) -> str:
        """
        Create a BIDS-compliant filename.

        Args:
            subject: Subject identifier (e.g., 'g01p01').
            session: Session identifier (e.g., '01').
            task: Task identifier (e.g., 'restingstate', 'therapy').
            datatype: Data type (e.g., 'physio').
            suffix: File suffix (e.g., 'bvp-clean', 'eda-metrics').
            extension: File extension (default: '.tsv').

        Returns:
            BIDS-compliant filename.
        """
        # Ensure proper prefixes
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"
        if not session.startswith("ses-"):
            session = f"ses-{session}"
        if not task.startswith("task-"):
            task = f"task-{task}"

        filename = f"{subject}_{session}_{task}_{datatype}-{suffix}{extension}"
        return filename

    def create_bids_path(
        self,
        subject: str,
        session: str,
        datatype: str = "physio",
        create_dirs: bool = True,
    ) -> Path:
        """
        Create BIDS-compliant directory path.

        Args:
            subject: Subject identifier.
            session: Session identifier.
            datatype: Data type directory name.
            create_dirs: Whether to create directories if they don't exist.

        Returns:
            Path to BIDS-compliant directory.
        """
        # Ensure proper prefixes
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"
        if not session.startswith("ses-"):
            session = f"ses-{session}"

        path = self.derivatives_root / subject / session / datatype

        if create_dirs:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created BIDS directory: {path}")

        return path

    def save_tsv_file(
        self,
        data: pd.DataFrame,
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save data as TSV file with optional JSON metadata.

        Args:
            data: DataFrame to save.
            filepath: Path for the TSV file.
            metadata: Optional metadata dictionary for JSON sidecar.
        """
        try:
            # Save TSV file
            data.to_csv(filepath, sep="\t", index=False, float_format="%.6f")
            logger.info(f"Saved TSV file: {filepath}")

            # Save JSON metadata if provided
            if metadata:
                json_path = filepath.with_suffix(".json")
                self.save_json_metadata(metadata, json_path)

        except Exception as e:
            raise BIDSError(f"Error saving TSV file {filepath}: {e}")

    def save_json_metadata(self, metadata: Dict[str, Any], filepath: Path) -> None:
        """
        Save metadata as JSON file.

        Args:
            metadata: Metadata dictionary.
            filepath: Path for the JSON file.
        """
        try:
            # Add generation timestamp
            metadata_with_timestamp = metadata.copy()
            metadata_with_timestamp["GeneratedOn"] = datetime.now().isoformat()
            metadata_with_timestamp["GeneratedBy"] = "TherasyncPipeline"

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata_with_timestamp, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved JSON metadata: {filepath}")

        except Exception as e:
            raise BIDSError(f"Error saving JSON metadata {filepath}: {e}")

    def load_tsv_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load TSV file as DataFrame.

        Args:
            filepath: Path to TSV file.

        Returns:
            Loaded DataFrame.
        """
        try:
            return pd.read_csv(filepath, sep="\t")
        except Exception as e:
            raise BIDSError(f"Error loading TSV file {filepath}: {e}")

    def load_json_metadata(self, filepath: Path) -> Dict[str, Any]:
        """
        Load JSON metadata file.

        Args:
            filepath: Path to JSON file.

        Returns:
            Metadata dictionary.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise BIDSError(f"Error loading JSON metadata {filepath}: {e}")

    def create_dataset_description(
        self,
        name: str = "Therasync Physiological Preprocessing",
        version: str = "1.0.0",
        description: str = "Preprocessed physiological data from family therapy sessions",
    ) -> None:
        """
        Create dataset_description.json file for the derivatives dataset.

        Args:
            name: Dataset name.
            version: Dataset version.
            description: Dataset description.
        """
        dataset_description = {
            "Name": name,
            "BIDSVersion": "1.7.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "TherasyncPipeline",
                    "Version": version,
                    "Description": description,
                }
            ],
            "SourceDatasets": [{"URL": "local", "Version": "1.0.0"}],
        }

        desc_path = self.derivatives_root / "dataset_description.json"
        self.save_json_metadata(dataset_description, desc_path)

    def find_source_files(
        self,
        subject: str,
        session: str,
        task: str,
        signal_type: str,
        source_root: str = "data",
    ) -> Dict[str, Path]:
        """
        Find source data files for processing.

        Args:
            subject: Subject identifier.
            session: Session identifier.
            task: Task identifier.
            signal_type: Signal type (e.g., 'bvp', 'eda', 'hr').
            source_root: Root directory for source data.

        Returns:
            Dictionary with 'tsv' and 'json' file paths.
        """
        # Ensure proper prefixes
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"
        if not session.startswith("ses-"):
            session = f"ses-{session}"
        if not task.startswith("task-"):
            task = f"task-{task}"

        source_path = Path(source_root) / subject / session / "physio"

        # Expected filename pattern
        base_name = f"{subject}_{session}_{task}_recording-{signal_type}"

        tsv_file = source_path / f"{base_name}.tsv"
        json_file = source_path / f"{base_name}.json"

        result = {}
        if tsv_file.exists():
            result["tsv"] = tsv_file
        if json_file.exists():
            result["json"] = json_file

        if not result:
            raise BIDSError(
                f"No source files found for {subject} {session} {task} {signal_type}"
            )

        return result

    def validate_bids_structure(self, check_path: Optional[Path] = None) -> List[str]:
        """
        Validate BIDS structure and return any issues found.

        Args:
            check_path: Path to check. If None, checks derivatives_root.

        Returns:
            List of validation issues (empty if valid).
        """
        if check_path is None:
            check_path = self.derivatives_root

        issues = []

        # Check if derivatives root exists
        if not check_path.exists():
            issues.append(f"Derivatives root does not exist: {check_path}")
            return issues

        # Check for dataset_description.json
        desc_file = check_path / "dataset_description.json"
        if not desc_file.exists():
            issues.append("Missing dataset_description.json")

        # Check subject directories
        for subject_dir in check_path.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith("sub-"):
                # Check session directories
                for session_dir in subject_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith("ses-"):
                        # Check datatype directories
                        physio_dir = session_dir / "physio"
                        if not physio_dir.exists():
                            issues.append(f"Missing physio directory: {physio_dir}")

        return issues


def create_bids_filename(
    subject: str,
    session: str,
    task: str,
    datatype: str,
    suffix: str,
    extension: str = ".tsv",
) -> str:
    """
    Convenience function to create BIDS-compliant filename.

    Args:
        subject: Subject identifier.
        session: Session identifier.
        task: Task identifier.
        datatype: Data type.
        suffix: File suffix.
        extension: File extension.

    Returns:
        BIDS-compliant filename.
    """
    bids_utils = BIDSUtils()
    return bids_utils.create_bids_filename(
        subject, session, task, datatype, suffix, extension
    )


if __name__ == "__main__":
    # Example usage
    bids_utils = BIDSUtils()

    # Create example filename
    filename = bids_utils.create_bids_filename(
        subject="g01p01",
        session="01",
        task="restingstate",
        datatype="physio",
        suffix="bvp-clean",
    )
    print(f"Example filename: {filename}")

    # Parse subject ID
    subject_info = bids_utils.parse_subject_id("g01p01")
    print(f"Subject info: {subject_info}")

    # Create BIDS path
    path = bids_utils.create_bids_path("g01p01", "01", create_dirs=False)
    print(f"BIDS path: {path}")
