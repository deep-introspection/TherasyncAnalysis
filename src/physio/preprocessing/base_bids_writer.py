"""
Base BIDS Writer for TherasyncPipeline.

This module provides an abstract base class for all physiological modality writers,
ensuring consistent API and BIDS compliance across BVP, EDA, and HR modalities.

Authors: Lena Adel, Remy Ramadour
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader
from src.core.bids_utils import BIDSUtils


logger = logging.getLogger(__name__)


class PhysioBIDSWriter(ABC):
    """
    Abstract base class for all physiological modality BIDS writers.

    This class defines the common interface and shared functionality for writing
    processed physiological data (BVP, EDA, HR) in BIDS-compliant format.

    All modality-specific writers (BVPBIDSWriter, EDABIDSWriter, HRBIDSWriter)
    must inherit from this class and implement the required abstract methods.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BIDS writer with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        self.bids_utils = BIDSUtils()

        # Get output configuration
        derivatives_dir = Path(self.config.get("paths.derivatives", "data/derivatives"))
        preprocessing_dir = self.config.get("output.preprocessing_dir", "preprocessing")

        # Get modality-specific subdirectory (must be set by subclass)
        self.modality = self._get_modality_name()
        modality_subdir = self.config.get(
            f"output.modality_subdirs.{self.modality}", self.modality
        )

        # Store base directories
        self.derivatives_base = derivatives_dir
        self.preprocessing_dir = preprocessing_dir
        self.modality_subdir = modality_subdir

        # Pipeline metadata
        self.pipeline_name = f"therasync-{self.modality}"
        self.pipeline_version = "1.0.0"

        # Ensure base preprocessing directory exists
        preprocessing_base = derivatives_dir / preprocessing_dir
        preprocessing_base.mkdir(parents=True, exist_ok=True)

        # Create dataset description for derivatives
        self._create_dataset_description()

        logger.info(
            f"{self.modality.upper()} BIDS Writer initialized "
            f"(output: {derivatives_dir}/{preprocessing_dir}/sub-{{subject}}/ses-{{session}}/{modality_subdir}/)"
        )

    @abstractmethod
    def _get_modality_name(self) -> str:
        """
        Get the modality name for this writer.

        Returns:
            Modality name (e.g., 'bvp', 'eda', 'hr')
        """
        pass

    @abstractmethod
    def save_processed_data(
        self,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, pd.DataFrame],
        session_metrics: pd.DataFrame,
        processing_metadata: Optional[Dict] = None,
    ) -> Dict[str, List[Path]]:
        """
        Save processed data and metrics in BIDS format.

        This is the main entry point for all writers. Subclasses must implement
        this method to save their modality-specific data.

        Args:
            subject_id: Subject identifier (format: 'sub-g01p01')
            session_id: Session identifier (format: 'ses-01')
            processed_results: Dictionary mapping moment names to processed DataFrames
                             Keys: moment names (e.g., 'restingstate', 'therapy')
                             Values: DataFrames with processed signals
            session_metrics: DataFrame with extracted metrics (rows = moments)
            processing_metadata: Optional additional metadata about processing

        Returns:
            Dictionary mapping file categories to lists of created file paths
            Example: {
                'processed_signals': [Path(...), Path(...)],
                'events': [Path(...)],
                'metrics': [Path(...), Path(...)],
                'metadata': [Path(...)],
                'summary': [Path(...)]
            }
        """
        pass

    def _get_subject_session_dir(self, subject_id: str, session_id: str) -> Path:
        """
        Get the output directory for a subject/session.

        Args:
            subject_id: Subject identifier (format: 'sub-g01p01')
            session_id: Session identifier (format: 'ses-01')

        Returns:
            Path to modality subdirectory
        """
        # Ensure IDs have correct prefixes
        subject_id = self._ensure_prefix(subject_id, "sub-")
        session_id = self._ensure_prefix(session_id, "ses-")

        # Create directory structure: derivatives/preprocessing/sub-xxx/ses-yyy/{modality}/
        output_dir = (
            self.derivatives_base
            / self.preprocessing_dir
            / subject_id
            / session_id
            / self.modality_subdir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def _ensure_prefix(self, identifier: str, prefix: str) -> str:
        """
        Ensure identifier has the correct prefix.

        Args:
            identifier: Identifier string (e.g., 'g01p01' or 'sub-g01p01')
            prefix: Required prefix (e.g., 'sub-' or 'ses-')

        Returns:
            Identifier with prefix
        """
        if not identifier.startswith(prefix):
            return f"{prefix}{identifier}"
        return identifier

    def _strip_prefix(self, identifier: str, prefix: str) -> str:
        """
        Remove prefix from identifier.

        Args:
            identifier: Identifier string (e.g., 'sub-g01p01')
            prefix: Prefix to remove (e.g., 'sub-')

        Returns:
            Identifier without prefix (e.g., 'g01p01')
        """
        if identifier.startswith(prefix):
            return identifier[len(prefix) :]
        return identifier

    def _create_dataset_description(self) -> None:
        """Create dataset_description.json for derivatives directory."""
        dataset_desc_file = (
            self.derivatives_base / self.preprocessing_dir / "dataset_description.json"
        )

        if dataset_desc_file.exists():
            return  # Already created

        dataset_description = {
            "Name": f"TherasyncPipeline {self.modality.upper()} Preprocessing Derivatives",
            "BIDSVersion": "1.7.0",
            "PipelineDescription": {
                "Name": self.pipeline_name,
                "Version": self.pipeline_version,
                "CodeURL": "https://github.com/Ramdam17/TherasyncAnalysis",
            },
            "GeneratedBy": [
                {
                    "Name": self.pipeline_name,
                    "Version": self.pipeline_version,
                    "Description": f"Automated preprocessing pipeline for {self.modality.upper()} physiological signals",
                }
            ],
            "SourceDatasets": [{"URL": "local", "Version": "1.0.0"}],
        }

        with open(dataset_desc_file, "w") as f:
            json.dump(dataset_description, f, indent=2)

        logger.debug(f"Created dataset_description.json at {dataset_desc_file}")

    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for numpy/pandas types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif pd.isna(obj):
            return None

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _create_base_metadata(self, subject_id: str, session_id: str) -> Dict:
        """
        Create base metadata common to all files.

        Args:
            subject_id: Subject identifier
            session_id: Session identifier

        Returns:
            Dictionary with base metadata
        """
        return {
            "Subject": self._strip_prefix(subject_id, "sub-"),
            "Session": self._strip_prefix(session_id, "ses-"),
            "Modality": self.modality.upper(),
            "Pipeline": {"Name": self.pipeline_name, "Version": self.pipeline_version},
            "ProcessingDate": datetime.now().isoformat(),
        }

    def _save_json_sidecar(self, json_path: Path, metadata: Dict) -> Path:
        """
        Save JSON sidecar file.

        Args:
            json_path: Path to JSON file
            metadata: Metadata dictionary

        Returns:
            Path to created file
        """
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=self._json_serializer)

        logger.debug(f"Saved JSON sidecar: {json_path}")
        return json_path

    def _save_tsv_file(
        self, tsv_path: Path, data: pd.DataFrame, compress: bool = False
    ) -> Path:
        """
        Save TSV file.

        Args:
            tsv_path: Path to TSV file
            data: DataFrame to save
            compress: Whether to compress (adds .gz extension)

        Returns:
            Path to created file
        """
        if compress:
            tsv_path = Path(str(tsv_path) + ".gz")
            data.to_csv(
                tsv_path, sep="\t", index=False, na_rep="n/a", compression="gzip"
            )
        else:
            data.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")

        logger.debug(f"Saved TSV file: {tsv_path}")
        return tsv_path

    def _add_epoch_columns(
        self, df: pd.DataFrame, task: str, time_column: str = "time"
    ) -> pd.DataFrame:
        """
        Add epoch ID columns to DataFrame if epoching is enabled in preprocessing mode.

        This method checks the configuration and, if epoching is enabled with
        mode='preprocessing', adds epoch columns using the EpochAssigner.

        Args:
            df: DataFrame with time series data
            task: Task/moment name (e.g., 'restingstate', 'therapy')
            time_column: Name of time column (default: 'time')

        Returns:
            DataFrame with added epoch columns (if enabled), or original DataFrame

        Note:
            Epoch columns are named automatically based on method and parameters:
            - epoch_fixed_duration{X}s_overlap{Y}s
            - epoch_nsplit{N}
            - epoch_sliding_duration{X}s_step{Y}s
        """
        epoching_config = self.config.get("epoching", {})

        # Check if epoching is enabled and in preprocessing mode
        if not epoching_config.get("enabled", False):
            logger.debug("Epoching not enabled, skipping epoch column addition")
            return df

        mode = epoching_config.get("mode", "separate")
        if mode != "preprocessing":
            logger.debug(
                f"Epoching mode is '{mode}', not 'preprocessing', skipping epoch column addition"
            )
            return df

        # Import EpochAssigner lazily (avoid circular imports)
        from src.physio.epoching.epoch_assigner import EpochAssigner

        logger.info(f"Adding epoch columns for task '{task}' (mode: preprocessing)")

        try:
            assigner = EpochAssigner(self.config.config_path)
            df_with_epochs = assigner.assign_all_epochs(df, task, time_column)

            # Count added columns
            epoch_cols = [c for c in df_with_epochs.columns if c.startswith("epoch_")]
            logger.info(
                f"Added {len(epoch_cols)} epoch columns: {', '.join(epoch_cols)}"
            )

            return df_with_epochs

        except Exception as e:
            logger.error(f"Failed to add epoch columns for task '{task}': {e}")
            logger.warning("Returning DataFrame without epoch columns")
            return df
