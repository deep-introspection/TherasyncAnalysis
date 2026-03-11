"""
MOI (Moments of Interest) Writer.

Saves epoched MOI annotations to derivatives directory.

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


class MOIWriter:
    """Writes epoched MOI annotations to derivatives."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize MOI writer.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.derivatives_path = Path(self.config["paths"]["derivatives"])
        self.alliance_dir = self.config["output"]["alliance_dir"]

    def save_epoched_moi(
        self, df: pd.DataFrame, metadata: Dict, group_id: str, session_id: str
    ) -> Path:
        """
        Save epoched MOI annotations.

        Args:
            df: DataFrame with epoched annotations
            metadata: Original metadata dictionary
            group_id: Group ID (e.g., 'g01')
            session_id: Session ID (e.g., '01')

        Returns:
            Path to saved TSV file
        """
        # Construct output directory
        subject_dir = f"sub-{group_id}shared"
        session_dir = f"ses-{session_id}"
        annotations_subdir = self.config["output"]["alliance_subdirs"]["annotations"]

        output_dir = (
            self.derivatives_path
            / self.alliance_dir
            / subject_dir
            / session_dir
            / annotations_subdir
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct filenames
        base_filename = (
            f"sub-{group_id}_ses-{session_id}_desc-alliance_annotations_epoched"
        )
        tsv_file = output_dir / f"{base_filename}.tsv"
        json_file = output_dir / f"{base_filename}.json"

        # Save TSV
        logger.info(f"Saving epoched MOI annotations to {tsv_file}")
        df.to_csv(tsv_file, sep="\t", index=False)

        # Update metadata with epoching info
        updated_metadata = metadata.copy()
        updated_metadata["Epoched"] = True
        updated_metadata["EpochingMethods"] = {
            "fixed": "Fixed window with overlap",
            "nsplit": "N-split equal duration epochs",
            "sliding": "Sliding window",
        }
        updated_metadata["EpochColumns"] = [
            "epoch_fixed",
            "epoch_nsplit",
            "epoch_sliding",
        ]
        updated_metadata["TimestampColumns"] = {
            "start": "Original start timestamp (HH:MM:SS)",
            "end": "Original end timestamp (HH:MM:SS)",
            "start_seconds": "Start time in seconds from session start",
            "end_seconds": "End time in seconds from session start",
        }

        # Add column info for new columns
        if "ColumnDescriptions" not in updated_metadata:
            updated_metadata["ColumnDescriptions"] = {}

        updated_metadata["ColumnDescriptions"].update(
            {
                "start_seconds": "Start time converted to seconds",
                "end_seconds": "End time converted to seconds",
                "epoch_fixed": "Fixed window epoch ID",
                "epoch_nsplit": "N-split epoch ID",
                "epoch_sliding": "Sliding window epoch ID",
            }
        )

        # Save JSON sidecar
        logger.info(f"Saving updated metadata to {json_file}")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(updated_metadata, f, indent=4, ensure_ascii=False)

        logger.info(f"✓ Saved epoched MOI file with {len(df)} annotations")

        return tsv_file
