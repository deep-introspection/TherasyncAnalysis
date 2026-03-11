"""
MOI (Moments of Interest) Epocher.

Applies epoching to MOI annotations using the same methods as physiological data.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader
from src.physio.epoching.epoch_assigner import EpochAssigner

logger = logging.getLogger(__name__)


class MOIEpocher:
    """Applies epoching to MOI annotations."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize MOI epocher.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.epoch_assigner = EpochAssigner(config_path)

    def add_epoch_columns(
        self, df: pd.DataFrame, metadata: Dict, task: str = "therapy"
    ) -> pd.DataFrame:
        """
        Add epoch columns to MOI annotations DataFrame.

        Args:
            df: DataFrame with MOI annotations (must have 'start_seconds' and 'end_seconds')
            metadata: Metadata dict with session duration
            task: Task name (default: 'therapy')

        Returns:
            DataFrame with added epoch columns:
                - epoch_fixed: Fixed window epochs
                - epoch_nsplit: N-split epochs
                - epoch_sliding: Sliding window epochs
        """
        df = df.copy()

        # Get session duration from metadata
        duration = metadata.get("Duration", 0)
        if duration <= 0:
            logger.warning("Invalid session duration in metadata, using 3600s default")
            duration = 3600

        logger.info(
            f"Adding epoch columns for {len(df)} annotations (duration: {duration:.1f}s)"
        )

        # Use both start and end times to find all overlapping epochs
        start_times = df["start_seconds"].values
        end_times = df["end_seconds"].values

        # Assign epochs for each method
        for method in ["fixed", "nsplit", "sliding"]:
            try:
                # Get method config
                method_config = self.config["epoching"]["methods"].get(method, {})
                if not method_config.get("enabled", False):
                    logger.debug(f"Epoching method '{method}' is disabled, skipping")
                    df[f"epoch_{method}"] = [[] for _ in range(len(df))]
                    continue

                task_config = method_config.get(task, {})
                if not task_config:
                    logger.warning(f"No config for task '{task}' in method '{method}'")
                    df[f"epoch_{method}"] = [[] for _ in range(len(df))]
                    continue

                # Assign epochs based on interval intersection [start, end]
                if method == "fixed":
                    epoch_lists = self._assign_fixed_epochs(
                        start_times, end_times, duration, task_config
                    )
                elif method == "nsplit":
                    epoch_lists = self._assign_nsplit_epochs(
                        start_times, end_times, duration, task_config
                    )
                elif method == "sliding":
                    epoch_lists = self._assign_sliding_epochs(
                        start_times, end_times, duration, task_config
                    )
                else:
                    epoch_lists = [[] for _ in range(len(start_times))]

                df[f"epoch_{method}"] = epoch_lists

                # Log epoch statistics
                num_epochs_per_annotation = [len(epochs) for epochs in epoch_lists]
                all_epoch_ids = [eid for epochs in epoch_lists for eid in epochs]
                if all_epoch_ids:
                    logger.debug(
                        f"Assigned {method} epochs: range [{min(all_epoch_ids)}, {max(all_epoch_ids)}], "
                        f"avg {np.mean(num_epochs_per_annotation):.1f} epochs/annotation"
                    )

            except Exception as e:
                logger.error(f"Error assigning {method} epochs: {e}")
                df[f"epoch_{method}"] = [[] for _ in range(len(df))]

        return df

    def _assign_fixed_epochs(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
        duration: float,
        config: Dict,
    ) -> list:
        """
        Assign fixed window epochs based on interval intersection.

        For each annotation [start, end], finds all epochs whose interval
        [epoch_start, epoch_end] has non-null intersection with the annotation.

        Args:
            start_times: Array of annotation start times in seconds
            end_times: Array of annotation end times in seconds
            duration: Total session duration
            config: Fixed method configuration

        Returns:
            List of lists of epoch IDs (one list per annotation)
        """
        epoch_duration = config.get("duration", 30)
        overlap = config.get("overlap", 5)
        step = epoch_duration - overlap

        # Calculate total number of epochs
        max_epoch = int(np.floor(duration / step))

        epoch_lists = []
        for start, end in zip(start_times, end_times):
            epochs_for_annotation = []

            # Test intersection with each possible epoch
            for epoch_id in range(max_epoch + 1):
                epoch_start = epoch_id * step
                epoch_end = epoch_start + epoch_duration

                # Check if intervals intersect: [start, end] ∩ [epoch_start, epoch_end] ≠ ∅
                # Intersection exists if: start < epoch_end AND end > epoch_start
                if start < epoch_end and end > epoch_start:
                    epochs_for_annotation.append(epoch_id)

            epoch_lists.append(epochs_for_annotation)

        return epoch_lists

    def _assign_nsplit_epochs(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
        duration: float,
        config: Dict,
    ) -> list:
        """
        Assign N-split epochs based on interval intersection.

        For each annotation [start, end], finds all epochs whose interval
        [epoch_start, epoch_end] has non-null intersection with the annotation.

        Args:
            start_times: Array of annotation start times in seconds
            end_times: Array of annotation end times in seconds
            duration: Total session duration
            config: N-split method configuration

        Returns:
            List of lists of epoch IDs (one list per annotation)
        """
        n_epochs = config.get("n_epochs", 120)

        if n_epochs <= 0:
            return [[] for _ in range(len(start_times))]

        # Duration of each epoch
        epoch_duration = duration / n_epochs

        epoch_lists = []
        for start, end in zip(start_times, end_times):
            epochs_for_annotation = []

            # Test intersection with each possible epoch
            for epoch_id in range(n_epochs):
                epoch_start = epoch_id * epoch_duration
                epoch_end = epoch_start + epoch_duration

                # Check if intervals intersect: [start, end] ∩ [epoch_start, epoch_end] ≠ ∅
                if start < epoch_end and end > epoch_start:
                    epochs_for_annotation.append(epoch_id)

            epoch_lists.append(epochs_for_annotation)

        return epoch_lists

    def _assign_sliding_epochs(
        self,
        start_times: np.ndarray,
        end_times: np.ndarray,
        duration: float,
        config: Dict,
    ) -> list:
        """
        Assign sliding window epochs based on interval intersection.

        For each annotation [start, end], finds all epochs whose interval
        [epoch_start, epoch_end] has non-null intersection with the annotation.

        Args:
            start_times: Array of annotation start times in seconds
            end_times: Array of annotation end times in seconds
            duration: Total session duration
            config: Sliding method configuration

        Returns:
            List of lists of epoch IDs (one list per annotation)
        """
        step = config.get("step", 5)
        window_duration = config.get("duration", 30)

        # Calculate total number of epochs
        max_epoch = int(np.floor(duration / step))

        epoch_lists = []
        for start, end in zip(start_times, end_times):
            epochs_for_annotation = []

            # Test intersection with each possible epoch
            for epoch_id in range(max_epoch + 1):
                epoch_start = epoch_id * step
                epoch_end = epoch_start + window_duration

                # Check if intervals intersect: [start, end] ∩ [epoch_start, epoch_end] ≠ ∅
                if start < epoch_end and end > epoch_start:
                    epochs_for_annotation.append(epoch_id)

            epoch_lists.append(epochs_for_annotation)

        return epoch_lists
