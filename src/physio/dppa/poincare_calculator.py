"""
Poincaré Calculator for DPPA (Dyadic Poincaré Plot Analysis).

Computes Poincaré plot centroids and metrics for RR interval data with epoch columns.
Each centroid represents the center of the Poincaré cloud (RRn vs RRn+1)
within a given epoch.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import ast
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class PoincareCalculator:
    """
    Calculate Poincaré plot centroids and metrics from RR intervals with epoch columns.

    Reads RR interval files from derivatives/preprocessing/ directory.
    Files contain epoch_id columns added during preprocessing.

    Poincaré Plot Metrics:
    - centroid_x: mean(RRn) - average of current RR intervals
    - centroid_y: mean(RRn+1) - average of next RR intervals
    - SD1: Short-term variability (perpendicular to identity line)
    - SD2: Long-term variability (along identity line)
    - sd_ratio: SD1/SD2 (sympatho-vagal balance)
    - n_intervals: Number of RR intervals in epoch
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Poincaré Calculator.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = ConfigLoader(config_path)
        # Read from preprocessing directory (files now have epoch columns)
        derivatives_dir = Path(self.config.get("paths.derivatives", "data/derivatives"))
        preprocessing_dir = self.config.get("output.preprocessing_dir", "preprocessing")
        self.preprocessing_dir = derivatives_dir / preprocessing_dir

        logger.info(
            f"Poincaré Calculator initialized (reading from {self.preprocessing_dir})"
        )

    def compute_poincare_metrics(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Compute Poincaré plot metrics for a single epoch.

        Args:
            rr_intervals: Array of RR intervals (in milliseconds)

        Returns:
            Dictionary with centroid_x, centroid_y, sd1, sd2, sd_ratio, n_intervals
            Returns NaN values if < 2 intervals
        """
        n_intervals = len(rr_intervals)

        # Need at least 2 intervals for Poincaré plot (RRn, RRn+1)
        if n_intervals < 2:
            return {
                "centroid_x": np.nan,
                "centroid_y": np.nan,
                "sd1": np.nan,
                "sd2": np.nan,
                "sd_ratio": np.nan,
                "n_intervals": n_intervals,
            }

        # Create Poincaré pairs: (RRn, RRn+1)
        rr_n = rr_intervals[:-1]  # Current RR intervals
        rr_n1 = rr_intervals[1:]  # Next RR intervals

        # Centroid coordinates
        centroid_x = np.mean(rr_n)
        centroid_y = np.mean(rr_n1)

        # SD1: Standard deviation perpendicular to identity line
        # SD1 = sqrt(var(RRn - RRn+1) / 2)
        diff = rr_n - rr_n1
        sd1 = np.sqrt(np.var(diff, ddof=1) / 2)

        # SD2: Standard deviation along identity line
        # SD2 = sqrt(var(RRn + RRn+1) / 2)
        sum_rr = rr_n + rr_n1
        sd2 = np.sqrt(np.var(sum_rr, ddof=1) / 2)

        # SD ratio (sympatho-vagal balance)
        sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan

        return {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "sd1": sd1,
            "sd2": sd2,
            "sd_ratio": sd_ratio,
            "n_intervals": n_intervals - 1,  # Number of pairs (RRn, RRn+1)
        }

    def compute_centroids_for_file(self, rr_file: Path, method: str) -> pd.DataFrame:
        """
        Compute Poincaré centroids for all epochs in an RR interval file.

        Args:
            rr_file: Path to epoched RR interval file
            method: Epoching method name (e.g., 'nsplit120', 'sliding_duration30s_step5s')

        Returns:
            DataFrame with columns: epoch_id, centroid_x, centroid_y, sd1, sd2,
                                   sd_ratio, n_intervals
        """
        epoch_col = f"epoch_{method}"

        # Load RR intervals with epoch assignments
        df = pd.read_csv(rr_file, sep="\t")

        if epoch_col not in df.columns:
            raise ValueError(f"Epoch column '{epoch_col}' not found in {rr_file}")

        # Parse epoch lists (stored as strings like "[0]", "[0, 1, 2]")
        df[epoch_col] = df[epoch_col].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )

        # Get unique epoch IDs
        all_epoch_ids = set()
        for epoch_list in df[epoch_col]:
            all_epoch_ids.update(epoch_list)
        all_epoch_ids = sorted(all_epoch_ids)

        # Compute centroids for each epoch
        results = []
        for epoch_id in all_epoch_ids:
            # Get RR intervals for this epoch
            mask = df[epoch_col].apply(lambda x: epoch_id in x)
            epoch_rr = df.loc[mask & df["is_valid"], "rr_interval_ms"].values

            # Compute Poincaré metrics
            metrics = self.compute_poincare_metrics(epoch_rr)
            metrics["epoch_id"] = epoch_id
            results.append(metrics)

        # Create DataFrame
        result_df = pd.DataFrame(results)
        result_df = result_df[
            [
                "epoch_id",
                "centroid_x",
                "centroid_y",
                "sd1",
                "sd2",
                "sd_ratio",
                "n_intervals",
            ]
        ]

        logger.debug(
            f"Computed {len(results)} centroids for {rr_file.name} (method: {method})"
        )
        return result_df

    def compute_subject_session(
        self,
        subject: str,
        session: str,
        task: Optional[str] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute Poincaré centroids for a subject/session.

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., 'ses-01')
            task: Optional task filter (e.g., 'restingstate', 'therapy')
            method: Optional epoching method filter

        Returns:
            Nested dict: {task: {method: DataFrame}}
        """
        subject_dir = self.preprocessing_dir / f"sub-{subject}" / session / "bvp"

        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        # Find RR interval files
        pattern = f"sub-{subject}_{session}_*_desc-rrintervals_physio.tsv"
        rr_files = list(subject_dir.glob(pattern))

        if not rr_files:
            raise FileNotFoundError(f"No RR interval files found in {subject_dir}")

        results = {}
        for rr_file in rr_files:
            # Extract task from filename
            parts = rr_file.stem.split("_")
            file_task = parts[2].replace("task-", "")

            # Filter by task if specified
            if task and file_task != task:
                continue

            # Get available epoch methods from file
            df_sample = pd.read_csv(rr_file, sep="\t", nrows=1)
            epoch_cols = [col for col in df_sample.columns if col.startswith("epoch_")]
            methods = [col.replace("epoch_", "") for col in epoch_cols]

            # Filter by method if specified
            if method:
                methods = [m for m in methods if m == method]

            # Compute for each method
            results[file_task] = {}
            for m in methods:
                logger.info(f"Computing centroids: {subject}/{session}/{file_task}/{m}")
                results[file_task][m] = self.compute_centroids_for_file(rr_file, m)

        return results
