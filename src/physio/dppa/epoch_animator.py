"""
Epoch Animator Module

Generate frame-by-frame Poincaré plot animations from RR intervals with epoch columns.
Loads RR interval data from preprocessing directory (with inline epoch columns),
computes Poincaré points per epoch, and prepares data for visualization.

Author: Lena Adel, Remy Ramadour
Date: 2025-11-12
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class EpochAnimator:
    """
    Prepare epoch-by-epoch data for Poincaré plot animations.

    Loads RR intervals from preprocessing files (with inline epoch columns)
    and computes Poincaré points (RRn, RRn+1 pairs) for each epoch.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Epoch Animator.

        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        self.config = ConfigLoader(config_path).config
        self.data_root = Path(self.config.get("paths", {}).get("data_root", "data"))
        logger.info("Epoch Animator initialized")

    def load_rr_intervals(
        self, subject: str, session: str, task: str, method: str
    ) -> pd.DataFrame:
        """
        Load RR intervals from preprocessing directory (with epoch columns).

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., 'ses-01', auto-prefix if needed)
            task: Task name ('therapy', 'restingstate')
            method: Epoching method ('sliding_duration30s_step5s')

        Returns:
            DataFrame with columns:
                - time_peak_start: Start time of RR interval (s)
                - time_peak_end: End time of RR interval (s)
                - rr_interval_ms: RR interval duration (ms)
                - is_valid: Validity flag (1=valid)
                - epoch_{method}: List of epoch IDs (JSON string format)

        Raises:
            FileNotFoundError: If RR intervals file doesn't exist
        """
        # Ensure session has 'ses-' prefix
        if not session.startswith("ses-"):
            session = f"ses-{session}"

        # Build file path (preprocessing directory)
        rr_file = (
            self.data_root
            / "derivatives"
            / "preprocessing"
            / f"sub-{subject}"
            / session
            / "bvp"
            / f"sub-{subject}_{session}_task-{task}_desc-rrintervals_physio.tsv"
        )

        if not rr_file.exists():
            raise FileNotFoundError(f"RR intervals file not found: {rr_file}")

        # Load TSV file
        df = pd.read_csv(rr_file, sep="\t")

        logger.info(
            f"Loaded RR intervals for {subject}/{session}/{task}: {len(df)} intervals"
        )

        return df

    def get_rr_for_epoch(
        self, rr_df: pd.DataFrame, epoch_id: int, method: str
    ) -> np.ndarray:
        """
        Extract RR intervals belonging to a specific epoch.

        Args:
            rr_df: DataFrame from load_rr_intervals()
            epoch_id: Epoch ID to extract
            method: Epoching method (used to select correct column)

        Returns:
            Array of RR intervals (ms) for the specified epoch

        Notes:
            - Epoch column contains JSON strings like "[0]" or "[0, 1]"
            - Returns intervals where epoch_id appears in the list
        """
        # Get column name
        epoch_col = f"epoch_{method}"

        if epoch_col not in rr_df.columns:
            raise ValueError(f"Column '{epoch_col}' not found in DataFrame")

        # Parse JSON strings and filter
        mask = rr_df[epoch_col].apply(
            lambda x: epoch_id in json.loads(x) if pd.notna(x) else False
        )

        epoch_rr = rr_df.loc[mask, "rr_interval_ms"].values

        return epoch_rr

    def compute_poincare_points(
        self, rr_intervals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Poincaré plot points from RR intervals.

        Args:
            rr_intervals: Array of RR intervals (ms)

        Returns:
            Tuple of (rr_n, rr_n_plus_1):
                - rr_n: RRn values (current intervals)
                - rr_n_plus_1: RRn+1 values (next intervals)

        Notes:
            - For N intervals, returns N-1 point pairs
            - Example: [750, 760, 755] → ([750, 760], [760, 755])
        """
        if len(rr_intervals) < 2:
            logger.warning("Need at least 2 RR intervals for Poincaré plot")
            return np.array([]), np.array([])

        rr_n = rr_intervals[:-1]
        rr_n_plus_1 = rr_intervals[1:]

        return rr_n, rr_n_plus_1

    def compute_poincare_for_epoch(
        self, rr_df: pd.DataFrame, epoch_id: int, method: str
    ) -> Dict[str, np.ndarray]:
        """
        Compute Poincaré points for a specific epoch.

        Args:
            rr_df: DataFrame from load_rr_intervals()
            epoch_id: Epoch ID to process
            method: Epoching method

        Returns:
            Dictionary with keys:
                - 'rr_n': RRn values (current intervals)
                - 'rr_n_plus_1': RRn+1 values (next intervals)
                - 'n_points': Number of Poincaré points
        """
        # Get RR intervals for this epoch
        rr_intervals = self.get_rr_for_epoch(rr_df, epoch_id, method)

        # Compute Poincaré points
        rr_n, rr_n_plus_1 = self.compute_poincare_points(rr_intervals)

        return {"rr_n": rr_n, "rr_n_plus_1": rr_n_plus_1, "n_points": len(rr_n)}
