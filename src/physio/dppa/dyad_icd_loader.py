"""
Module for loading Inter-Centroid Distance (ICD) data for dyadic analysis.

This module provides functionality to load ICD time series for specific dyads,
supporting both resting state and therapy tasks.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class DyadICDLoader:
    """
    Load Inter-Centroid Distance (ICD) data for dyadic analysis.

    This class handles loading ICD CSV files for specific dyad pairs,
    supporting both inter-session and intra-family dyad types.

    Attributes:
        config: Configuration object containing paths and settings.
        derivatives_path: Path to derivatives directory.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DyadICDLoader.

        Args:
            config_path: Optional path to configuration file.
                        If None, uses default config.yaml.
        """
        self.config = ConfigLoader(config_path)
        self.derivatives_path = Path(
            self.config.get("paths.derivatives", "data/derivatives")
        )
        logger.info("DyadICDLoader initialized")

    def parse_dyad_info(self, dyad_pair: str) -> Dict[str, str]:
        """
        Parse dyad pair string to extract subject and session information.

        Args:
            dyad_pair: Dyad identifier in format:
                      "sub1_ses-XX_vs_sub2_ses-YY"
                      Example: "g01p01_ses-01_vs_g01p02_ses-01"

        Returns:
            Dictionary with keys: 'sub1', 'ses1', 'sub2', 'ses2'

        Raises:
            ValueError: If dyad_pair format is invalid.

        Example:
            >>> loader = DyadICDLoader()
            >>> info = loader.parse_dyad_info("g01p01_ses-01_vs_g01p02_ses-01")
            >>> print(info)
            {'sub1': 'g01p01', 'ses1': '01', 'sub2': 'g01p02', 'ses2': '01'}
        """
        if not dyad_pair or "_vs_" not in dyad_pair:
            raise ValueError(
                f"Invalid dyad pair format: '{dyad_pair}'. "
                "Expected format: 'sub1_ses-XX_vs_sub2_ses-YY'"
            )

        try:
            # Split on "_vs_" to get both sides
            parts = dyad_pair.split("_vs_")
            if len(parts) != 2:
                raise ValueError("Expected exactly one '_vs_' separator")

            # Parse first subject
            left_parts = parts[0].split("_ses-")
            if len(left_parts) != 2:
                raise ValueError("Missing session in first subject")
            sub1, ses1 = left_parts[0], left_parts[1]

            # Parse second subject
            right_parts = parts[1].split("_ses-")
            if len(right_parts) != 2:
                raise ValueError("Missing session in second subject")
            sub2, ses2 = right_parts[0], right_parts[1]

            return {
                "sub1": sub1,
                "ses1": ses1,
                "sub2": sub2,
                "ses2": ses2,
            }

        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid dyad pair format: '{dyad_pair}'. "
                f"Expected format: 'sub1_ses-XX_vs_sub2_ses-YY'. Error: {e}"
            )

    def load_icd(
        self, dyad_pair: str, task: str, method: str, dyad_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load ICD time series for a specific dyad and task.

        Args:
            dyad_pair: Dyad identifier (e.g., "g01p01_ses-01_vs_g01p02_ses-01")
            task: Task name ('restingstate' or 'therapy')
            method: Epoching method (e.g., 'nsplit120', 'sliding_duration30s_step5s')
            dyad_type: Type of dyad comparison ('inter_session' or 'intra_family').
                      If None, auto-detects based on session comparison.

        Returns:
            DataFrame with columns: ['epoch_id', 'icd_value']
            Note: restingstate will have 1 row (epoch_id=0),
                  therapy will have multiple rows.

        Raises:
            FileNotFoundError: If ICD file does not exist.
            ValueError: If required columns are missing or invalid format.

        Example:
            >>> loader = DyadICDLoader()
            >>> df = loader.load_icd("g01p01_ses-01_vs_g01p02_ses-01", "therapy", "nsplit120")
            >>> print(df.head())
               epoch_id  icd_value
            0         0      50.23
            1         1      48.91
        """
        # Parse dyad info
        dyad_info = self.parse_dyad_info(dyad_pair)

        # Auto-detect dyad type if not provided
        if dyad_type is None:
            # If same session, likely intra_family; if different sessions, inter_session
            if dyad_info["ses1"] == dyad_info["ses2"]:
                # Same session - could be either, try intra_family first for real dyads
                dyad_type = "intra_family"
            else:
                dyad_type = "inter_session"

        # Construct file path
        icd_file = (
            self.derivatives_path
            / "dppa"
            / dyad_type
            / f"{dyad_type}_icd_task-{task}_method-{method}.csv"
        )

        # If file not found with intra_family, try inter_session as fallback
        if not icd_file.exists() and dyad_type == "intra_family":
            dyad_type = "inter_session"
            icd_file = (
                self.derivatives_path
                / "dppa"
                / dyad_type
                / f"{dyad_type}_icd_task-{task}_method-{method}.csv"
            )

        if not icd_file.exists():
            raise FileNotFoundError(
                f"ICD file not found: {icd_file}\n"
                f"Dyad: {dyad_pair}, Task: {task}, Method: {method}"
            )

        logger.info(f"Loading ICD data from: {icd_file}")

        # Load CSV
        df = pd.read_csv(icd_file)

        # Validate columns
        if "epoch_id" not in df.columns:
            raise ValueError(f"Missing 'epoch_id' column in {icd_file}")

        # Try original dyad_pair, then alternative column formats (ICD is symmetric)
        actual_dyad_pair = dyad_pair
        if dyad_pair not in df.columns and "_vs_" in dyad_pair:
            info = self.parse_dyad_info(dyad_pair)
            # Build candidate column names in all possible formats
            candidates = [
                # Reversed: "sub2_ses-YY_vs_sub1_ses-XX"
                f"{info['sub2']}_ses-{info['ses2']}_vs_{info['sub1']}_ses-{info['ses1']}",
                # Intra-family format: "sub1_vs_sub2_ses-XX"
                f"{info['sub1']}_vs_{info['sub2']}_ses-{info['ses1']}",
                # Intra-family reversed: "sub2_vs_sub1_ses-XX"
                f"{info['sub2']}_vs_{info['sub1']}_ses-{info['ses1']}",
            ]
            matched = False
            for candidate in candidates:
                if candidate in df.columns:
                    actual_dyad_pair = candidate
                    logger.debug(
                        f"Dyad pair '{dyad_pair}' not found, using: '{candidate}'"
                    )
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Dyad pair '{dyad_pair}' not found in {icd_file}. "
                    f"Tried: {candidates}. "
                    f"Available columns: {list(df.columns)}"
                )
        elif dyad_pair not in df.columns:
            raise ValueError(
                f"Dyad pair '{dyad_pair}' not found in {icd_file}. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract relevant columns and rename
        result = df[["epoch_id", actual_dyad_pair]].copy()
        result.rename(columns={actual_dyad_pair: "icd_value"}, inplace=True)

        logger.info(
            f"Loaded {len(result)} epochs for dyad {dyad_pair} "
            f"(task={task}, method={method})"
        )

        return result

    def load_both_tasks(
        self, dyad_pair: str, methods: Union[str, Dict[str, str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ICD data for both restingstate and therapy tasks.

        Args:
            dyad_pair: Dyad identifier (e.g., "g01p01_ses-01_vs_g01p02_ses-01")
            methods: Either a single method string (applied to all tasks) or
                    a dict mapping task -> method (e.g., {'restingstate': 'nsplit1', 'therapy': 'nsplit120'})

        Returns:
            Dictionary with keys 'restingstate' and 'therapy',
            each containing a DataFrame with ICD time series.

        Raises:
            FileNotFoundError: If either task file is missing.

        Example:
            >>> loader = DyadICDLoader()
            >>> # Old style (single method)
            >>> data = loader.load_both_tasks("g01p01_ses-01_vs_g01p02_ses-01", "nsplit120")
            >>> # New style (per-task methods)
            >>> methods = {'restingstate': 'nsplit1', 'therapy': 'nsplit120'}
            >>> data = loader.load_both_tasks("g01p01_ses-01_vs_g01p02_ses-01", methods)
        """
        # Normalize methods to dict format
        if isinstance(methods, str):
            methods_dict = {"restingstate": methods, "therapy": methods}
        else:
            methods_dict = methods

        logger.info(f"Loading both tasks for dyad: {dyad_pair}")
        for task, method in methods_dict.items():
            logger.debug(f"  {task}: {method}")

        result = {}
        for task in ["restingstate", "therapy"]:
            method = methods_dict.get(task)
            if method:
                try:
                    result[task] = self.load_icd(dyad_pair, task, method)
                except FileNotFoundError as e:
                    logger.warning(f"Could not load {task}: {e}")
                    raise

        logger.info(
            "Successfully loaded tasks: "
            + ", ".join(f"{t}={len(df)} epochs" for t, df in result.items())
        )

        return result
