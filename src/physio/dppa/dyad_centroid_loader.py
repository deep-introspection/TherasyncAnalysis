"""
Module for loading Poincaré centroid data for dyadic analysis.

This module provides functionality to load centroid time series for dyad members,
supporting both resting state and therapy tasks.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class DyadCentroidLoader:
    """
    Load Poincaré centroid data for dyadic analysis.

    This class handles loading centroid TSV files for both members of a dyad,
    supporting both restingstate and therapy tasks.

    Attributes:
        config: Configuration object containing paths and settings.
        derivatives_path: Path to derivatives directory.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DyadCentroidLoader.

        Args:
            config_path: Optional path to configuration file.
                        If None, uses default config.yaml.
        """
        self.config = ConfigLoader(config_path)
        self.derivatives_path = Path(
            self.config.get("paths.derivatives", "data/derivatives")
        )
        logger.info("DyadCentroidLoader initialized")

    def load_centroids(
        self, dyad_info: Dict[str, str], task: str, method: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load centroid time series for both members of a dyad.

        Args:
            dyad_info: Dictionary with keys: 'sub1', 'ses1', 'sub2', 'ses2'
                      Example: {"sub1": "g01p01", "ses1": "01", "sub2": "g01p02", "ses2": "01"}
            task: Task name ('restingstate' or 'therapy')
            method: Epoching method (e.g., 'nsplit120', 'sliding_duration30s_step5s')

        Returns:
            Tuple of (df_subject1, df_subject2), each DataFrame containing:
            - epoch_id: Epoch identifier
            - centroid_x: X coordinate of Poincaré centroid
            - centroid_y: Y coordinate of Poincaré centroid
            - sd1: Standard deviation 1 (short-term variability)
            - sd2: Standard deviation 2 (long-term variability)
            - sd_ratio: Ratio SD1/SD2
            - n_intervals: Number of RR intervals in epoch

        Raises:
            FileNotFoundError: If centroid file does not exist for either subject.
            ValueError: If required columns are missing.

        Example:
            >>> loader = DyadCentroidLoader()
            >>> dyad_info = {"sub1": "g01p01", "ses1": "01", "sub2": "g01p02", "ses2": "01"}
            >>> df1, df2 = loader.load_centroids(dyad_info, "therapy", "nsplit120")
            >>> print(len(df1), len(df2))
            120 120
        """
        # Load centroids for subject 1
        file1 = self._construct_centroid_path(
            dyad_info["sub1"], dyad_info["ses1"], task, method
        )
        df1 = self._load_centroid_file(file1, dyad_info["sub1"], dyad_info["ses1"])

        # Load centroids for subject 2
        file2 = self._construct_centroid_path(
            dyad_info["sub2"], dyad_info["ses2"], task, method
        )
        df2 = self._load_centroid_file(file2, dyad_info["sub2"], dyad_info["ses2"])

        logger.info(
            f"Loaded centroids for dyad: {dyad_info['sub1']}_ses-{dyad_info['ses1']} "
            f"vs {dyad_info['sub2']}_ses-{dyad_info['ses2']} "
            f"(task={task}, method={method}, epochs: {len(df1)} vs {len(df2)})"
        )

        return df1, df2

    def _construct_centroid_path(
        self, subject: str, session: str, task: str, method: str
    ) -> Path:
        """
        Construct path to centroid file for a subject/session/task.

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., '01')
            task: Task name ('restingstate' or 'therapy')
            method: Epoching method

        Returns:
            Path to centroid TSV file.
        """
        return (
            self.derivatives_path
            / "dppa"
            / f"sub-{subject}"
            / f"ses-{session}"
            / "poincare"
            / f"sub-{subject}_ses-{session}_task-{task}_method-{method}_desc-poincare_physio.tsv"
        )

    def _load_centroid_file(
        self, file_path: Path, subject: str, session: str
    ) -> pd.DataFrame:
        """
        Load centroid TSV file and validate columns.

        Args:
            file_path: Path to centroid file
            subject: Subject ID (for error messages)
            session: Session ID (for error messages)

        Returns:
            DataFrame with centroid data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If required columns are missing.
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"Centroid file not found: {file_path}\n"
                f"Subject: {subject}, Session: {session}"
            )

        logger.debug(f"Loading centroid file: {file_path}")

        # Load TSV (tab-separated)
        df = pd.read_csv(file_path, sep="\t")

        # Validate required columns
        required_cols = [
            "epoch_id",
            "centroid_x",
            "centroid_y",
            "sd1",
            "sd2",
            "sd_ratio",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns in {file_path}: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        return df

    def validate_epoch_alignment(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Validate that two DataFrames have matching epoch_ids.

        Args:
            df1: First subject's centroid DataFrame
            df2: Second subject's centroid DataFrame

        Returns:
            True if epoch_ids match, False otherwise.

        Example:
            >>> loader = DyadCentroidLoader()
            >>> dyad_info = {"sub1": "g01p01", "ses1": "01", "sub2": "g01p02", "ses2": "01"}
            >>> df1, df2 = loader.load_centroids(dyad_info, "therapy", "nsplit120")
            >>> is_aligned = loader.validate_epoch_alignment(df1, df2)
            >>> print(is_aligned)
            True
        """
        if len(df1) != len(df2):
            logger.warning(f"Epoch count mismatch: {len(df1)} vs {len(df2)} epochs")
            return False

        epochs1 = set(df1["epoch_id"])
        epochs2 = set(df2["epoch_id"])

        if epochs1 != epochs2:
            missing_in_2 = epochs1 - epochs2
            missing_in_1 = epochs2 - epochs1
            logger.warning(
                f"Epoch alignment mismatch:\n"
                f"  Missing in subject 2: {missing_in_2}\n"
                f"  Missing in subject 1: {missing_in_1}"
            )
            return False

        logger.debug(f"Epochs aligned: {len(epochs1)} matching epochs")
        return True

    def load_both_tasks(
        self, dyad_info: Dict[str, str], methods: Union[str, Dict[str, str]]
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load centroid data for both restingstate and therapy tasks.

        Args:
            dyad_info: Dictionary with keys: 'sub1', 'ses1', 'sub2', 'ses2'
            methods: Either a single method string (applied to all tasks) or
                    a dict mapping task -> method (e.g., {'restingstate': 'nsplit1', 'therapy': 'nsplit120'})

        Returns:
            Dictionary with keys 'restingstate' and 'therapy',
            each containing a tuple (df_subject1, df_subject2).

        Raises:
            FileNotFoundError: If any centroid file is missing.

        Example:
            >>> loader = DyadCentroidLoader()
            >>> dyad_info = {"sub1": "g01p01", "ses1": "01", "sub2": "g01p02", "ses2": "01"}
            >>> # Old style (single method)
            >>> data = loader.load_both_tasks(dyad_info, "nsplit120")
            >>> # New style (per-task methods)
            >>> methods = {'restingstate': 'nsplit1', 'therapy': 'nsplit120'}
            >>> data = loader.load_both_tasks(dyad_info, methods)
        """
        # Normalize methods to dict format
        if isinstance(methods, str):
            methods_dict = {"restingstate": methods, "therapy": methods}
        else:
            methods_dict = methods

        logger.info(
            f"Loading both tasks for dyad: {dyad_info['sub1']}_ses-{dyad_info['ses1']} "
            f"vs {dyad_info['sub2']}_ses-{dyad_info['ses2']}"
        )
        for task, method in methods_dict.items():
            logger.debug(f"  {task}: {method}")

        result = {}
        for task in ["restingstate", "therapy"]:
            method = methods_dict.get(task)
            if method:
                try:
                    result[task] = self.load_centroids(dyad_info, task, method)
                except FileNotFoundError as e:
                    logger.warning(f"Could not load {task}: {e}")
                    raise

        logger.info(
            "Successfully loaded tasks: "
            + ", ".join(
                f"{t}=({len(dfs[0])}, {len(dfs[1])}) epochs"
                for t, dfs in result.items()
            )
        )

        return result
