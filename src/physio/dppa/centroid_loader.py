"""
Centroid Loader for DPPA Analysis.

This module loads pre-computed Poincaré centroid files from disk and provides
caching for efficient repeated access.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class CentroidLoader:
    """
    Load and cache Poincaré centroid files.

    This class provides methods to load pre-computed Poincaré centroids
    from BIDS-formatted TSV files. Results are cached in memory for
    performance optimization.

    Attributes:
        dppa_dir: Path to DPPA derivatives directory
        cache: Dictionary storing loaded DataFrames

    Example:
        >>> loader = CentroidLoader()
        >>> df = loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
        >>> if df is not None:
        ...     print(f"Loaded {len(df)} epochs")
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize CentroidLoader.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config = ConfigLoader(config_path)

        # Get DPPA directory path
        paths = self.config.get("paths", {})
        self.dppa_dir = Path(paths.get("dppa", "data/derivatives/dppa"))

        # Cache for loaded centroids: (subject, session, task, method) -> DataFrame
        self.cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}

        logger.info("Centroid Loader initialized")

    def load_centroid(
        self, subject: str, session: str, task: str, method: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load Poincaré centroid data for a specific subject/session/task/method.

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., 'ses-01')
            task: Task name (e.g., 'therapy', 'restingstate')
            method: Epoching method (e.g., 'nsplit120', 'sliding_duration30s_step5s')
            use_cache: If True, return cached result if available

        Returns:
            DataFrame with columns: epoch_id, centroid_x, centroid_y, sd1, sd2,
            sd_ratio, n_intervals. Returns None if file doesn't exist.

        Example:
            >>> loader = CentroidLoader()
            >>> df = loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
        """
        # Normalize session format
        if not session.startswith("ses-"):
            session = f"ses-{session}"

        # Check cache
        cache_key = (subject, session, task, method)
        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit: {subject}/{session}/{task}/{method}")
            return self.cache[cache_key]

        # Build file path
        file_path = (
            self.dppa_dir
            / f"sub-{subject}"
            / session
            / "poincare"
            / f"sub-{subject}_{session}_task-{task}_method-{method}_desc-poincare_physio.tsv"
        )

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Centroid file not found: {file_path}")
            return None

        try:
            # Load centroid data
            df = pd.read_csv(file_path, sep="\t")

            # Validate columns
            required_cols = [
                "epoch_id",
                "centroid_x",
                "centroid_y",
                "sd1",
                "sd2",
                "sd_ratio",
                "n_intervals",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {file_path}: {missing_cols}")
                return None

            # Cache result
            if use_cache:
                self.cache[cache_key] = df

            logger.debug(
                f"Loaded centroids: {subject}/{session}/{task}/{method} ({len(df)} epochs)"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to load centroid file {file_path}: {e}")
            return None

    def load_subject_session(
        self,
        subject: str,
        session: str,
        task: Optional[str] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all centroid files for a subject/session.

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., 'ses-01')
            task: Optional task filter (if None, load all tasks)
            method: Optional method filter (if None, load all methods)

        Returns:
            Nested dictionary: {task: {method: DataFrame}}

        Example:
            >>> loader = CentroidLoader()
            >>> results = loader.load_subject_session('g01p01', 'ses-01')
            >>> therapy_nsplit = results['therapy']['nsplit120']
        """
        # Normalize session format
        if not session.startswith("ses-"):
            session = f"ses-{session}"

        # Build directory path
        poincare_dir = self.dppa_dir / f"sub-{subject}" / session / "poincare"

        if not poincare_dir.exists():
            logger.warning(f"Poincaré directory not found: {poincare_dir}")
            return {}

        # Find all centroid files
        results = {}

        for tsv_file in sorted(poincare_dir.glob("*_desc-poincare_physio.tsv")):
            # Parse filename
            parts = tsv_file.stem.split("_")
            file_task = None
            file_method = None

            for i, part in enumerate(parts):
                if part.startswith("task-"):
                    file_task = part.replace("task-", "")
                elif part.startswith("method-"):
                    file_method = part.replace("method-", "")

            if not file_task or not file_method:
                logger.warning(f"Could not parse filename: {tsv_file.name}")
                continue

            # Apply filters
            if task and file_task != task:
                continue
            if method and file_method != method:
                continue

            # Load centroid data
            df = self.load_centroid(subject, session, file_task, file_method)
            if df is not None:
                if file_task not in results:
                    results[file_task] = {}
                results[file_task][file_method] = df

        logger.info(f"Loaded centroids for {subject}/{session}: {len(results)} tasks")
        return results

    def clear_cache(self):
        """Clear the centroid cache to free memory."""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared: {cache_size} entries removed")

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and memory info
        """
        return {
            "entries": len(self.cache),
            "subjects": len(set(key[0] for key in self.cache.keys())),
            "sessions": len(set(key[0:2] for key in self.cache.keys())),
        }
