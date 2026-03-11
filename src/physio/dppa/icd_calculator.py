"""
Inter-Centroid Distance (ICD) Calculator for DPPA Analysis.

This module computes Euclidean distances between Poincaré plot centroids
of two participants across epochs.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class ICDCalculator:
    """
    Calculate Inter-Centroid Distances between dyads.

    This class computes the Euclidean distance between Poincaré centroids
    of two participants for each epoch. The formula is:

    ICD = √[(centroid_x₁ - centroid_x₂)² + (centroid_y₁ - centroid_y₂)²]

    NaN handling: If either centroid is NaN, the resulting ICD is NaN.

    Example:
        >>> calculator = ICDCalculator()
        >>> result = calculator.compute_icd(centroids1, centroids2)
        >>> print(f"Mean ICD: {result['icd'].mean():.2f} ms")
    """

    def __init__(self):
        """Initialize ICD Calculator."""
        logger.info("ICD Calculator initialized")

    def compute_icd(
        self,
        centroids1: pd.DataFrame,
        centroids2: pd.DataFrame,
        align_on: str = "epoch_id",
    ) -> pd.DataFrame:
        """
        Compute Inter-Centroid Distances between two centroid series.

        Args:
            centroids1: First participant's centroids (from CentroidLoader)
            centroids2: Second participant's centroids (from CentroidLoader)
            align_on: Column to align on (default: 'epoch_id')

        Returns:
            DataFrame with columns: epoch_id, icd, centroid_x1, centroid_y1,
            centroid_x2, centroid_y2, n_intervals1, n_intervals2

        Example:
            >>> calc = ICDCalculator()
            >>> df1 = loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
            >>> df2 = loader.load_centroid('g01p02', 'ses-01', 'therapy', 'nsplit120')
            >>> result = calc.compute_icd(df1, df2)
        """
        # Validate input
        required_cols = ["epoch_id", "centroid_x", "centroid_y", "n_intervals"]
        for col in required_cols:
            if col not in centroids1.columns:
                raise ValueError(f"Missing column '{col}' in centroids1")
            if col not in centroids2.columns:
                raise ValueError(f"Missing column '{col}' in centroids2")

        # Merge centroids on epoch_id
        merged = pd.merge(
            centroids1[["epoch_id", "centroid_x", "centroid_y", "n_intervals"]],
            centroids2[["epoch_id", "centroid_x", "centroid_y", "n_intervals"]],
            on=align_on,
            suffixes=("1", "2"),
            how="outer",  # Keep all epochs, even if missing in one participant
        )

        # Sort by epoch_id
        merged = merged.sort_values("epoch_id").reset_index(drop=True)

        # Compute Euclidean distance
        # ICD = √[(x₁ - x₂)² + (y₁ - y₂)²]
        merged["icd"] = np.sqrt(
            (merged["centroid_x1"] - merged["centroid_x2"]) ** 2
            + (merged["centroid_y1"] - merged["centroid_y2"]) ** 2
        )

        # Count valid ICDs (both centroids non-NaN)
        valid_count = merged["icd"].notna().sum()
        total_count = len(merged)

        logger.debug(f"Computed ICD: {valid_count}/{total_count} valid epochs")

        return merged

    def compute_icd_summary(self, icd_df: pd.DataFrame) -> dict:
        """
        Compute summary statistics for ICD values.

        Args:
            icd_df: DataFrame from compute_icd()

        Returns:
            Dictionary with mean, std, min, max, median, valid_count

        Example:
            >>> result = calc.compute_icd(df1, df2)
            >>> summary = calc.compute_icd_summary(result)
            >>> print(f"Mean ICD: {summary['mean']:.2f} ms")
        """
        icd_values = icd_df["icd"].dropna()

        if len(icd_values) == 0:
            logger.warning("No valid ICD values to summarize")
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "valid_count": 0,
                "total_count": len(icd_df),
            }

        summary = {
            "mean": float(icd_values.mean()),
            "std": float(icd_values.std()),
            "min": float(icd_values.min()),
            "max": float(icd_values.max()),
            "median": float(icd_values.median()),
            "valid_count": len(icd_values),
            "total_count": len(icd_df),
        }

        return summary

    def compute_batch_icd(
        self,
        centroid_dict1: dict,
        centroid_dict2: dict,
        task: Optional[str] = None,
        method: Optional[str] = None,
    ) -> dict:
        """
        Compute ICD for multiple tasks/methods between two participants.

        Args:
            centroid_dict1: Dict from CentroidLoader.load_subject_session()
            centroid_dict2: Dict from CentroidLoader.load_subject_session()
            task: Optional task filter
            method: Optional method filter

        Returns:
            Nested dict: {task: {method: DataFrame}}

        Example:
            >>> centroids1 = loader.load_subject_session('g01p01', 'ses-01')
            >>> centroids2 = loader.load_subject_session('g01p02', 'ses-01')
            >>> results = calc.compute_batch_icd(centroids1, centroids2)
        """
        results = {}

        # Find common tasks
        tasks1 = set(centroid_dict1.keys())
        tasks2 = set(centroid_dict2.keys())
        common_tasks = tasks1 & tasks2

        if not common_tasks:
            logger.warning("No common tasks found between participants")
            return results

        # Apply task filter
        if task:
            if task not in common_tasks:
                logger.warning(f"Task '{task}' not common to both participants")
                return results
            common_tasks = {task}

        for task_name in common_tasks:
            results[task_name] = {}

            # Find common methods
            methods1 = set(centroid_dict1[task_name].keys())
            methods2 = set(centroid_dict2[task_name].keys())
            common_methods = methods1 & methods2

            # Apply method filter
            if method:
                if method not in common_methods:
                    logger.warning(
                        f"Method '{method}' not common for task '{task_name}'"
                    )
                    continue
                common_methods = {method}

            for method_name in common_methods:
                df1 = centroid_dict1[task_name][method_name]
                df2 = centroid_dict2[task_name][method_name]

                icd_df = self.compute_icd(df1, df2)
                results[task_name][method_name] = icd_df

        logger.info(f"Computed batch ICD: {len(results)} tasks")
        return results
