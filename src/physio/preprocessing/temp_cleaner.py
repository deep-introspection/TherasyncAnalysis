"""
Temperature Data Cleaner for TherasyncPipeline.

This module provides functionality to clean peripheral skin temperature data
from Empatica devices, including outlier removal, artifact detection,
gap interpolation, and quality assessment.

Authors: Lena Adel, Remy Ramadour
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class TEMPCleaner:
    """
    Clean and preprocess peripheral skin temperature data with conservative approach.

    This class handles:
    - Physiological outlier removal (< 25°C or > 40°C)
    - Artifact detection (sudden jumps > 2°C between samples)
    - Short gap interpolation (linear interpolation for gaps < 10 seconds)
    - Quality assessment and scoring
    - Data validation

    Temperature data characteristics:
    - Sampling rate: 4 Hz
    - Unit: degrees Celsius (°C)
    - Expected range: 25-40°C for peripheral skin temperature
    - Slow dynamics: temperature changes are gradual (< 0.5°C per minute typical)
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the temperature cleaner with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)

        # Get temperature cleaning configuration
        temp_config = self.config.get("physio.temp.processing", {})

        # Outlier thresholds (physiological limits)
        self.outlier_threshold = temp_config.get("outlier_threshold", [25.0, 40.0])

        # Artifact detection (sudden jumps)
        self.jump_threshold = temp_config.get(
            "jump_threshold", 2.0
        )  # °C between consecutive samples

        # Interpolation settings
        self.interpolation_max_gap = temp_config.get(
            "interpolation_max_gap", 10
        )  # seconds

        # Quality settings
        self.quality_min_samples = temp_config.get("quality_min_samples", 30)

        # Get sampling rate
        self.sampling_rate = self.config.get("physio.temp.sampling_rate", 4)

        logger.info(
            f"Temperature Cleaner initialized (outliers: {self.outlier_threshold}°C, "
            f"jump threshold: {self.jump_threshold}°C, max gap: {self.interpolation_max_gap}s, "
            f"sampling rate: {self.sampling_rate} Hz)"
        )

    def clean_signal(
        self, data: pd.DataFrame, moment: str = "unknown"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean temperature signal with conservative approach.

        Args:
            data: DataFrame with columns ['time', 'temp']
            moment: Moment/task name for logging

        Returns:
            Tuple of:
                - Cleaned DataFrame with standardized columns:
                  ['time', 'TEMP_Raw', 'TEMP_Clean', 'TEMP_Outliers', 'TEMP_Interpolated', 'TEMP_Quality']
                - Processing metadata dictionary

        Example:
            >>> cleaner = TEMPCleaner()
            >>> cleaned_data, metadata = cleaner.clean_signal(raw_data, moment='restingstate')
            >>> print(f"Quality score: {metadata['quality_score']:.3f}")
        """
        logger.info(
            f"Cleaning temperature signal for moment '{moment}' ({len(data)} samples)"
        )

        if len(data) == 0:
            raise ValueError("Empty temperature data provided")

        # Create working copy with internal column names (for processing)
        result = data.copy()

        # Initialize processing columns with internal names
        result["temp_clean"] = result["temp"].copy()
        result["temp_outliers"] = False
        result["temp_interpolated"] = False
        result["temp_quality"] = 1.0  # Will be updated per sample

        # Step 1: Remove physiological outliers
        outlier_mask = self._detect_outliers(result["temp"])
        result.loc[outlier_mask, "temp_outliers"] = True
        result.loc[outlier_mask, "temp_clean"] = np.nan

        outlier_count = outlier_mask.sum()
        logger.info(
            f"Detected {outlier_count} outliers ({outlier_count / len(data) * 100:.1f}%)"
        )

        # Step 2: Detect and mark artifacts (sudden jumps)
        artifact_mask = self._detect_artifacts(result["temp_clean"])
        result.loc[artifact_mask, "temp_outliers"] = True
        result.loc[artifact_mask, "temp_clean"] = np.nan

        artifact_count = artifact_mask.sum()
        logger.info(
            f"Detected {artifact_count} artifact samples ({artifact_count / len(data) * 100:.1f}%)"
        )

        # Step 3: Interpolate short gaps
        interpolated_mask = self._interpolate_gaps(result)
        interpolated_count = interpolated_mask.sum()
        logger.info(f"Interpolated {interpolated_count} samples in short gaps")

        # Step 4: Calculate quality scores
        quality_scores = self._calculate_quality(result)
        result["temp_quality"] = quality_scores

        # Step 5: Calculate overall metrics
        metadata = self._calculate_cleaning_metadata(result, moment)

        # Step 6: Rename columns to standardized convention (BIDS-compliant)
        result = result.rename(
            columns={
                "temp": "TEMP_Raw",
                "temp_clean": "TEMP_Clean",
                "temp_outliers": "TEMP_Outliers",
                "temp_interpolated": "TEMP_Interpolated",
                "temp_quality": "TEMP_Quality",
            }
        )

        logger.info(
            f"Temperature cleaning complete for '{moment}': "
            f"quality {metadata['quality_score']:.3f}, "
            f"{metadata['valid_samples']}/{metadata['total_samples']} valid samples"
        )

        return result, metadata

    def _detect_outliers(self, temp_series: pd.Series) -> pd.Series:
        """
        Detect physiological outliers in temperature data.

        Args:
            temp_series: Temperature values to check

        Returns:
            Boolean mask indicating outliers
        """
        min_temp, max_temp = self.outlier_threshold

        # Physiological outliers
        physiological_outliers = (temp_series < min_temp) | (temp_series > max_temp)

        # NaN values are also considered outliers
        nan_outliers = temp_series.isna()

        return physiological_outliers | nan_outliers

    def _detect_artifacts(self, temp_series: pd.Series) -> pd.Series:
        """
        Detect artifacts (sudden jumps) in temperature data.

        Temperature changes slowly, so sudden jumps > threshold indicate artifacts.

        Args:
            temp_series: Temperature values to check

        Returns:
            Boolean mask indicating artifact samples
        """
        artifact_mask = pd.Series(False, index=temp_series.index)

        # Calculate differences between consecutive samples
        temp_diff = temp_series.diff().abs()

        # Mark samples where jump exceeds threshold
        # Mark both the current sample and the next one as artifacts
        jump_indices = temp_diff[temp_diff > self.jump_threshold].index

        for idx in jump_indices:
            artifact_mask.loc[idx] = True

        return artifact_mask

    def _interpolate_gaps(self, data: pd.DataFrame) -> pd.Series:
        """
        Interpolate short gaps in temperature data.

        Args:
            data: DataFrame with temperature data and outlier flags

        Returns:
            Boolean mask indicating interpolated samples
        """
        interpolated_mask = pd.Series(False, index=data.index)

        # Find gaps (NaN values in temp_clean)
        nan_mask = data["temp_clean"].isna()

        if not nan_mask.any():
            return interpolated_mask

        # Find consecutive gaps
        gap_groups = self._find_gap_groups(nan_mask, data["time"])

        for gap_start, gap_end, gap_duration in gap_groups:
            if gap_duration <= self.interpolation_max_gap:
                # Interpolate this gap
                self._interpolate_gap(data, gap_start, gap_end, interpolated_mask)

        return interpolated_mask

    def _find_gap_groups(self, nan_mask: pd.Series, time_series: pd.Series) -> list:
        """
        Find consecutive groups of NaN values and their durations.

        Args:
            nan_mask: Boolean mask of NaN positions
            time_series: Time values

        Returns:
            List of tuples (start_idx, end_idx, duration_seconds)
        """
        gap_groups = []

        # Find start and end positions of consecutive NaN groups
        nan_diff = nan_mask.astype(int).diff()
        gap_starts = nan_diff[nan_diff == 1].index
        gap_ends = nan_diff[nan_diff == -1].index

        # Handle edge cases
        if nan_mask.iloc[0]:
            gap_starts = [nan_mask.index[0]] + list(gap_starts)
        if nan_mask.iloc[-1]:
            gap_ends = list(gap_ends) + [nan_mask.index[-1]]

        # Pair starts and ends
        for start_idx, end_idx in zip(gap_starts, gap_ends):
            duration = time_series.loc[end_idx] - time_series.loc[start_idx]
            gap_groups.append((start_idx, end_idx, duration))

        return gap_groups

    def _interpolate_gap(
        self,
        data: pd.DataFrame,
        gap_start: int,
        gap_end: int,
        interpolated_mask: pd.Series,
    ) -> None:
        """
        Interpolate a specific gap using linear interpolation.

        Args:
            data: DataFrame to modify
            gap_start: Start index of gap
            gap_end: End index of gap
            interpolated_mask: Mask to update with interpolated positions
        """
        # Find valid data points before and after gap
        before_idx = gap_start - 1 if gap_start > 0 else None
        after_idx = gap_end + 1 if gap_end < len(data) - 1 else None

        # Skip interpolation if we don't have both endpoints
        if before_idx is None or after_idx is None:
            return

        if pd.isna(data.loc[before_idx, "temp_clean"]) or pd.isna(
            data.loc[after_idx, "temp_clean"]
        ):
            return

        # Linear interpolation
        gap_indices = list(range(gap_start, gap_end + 1))
        time_points = data.loc[gap_indices, "time"].values

        # Interpolate between valid endpoints
        start_time = data.loc[before_idx, "time"]
        end_time = data.loc[after_idx, "time"]
        start_temp = data.loc[before_idx, "temp_clean"]
        end_temp = data.loc[after_idx, "temp_clean"]

        # Linear interpolation
        interpolated_temp = np.interp(
            time_points, [start_time, end_time], [start_temp, end_temp]
        )

        # Update data
        data.loc[gap_indices, "temp_clean"] = interpolated_temp
        data.loc[gap_indices, "temp_interpolated"] = True
        interpolated_mask.loc[gap_indices] = True

    def _calculate_quality(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate quality scores for each sample.

        Quality is based on:
        - Original data: 1.0
        - Interpolated data: 0.7
        - Missing data (NaN): 0.0
        - Data at edges of physiological range: reduced quality

        Args:
            data: DataFrame with cleaned temperature data

        Returns:
            Series of quality scores (0-1)
        """
        quality_scores = pd.Series(1.0, index=data.index)

        # Reduce quality for interpolated samples
        quality_scores.loc[data["temp_interpolated"]] = 0.7

        # Zero quality for remaining NaN values
        quality_scores.loc[data["temp_clean"].isna()] = 0.0

        # Reduce quality for values at the edges of physiological range
        min_temp, max_temp = self.outlier_threshold
        edge_threshold = 2.0  # °C from edges

        low_edge = (data["temp_clean"] < min_temp + edge_threshold) & (
            data["temp_clean"] >= min_temp
        )
        high_edge = (data["temp_clean"] > max_temp - edge_threshold) & (
            data["temp_clean"] <= max_temp
        )

        quality_scores.loc[low_edge | high_edge] *= 0.8

        return quality_scores

    def _calculate_cleaning_metadata(self, data: pd.DataFrame, moment: str) -> Dict:
        """
        Calculate comprehensive metadata about the cleaning process.

        Args:
            data: Cleaned DataFrame
            moment: Moment name

        Returns:
            Dictionary with cleaning statistics
        """
        total_samples = len(data)
        valid_samples = (~data["temp_clean"].isna()).sum()
        outlier_samples = data["temp_outliers"].sum()
        interpolated_samples = data["temp_interpolated"].sum()

        # Calculate overall quality score
        mean_quality = data["temp_quality"].mean()

        # Temperature statistics for valid data
        valid_temp = data["temp_clean"].dropna()
        temp_stats = {}
        if len(valid_temp) > 0:
            temp_stats = {
                "temp_mean": float(valid_temp.mean()),
                "temp_std": float(valid_temp.std()),
                "temp_min": float(valid_temp.min()),
                "temp_max": float(valid_temp.max()),
                "temp_range": float(valid_temp.max() - valid_temp.min()),
            }

        metadata = {
            "moment": moment,
            "total_samples": int(total_samples),
            "valid_samples": int(valid_samples),
            "outlier_samples": int(outlier_samples),
            "interpolated_samples": int(interpolated_samples),
            "data_completeness": float(valid_samples / total_samples)
            if total_samples > 0
            else 0.0,
            "quality_score": float(mean_quality),
            "outlier_percentage": float(outlier_samples / total_samples * 100)
            if total_samples > 0
            else 0.0,
            "interpolated_percentage": float(interpolated_samples / total_samples * 100)
            if total_samples > 0
            else 0.0,
            "processing_parameters": {
                "outlier_threshold_celsius": self.outlier_threshold,
                "jump_threshold_celsius": self.jump_threshold,
                "interpolation_max_gap_seconds": self.interpolation_max_gap,
                "sampling_rate_hz": self.sampling_rate,
            },
        }

        # Add temperature statistics
        metadata.update(temp_stats)

        return metadata

    def validate_cleaning_quality(self, metadata: Dict) -> Tuple[bool, str]:
        """
        Validate the quality of cleaning results.

        Args:
            metadata: Cleaning metadata from clean_signal()

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []

        # Check minimum samples
        if metadata["valid_samples"] < self.quality_min_samples:
            issues.append(
                f"Too few valid samples: {metadata['valid_samples']} < {self.quality_min_samples}"
            )

        # Check data completeness
        if metadata["data_completeness"] < 0.7:
            issues.append(f"Low data completeness: {metadata['data_completeness']:.1%}")

        # Check quality score
        if metadata["quality_score"] < 0.6:
            issues.append(f"Low quality score: {metadata['quality_score']:.3f}")

        # Check excessive outliers
        if metadata["outlier_percentage"] > 20:
            issues.append(f"Excessive outliers: {metadata['outlier_percentage']:.1f}%")

        is_valid = len(issues) == 0
        message = "Quality validation passed" if is_valid else "; ".join(issues)

        return is_valid, message
