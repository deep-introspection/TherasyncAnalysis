"""
HR Data Cleaner for TherasyncPipeline.

This module provides functionality to clean Heart Rate (HR) data from Empatica devices,
including outlier removal, gap interpolation, and quality assessment.

Authors: Lena Adel, Remy Ramadour
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class HRCleaner:
    """
    Clean and preprocess HR data with conservative approach.
    
    This class handles:
    - Physiological outlier removal (< 40 or > 180 BPM)
    - Short gap interpolation (< 5 seconds)
    - Quality assessment and scoring
    - Data validation
    
    HR data characteristics:
    - Sampling rate: 1 Hz
    - Unit: BPM (beats per minute)
    - Expected range: 40-180 BPM for most populations
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the HR cleaner with configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        
        # Get HR cleaning configuration
        hr_config = self.config.get('physio.hr.processing', {})
        
        self.outlier_threshold = hr_config.get('outlier_threshold', [40, 180])
        self.interpolation_max_gap = hr_config.get('interpolation_max_gap', 5)  # seconds
        self.quality_min_samples = hr_config.get('quality_min_samples', 30)
        
        # Get sampling rate
        self.sampling_rate = self.config.get('physio.hr.sampling_rate', 1)
        
        logger.info(
            f"HR Cleaner initialized (outliers: {self.outlier_threshold} BPM, "
            f"max gap: {self.interpolation_max_gap}s, sampling rate: {self.sampling_rate} Hz)"
        )
    
    def clean_signal(
        self,
        data: pd.DataFrame,
        moment: str = "unknown"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean HR signal with conservative approach.
        
        Args:
            data: DataFrame with columns ['time', 'hr']
            moment: Moment/task name for logging
        
        Returns:
            Tuple of:
                - Cleaned DataFrame with standardized columns:
                  ['time', 'HR_Raw', 'HR_Clean', 'HR_Outliers', 'HR_Interpolated', 'HR_Quality']
                - Processing metadata dictionary
        
        Example:
            >>> cleaner = HRCleaner()
            >>> cleaned_data, metadata = cleaner.clean_signal(raw_data, moment='restingstate')
            >>> print(f"Quality score: {metadata['quality_score']:.3f}")
        """
        logger.info(f"Cleaning HR signal for moment '{moment}' ({len(data)} samples)")
        
        if len(data) == 0:
            raise ValueError("Empty HR data provided")
        
        # Create working copy with internal column names (for processing)
        result = data.copy()
        
        # Initialize processing columns with internal names
        result['hr_clean'] = result['hr'].copy()
        result['hr_outliers'] = False
        result['hr_interpolated'] = False
        result['hr_quality'] = 1.0  # Will be updated per sample
        
        # Step 1: Remove physiological outliers
        outlier_mask = self._detect_outliers(result['hr'])
        result.loc[outlier_mask, 'hr_outliers'] = True
        result.loc[outlier_mask, 'hr_clean'] = np.nan
        
        outlier_count = outlier_mask.sum()
        logger.info(f"Detected {outlier_count} outliers ({outlier_count/len(data)*100:.1f}%)")
        
        # Step 2: Interpolate short gaps
        interpolated_mask = self._interpolate_gaps(result)
        interpolated_count = interpolated_mask.sum()
        logger.info(f"Interpolated {interpolated_count} samples in short gaps")
        
        # Step 3: Calculate quality scores
        quality_scores = self._calculate_quality(result)
        result['hr_quality'] = quality_scores
        
        # Step 4: Calculate overall metrics
        metadata = self._calculate_cleaning_metadata(result, moment)
        
        # Step 5: Rename columns to standardized convention (BIDS-compliant)
        # From internal names to final output names
        result = result.rename(columns={
            'hr': 'HR_Raw',
            'hr_clean': 'HR_Clean',
            'hr_outliers': 'HR_Outliers',
            'hr_interpolated': 'HR_Interpolated',
            'hr_quality': 'HR_Quality'
        })
        
        logger.info(
            f"HR cleaning complete for '{moment}': "
            f"quality {metadata['quality_score']:.3f}, "
            f"{metadata['valid_samples']}/{metadata['total_samples']} valid samples"
        )
        
        return result, metadata
    
    def _detect_outliers(self, hr_series: pd.Series) -> pd.Series:
        """
        Detect physiological outliers in HR data.
        
        Args:
            hr_series: HR values to check
        
        Returns:
            Boolean mask indicating outliers
        """
        min_hr, max_hr = self.outlier_threshold
        
        # Physiological outliers
        physiological_outliers = (hr_series < min_hr) | (hr_series > max_hr)
        
        # NaN values are also considered outliers
        nan_outliers = hr_series.isna()
        
        return physiological_outliers | nan_outliers
    
    def _interpolate_gaps(self, data: pd.DataFrame) -> pd.Series:
        """
        Interpolate short gaps in HR data.
        
        Args:
            data: DataFrame with HR data and outlier flags
        
        Returns:
            Boolean mask indicating interpolated samples
        """
        interpolated_mask = pd.Series(False, index=data.index)
        
        # Find gaps (NaN values in hr_clean)
        nan_mask = data['hr_clean'].isna()
        
        if not nan_mask.any():
            return interpolated_mask
        
        # Find consecutive gaps
        gap_groups = self._find_gap_groups(nan_mask, data['time'])
        
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
            gap_starts = [nan_mask.index[0]] + gap_starts.tolist()
        if nan_mask.iloc[-1]:
            gap_ends = gap_ends.tolist() + [nan_mask.index[-1]]
        
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
        interpolated_mask: pd.Series
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
        
        if pd.isna(data.loc[before_idx, 'hr_clean']) or pd.isna(data.loc[after_idx, 'hr_clean']):
            return
        
        # Linear interpolation
        gap_indices = list(range(gap_start, gap_end + 1))
        time_points = data.loc[gap_indices, 'time'].values
        
        # Interpolate between valid endpoints
        start_time = data.loc[before_idx, 'time']
        end_time = data.loc[after_idx, 'time']
        start_hr = data.loc[before_idx, 'hr_clean']
        end_hr = data.loc[after_idx, 'hr_clean']
        
        # Linear interpolation
        interpolated_hr = np.interp(time_points, [start_time, end_time], [start_hr, end_hr])
        
        # Update data
        data.loc[gap_indices, 'hr_clean'] = interpolated_hr
        data.loc[gap_indices, 'hr_interpolated'] = True
        interpolated_mask.loc[gap_indices] = True
    
    def _calculate_quality(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate quality scores for each sample.
        
        Args:
            data: DataFrame with cleaned HR data
        
        Returns:
            Series of quality scores (0-1)
        """
        quality_scores = pd.Series(1.0, index=data.index)
        
        # Reduce quality for interpolated samples
        quality_scores.loc[data['hr_interpolated']] = 0.7
        
        # Zero quality for remaining NaN values
        quality_scores.loc[data['hr_clean'].isna()] = 0.0
        
        # Reduce quality for values at the edges of physiological range
        min_hr, max_hr = self.outlier_threshold
        edge_threshold = 10  # BPM from edges
        
        low_edge = (data['hr_clean'] < min_hr + edge_threshold) & (data['hr_clean'] >= min_hr)
        high_edge = (data['hr_clean'] > max_hr - edge_threshold) & (data['hr_clean'] <= max_hr)
        
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
        valid_samples = (~data['hr_clean'].isna()).sum()
        outlier_samples = data['hr_outliers'].sum()
        interpolated_samples = data['hr_interpolated'].sum()
        
        # Calculate overall quality score
        mean_quality = data['hr_quality'].mean()
        
        # HR statistics for valid data
        valid_hr = data['hr_clean'].dropna()
        hr_stats = {}
        if len(valid_hr) > 0:
            hr_stats = {
                'hr_mean': float(valid_hr.mean()),
                'hr_std': float(valid_hr.std()),
                'hr_min': float(valid_hr.min()),
                'hr_max': float(valid_hr.max()),
                'hr_range': float(valid_hr.max() - valid_hr.min())
            }
        
        metadata = {
            'moment': moment,
            'total_samples': int(total_samples),
            'valid_samples': int(valid_samples),
            'outlier_samples': int(outlier_samples),
            'interpolated_samples': int(interpolated_samples),
            'data_completeness': float(valid_samples / total_samples) if total_samples > 0 else 0.0,
            'quality_score': float(mean_quality),
            'outlier_percentage': float(outlier_samples / total_samples * 100) if total_samples > 0 else 0.0,
            'interpolated_percentage': float(interpolated_samples / total_samples * 100) if total_samples > 0 else 0.0,
            'processing_parameters': {
                'outlier_threshold_bpm': self.outlier_threshold,
                'interpolation_max_gap_seconds': self.interpolation_max_gap,
                'sampling_rate_hz': self.sampling_rate
            }
        }
        
        # Add HR statistics
        metadata.update(hr_stats)
        
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
        if metadata['valid_samples'] < self.quality_min_samples:
            issues.append(f"Too few valid samples: {metadata['valid_samples']} < {self.quality_min_samples}")
        
        # Check data completeness
        if metadata['data_completeness'] < 0.7:
            issues.append(f"Low data completeness: {metadata['data_completeness']:.1%}")
        
        # Check quality score
        if metadata['quality_score'] < 0.6:
            issues.append(f"Low quality score: {metadata['quality_score']:.3f}")
        
        # Check excessive outliers
        if metadata['outlier_percentage'] > 20:
            issues.append(f"Excessive outliers: {metadata['outlier_percentage']:.1f}%")
        
        is_valid = len(issues) == 0
        message = "Quality validation passed" if is_valid else "; ".join(issues)
        
        return is_valid, message