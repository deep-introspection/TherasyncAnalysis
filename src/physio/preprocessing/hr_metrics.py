"""
HR Metrics Extractor for TherasyncPipeline.

This module extracts comprehensive Heart Rate (HR) metrics from cleaned HR data,
providing 25 metrics across 5 categories for psychophysiological analysis.

Authors: Lena Adel, Remy Ramadour
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np
from scipy import stats

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class HRMetricsExtractor:
    """
    Extract comprehensive HR metrics from cleaned data.
    
    This class extracts 25 HR metrics across 5 categories:
    1. Descriptive Statistics (7): Mean, SD, Min, Max, Range, Median, IQR
    2. Trend Analysis (5): Slope, Initial_HR, Final_HR, HR_Change, Peak_Time
    3. Stability Metrics (4): HR_Stability, RMSSD_Simple, CV, MAD
    4. Response Patterns (6): Elevated_Percent, Recovery_Rate, Acceleration, etc.
    5. Contextual Metrics (3): Duration, Valid_Samples, Quality_Score
    
    Note: This is distinct from HRV metrics (already extracted from BVP pipeline).
    HR metrics focus on beat-to-beat heart rate patterns, not inter-beat intervals.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the HR metrics extractor.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        
        # Get HR metrics configuration
        hr_config = self.config.get('physio.hr.metrics', {})
        
        self.baseline_window = hr_config.get('baseline_window', 60)  # seconds
        self.elevated_threshold = hr_config.get('elevated_threshold', 1.2)  # 20% above baseline
        self.stability_window = hr_config.get('stability_window', 10)  # seconds
        
        logger.info(f"HR Metrics Extractor initialized")
    
    def extract_metrics(
        self,
        data: pd.DataFrame,
        moment: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Extract comprehensive HR metrics from cleaned data.
        
        Args:
            data: DataFrame with columns ['time', 'hr_clean', 'hr_quality']
            moment: Moment/task name for context
        
        Returns:
            Dictionary with 25 HR metrics organized by category
        
        Example:
            >>> extractor = HRMetricsExtractor()
            >>> metrics = extractor.extract_metrics(cleaned_data, moment='therapy')
            >>> print(f"Mean HR: {metrics['descriptive']['hr_mean']:.1f} BPM")
        """
        logger.info(f"Extracting HR metrics for moment '{moment}' ({len(data)} samples)")
        
        if len(data) == 0:
            return self._empty_metrics(moment)
        
        # Get valid HR data (uppercase = NeuroKit/BIDS convention)
        hr_col = 'HR_Clean'
        quality_col = 'HR_Quality'
        
        valid_mask = ~data[hr_col].isna()
        valid_data = data[valid_mask].copy()
        
        if len(valid_data) == 0:
            logger.warning(f"No valid HR data for moment '{moment}'")
            return self._empty_metrics(moment)
        
        hr_values = valid_data[hr_col].values
        time_values = valid_data['time'].values
        quality_values = valid_data[quality_col].values if quality_col in valid_data.columns else np.ones(len(valid_data))
        
        # Extract metrics by category
        metrics = {
            'moment': moment,
            'descriptive': self._extract_descriptive_metrics(hr_values),
            'trend': self._extract_trend_metrics(hr_values, time_values),
            'stability': self._extract_stability_metrics(hr_values),
            'response': self._extract_response_metrics(hr_values, time_values),
            'contextual': self._extract_contextual_metrics(data, valid_data, quality_values)
        }
        
        # Add summary statistics
        metrics['summary'] = self._calculate_summary_stats(metrics)
        
        logger.info(
            f"HR metrics extracted for '{moment}': "
            f"mean {metrics['descriptive']['hr_mean']:.1f} BPM, "
            f"stability {metrics['stability']['hr_stability']:.3f}"
        )
        
        return metrics
    
    def _extract_descriptive_metrics(self, hr_values: np.ndarray) -> Dict[str, float]:
        """
        Extract descriptive statistics (7 metrics).
        
        Args:
            hr_values: Array of HR values in BPM
        
        Returns:
            Dictionary with descriptive metrics
        """
        return {
            'hr_mean': float(np.mean(hr_values)),
            'hr_std': float(np.std(hr_values, ddof=1)),
            'hr_min': float(np.min(hr_values)),
            'hr_max': float(np.max(hr_values)),
            'hr_range': float(np.max(hr_values) - np.min(hr_values)),
            'hr_median': float(np.median(hr_values)),
            'hr_iqr': float(np.percentile(hr_values, 75) - np.percentile(hr_values, 25))
        }
    
    def _extract_trend_metrics(self, hr_values: np.ndarray, time_values: np.ndarray) -> Dict[str, float]:
        """
        Extract trend analysis metrics (5 metrics).
        
        Args:
            hr_values: Array of HR values in BPM
            time_values: Array of time values in seconds
        
        Returns:
            Dictionary with trend metrics
        """
        # Linear regression for overall slope
        slope, _, r_value, _, _ = stats.linregress(time_values, hr_values)
        
        # Initial and final HR (average of first/last 10% of data)
        n_samples = len(hr_values)
        edge_samples = max(1, n_samples // 10)
        
        initial_hr = np.mean(hr_values[:edge_samples])
        final_hr = np.mean(hr_values[-edge_samples:])
        
        # Peak time (time of maximum HR)
        peak_idx = np.argmax(hr_values)
        peak_time = time_values[peak_idx] - time_values[0]  # Relative to start
        
        return {
            'hr_slope': float(slope),  # BPM per second
            'hr_slope_r2': float(r_value ** 2),  # R-squared of linear fit
            'hr_initial': float(initial_hr),
            'hr_final': float(final_hr),
            'hr_change': float(final_hr - initial_hr),
            'hr_peak_time': float(peak_time)  # Seconds from start
        }
    
    def _extract_stability_metrics(self, hr_values: np.ndarray) -> Dict[str, float]:
        """
        Extract stability metrics (4 metrics).
        
        Args:
            hr_values: Array of HR values in BPM
        
        Returns:
            Dictionary with stability metrics
        """
        # HR Stability (inverse of coefficient of variation)
        cv = np.std(hr_values) / np.mean(hr_values)
        hr_stability = 1.0 / (1.0 + cv)
        
        # RMSSD for HR (simple successive differences)
        hr_diff = np.diff(hr_values)
        rmssd_simple = np.sqrt(np.mean(hr_diff ** 2))
        
        # Coefficient of variation
        hr_cv = cv
        
        # Median Absolute Deviation
        median_hr = np.median(hr_values)
        mad = np.median(np.abs(hr_values - median_hr))
        
        return {
            'hr_stability': float(hr_stability),
            'hr_rmssd_simple': float(rmssd_simple),
            'hr_cv': float(hr_cv),
            'hr_mad': float(mad)
        }
    
    def _extract_response_metrics(self, hr_values: np.ndarray, time_values: np.ndarray) -> Dict[str, float]:
        """
        Extract response pattern metrics (6 metrics).
        
        Args:
            hr_values: Array of HR values in BPM
            time_values: Array of time values in seconds
        
        Returns:
            Dictionary with response metrics
        """
        # Baseline HR (first portion of data)
        baseline_duration = min(self.baseline_window, (time_values[-1] - time_values[0]) / 3)
        baseline_mask = time_values <= (time_values[0] + baseline_duration)
        baseline_hr = np.mean(hr_values[baseline_mask]) if baseline_mask.any() else np.mean(hr_values)
        
        # Elevated HR percentage (above threshold)
        elevated_threshold = baseline_hr * self.elevated_threshold
        elevated_percent = np.mean(hr_values > elevated_threshold) * 100
        
        # Recovery rate (slope of HR decline after peak)
        peak_idx = np.argmax(hr_values)
        if peak_idx < len(hr_values) - 5:  # Need at least 5 points for recovery
            recovery_data = hr_values[peak_idx:]
            recovery_time = time_values[peak_idx:] - time_values[peak_idx]
            recovery_slope, _, _, _, _ = stats.linregress(recovery_time, recovery_data)
        else:
            recovery_slope = 0.0
        
        # HR acceleration (second derivative)
        if len(hr_values) >= 3:
            hr_velocity = np.gradient(hr_values)
            hr_acceleration = np.gradient(hr_velocity)
            mean_acceleration = np.mean(np.abs(hr_acceleration))
            max_acceleration = np.max(np.abs(hr_acceleration))
        else:
            mean_acceleration = 0.0
            max_acceleration = 0.0
        
        # Response latency (time to reach 10% of max response)
        max_response = np.max(hr_values) - baseline_hr
        target_hr = baseline_hr + 0.1 * max_response
        response_mask = hr_values >= target_hr
        response_latency = time_values[response_mask][0] - time_values[0] if response_mask.any() else 0.0
        
        return {
            'hr_baseline': float(baseline_hr),
            'hr_elevated_percent': float(elevated_percent),
            'hr_recovery_rate': float(recovery_slope),  # BPM per second
            'hr_mean_acceleration': float(mean_acceleration),
            'hr_max_acceleration': float(max_acceleration),
            'hr_response_latency': float(response_latency)  # Seconds
        }
    
    def _extract_contextual_metrics(
        self,
        original_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        quality_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract contextual metrics (3 metrics).
        
        Args:
            original_data: Original DataFrame with all samples
            valid_data: DataFrame with only valid samples
            quality_values: Array of quality scores
        
        Returns:
            Dictionary with contextual metrics
        """
        total_duration = original_data['time'].iloc[-1] - original_data['time'].iloc[0]
        valid_samples = len(valid_data)
        mean_quality = np.mean(quality_values)
        
        return {
            'hr_duration': float(total_duration),  # Total duration in seconds
            'hr_valid_samples': int(valid_samples),  # Number of valid samples
            'hr_mean_quality': float(mean_quality)  # Average quality score
        }
    
    def _calculate_summary_stats(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics across all metrics.
        
        Args:
            metrics: Dictionary with all extracted metrics
        
        Returns:
            Dictionary with summary statistics
        """
        # Count metrics by category
        category_counts = {}
        total_metrics = 0
        
        for category, category_metrics in metrics.items():
            if category in ['moment', 'summary']:
                continue
            count = len(category_metrics)
            category_counts[f'{category}_count'] = count
            total_metrics += count
        
        # Quality assessment
        quality_score = metrics['contextual']['hr_mean_quality']
        stability_score = metrics['stability']['hr_stability']
        
        # Overall assessment
        if quality_score >= 0.9 and stability_score >= 0.7:
            overall_quality = "excellent"
        elif quality_score >= 0.8 and stability_score >= 0.6:
            overall_quality = "good"
        elif quality_score >= 0.7 and stability_score >= 0.5:
            overall_quality = "acceptable"
        else:
            overall_quality = "poor"
        
        return {
            'total_metrics_extracted': total_metrics,
            'overall_quality_assessment': overall_quality,
            'extraction_success': True,
            **category_counts
        }
    
    def _empty_metrics(self, moment: str) -> Dict[str, Any]:
        """
        Return empty metrics structure for failed extractions.
        
        Args:
            moment: Moment name
        
        Returns:
            Dictionary with NaN values for all metrics
        """
        nan_value = float('nan')
        
        return {
            'moment': moment,
            'descriptive': {
                'hr_mean': nan_value,
                'hr_std': nan_value,
                'hr_min': nan_value,
                'hr_max': nan_value,
                'hr_range': nan_value,
                'hr_median': nan_value,
                'hr_iqr': nan_value
            },
            'trend': {
                'hr_slope': nan_value,
                'hr_slope_r2': nan_value,
                'hr_initial': nan_value,
                'hr_final': nan_value,
                'hr_change': nan_value,
                'hr_peak_time': nan_value
            },
            'stability': {
                'hr_stability': nan_value,
                'hr_rmssd_simple': nan_value,
                'hr_cv': nan_value,
                'hr_mad': nan_value
            },
            'response': {
                'hr_baseline': nan_value,
                'hr_elevated_percent': nan_value,
                'hr_recovery_rate': nan_value,
                'hr_mean_acceleration': nan_value,
                'hr_max_acceleration': nan_value,
                'hr_response_latency': nan_value
            },
            'contextual': {
                'hr_duration': 0.0,
                'hr_valid_samples': 0,
                'hr_mean_quality': 0.0
            },
            'summary': {
                'total_metrics_extracted': 0,
                'overall_quality_assessment': "failed",
                'extraction_success': False,
                'descriptive_count': 0,
                'trend_count': 0,
                'stability_count': 0,
                'response_count': 0,
                'contextual_count': 0
            }
        }
    
    def extract_session_metrics(
        self,
        processed_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract HR metrics for multiple moments, returning flat Dict format.
        
        This method provides the same output format as BVPMetricsExtractor.extract_session_metrics()
        for consistency across modalities.
        
        Args:
            processed_results: Dictionary mapping moment names to cleaned DataFrames
                             (output from HRCleaner with HR_Clean column)
        
        Returns:
            Dictionary with flattened metrics for each moment.
            Format: {moment: {metric_name: value}}
        
        Example:
            >>> results = {'restingstate': cleaned_rest, 'therapy': cleaned_therapy}
            >>> all_metrics = extractor.extract_session_metrics(results)
            >>> print(all_metrics['restingstate']['hr_mean'])
        """
        logger.info(f"Extracting HR metrics for {len(processed_results)} moments (dict format)")
        
        session_metrics = {}
        
        for moment_name, cleaned_data in processed_results.items():
            try:
                # Get nested metrics structure
                nested_metrics = self.extract_metrics(cleaned_data, moment=moment_name)
                
                # Flatten to single-level dict
                flat_metrics = self._flatten_metrics(nested_metrics)
                session_metrics[moment_name] = flat_metrics
                
            except Exception as e:
                logger.error(f"Error extracting metrics for moment '{moment_name}': {str(e)}")
                session_metrics[moment_name] = {}
        
        logger.info(f"Successfully extracted metrics for {len(session_metrics)} moments")
        
        return session_metrics
    
    def extract_metrics_dataframe(
        self,
        processed_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extract HR metrics and return as DataFrame.
        
        This method provides the same output format as EDAMetricsExtractor.extract_multiple_moments()
        for consistency across modalities.
        
        Args:
            processed_results: Dictionary mapping moment names to cleaned DataFrames
        
        Returns:
            DataFrame with one row per moment containing all metrics
        
        Example:
            >>> metrics_df = extractor.extract_metrics_dataframe(results)
            >>> print(metrics_df[['moment', 'hr_mean', 'hr_stability']])
        """
        # Get metrics as dict
        session_metrics = self.extract_session_metrics(processed_results)
        
        # Convert to DataFrame
        rows = []
        for moment, metrics in session_metrics.items():
            row = {'moment': moment}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        logger.info(f"Extracted metrics DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def _flatten_metrics(self, nested_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Flatten nested metrics structure to single-level dictionary.
        
        Args:
            nested_metrics: Nested dictionary with category keys
        
        Returns:
            Flat dictionary with all metrics
        """
        flat = {}
        
        for key, value in nested_metrics.items():
            if key in ['moment', 'summary']:
                # Skip moment (redundant) and summary (metadata)
                continue
            
            if isinstance(value, dict):
                # Flatten nested category
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, (int, float)):
                        flat[metric_name] = float(metric_value)
            elif isinstance(value, (int, float)):
                flat[key] = float(value)
        
        return flat
    
    def get_metrics_description(self) -> Dict[str, Dict[str, str]]:
        """
        Get detailed descriptions of all HR metrics.
        
        Returns:
            Dictionary with metric descriptions by category
        """
        return {
            'descriptive': {
                'hr_mean': 'Mean heart rate in BPM',
                'hr_std': 'Standard deviation of heart rate',
                'hr_min': 'Minimum heart rate in BPM',
                'hr_max': 'Maximum heart rate in BPM',
                'hr_range': 'Heart rate range (max - min) in BPM',
                'hr_median': 'Median heart rate in BPM',
                'hr_iqr': 'Interquartile range of heart rate'
            },
            'trend': {
                'hr_slope': 'Linear trend slope in BPM per second',
                'hr_slope_r2': 'R-squared of linear trend fit',
                'hr_initial': 'Initial heart rate (first 10%) in BPM',
                'hr_final': 'Final heart rate (last 10%) in BPM',
                'hr_change': 'Overall change (final - initial) in BPM',
                'hr_peak_time': 'Time of peak heart rate in seconds'
            },
            'stability': {
                'hr_stability': 'Heart rate stability index (0-1)',
                'hr_rmssd_simple': 'Root mean square of successive differences',
                'hr_cv': 'Coefficient of variation',
                'hr_mad': 'Median absolute deviation'
            },
            'response': {
                'hr_baseline': 'Baseline heart rate in BPM',
                'hr_elevated_percent': 'Percentage of time above elevated threshold',
                'hr_recovery_rate': 'Recovery rate after peak in BPM/second',
                'hr_mean_acceleration': 'Mean heart rate acceleration',
                'hr_max_acceleration': 'Maximum heart rate acceleration',
                'hr_response_latency': 'Response latency in seconds'
            },
            'contextual': {
                'hr_duration': 'Total duration of recording in seconds',
                'hr_valid_samples': 'Number of valid samples',
                'hr_mean_quality': 'Mean quality score (0-1)'
            }
        }