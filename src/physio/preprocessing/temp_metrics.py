"""
Temperature Metrics Extractor for TherasyncPipeline.

This module extracts comprehensive peripheral skin temperature metrics from
cleaned temperature data, providing 14 metrics across 4 categories for
psychophysiological analysis.

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


class TEMPMetricsExtractor:
    """
    Extract comprehensive temperature metrics from cleaned data.
    
    This class extracts 14 temperature metrics across 4 categories:
    1. Descriptive Statistics (7): Mean, SD, Min, Max, Range, Median, IQR
    2. Trend Analysis (5): Slope, Initial, Final, Change, Slope_R2
    3. Stability Metrics (2): Stability, CV (coefficient of variation)
    4. Contextual Metrics (3): Duration, Valid_Samples, Quality_Score
    
    Note: Temperature signals are slow-changing and reflect peripheral
    vasoconstriction/vasodilation patterns related to autonomic activity.
    Temperature typically:
    - Decreases during stress/anxiety (peripheral vasoconstriction)
    - Increases during relaxation (peripheral vasodilation)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the temperature metrics extractor.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        
        # Get temperature metrics configuration
        temp_config = self.config.get('physio.temp.metrics', {})
        
        self.baseline_window = temp_config.get('baseline_window', 60)  # seconds
        
        logger.info("Temperature Metrics Extractor initialized")
    
    def extract_metrics(
        self,
        data: pd.DataFrame,
        moment: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Extract comprehensive temperature metrics from cleaned data.
        
        Args:
            data: DataFrame with columns ['time', 'TEMP_Clean', 'TEMP_Quality']
            moment: Moment/task name for context
        
        Returns:
            Dictionary with 14+ temperature metrics organized by category
        
        Example:
            >>> extractor = TEMPMetricsExtractor()
            >>> metrics = extractor.extract_metrics(cleaned_data, moment='therapy')
            >>> print(f"Mean Temp: {metrics['descriptive']['temp_mean']:.2f} °C")
        """
        logger.info(f"Extracting temperature metrics for moment '{moment}' ({len(data)} samples)")
        
        if len(data) == 0:
            return self._empty_metrics(moment)
        
        # Column names (uppercase = NeuroKit/BIDS convention)
        clean_col = 'TEMP_Clean'
        quality_col = 'TEMP_Quality'
        
        if clean_col not in data.columns:
            logger.error(f"Required column {clean_col} not found in data")
            return self._empty_metrics(moment)
        
        # Get valid temperature data
        valid_mask = ~data[clean_col].isna()
        valid_data = data[valid_mask].copy()
        
        if len(valid_data) == 0:
            logger.warning(f"No valid temperature data for moment '{moment}'")
            return self._empty_metrics(moment)
        
        temp_values = valid_data[clean_col].values
        time_values = valid_data['time'].values
        
        # Quality values (use default 1.0 if not available)
        if quality_col in valid_data.columns:
            quality_values = valid_data[quality_col].values
        else:
            quality_values = np.ones(len(valid_data))
        
        # Extract metrics by category
        metrics = {
            'moment': moment,
            'descriptive': self._extract_descriptive_metrics(temp_values),
            'trend': self._extract_trend_metrics(temp_values, time_values),
            'stability': self._extract_stability_metrics(temp_values),
            'contextual': self._extract_contextual_metrics(data, valid_data, quality_values, clean_col)
        }
        
        # Add summary statistics
        metrics['summary'] = self._calculate_summary_stats(metrics)
        
        logger.info(
            f"Temperature metrics extracted for '{moment}': "
            f"mean {metrics['descriptive']['temp_mean']:.2f} °C, "
            f"change {metrics['trend']['temp_change']:.2f} °C"
        )
        
        return metrics
    
    def _extract_descriptive_metrics(self, temp_values: np.ndarray) -> Dict[str, float]:
        """
        Extract descriptive statistics (7 metrics).
        
        Args:
            temp_values: Array of temperature values in °C
        
        Returns:
            Dictionary with descriptive metrics
        """
        return {
            'temp_mean': float(np.mean(temp_values)),
            'temp_std': float(np.std(temp_values, ddof=1)) if len(temp_values) > 1 else 0.0,
            'temp_min': float(np.min(temp_values)),
            'temp_max': float(np.max(temp_values)),
            'temp_range': float(np.max(temp_values) - np.min(temp_values)),
            'temp_median': float(np.median(temp_values)),
            'temp_iqr': float(np.percentile(temp_values, 75) - np.percentile(temp_values, 25))
        }
    
    def _extract_trend_metrics(self, temp_values: np.ndarray, time_values: np.ndarray) -> Dict[str, float]:
        """
        Extract trend analysis metrics (5 metrics).
        
        Temperature trends are particularly meaningful:
        - Increasing trend suggests relaxation (vasodilation)
        - Decreasing trend suggests stress/anxiety (vasoconstriction)
        
        Args:
            temp_values: Array of temperature values in °C
            time_values: Array of time values in seconds
        
        Returns:
            Dictionary with trend metrics
        """
        # Linear regression for overall slope
        if len(temp_values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_values, temp_values)
        else:
            slope, r_value = 0.0, 0.0
        
        # Initial and final temperature (average of first/last 10% of data)
        n_samples = len(temp_values)
        edge_samples = max(1, n_samples // 10)
        
        initial_temp = np.mean(temp_values[:edge_samples])
        final_temp = np.mean(temp_values[-edge_samples:])
        
        return {
            'temp_slope': float(slope),  # °C per second
            'temp_slope_per_minute': float(slope * 60),  # °C per minute (more interpretable)
            'temp_slope_r2': float(r_value ** 2),  # R-squared of linear fit
            'temp_initial': float(initial_temp),
            'temp_final': float(final_temp),
            'temp_change': float(final_temp - initial_temp)
        }
    
    def _extract_stability_metrics(self, temp_values: np.ndarray) -> Dict[str, float]:
        """
        Extract stability metrics (2 metrics).
        
        Temperature is typically very stable (low variability),
        so high variability may indicate measurement issues or
        strong autonomic reactivity.
        
        Args:
            temp_values: Array of temperature values in °C
        
        Returns:
            Dictionary with stability metrics
        """
        mean_temp = np.mean(temp_values)
        std_temp = np.std(temp_values, ddof=1) if len(temp_values) > 1 else 0.0
        
        # Coefficient of variation
        cv = std_temp / mean_temp if mean_temp > 0 else 0.0
        
        # Stability index (inverse of CV, normalized)
        stability = 1.0 / (1.0 + cv)
        
        return {
            'temp_stability': float(stability),
            'temp_cv': float(cv)
        }
    
    def _extract_contextual_metrics(
        self,
        original_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        quality_values: np.ndarray,
        clean_col: str
    ) -> Dict[str, float]:
        """
        Extract contextual metrics (3 metrics).
        
        Args:
            original_data: Original DataFrame with all samples
            valid_data: DataFrame with only valid samples
            quality_values: Array of quality scores
            clean_col: Name of the clean temperature column
        
        Returns:
            Dictionary with contextual metrics
        """
        total_duration = original_data['time'].iloc[-1] - original_data['time'].iloc[0]
        valid_samples = len(valid_data)
        mean_quality = np.mean(quality_values)
        
        return {
            'temp_duration': float(total_duration),  # Total duration in seconds
            'temp_valid_samples': int(valid_samples),  # Number of valid samples
            'temp_mean_quality': float(mean_quality)  # Average quality score
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
        
        # Quality assessment based on data quality and stability
        quality_score = metrics['contextual']['temp_mean_quality']
        stability_score = metrics['stability']['temp_stability']
        
        # Overall assessment
        if quality_score >= 0.9 and stability_score >= 0.9:
            overall_quality = "excellent"
        elif quality_score >= 0.8 and stability_score >= 0.8:
            overall_quality = "good"
        elif quality_score >= 0.7 and stability_score >= 0.7:
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
                'temp_mean': nan_value,
                'temp_std': nan_value,
                'temp_min': nan_value,
                'temp_max': nan_value,
                'temp_range': nan_value,
                'temp_median': nan_value,
                'temp_iqr': nan_value
            },
            'trend': {
                'temp_slope': nan_value,
                'temp_slope_per_minute': nan_value,
                'temp_slope_r2': nan_value,
                'temp_initial': nan_value,
                'temp_final': nan_value,
                'temp_change': nan_value
            },
            'stability': {
                'temp_stability': nan_value,
                'temp_cv': nan_value
            },
            'contextual': {
                'temp_duration': 0.0,
                'temp_valid_samples': 0,
                'temp_mean_quality': 0.0
            },
            'summary': {
                'total_metrics_extracted': 0,
                'overall_quality_assessment': "failed",
                'extraction_success': False,
                'descriptive_count': 0,
                'trend_count': 0,
                'stability_count': 0,
                'contextual_count': 0
            }
        }
    
    def extract_session_metrics(
        self,
        processed_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract temperature metrics for multiple moments, returning flat Dict format.
        
        This method provides the same output format as other metrics extractors
        for consistency across modalities.
        
        Args:
            processed_results: Dictionary mapping moment names to cleaned DataFrames
                             (output from TEMPCleaner with TEMP_Clean column)
        
        Returns:
            Dictionary with flattened metrics for each moment.
            Format: {moment: {metric_name: value}}
        
        Example:
            >>> results = {'restingstate': cleaned_rest, 'therapy': cleaned_therapy}
            >>> all_metrics = extractor.extract_session_metrics(results)
            >>> print(all_metrics['restingstate']['temp_mean'])
        """
        logger.info(f"Extracting temperature metrics for {len(processed_results)} moments (dict format)")
        
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
        Extract temperature metrics and return as DataFrame.
        
        This method provides the same output format as other metrics extractors
        for consistency across modalities.
        
        Args:
            processed_results: Dictionary mapping moment names to cleaned DataFrames
        
        Returns:
            DataFrame with one row per moment containing all metrics
        
        Example:
            >>> metrics_df = extractor.extract_metrics_dataframe(results)
            >>> print(metrics_df[['moment', 'temp_mean', 'temp_change']])
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
        Get detailed descriptions of all temperature metrics.
        
        Returns:
            Dictionary with metric descriptions by category
        """
        return {
            'descriptive': {
                'temp_mean': 'Mean peripheral skin temperature in °C',
                'temp_std': 'Standard deviation of temperature',
                'temp_min': 'Minimum temperature in °C',
                'temp_max': 'Maximum temperature in °C',
                'temp_range': 'Temperature range (max - min) in °C',
                'temp_median': 'Median temperature in °C',
                'temp_iqr': 'Interquartile range of temperature'
            },
            'trend': {
                'temp_slope': 'Linear trend slope in °C per second',
                'temp_slope_per_minute': 'Linear trend slope in °C per minute (more interpretable)',
                'temp_slope_r2': 'R-squared of linear trend fit',
                'temp_initial': 'Initial temperature (first 10%) in °C',
                'temp_final': 'Final temperature (last 10%) in °C',
                'temp_change': 'Overall change (final - initial) in °C'
            },
            'stability': {
                'temp_stability': 'Temperature stability index (0-1, higher = more stable)',
                'temp_cv': 'Coefficient of variation (lower = more stable)'
            },
            'contextual': {
                'temp_duration': 'Total duration of recording in seconds',
                'temp_valid_samples': 'Number of valid samples',
                'temp_mean_quality': 'Mean quality score (0-1)'
            }
        }
