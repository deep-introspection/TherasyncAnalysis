"""
BVP Metrics Extractor for TherasyncPipeline.

This module provides functionality to extract Heart Rate Variability (HRV) and other
cardiovascular metrics from processed BVP data using NeuroKit2.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import neurokit2 as nk

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class BVPMetricsExtractor:
    """
    Extract HRV and cardiovascular metrics from processed BVP data.

    This class implements the essential HRV metrics extraction using NeuroKit2,
    supporting both session-level analysis and future epoched analysis capabilities.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BVP metrics extractor with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)

        # Get BVP metrics configuration
        self.bvp_config = self.config.get("physio.bvp", {})
        self.metrics_config = self.bvp_config.get("metrics", {})

        # Get selected metrics
        self.selected_metrics = self.metrics_config.get("selected_metrics", {})
        self.time_domain_metrics = self.selected_metrics.get("time_domain", [])
        self.frequency_domain_metrics = self.selected_metrics.get(
            "frequency_domain", []
        )
        self.nonlinear_metrics = self.selected_metrics.get("nonlinear", [])

        # Epoched analysis configuration (for future implementation)
        self.epoched_config = self.metrics_config.get("epoched_analysis", {})
        self.epoched_enabled = self.epoched_config.get("enabled", False)

        # Minimum duration for frequency-domain HRV (Task Force 1996: >=2 min)
        self.min_duration_freq = self.metrics_config.get(
            "min_duration_frequency_domain", 120
        )

        logger.info(
            f"BVP Metrics Extractor initialized: "
            f"{len(self.time_domain_metrics)} time-domain, "
            f"{len(self.frequency_domain_metrics)} frequency-domain, "
            f"{len(self.nonlinear_metrics)} nonlinear metrics"
        )

    def extract_session_metrics(
        self, processed_results: Dict[str, Tuple[pd.DataFrame, Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract HRV metrics for entire sessions/moments.

        Args:
            processed_results: Output from BVPCleaner.process_moment_signals()
                              Format: {moment: (processed_signals, processing_info)}

        Returns:
            Dictionary with extracted metrics for each moment.
            Format: {moment: {metric_name: value}}
        """
        session_metrics = {}

        for moment, (processed_signals, processing_info) in processed_results.items():
            try:
                # Extract peaks from processing info
                peaks = processing_info.get("PPG_Peaks", [])
                sampling_rate = processing_info.get("sampling_rate", 64)

                # Validate peaks for HRV analysis
                if not self._validate_peaks_for_hrv(peaks, sampling_rate, moment):
                    logger.warning(
                        f"Skipping HRV analysis for {moment}: insufficient peaks"
                    )
                    session_metrics[moment] = self._get_empty_metrics_dict()
                    continue

                # Extract HRV metrics
                moment_metrics = self._extract_hrv_metrics(peaks, sampling_rate, moment)

                # Add basic signal quality metrics
                quality_metrics = self._extract_signal_quality_metrics(
                    processed_signals, processing_info, moment
                )
                moment_metrics.update(quality_metrics)

                session_metrics[moment] = moment_metrics

                logger.info(
                    f"Extracted {len(moment_metrics)} metrics for {moment}: "
                    f"HRV_MeanNN={moment_metrics.get('HRV_MeanNN', 'N/A'):.1f}ms"
                )

            except Exception as e:
                logger.error(f"Failed to extract metrics for {moment}: {e}")
                session_metrics[moment] = self._get_empty_metrics_dict()
                continue

        return session_metrics

    def extract_metrics_dataframe(
        self, processed_results: Dict[str, Tuple[pd.DataFrame, Dict]]
    ) -> pd.DataFrame:
        """
        Extract HRV metrics and return as DataFrame.

        This method provides the same output format as EDAMetricsExtractor.extract_multiple_moments()
        for consistency across modalities.

        Args:
            processed_results: Output from BVPCleaner.process_moment_signals()
                              Format: {moment: (processed_signals, processing_info)}

        Returns:
            DataFrame with one row per moment containing all metrics

        Example:
            >>> metrics_df = extractor.extract_metrics_dataframe(processed_results)
            >>> print(metrics_df[['moment', 'HRV_MeanNN', 'HRV_RMSSD']])
        """
        # Get metrics as dict
        session_metrics = self.extract_session_metrics(processed_results)

        # Convert to DataFrame
        rows = []
        for moment, metrics in session_metrics.items():
            row = {"moment": moment}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        logger.info(
            f"Extracted metrics DataFrame with {len(df)} rows, {len(df.columns)} columns"
        )

        return df

    def _extract_hrv_metrics(
        self, peaks: Union[List, np.ndarray], sampling_rate: int, moment: str
    ) -> Dict[str, float]:
        """
        Extract HRV metrics from peaks using NeuroKit2.

        This method filters out physiologically invalid RR intervals (outside 300-2000ms)
        and passes the VALID RR intervals directly to NeuroKit2 for HRV computation.

        IMPORTANT: We pass RR intervals directly (not peaks) to avoid the bug where
        removing a peak creates a new invalid interval between its neighbors.

        Args:
            peaks: Array of peak indices
            sampling_rate: Sampling rate in Hz
            moment: Moment name for logging

        Returns:
            Dictionary of extracted HRV metrics
        """
        metrics = {}
        peaks_array = np.array(peaks)

        # Get RR interval configuration
        rr_config = self.bvp_config.get("rr_intervals", {})
        min_valid_ms = rr_config.get("min_valid_ms", 300)
        max_valid_ms = rr_config.get("max_valid_ms", 2000)

        try:
            # Calculate RR intervals in milliseconds
            rr_intervals_ms = np.diff(peaks_array) / sampling_rate * 1000

            # Filter to keep only physiologically valid RR intervals
            # Valid range: 300ms (200 BPM) to 2000ms (30 BPM)
            valid_mask = (rr_intervals_ms >= min_valid_ms) & (
                rr_intervals_ms <= max_valid_ms
            )
            valid_rr_ms = rr_intervals_ms[valid_mask]

            n_total = len(rr_intervals_ms)
            n_valid = len(valid_rr_ms)
            n_outliers = n_total - n_valid

            if n_outliers > 0:
                logger.info(
                    f"RR interval filtering for {moment}: "
                    f"kept {n_valid}/{n_total} valid intervals "
                    f"({100 * n_valid / n_total:.1f}%), "
                    f"removed {n_outliers} outside [{min_valid_ms}-{max_valid_ms}ms] range"
                )
            else:
                logger.debug(f"All {n_total} RR intervals valid for {moment}")

            # Check minimum requirements for HRV analysis
            if n_valid < 10:
                logger.warning(
                    f"Insufficient valid RR intervals for HRV analysis in {moment}: "
                    f"{n_valid} < 10 required"
                )
                return self._get_empty_hrv_metrics_dict()

            # Convert valid RR intervals to peaks for NeuroKit2
            # NeuroKit expects peak indices, so we reconstruct them from cumulative RR
            # Starting at 0, each peak is at cumsum of RR intervals (converted to samples)
            valid_rr_samples = valid_rr_ms / 1000 * sampling_rate
            cumulative_samples = np.concatenate([[0], np.cumsum(valid_rr_samples)])
            synthetic_peaks = cumulative_samples.astype(int)

            # Extract time-domain metrics
            if self.time_domain_metrics:
                time_metrics = nk.hrv_time(synthetic_peaks, sampling_rate=sampling_rate)
                for metric in self.time_domain_metrics:
                    if metric in time_metrics.columns:
                        metrics[metric] = float(time_metrics[metric].iloc[0])
                    else:
                        logger.warning(
                            f"Time-domain metric {metric} not found for {moment}"
                        )
                        metrics[metric] = np.nan

            # Extract frequency-domain metrics (gated on epoch duration)
            if self.frequency_domain_metrics:
                duration_s = valid_rr_ms.sum() / 1000.0
                if duration_s < self.min_duration_freq:
                    logger.warning(
                        f"Epoch duration {duration_s:.1f}s < {self.min_duration_freq}s "
                        f"minimum for frequency-domain HRV in {moment}; "
                        f"setting LF/HF metrics to NaN"
                    )
                    for metric in self.frequency_domain_metrics:
                        metrics[metric] = np.nan
                else:
                    try:
                        freq_metrics = nk.hrv_frequency(
                            synthetic_peaks, sampling_rate=sampling_rate
                        )
                        for metric in self.frequency_domain_metrics:
                            if metric in freq_metrics.columns:
                                metrics[metric] = float(freq_metrics[metric].iloc[0])
                            else:
                                logger.warning(
                                    f"Frequency-domain metric {metric} not found "
                                    f"for {moment}"
                                )
                                metrics[metric] = np.nan
                    except Exception as e:
                        logger.warning(
                            f"Frequency-domain analysis failed for {moment}: {e}"
                        )
                        for metric in self.frequency_domain_metrics:
                            metrics[metric] = np.nan

            # Extract nonlinear metrics
            if self.nonlinear_metrics:
                try:
                    nonlinear_metrics = nk.hrv_nonlinear(
                        synthetic_peaks, sampling_rate=sampling_rate
                    )
                    for metric in self.nonlinear_metrics:
                        if metric in nonlinear_metrics.columns:
                            metrics[metric] = float(nonlinear_metrics[metric].iloc[0])
                        else:
                            logger.warning(
                                f"Nonlinear metric {metric} not found for {moment}"
                            )
                            metrics[metric] = np.nan
                except Exception as e:
                    logger.warning(f"Nonlinear analysis failed for {moment}: {e}")
                    for metric in self.nonlinear_metrics:
                        metrics[metric] = np.nan

            # Log summary of computed metrics
            logger.info(
                f"HRV metrics for {moment}: "
                f"MeanNN={metrics.get('HRV_MeanNN', np.nan):.1f}ms, "
                f"SDNN={metrics.get('HRV_SDNN', np.nan):.1f}ms, "
                f"RMSSD={metrics.get('HRV_RMSSD', np.nan):.1f}ms"
            )

            return metrics

        except Exception as e:
            logger.error(f"HRV extraction failed for {moment}: {e}")
            return self._get_empty_hrv_metrics_dict()

    def _extract_signal_quality_metrics(
        self, processed_signals: pd.DataFrame, processing_info: Dict, moment: str
    ) -> Dict[str, float]:
        """
        Extract signal quality and basic metrics.

        Args:
            processed_signals: Processed signals DataFrame
            processing_info: Processing information dictionary
            moment: Moment name for logging

        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}

        try:
            # Number of detected peaks
            peaks = processing_info.get("PPG_Peaks", [])
            quality_metrics["BVP_NumPeaks"] = len(peaks)

            # Signal duration
            sampling_rate = processing_info.get("sampling_rate", 64)
            duration = len(processed_signals) / sampling_rate
            quality_metrics["BVP_Duration"] = duration

            # Peak rate (peaks per minute)
            if duration > 0:
                quality_metrics["BVP_PeakRate"] = (len(peaks) / duration) * 60
            else:
                quality_metrics["BVP_PeakRate"] = np.nan

            # Mean signal quality if available
            if "PPG_Quality" in processed_signals.columns:
                quality_scores = processed_signals["PPG_Quality"].dropna()
                if not quality_scores.empty:
                    quality_metrics["BVP_MeanQuality"] = float(quality_scores.mean())
                    quality_metrics["BVP_QualityStd"] = float(quality_scores.std())
                else:
                    quality_metrics["BVP_MeanQuality"] = np.nan
                    quality_metrics["BVP_QualityStd"] = np.nan

            # Signal amplitude metrics from cleaned signal
            if "PPG_Clean" in processed_signals.columns:
                clean_signal = processed_signals["PPG_Clean"].dropna()
                if not clean_signal.empty:
                    quality_metrics["BVP_MeanAmplitude"] = float(clean_signal.mean())
                    quality_metrics["BVP_StdAmplitude"] = float(clean_signal.std())
                    quality_metrics["BVP_RangeAmplitude"] = float(
                        clean_signal.max() - clean_signal.min()
                    )
                else:
                    quality_metrics["BVP_MeanAmplitude"] = np.nan
                    quality_metrics["BVP_StdAmplitude"] = np.nan
                    quality_metrics["BVP_RangeAmplitude"] = np.nan

        except Exception as e:
            logger.warning(f"Signal quality extraction failed for {moment}: {e}")

        return quality_metrics

    def _validate_peaks_for_hrv(
        self, peaks: Union[List, np.ndarray], sampling_rate: int, moment: str
    ) -> bool:
        """
        Validate that peaks are sufficient for HRV analysis.

        Args:
            peaks: Array of peak indices
            sampling_rate: Sampling rate in Hz
            moment: Moment name for logging

        Returns:
            True if peaks are sufficient for HRV analysis
        """
        if len(peaks) < 10:
            logger.warning(
                f"Insufficient peaks for HRV analysis in {moment}: {len(peaks)} < 10"
            )
            return False

        # Check for reasonable peak intervals (avoid artifacts)
        peaks_array = np.array(peaks)
        if len(peaks_array) > 1:
            intervals = np.diff(peaks_array)
            # Check for very short intervals (< 200ms at 64Hz = 12.8 samples)
            min_interval = 0.2 * sampling_rate  # 200ms in samples
            if np.any(intervals < min_interval):
                short_intervals = np.sum(intervals < min_interval)
                logger.warning(
                    f"Found {short_intervals} very short intervals in {moment}, "
                    f"may indicate artifacts"
                )

        return True

    def _get_empty_metrics_dict(self) -> Dict[str, float]:
        """Get dictionary with all configured metrics set to NaN."""
        metrics = {}

        # Add all configured metrics with NaN values
        for metric in self.time_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.frequency_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.nonlinear_metrics:
            metrics[metric] = np.nan

        # Add quality metrics
        quality_metrics = [
            "BVP_NumPeaks",
            "BVP_Duration",
            "BVP_PeakRate",
            "BVP_MeanQuality",
            "BVP_QualityStd",
            "BVP_MeanAmplitude",
            "BVP_StdAmplitude",
            "BVP_RangeAmplitude",
        ]
        for metric in quality_metrics:
            metrics[metric] = np.nan

        return metrics

    def _get_empty_hrv_metrics_dict(self) -> Dict[str, float]:
        """Get dictionary with HRV metrics set to NaN."""
        metrics = {}

        for metric in self.time_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.frequency_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.nonlinear_metrics:
            metrics[metric] = np.nan

        return metrics

    def get_metrics_summary(
        self, session_metrics: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Convert session metrics to a summary DataFrame.

        Args:
            session_metrics: Output from extract_session_metrics()

        Returns:
            DataFrame with moments as rows and metrics as columns
        """
        if not session_metrics:
            logger.warning("No session metrics to summarize")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(session_metrics, orient="index")

        # Add moment names as a column for convenience
        df.index.name = "moment"
        df = df.reset_index()

        logger.info(
            f"Created metrics summary: {len(df)} moments × {len(df.columns) - 1} metrics"
        )

        return df

    def compare_moments(
        self,
        session_metrics: Dict[str, Dict[str, float]],
        baseline_moment: str = "restingstate",
        comparison_moment: str = "therapy",
    ) -> Dict[str, float]:
        """
        Compare metrics between two moments (e.g., resting vs therapy).

        Args:
            session_metrics: Output from extract_session_metrics()
            baseline_moment: Name of baseline moment
            comparison_moment: Name of comparison moment

        Returns:
            Dictionary of differences (comparison - baseline)
        """
        if baseline_moment not in session_metrics:
            logger.error(
                f"Baseline moment '{baseline_moment}' not found in session metrics"
            )
            return {}

        if comparison_moment not in session_metrics:
            logger.error(
                f"Comparison moment '{comparison_moment}' not found in session metrics"
            )
            return {}

        baseline_metrics = session_metrics[baseline_moment]
        comparison_metrics = session_metrics[comparison_moment]

        differences = {}

        for metric in baseline_metrics:
            baseline_val = baseline_metrics.get(metric, np.nan)
            comparison_val = comparison_metrics.get(metric, np.nan)

            # Ensure we have valid float values
            if baseline_val is not None and comparison_val is not None:
                baseline_float = float(baseline_val)
                comparison_float = float(comparison_val)

                if not (np.isnan(baseline_float) or np.isnan(comparison_float)):
                    differences[f"{metric}_diff"] = comparison_float - baseline_float
                    if baseline_float != 0:
                        differences[f"{metric}_pct_change"] = (
                            (comparison_float - baseline_float) / baseline_float
                        ) * 100
                    else:
                        differences[f"{metric}_pct_change"] = np.nan
                else:
                    differences[f"{metric}_diff"] = np.nan
                    differences[f"{metric}_pct_change"] = np.nan
            else:
                differences[f"{metric}_diff"] = np.nan
                differences[f"{metric}_pct_change"] = np.nan

        logger.info(
            f"Computed {len(differences)} comparison metrics: "
            f"{comparison_moment} vs {baseline_moment}"
        )

        return differences

    def get_configured_metrics_list(self) -> List[str]:
        """
        Get list of all configured metrics that will be extracted.

        Returns:
            List of metric names
        """
        all_metrics = []
        all_metrics.extend(self.time_domain_metrics)
        all_metrics.extend(self.frequency_domain_metrics)
        all_metrics.extend(self.nonlinear_metrics)

        # Add quality metrics
        quality_metrics = [
            "BVP_NumPeaks",
            "BVP_Duration",
            "BVP_PeakRate",
            "BVP_MeanQuality",
            "BVP_QualityStd",
            "BVP_MeanAmplitude",
            "BVP_StdAmplitude",
            "BVP_RangeAmplitude",
        ]
        all_metrics.extend(quality_metrics)

        return all_metrics

    # TODO: Future implementation for epoched analysis
    def extract_epoched_metrics(
        self, processed_signals: pd.DataFrame, processing_info: Dict, moment: str
    ) -> pd.DataFrame:
        """
        Extract HRV metrics from sliding windows (future implementation).

        This method will implement the 30-second sliding window approach
        with 1-second steps for dynamic HRV analysis.

        Args:
            processed_signals: Processed signals DataFrame
            processing_info: Processing information dictionary
            moment: Moment name

        Returns:
            DataFrame with time-series of HRV metrics
        """
        if not self.epoched_enabled:
            logger.info("Epoched analysis not enabled in configuration")
            return pd.DataFrame()

            # TODO: Implement epoched analysis
        logger.info("Epoched HRV analysis not yet implemented")
        return pd.DataFrame()

    def extract_rr_intervals(
        self,
        peaks: Union[List, np.ndarray],
        sampling_rate: int,
        moment: str = "unknown",
    ) -> pd.DataFrame:
        """
        Extract RR intervals with timestamps and validity flags.

        This method computes RR intervals (peak-to-peak intervals) from detected
        peaks and creates a time-series dataframe with timestamps and validity flags.
        Invalid intervals (outside physiological range) are kept but marked.

        Args:
            peaks: Array of peak indices from BVP signal processing
            sampling_rate: Sampling rate in Hz (e.g., 64 for Empatica E4)
            moment: Moment/task name for logging

        Returns:
            DataFrame with columns:
                - time_peak_start: Timestamp of start peak (seconds)
                - time_peak_end: Timestamp of end peak (seconds)
                - rr_interval_ms: RR interval duration (milliseconds)
                - is_valid: 1 if valid, 0 if outside physiological range

        Example:
            >>> extractor = BVPMetricsExtractor()
            >>> rr_df = extractor.extract_rr_intervals(peaks, sampling_rate=64, moment='restingstate')
            >>> valid_rr = rr_df[rr_df['is_valid'] == 1]
            >>> print(f"Valid RR intervals: {len(valid_rr)}/{len(rr_df)}")
        """
        peaks_array = np.array(peaks)

        # Get RR interval configuration
        rr_config = self.bvp_config.get("rr_intervals", {})
        min_valid_ms = rr_config.get("min_valid_ms", 300)
        max_valid_ms = rr_config.get("max_valid_ms", 2000)

        if len(peaks_array) < 2:
            logger.warning(
                f"Insufficient peaks for RR interval extraction in {moment}: {len(peaks_array)} peaks"
            )
            return pd.DataFrame(
                columns=[
                    "time_peak_start",
                    "time_peak_end",
                    "rr_interval_ms",
                    "is_valid",
                ]
            )

        # Calculate RR intervals in milliseconds
        rr_intervals_ms = np.diff(peaks_array) / sampling_rate * 1000

        # Calculate timestamps for each peak (in seconds)
        time_peaks = peaks_array / sampling_rate

        # Create arrays for start and end times
        time_start = time_peaks[:-1]  # All peaks except last
        time_end = time_peaks[1:]  # All peaks except first

        # Determine validity based on physiological range
        is_valid = (rr_intervals_ms >= min_valid_ms) & (rr_intervals_ms <= max_valid_ms)
        is_valid_int = is_valid.astype(int)

        # Count invalid intervals
        n_invalid = (~is_valid).sum()
        n_total = len(rr_intervals_ms)

        if n_invalid > 0:
            logger.info(
                f"RR intervals for {moment}: "
                f"{n_invalid}/{n_total} ({100 * n_invalid / n_total:.1f}%) outside "
                f"valid range [{min_valid_ms}-{max_valid_ms}ms]"
            )
        else:
            logger.debug(f"All {n_total} RR intervals valid for {moment}")

        # Create DataFrame
        rr_df = pd.DataFrame(
            {
                "time_peak_start": time_start,
                "time_peak_end": time_end,
                "rr_interval_ms": rr_intervals_ms,
                "is_valid": is_valid_int,
            }
        )

        logger.info(
            f"Extracted {len(rr_df)} RR intervals for {moment}: "
            f"{is_valid_int.sum()} valid, {n_invalid} invalid"
        )

        return rr_df
