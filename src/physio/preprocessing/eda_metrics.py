"""
EDA Metrics Extractor for TherasyncPipeline.

This module extracts comprehensive EDA (Electrodermal Activity) metrics from processed signals,
including SCR (Skin Conductance Response) features and tonic/phasic component statistics.

Authors: Lena Adel, Remy Ramadour
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class EDAMetricsExtractor:
    """
    Extract EDA metrics from processed signals.

    This class computes comprehensive metrics from EDA data:
    - SCR features: count, amplitude (mean/max), rise time, recovery time
    - Tonic component: mean, standard deviation
    - Phasic component: mean, standard deviation, rate

    Metrics can be extracted for full recordings or individual moments.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the EDA metrics extractor with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)

        # Get EDA configuration
        self.sampling_rate = self.config.get("physio.eda.sampling_rate", 4)
        self.metrics_config = self.config.get("physio.eda.metrics", {})

        # Get selected metrics - handle both dict and list formats for backward compatibility
        if isinstance(self.metrics_config, list):
            # Old format: metrics is a list of metric names
            self.selected_metrics = {}
            self.extract_all = True  # If list provided, extract those metrics
        elif isinstance(self.metrics_config, dict):
            # New format: metrics is a dict with extract_all and selected_metrics
            selected_metrics_raw = self.metrics_config.get("selected_metrics", {})
            if isinstance(selected_metrics_raw, list):
                # Empty list or list of strings - use default empty dict
                self.selected_metrics = {}
            else:
                self.selected_metrics = selected_metrics_raw
            self.extract_all = self.metrics_config.get("extract_all", False)
        else:
            # Fallback
            self.selected_metrics = {}
            self.extract_all = False

        logger.info(
            f"EDA Metrics Extractor initialized "
            f"(extract_all: {self.extract_all}, sampling rate: {self.sampling_rate} Hz)"
        )

    def extract_eda_metrics(
        self, processed_signals: pd.DataFrame, moment: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract EDA metrics from processed signals.

        Args:
            processed_signals: Output from EDACleaner.clean_signal()
                Must contain columns: EDA_Tonic, EDA_Phasic, SCR_Peaks, SCR_Amplitude,
                SCR_RiseTime, SCR_RecoveryTime
            moment: Optional moment name to include in output

        Returns:
            DataFrame with one row containing all EDA metrics

        Example:
            >>> extractor = EDAMetricsExtractor()
            >>> metrics = extractor.extract_eda_metrics(processed_signals, moment='restingstate')
            >>> print(metrics[['SCR_Peaks_N', 'SCR_Peaks_Amplitude_Mean', 'EDA_Tonic_Mean']])
        """
        logger.info(f"Extracting EDA metrics{' for ' + moment if moment else ''}")

        # Validate input
        self._validate_input(processed_signals)

        # Initialize metrics dictionary
        metrics = {}

        if moment:
            metrics["moment"] = moment

        # Extract signal duration
        duration_seconds = len(processed_signals) / self.sampling_rate

        # Extract SCR metrics
        scr_metrics = self._extract_scr_metrics(processed_signals, duration_seconds)
        metrics.update(scr_metrics)

        # Extract tonic component metrics
        tonic_metrics = self._extract_tonic_metrics(processed_signals)
        metrics.update(tonic_metrics)

        # Extract phasic component metrics
        phasic_metrics = self._extract_phasic_metrics(
            processed_signals, duration_seconds
        )
        metrics.update(phasic_metrics)

        # Add signal quality metadata
        metrics["EDA_Duration"] = duration_seconds
        metrics["EDA_SamplingRate"] = self.sampling_rate

        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics])

        logger.info(
            f"Extracted {len(metrics)} EDA metrics "
            f"({metrics.get('SCR_Peaks_N', 0)} SCRs, "
            f"{metrics.get('EDA_Tonic_Mean', 0):.3f} μS tonic)"
        )

        return metrics_df

    def _validate_input(self, processed_signals: pd.DataFrame) -> None:
        """
        Validate that processed signals have required columns.

        Args:
            processed_signals: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            "EDA_Tonic",
            "EDA_Phasic",
            "SCR_Peaks",
            "SCR_Amplitude",
            "SCR_RiseTime",
            "SCR_RecoveryTime",
        ]

        missing_columns = set(required_columns) - set(processed_signals.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in processed signals: {missing_columns}. "
                f"Found columns: {list(processed_signals.columns)}"
            )

        if len(processed_signals) == 0:
            raise ValueError("Processed signals DataFrame is empty")

    def _extract_scr_metrics(
        self, processed_signals: pd.DataFrame, duration_seconds: float
    ) -> Dict[str, float]:
        """
        Extract SCR (Skin Conductance Response) metrics.

        Args:
            processed_signals: Processed EDA signals
            duration_seconds: Signal duration in seconds

        Returns:
            Dictionary with SCR metrics
        """
        metrics = {}

        # Count SCR peaks
        num_scrs = int(processed_signals["SCR_Peaks"].sum())
        metrics["SCR_Peaks_N"] = num_scrs

        # SCR rate (per minute)
        duration_minutes = duration_seconds / 60
        metrics["SCR_Peaks_Rate"] = (
            (num_scrs / duration_minutes) if duration_minutes > 0 else 0.0
        )

        if num_scrs > 0:
            # Get SCR amplitudes (only for peaks)
            scr_amplitudes = processed_signals.loc[
                processed_signals["SCR_Peaks"] == 1, "SCR_Amplitude"
            ].dropna()

            if len(scr_amplitudes) > 0:
                metrics["SCR_Peaks_Amplitude_Mean"] = float(scr_amplitudes.mean())
                metrics["SCR_Peaks_Amplitude_Max"] = float(scr_amplitudes.max())
                metrics["SCR_Peaks_Amplitude_SD"] = float(scr_amplitudes.std())
            else:
                metrics["SCR_Peaks_Amplitude_Mean"] = 0.0
                metrics["SCR_Peaks_Amplitude_Max"] = 0.0
                metrics["SCR_Peaks_Amplitude_SD"] = 0.0

            # Get SCR rise times
            scr_rise_times = processed_signals.loc[
                processed_signals["SCR_Peaks"] == 1, "SCR_RiseTime"
            ].dropna()

            if len(scr_rise_times) > 0:
                metrics["SCR_RiseTime_Mean"] = float(scr_rise_times.mean())
                metrics["SCR_RiseTime_SD"] = float(scr_rise_times.std())
            else:
                metrics["SCR_RiseTime_Mean"] = 0.0
                metrics["SCR_RiseTime_SD"] = 0.0

            # Get SCR recovery times
            scr_recovery_times = processed_signals.loc[
                processed_signals["SCR_Peaks"] == 1, "SCR_RecoveryTime"
            ].dropna()

            if len(scr_recovery_times) > 0:
                metrics["SCR_RecoveryTime_Mean"] = float(scr_recovery_times.mean())
                metrics["SCR_RecoveryTime_SD"] = float(scr_recovery_times.std())
            else:
                metrics["SCR_RecoveryTime_Mean"] = 0.0
                metrics["SCR_RecoveryTime_SD"] = 0.0
        else:
            # No SCRs detected - set all metrics to 0
            metrics["SCR_Peaks_Amplitude_Mean"] = 0.0
            metrics["SCR_Peaks_Amplitude_Max"] = 0.0
            metrics["SCR_Peaks_Amplitude_SD"] = 0.0
            metrics["SCR_RiseTime_Mean"] = 0.0
            metrics["SCR_RiseTime_SD"] = 0.0
            metrics["SCR_RecoveryTime_Mean"] = 0.0
            metrics["SCR_RecoveryTime_SD"] = 0.0

        return metrics

    def _extract_tonic_metrics(
        self, processed_signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract tonic (baseline) component metrics.

        The tonic component represents the slow-varying baseline skin conductance level,
        reflecting overall arousal state.

        Args:
            processed_signals: Processed EDA signals

        Returns:
            Dictionary with tonic metrics
        """
        metrics = {}

        tonic = processed_signals["EDA_Tonic"]

        metrics["EDA_Tonic_Mean"] = float(tonic.mean())
        metrics["EDA_Tonic_SD"] = float(tonic.std())
        metrics["EDA_Tonic_Min"] = float(tonic.min())
        metrics["EDA_Tonic_Max"] = float(tonic.max())
        metrics["EDA_Tonic_Range"] = float(tonic.max() - tonic.min())

        return metrics

    def _extract_phasic_metrics(
        self, processed_signals: pd.DataFrame, duration_seconds: float
    ) -> Dict[str, float]:
        """
        Extract phasic (response) component metrics.

        The phasic component represents rapid skin conductance responses (SCRs)
        associated with sympathetic nervous system activity.

        Args:
            processed_signals: Processed EDA signals
            duration_seconds: Signal duration in seconds

        Returns:
            Dictionary with phasic metrics
        """
        metrics = {}

        phasic = processed_signals["EDA_Phasic"]

        metrics["EDA_Phasic_Mean"] = float(phasic.mean())
        metrics["EDA_Phasic_SD"] = float(phasic.std())
        metrics["EDA_Phasic_Min"] = float(phasic.min())
        metrics["EDA_Phasic_Max"] = float(phasic.max())
        metrics["EDA_Phasic_Range"] = float(phasic.max() - phasic.min())

        # Phasic activity rate (responses per minute)
        # Count positive phasic values as activity
        positive_phasic = (phasic > 0).sum()
        duration_minutes = duration_seconds / 60
        metrics["EDA_Phasic_Rate"] = (
            (positive_phasic / duration_minutes) if duration_minutes > 0 else 0.0
        )

        return metrics

    def extract_multiple_moments(
        self, moment_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extract EDA metrics for multiple moments at once.

        Args:
            moment_data: Dictionary mapping moment names to processed signals
                Example: {'restingstate': processed_rest, 'therapy': processed_therapy}

        Returns:
            DataFrame with one row per moment containing all metrics

        Example:
            >>> moments = {
            ...     'restingstate': processed_rest,
            ...     'therapy': processed_therapy
            ... }
            >>> all_metrics = extractor.extract_multiple_moments(moments)
            >>> print(all_metrics[['moment', 'SCR_Peaks_N', 'EDA_Tonic_Mean']])
        """
        logger.info(f"Extracting EDA metrics for {len(moment_data)} moments")

        all_metrics = []

        for moment_name, processed_signals in moment_data.items():
            try:
                metrics = self.extract_eda_metrics(
                    processed_signals, moment=moment_name
                )
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(
                    f"Error extracting metrics for moment '{moment_name}': {str(e)}"
                )
                raise

        # Concatenate all metrics
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        logger.info(
            f"Successfully extracted metrics for {len(combined_metrics)} moments"
        )

        return combined_metrics

    def extract_session_metrics(
        self, moment_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract EDA metrics for multiple moments, returning Dict format.

        This method provides the same output format as BVPMetricsExtractor.extract_session_metrics()
        for consistency across modalities.

        Args:
            moment_data: Dictionary mapping moment names to processed signals
                Example: {'restingstate': processed_rest, 'therapy': processed_therapy}

        Returns:
            Dictionary with extracted metrics for each moment.
            Format: {moment: {metric_name: value}}

        Example:
            >>> moments = {
            ...     'restingstate': processed_rest,
            ...     'therapy': processed_therapy
            ... }
            >>> all_metrics = extractor.extract_session_metrics(moments)
            >>> print(all_metrics['restingstate']['SCR_Peaks_N'])
        """
        logger.info(
            f"Extracting EDA metrics for {len(moment_data)} moments (dict format)"
        )

        session_metrics = {}

        for moment_name, processed_signals in moment_data.items():
            try:
                # Get DataFrame with one row
                metrics_df = self.extract_eda_metrics(
                    processed_signals, moment=moment_name
                )

                # Convert to dict (excluding 'moment' column)
                if len(metrics_df) > 0:
                    metrics_dict = metrics_df.iloc[0].to_dict()
                    # Remove 'moment' key if present (redundant with dict key)
                    metrics_dict.pop("moment", None)
                    session_metrics[moment_name] = metrics_dict
                else:
                    session_metrics[moment_name] = {}

            except Exception as e:
                logger.error(
                    f"Error extracting metrics for moment '{moment_name}': {str(e)}"
                )
                session_metrics[moment_name] = {}

        logger.info(
            f"Successfully extracted metrics for {len(session_metrics)} moments"
        )

        return session_metrics

    def get_metric_descriptions(self) -> Dict[str, dict]:
        """
        Get detailed descriptions of all EDA metrics.

        Returns:
            Dictionary with metric names as keys and description dicts as values.
            Each description contains: name, unit, domain, description, interpretation

        Example:
            >>> extractor = EDAMetricsExtractor()
            >>> descriptions = extractor.get_metric_descriptions()
            >>> print(descriptions['SCR_Peaks_N'])
        """
        descriptions = {
            # SCR Metrics
            "SCR_Peaks_N": {
                "name": "Number of SCR Peaks",
                "unit": "count",
                "domain": "scr",
                "description": "Total number of skin conductance responses detected",
                "interpretation": "Higher values indicate more frequent sympathetic arousal responses",
            },
            "SCR_Peaks_Rate": {
                "name": "SCR Rate",
                "unit": "per minute",
                "domain": "scr",
                "description": "Frequency of skin conductance responses",
                "interpretation": "Reflects overall sympathetic activity level",
            },
            "SCR_Peaks_Amplitude_Mean": {
                "name": "Mean SCR Amplitude",
                "unit": "μS",
                "domain": "scr",
                "description": "Average amplitude of detected SCRs",
                "interpretation": "Higher values indicate stronger sympathetic responses",
            },
            "SCR_Peaks_Amplitude_Max": {
                "name": "Maximum SCR Amplitude",
                "unit": "μS",
                "domain": "scr",
                "description": "Largest SCR amplitude detected",
                "interpretation": "Reflects peak sympathetic arousal",
            },
            "SCR_Peaks_Amplitude_SD": {
                "name": "SCR Amplitude Variability",
                "unit": "μS",
                "domain": "scr",
                "description": "Standard deviation of SCR amplitudes",
                "interpretation": "Higher values indicate more variable responses",
            },
            "SCR_RiseTime_Mean": {
                "name": "Mean SCR Rise Time",
                "unit": "seconds",
                "domain": "scr",
                "description": "Average time from SCR onset to peak",
                "interpretation": "Faster rise times may indicate stronger arousal",
            },
            "SCR_RiseTime_SD": {
                "name": "SCR Rise Time Variability",
                "unit": "seconds",
                "domain": "scr",
                "description": "Standard deviation of SCR rise times",
                "interpretation": "Reflects consistency of sympathetic responses",
            },
            "SCR_RecoveryTime_Mean": {
                "name": "Mean SCR Recovery Time",
                "unit": "seconds",
                "domain": "scr",
                "description": "Average time from SCR peak to baseline",
                "interpretation": "Longer recovery may indicate sustained arousal",
            },
            "SCR_RecoveryTime_SD": {
                "name": "SCR Recovery Time Variability",
                "unit": "seconds",
                "domain": "scr",
                "description": "Standard deviation of SCR recovery times",
                "interpretation": "Reflects variability in arousal regulation",
            },
            # Tonic Component Metrics
            "EDA_Tonic_Mean": {
                "name": "Mean Tonic EDA Level",
                "unit": "μS",
                "domain": "tonic",
                "description": "Average baseline skin conductance level",
                "interpretation": "Higher values indicate higher overall arousal state",
            },
            "EDA_Tonic_SD": {
                "name": "Tonic EDA Variability",
                "unit": "μS",
                "domain": "tonic",
                "description": "Variability in baseline skin conductance",
                "interpretation": "Reflects changes in arousal state over time",
            },
            "EDA_Tonic_Min": {
                "name": "Minimum Tonic EDA",
                "unit": "μS",
                "domain": "tonic",
                "description": "Lowest baseline conductance level",
                "interpretation": "Reflects minimum arousal state",
            },
            "EDA_Tonic_Max": {
                "name": "Maximum Tonic EDA",
                "unit": "μS",
                "domain": "tonic",
                "description": "Highest baseline conductance level",
                "interpretation": "Reflects peak arousal state",
            },
            "EDA_Tonic_Range": {
                "name": "Tonic EDA Range",
                "unit": "μS",
                "domain": "tonic",
                "description": "Difference between max and min tonic levels",
                "interpretation": "Larger range indicates greater arousal fluctuation",
            },
            # Phasic Component Metrics
            "EDA_Phasic_Mean": {
                "name": "Mean Phasic EDA",
                "unit": "μS",
                "domain": "phasic",
                "description": "Average phasic (response) component",
                "interpretation": "Higher values indicate more phasic activity",
            },
            "EDA_Phasic_SD": {
                "name": "Phasic EDA Variability",
                "unit": "μS",
                "domain": "phasic",
                "description": "Variability in phasic responses",
                "interpretation": "Reflects consistency of sympathetic responses",
            },
            "EDA_Phasic_Min": {
                "name": "Minimum Phasic EDA",
                "unit": "μS",
                "domain": "phasic",
                "description": "Lowest phasic activity level",
                "interpretation": "Baseline phasic activity",
            },
            "EDA_Phasic_Max": {
                "name": "Maximum Phasic EDA",
                "unit": "μS",
                "domain": "phasic",
                "description": "Highest phasic activity level",
                "interpretation": "Peak phasic response",
            },
            "EDA_Phasic_Range": {
                "name": "Phasic EDA Range",
                "unit": "μS",
                "domain": "phasic",
                "description": "Difference between max and min phasic levels",
                "interpretation": "Reflects dynamic range of responses",
            },
            "EDA_Phasic_Rate": {
                "name": "Phasic Activity Rate",
                "unit": "per minute",
                "domain": "phasic",
                "description": "Frequency of positive phasic activity",
                "interpretation": "Higher rates indicate more frequent responses",
            },
            # Metadata
            "EDA_Duration": {
                "name": "Signal Duration",
                "unit": "seconds",
                "domain": "metadata",
                "description": "Duration of EDA recording",
                "interpretation": "Recording length",
            },
            "EDA_SamplingRate": {
                "name": "Sampling Rate",
                "unit": "Hz",
                "domain": "metadata",
                "description": "EDA signal sampling frequency",
                "interpretation": "Data acquisition rate",
            },
        }

        return descriptions

    def get_selected_metrics(self) -> List[str]:
        """
        Get list of metrics configured to be extracted.

        Returns:
            List of metric names to extract
        """
        if self.extract_all:
            descriptions = self.get_metric_descriptions()
            return list(descriptions.keys())
        else:
            # Get selected metrics from config
            selected = []
            for domain, metrics in self.selected_metrics.items():
                if isinstance(metrics, list):
                    selected.extend(metrics)
            return selected
