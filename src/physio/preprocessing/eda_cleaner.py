"""
EDA Signal Cleaner for TherasyncPipeline.

This module provides functionality to clean and process Electrodermal Activity (EDA) signals,
decomposing them into tonic (baseline) and phasic (response) components, and detecting
Skin Conductance Responses (SCRs).

Authors: Lena Adel, Remy Ramadour
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np
import neurokit2 as nk

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class EDACleaner:
    """
    Clean and process EDA signals using NeuroKit2.

    This class handles:
    - Signal cleaning and filtering
    - Tonic-phasic decomposition
    - SCR (Skin Conductance Response) peak detection
    - Quality assessment

    The tonic component represents the baseline skin conductance level,
    while the phasic component represents rapid SCRs (sympathetic arousal responses).
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the EDA cleaner with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)

        # Get EDA processing configuration
        self.sampling_rate = self.config.get("physio.eda.sampling_rate", 4)
        self.processing_config = self.config.get("physio.eda.processing", {})

        # Processing parameters
        self.method = self.processing_config.get("method", "neurokit")
        self.method_scr = self.processing_config.get("method_scr", "neurokit")
        self.scr_threshold = self.processing_config.get("scr_threshold", 0.01)
        self.scr_min_amplitude = self.processing_config.get("scr_min_amplitude", 0.01)

        logger.info(
            f"EDA Cleaner initialized (method: {self.method}, "
            f"sampling rate: {self.sampling_rate} Hz, "
            f"SCR threshold: {self.scr_threshold} μS)"
        )

    def clean_signal(
        self, eda_data: pd.DataFrame, moment: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process raw EDA signal and decompose into tonic and phasic components.

        This method:
        1. Cleans the raw EDA signal
        2. Decomposes into tonic (slow-changing baseline) and phasic (fast responses) components
        3. Detects SCR peaks in the phasic component
        4. Computes quality metrics

        Args:
            eda_data: DataFrame with columns ['time', 'eda']
            moment: Optional moment name for logging

        Returns:
            DataFrame with processed EDA signals including:
                - EDA_Raw: Original signal
                - EDA_Clean: Cleaned signal
                - EDA_Tonic: Slow-varying baseline (tonic component)
                - EDA_Phasic: Fast-varying responses (phasic component)
                - SCR_Peaks: Binary indicators of SCR peaks (1 = peak, 0 = no peak)
                - SCR_Amplitude: Amplitude of detected SCRs (0 if no peak)
                - SCR_RiseTime: Rise time of SCRs (0 if no peak)
                - SCR_RecoveryTime: Recovery time of SCRs (0 if no peak)

        Raises:
            ValueError: If input data is invalid

        Example:
            >>> cleaner = EDACleaner()
            >>> processed = cleaner.clean_signal(raw_data, moment='restingstate')
            >>> print(f"Detected {processed['SCR_Peaks'].sum()} SCRs")
        """
        moment_str = f" for moment '{moment}'" if moment else ""
        logger.info(f"Processing EDA signal{moment_str} ({len(eda_data)} samples)")

        # Validate input
        self._validate_input(eda_data)

        # Extract raw signal
        raw_signal = eda_data["eda"].values

        try:
            # Process EDA signal with NeuroKit2
            # This performs cleaning and tonic-phasic decomposition using cvxEDA
            # Note: Negative values in phasic/tonic components are NORMAL after
            # mathematical decomposition (centered signals, filtering) - they do NOT
            # indicate errors. What matters is the relative change, not absolute values.
            signals, info = nk.eda_process(
                raw_signal,
                sampling_rate=self.sampling_rate,
                method=self.method,  # Uses cvxEDA for gold-standard decomposition
            )

            # Create output DataFrame with all components
            processed_signals = pd.DataFrame(
                {
                    "EDA_Raw": raw_signal,
                    "EDA_Clean": signals["EDA_Clean"].values,
                    "EDA_Tonic": signals["EDA_Tonic"].values,
                    "EDA_Phasic": signals["EDA_Phasic"].values,
                    "SCR_Peaks": signals["SCR_Peaks"].values.astype(int),
                }
            )

            # Add SCR-specific features if peaks were detected
            if "SCR_Amplitude" in signals.columns:
                processed_signals["SCR_Amplitude"] = signals["SCR_Amplitude"].values
            else:
                processed_signals["SCR_Amplitude"] = 0.0

            if "SCR_RiseTime" in signals.columns:
                processed_signals["SCR_RiseTime"] = signals["SCR_RiseTime"].values
            else:
                processed_signals["SCR_RiseTime"] = 0.0

            if "SCR_RecoveryTime" in signals.columns:
                processed_signals["SCR_RecoveryTime"] = signals[
                    "SCR_RecoveryTime"
                ].values
            else:
                processed_signals["SCR_RecoveryTime"] = 0.0

            # Log processing results
            num_scrs = processed_signals["SCR_Peaks"].sum()
            duration = len(eda_data) / self.sampling_rate
            scr_rate = (num_scrs / duration) * 60  # SCRs per minute

            logger.info(
                f"EDA processing complete{moment_str}: "
                f"{num_scrs} SCRs detected ({scr_rate:.2f} SCRs/min)"
            )

            # Log signal quality
            self._log_signal_quality(processed_signals, moment)

            return processed_signals

        except Exception as e:
            logger.error(f"Error processing EDA signal{moment_str}: {str(e)}")
            raise

    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input EDA data.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        if "eda" not in data.columns:
            raise ValueError(
                f"Input data must have 'eda' column. Found columns: {list(data.columns)}"
            )

        # Check for empty data
        if len(data) == 0:
            raise ValueError("Input data is empty")

        # Check for minimum length (need at least a few seconds for decomposition)
        min_duration = self.processing_config.get("min_duration", 5)  # seconds
        min_samples = self.sampling_rate * min_duration
        if len(data) < min_samples:
            logger.warning(
                f"Short EDA signal: {len(data)} samples ({len(data) / self.sampling_rate:.1f}s). "
                f"Minimum recommended: {min_samples} samples ({min_duration}s)"
            )

        # Check for all NaN
        if data["eda"].isna().all():
            raise ValueError("All EDA values are NaN")

        # Warn about NaN values
        if data["eda"].isna().any():
            n_nan = data["eda"].isna().sum()
            logger.warning(
                f"EDA signal contains {n_nan} NaN values ({n_nan / len(data) * 100:.1f}%). "
                "These will be interpolated during processing."
            )

    def _log_signal_quality(
        self, processed_signals: pd.DataFrame, moment: Optional[str] = None
    ) -> None:
        """
        Log quality metrics for processed EDA signal.

        Args:
            processed_signals: Processed EDA DataFrame
            moment: Optional moment name for logging
        """
        moment_str = f" for {moment}" if moment else ""

        # Compute quality indicators
        tonic_mean = processed_signals["EDA_Tonic"].mean()
        tonic_std = processed_signals["EDA_Tonic"].std()
        phasic_mean = processed_signals["EDA_Phasic"].mean()
        phasic_std = processed_signals["EDA_Phasic"].std()

        # Check for unusual values
        if tonic_mean < 0.01:
            logger.warning(
                f"Very low tonic EDA level{moment_str}: {tonic_mean:.4f} μS. "
                "This may indicate poor sensor contact."
            )

        if tonic_mean > 20:
            logger.warning(
                f"Very high tonic EDA level{moment_str}: {tonic_mean:.2f} μS. "
                "This is unusual and may indicate sensor issues."
            )

        # Check phasic component
        num_scrs = processed_signals["SCR_Peaks"].sum()
        if num_scrs == 0:
            logger.warning(
                f"No SCRs detected{moment_str}. "
                f"Tonic: {tonic_mean:.3f}±{tonic_std:.3f} μS, "
                f"Phasic: {phasic_mean:.4f}±{phasic_std:.4f} μS"
            )
        else:
            logger.debug(
                f"EDA quality{moment_str}: "
                f"Tonic: {tonic_mean:.3f}±{tonic_std:.3f} μS, "
                f"Phasic: {phasic_mean:.4f}±{phasic_std:.4f} μS, "
                f"SCRs: {num_scrs}"
            )

    def get_scr_peaks(self, processed_signals: pd.DataFrame) -> np.ndarray:
        """
        Extract SCR peak indices from processed signals.

        Args:
            processed_signals: Output from clean_signal()

        Returns:
            Array of sample indices where SCR peaks occur

        Example:
            >>> peaks = cleaner.get_scr_peaks(processed_signals)
            >>> print(f"SCR peaks at indices: {peaks}")
        """
        peak_indices = np.where(processed_signals["SCR_Peaks"] == 1)[0]
        return peak_indices

    def get_scr_metadata(
        self, processed_signals: pd.DataFrame, moment: Optional[str] = None
    ) -> dict:
        """
        Compute comprehensive SCR statistics and quality metrics.

        Args:
            processed_signals: Output from clean_signal()
            moment: Optional moment name to include in metadata

        Returns:
            Dictionary with SCR statistics:
                - num_scrs: Total number of detected SCRs
                - scr_indices: List of sample indices where SCRs occur
                - scr_rate: SCRs per minute
                - mean_scr_amplitude: Average SCR amplitude (μS)
                - max_scr_amplitude: Maximum SCR amplitude (μS)
                - mean_rise_time: Average rise time (seconds)
                - mean_recovery_time: Average recovery time (seconds)
                - tonic_mean: Mean tonic level (μS)
                - tonic_std: Tonic level standard deviation (μS)
                - phasic_mean: Mean phasic activity (μS)
                - phasic_std: Phasic activity standard deviation (μS)
                - processing_method: Method used for processing
                - sampling_rate: Sampling rate (Hz)
                - duration_seconds: Signal duration

        Example:
            >>> metadata = cleaner.get_scr_metadata(processed_signals, moment='therapy')
            >>> print(f"SCR rate: {metadata['scr_rate']:.2f} per minute")
        """
        # Basic counts
        num_scrs = int(processed_signals["SCR_Peaks"].sum())
        duration = len(processed_signals) / self.sampling_rate
        scr_rate = (num_scrs / duration) * 60 if duration > 0 else 0

        # SCR peak indices
        scr_indices = self.get_scr_peaks(processed_signals).tolist()

        # SCR amplitude statistics
        scr_amplitudes = processed_signals.loc[
            processed_signals["SCR_Peaks"] == 1, "SCR_Amplitude"
        ]
        mean_scr_amplitude = (
            float(scr_amplitudes.mean()) if len(scr_amplitudes) > 0 else 0.0
        )
        max_scr_amplitude = (
            float(scr_amplitudes.max()) if len(scr_amplitudes) > 0 else 0.0
        )

        # SCR timing statistics
        scr_rise_times = processed_signals.loc[
            processed_signals["SCR_Peaks"] == 1, "SCR_RiseTime"
        ]
        mean_rise_time = (
            float(scr_rise_times.mean()) if len(scr_rise_times) > 0 else 0.0
        )

        scr_recovery_times = processed_signals.loc[
            processed_signals["SCR_Peaks"] == 1, "SCR_RecoveryTime"
        ]
        mean_recovery_time = (
            float(scr_recovery_times.mean()) if len(scr_recovery_times) > 0 else 0.0
        )

        # Tonic and phasic statistics
        tonic_mean = float(processed_signals["EDA_Tonic"].mean())
        tonic_std = float(processed_signals["EDA_Tonic"].std())
        phasic_mean = float(processed_signals["EDA_Phasic"].mean())
        phasic_std = float(processed_signals["EDA_Phasic"].std())

        metadata = {
            "num_scrs": num_scrs,
            "scr_indices": scr_indices,
            "scr_rate": scr_rate,
            "mean_scr_amplitude": mean_scr_amplitude,
            "max_scr_amplitude": max_scr_amplitude,
            "mean_rise_time": mean_rise_time,
            "mean_recovery_time": mean_recovery_time,
            "tonic_mean": tonic_mean,
            "tonic_std": tonic_std,
            "phasic_mean": phasic_mean,
            "phasic_std": phasic_std,
            "processing_method": self.method,
            "scr_detection_method": self.method_scr,
            "scr_threshold": self.scr_threshold,
            "sampling_rate": self.sampling_rate,
            "duration_seconds": duration,
        }

        if moment:
            metadata["moment"] = moment

        return metadata

    def get_tonic_component(self, processed_signals: pd.DataFrame) -> pd.Series:
        """
        Extract tonic (baseline) component from processed signals.

        The tonic component represents the slow-varying baseline skin conductance level.
        It reflects overall arousal state and changes gradually over time.

        Args:
            processed_signals: Output from clean_signal()

        Returns:
            Series containing tonic EDA values

        Example:
            >>> tonic = cleaner.get_tonic_component(processed_signals)
            >>> print(f"Mean tonic level: {tonic.mean():.3f} μS")
        """
        return processed_signals["EDA_Tonic"]

    def get_phasic_component(self, processed_signals: pd.DataFrame) -> pd.Series:
        """
        Extract phasic (response) component from processed signals.

        The phasic component represents rapid skin conductance responses (SCRs)
        associated with sympathetic nervous system activity and emotional arousal.

        Args:
            processed_signals: Output from clean_signal()

        Returns:
            Series containing phasic EDA values

        Example:
            >>> phasic = cleaner.get_phasic_component(processed_signals)
            >>> print(f"Mean phasic activity: {phasic.mean():.4f} μS")
        """
        return processed_signals["EDA_Phasic"]

    def compute_scr_features(self, processed_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute detailed features for each detected SCR.

        Args:
            processed_signals: Output from clean_signal()

        Returns:
            DataFrame with one row per SCR, containing:
                - scr_index: Sample index of SCR peak
                - scr_time: Time of SCR peak (seconds)
                - scr_amplitude: SCR amplitude (μS)
                - scr_rise_time: Rise time (seconds)
                - scr_recovery_time: Recovery time (seconds)

        Example:
            >>> scr_features = cleaner.compute_scr_features(processed_signals)
            >>> print(f"Found {len(scr_features)} SCRs")
            >>> print(scr_features[['scr_time', 'scr_amplitude']].head())
        """
        # Get SCR peak indices
        scr_indices = self.get_scr_peaks(processed_signals)

        if len(scr_indices) == 0:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(
                columns=[
                    "scr_index",
                    "scr_time",
                    "scr_amplitude",
                    "scr_rise_time",
                    "scr_recovery_time",
                ]
            )

        # Extract features for each SCR
        features = []
        for idx in scr_indices:
            scr_time = idx / self.sampling_rate
            scr_amplitude = processed_signals.loc[idx, "SCR_Amplitude"]
            scr_rise_time = processed_signals.loc[idx, "SCR_RiseTime"]
            scr_recovery_time = processed_signals.loc[idx, "SCR_RecoveryTime"]

            features.append(
                {
                    "scr_index": int(idx),
                    "scr_time": scr_time,
                    "scr_amplitude": scr_amplitude,
                    "scr_rise_time": scr_rise_time,
                    "scr_recovery_time": scr_recovery_time,
                }
            )

        return pd.DataFrame(features)

    def calculate_quality(self, processed_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EDA signal quality score based on signal stability and physiological plausibility.

        Quality is calculated using a sliding window approach combining:
        - Tonic stability (coefficient of variation in window)
        - Phasic activity reasonableness (z-score based penalty for extreme values)

        The quality score ranges from 0 to 1, where higher values indicate better quality.

        Args:
            processed_signals: DataFrame with EDA_Tonic and EDA_Phasic columns
                             (output from clean_signal())

        Returns:
            DataFrame with added EDA_Quality column (0-1, higher is better)

        Example:
            >>> cleaner = EDACleaner()
            >>> processed = cleaner.clean_signal(raw_data)
            >>> processed_with_quality = cleaner.calculate_quality(processed)
            >>> print(f"Mean quality: {processed_with_quality['EDA_Quality'].mean():.3f}")
        """
        # Validate input
        required_cols = ["EDA_Tonic", "EDA_Phasic"]
        missing_cols = [
            col for col in required_cols if col not in processed_signals.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create a copy to avoid modifying the original
        result = processed_signals.copy()

        quality_scores = []
        quality_window = self.processing_config.get("quality_window", 4)  # seconds
        window_size = self.sampling_rate * quality_window

        for i in range(len(result)):
            # Get window around current sample
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(result), i + window_size // 2)
            window = result.iloc[start_idx:end_idx]

            # Factor 1: Tonic stability (low CV = more stable = better)
            tonic_values = window["EDA_Tonic"]
            tonic_mean = tonic_values.mean()
            tonic_cv = tonic_values.std() / (abs(tonic_mean) + 1e-10)
            tonic_quality = 1.0 / (1.0 + tonic_cv)  # 0-1, higher is better

            # Factor 2: Phasic reasonableness (not too extreme)
            phasic_value = result.iloc[i]["EDA_Phasic"]
            phasic_std = window["EDA_Phasic"].std()
            phasic_z = abs(phasic_value) / (phasic_std + 1e-10)
            phasic_quality = 1.0 / (1.0 + phasic_z / 3.0)  # Penalize extreme values

            # Combined quality (weighted average: tonic stability more important)
            quality = 0.6 * tonic_quality + 0.4 * phasic_quality
            quality_scores.append(quality)

        result["EDA_Quality"] = quality_scores

        logger.debug(
            f"EDA quality calculated: mean={np.mean(quality_scores):.3f}, "
            f"min={np.min(quality_scores):.3f}, max={np.max(quality_scores):.3f}"
        )

        return result
