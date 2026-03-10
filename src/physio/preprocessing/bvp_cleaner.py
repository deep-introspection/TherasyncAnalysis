"""
BVP Cleaner for TherasyncPipeline.

This module provides functionality to clean and process Blood Volume Pulse (BVP) data
using NeuroKit2, following the established Therasync methodology.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import neurokit2 as nk

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class BVPCleaner:
    """
    Clean and process BVP data using NeuroKit2 PPG processing pipeline.
    
    This class implements the BVP preprocessing approach established in the original
    Therasync project, using nk.ppg_process with elgendi method for peak detection
    and templatematch for quality assessment.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BVP cleaner with configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        
        # Get BVP-specific configuration
        self.bvp_config = self.config.get('physio.bvp', {})
        self.processing_config = self.bvp_config.get('processing', {})
        
        # Set processing parameters from config
        self.method = self.processing_config.get('method', 'elgendi')
        self.method_quality = self.processing_config.get('method_quality', 'templatematch')
        self.quality_threshold = self.processing_config.get('quality_threshold', 0.8)
        self.lowcut_freq = self.processing_config.get('lowcut_freq', 0.5)
        self.highcut_freq = self.processing_config.get('highcut_freq', 8.0)
        
        logger.info(f"BVP Cleaner initialized with method={self.method}, quality={self.method_quality}")
    
    def process_signal(
        self, 
        bvp_signal: Union[pd.Series, np.ndarray, List], 
        sampling_rate: Optional[int] = None,
        moment: str = "unknown"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a BVP signal using NeuroKit2 PPG processing pipeline.
        
        This method follows the established Therasync approach:
        1. Apply nk.ppg_process with elgendi method and templatematch quality
        2. Extract cleaned signal and processing information
        3. Validate results and apply quality checks
        
        Args:
            bvp_signal: Raw BVP signal data
            sampling_rate: Sampling rate in Hz. If None, uses config default.
            moment: Name of the moment/task being processed (for logging)
            
        Returns:
            Tuple of (processed_signals_dataframe, processing_info_dict)
            
        Raises:
            ValueError: If signal is empty or processing fails
            RuntimeError: If NeuroKit2 processing encounters errors
        """
        if sampling_rate is None:
            sampling_rate = self.bvp_config.get('sampling_rate', 64)
        
        # Ensure sampling_rate is int for type safety
        if not isinstance(sampling_rate, int):
            sampling_rate = int(sampling_rate) if sampling_rate is not None else 64
        
        # Convert input to numpy array for consistency
        if isinstance(bvp_signal, pd.Series):
            signal_array = bvp_signal.values
        elif isinstance(bvp_signal, list):
            signal_array = np.array(bvp_signal)
        else:
            signal_array = np.asarray(bvp_signal)
        
        # Validate input signal
        self._validate_input_signal(signal_array, sampling_rate, moment)
        
        logger.info(
            f"Processing BVP signal for {moment}: {len(signal_array)} samples "
            f"at {sampling_rate} Hz"
        )
        
        try:
            # Apply NeuroKit2 PPG processing with established parameters
            processed_signals, processing_info = nk.ppg_process(
                signal_array,
                sampling_rate=sampling_rate,
                method=self.method,
                method_quality=self.method_quality
            )
            
            # Validate processing results
            self._validate_processing_results(processed_signals, processing_info, moment)
            
            # Add metadata to processing info
            processing_info.update({
                'moment': moment,
                'original_length': len(signal_array),
                'sampling_rate': sampling_rate,
                'processing_method': self.method,
                'quality_method': self.method_quality,
                'config_used': {
                    'method': self.method,
                    'method_quality': self.method_quality,
                    'quality_threshold': self.quality_threshold
                }
            })
            
            logger.info(
                f"Successfully processed BVP {moment}: {len(processed_signals)} samples, "
                f"{len(processing_info.get('PPG_Peaks', []))} peaks detected"
            )
            
            return processed_signals, processing_info
            
        except Exception as e:
            logger.error(f"Error processing BVP signal for {moment}: {e}")
            raise RuntimeError(f"BVP processing failed for {moment}: {e}") from e
    
    def process_moment_signals(
        self,
        moment_data: Dict[str, Union[pd.DataFrame, Dict]]
    ) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        Process BVP signals for all moments in a dataset.
        
        Args:
            moment_data: Dictionary with moment names as keys and data/metadata as values.
                        Expected format: {moment: {'data': DataFrame, 'metadata': Dict}}
            
        Returns:
            Dictionary with processed signals and info for each moment.
            Format: {moment: (processed_signals, processing_info)}
        """
        processed_results = {}
        
        for moment, moment_info in moment_data.items():
            try:
                # Extract BVP signal from loaded data
                data_df = moment_info['data']
                metadata = moment_info['metadata']
                
                if 'bvp' not in data_df.columns:
                    logger.warning(f"No BVP column found in {moment} data, skipping")
                    continue
                
                bvp_signal = data_df['bvp']
                sampling_rate = metadata.get('SamplingFrequency', 
                                           self.bvp_config.get('sampling_rate', 64))
                
                # Process the signal
                processed_signals, processing_info = self.process_signal(
                    bvp_signal, sampling_rate, moment
                )
                
                processed_results[moment] = (processed_signals, processing_info)
                
            except Exception as e:
                logger.error(f"Failed to process {moment}: {e}")
                # Continue processing other moments even if one fails
                continue
        
        logger.info(f"Processed BVP signals for {len(processed_results)} moments")
        return processed_results
    
    def _validate_input_signal(
        self, 
        signal: np.ndarray, 
        sampling_rate: int, 
        moment: str
    ) -> None:
        """
        Validate input BVP signal before processing.
        
        Args:
            signal: BVP signal array
            sampling_rate: Sampling rate in Hz
            moment: Moment name for error reporting
            
        Raises:
            ValueError: If signal validation fails
        """
        if len(signal) == 0:
            raise ValueError(f"Empty BVP signal for {moment}")
        
        if sampling_rate <= 0:
            raise ValueError(f"Invalid sampling rate {sampling_rate} for {moment}")
        
        # Check for all NaN values
        if np.all(np.isnan(signal)):
            raise ValueError(f"BVP signal contains only NaN values for {moment}")
        
        # Check minimum duration for meaningful analysis
        min_duration = self.processing_config.get('min_duration', 10)  # seconds
        if len(signal) < sampling_rate * min_duration:
            logger.warning(
                f"Short BVP signal for {moment}: {len(signal)/sampling_rate:.1f}s "
                f"(minimum recommended: {min_duration}s)"
            )
        
        # Check for excessive NaN values
        nan_percentage = np.sum(np.isnan(signal)) / len(signal) * 100
        if nan_percentage > 50:
            logger.warning(
                f"High percentage of NaN values in BVP signal for {moment}: "
                f"{nan_percentage:.1f}%"
            )
    
    def _validate_processing_results(
        self, 
        processed_signals: pd.DataFrame, 
        processing_info: Dict, 
        moment: str
    ) -> None:
        """
        Validate the results of BVP processing.
        
        Args:
            processed_signals: DataFrame with processed signals
            processing_info: Dictionary with processing information
            moment: Moment name for error reporting
            
        Raises:
            ValueError: If processing results are invalid
        """
        # Check if processed signals DataFrame is empty
        if processed_signals.empty:
            raise ValueError(f"BVP processing returned empty DataFrame for {moment}")
        
        # Check for required columns
        required_columns = ['PPG_Clean']
        missing_columns = [col for col in required_columns if col not in processed_signals.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in processed BVP for {moment}: {missing_columns}"
            )
        
        # Check if peaks were detected
        peaks = processing_info.get('PPG_Peaks', [])
        if len(peaks) == 0:
            logger.warning(f"No peaks detected in BVP signal for {moment}")
        elif len(peaks) < 5:  # Very few peaks might indicate poor signal quality
            logger.warning(
                f"Very few peaks detected in BVP signal for {moment}: {len(peaks)} peaks"
            )
        
        # Check signal quality if available
        if 'PPG_Quality' in processed_signals.columns:
            quality_scores = processed_signals['PPG_Quality'].dropna()
            if not quality_scores.empty:
                mean_quality = quality_scores.mean()
                if mean_quality < self.quality_threshold:
                    logger.warning(
                        f"Low signal quality for BVP {moment}: "
                        f"mean quality {mean_quality:.3f} < threshold {self.quality_threshold}. "
                        f"Note: Lower quality scores are common for longer recordings and may still be acceptable for analysis."
                    )
    
    def get_clean_signal(self, processed_signals: pd.DataFrame) -> pd.Series:
        """
        Extract the cleaned BVP signal from processed results.
        
        Args:
            processed_signals: DataFrame from process_signal()
            
        Returns:
            Series containing the cleaned BVP signal
        """
        return processed_signals['PPG_Clean']
    
    def get_peaks(self, processing_info: Dict) -> np.ndarray:
        """
        Extract detected peaks from processing results.
        
        Args:
            processing_info: Info dictionary from process_signal()
            
        Returns:
            Array of peak indices
        """
        return np.array(processing_info.get('PPG_Peaks', []))
    
    def get_quality_scores(self, processed_signals: pd.DataFrame) -> Optional[pd.Series]:
        """
        Extract signal quality scores if available.
        
        Args:
            processed_signals: DataFrame from process_signal()
            
        Returns:
            Series containing quality scores, or None if not available
        """
        if 'PPG_Quality' in processed_signals.columns:
            return processed_signals['PPG_Quality']
        return None
    
    def apply_additional_filtering(
        self, 
        signal: Union[pd.Series, np.ndarray], 
        sampling_rate: int
    ) -> np.ndarray:
        """
        Apply additional filtering to BVP signal if needed.
        
        This method provides optional additional filtering using the configured
        frequency bounds, separate from the main nk.ppg_process pipeline.
        
        Args:
            signal: BVP signal to filter
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Filtered signal array
        """
        try:
            filtered_signal = nk.signal_filter(
                signal,
                sampling_rate=sampling_rate,
                lowcut=self.lowcut_freq,
                highcut=self.highcut_freq,
                method='butterworth',
                order=4
            )
            
            logger.debug(
                f"Applied additional filtering: {self.lowcut_freq}-{self.highcut_freq} Hz"
            )
            
            return filtered_signal
            
        except Exception as e:
            logger.warning(f"Additional filtering failed: {e}, returning original signal")
            return np.asarray(signal)