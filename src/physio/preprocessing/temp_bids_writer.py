"""
Temperature BIDS Writer for TherasyncPipeline.

This module writes peripheral skin temperature processing results to BIDS-compliant
output format, creating standardized files for processed signals, metrics, and metadata.

Authors: Lena Adel, Remy Ramadour
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np

from src.physio.preprocessing.base_bids_writer import PhysioBIDSWriter


logger = logging.getLogger(__name__)


class TEMPBIDSWriter(PhysioBIDSWriter):
    """
    Write temperature processing results in BIDS-compliant format.

    This class creates 6 file types per moment following the BIDS specification:
    1. _desc-processed_recording-temp.tsv: Processed temperature signals (uncompressed)
    2. _desc-processed_recording-temp.json: Signal metadata and processing parameters
    3. _desc-temp-metrics.tsv: Extracted temperature metrics
    4. _desc-temp-metrics.json: Metrics metadata and descriptions
    5. _desc-temp-summary.json: Processing summary and quality assessment

    Output structure:
    derivatives/preprocessing/
    ├── sub-{subject}/
    │   ├── ses-{session}/
    │   │   ├── temp/
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-processed_recording-temp.tsv
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-processed_recording-temp.json
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-temp-metrics.tsv
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-temp-metrics.json
    │   │   │   └── sub-{subject}_ses-{session}_task-{moment}_desc-temp-summary.json

    Notes:
    - Inherits from PhysioBIDSWriter base class
    - Files are PER MOMENT (restingstate, therapy)
    - Files are UNCOMPRESSED (.tsv instead of .tsv.gz)
    - Unified API with save_processed_data() method
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the temperature BIDS writer.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        super().__init__(config_path)
        logger.info(
            f"Temperature BIDS Writer initialized "
            f"(modality: temp, output: {self.derivatives_base}/{self.preprocessing_dir}/)"
        )

    def _get_modality_name(self) -> str:
        """Get the modality identifier for temperature."""
        return "temp"

    def save_processed_data(
        self,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, pd.DataFrame],
        session_metrics: Optional[pd.DataFrame] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Path]]:
        """
        Write complete temperature processing results in BIDS format.

        Args:
            subject_id: Subject identifier WITH prefix (e.g., 'sub-g01p01')
            session_id: Session identifier WITH prefix (e.g., 'ses-01')
            processed_results: Dictionary mapping moment names to processed DataFrames
                             Expected columns: time, TEMP_Raw, TEMP_Clean, TEMP_Quality,
                                              TEMP_Outliers, TEMP_Interpolated
            session_metrics: DataFrame with session-level metrics (optional)
            processing_metadata: Additional processing metadata (optional)

        Returns:
            Dictionary mapping file types to lists of paths (one per moment)

        Example:
            >>> writer = TEMPBIDSWriter()
            >>> processed_results = {
            ...     'restingstate': df_resting,
            ...     'therapy': df_therapy
            ... }
            >>> file_paths = writer.save_processed_data(
            ...     'sub-g01p01', 'ses-01', processed_results, metrics, metadata
            ... )
            >>> print(f"Physio files: {file_paths['physio']}")
        """
        # Ensure IDs have prefixes
        subject_id = self._ensure_prefix(subject_id, "sub")
        session_id = self._ensure_prefix(session_id, "ses")

        logger.info(
            f"Writing temperature results for {subject_id} {session_id} "
            f"({len(processed_results)} moments)"
        )

        # Get subject/session directory
        output_dir = self._get_subject_session_dir(subject_id, session_id)

        # Initialize file paths dictionary
        all_file_paths: Dict[str, List[Path]] = {
            "physio": [],
            "physio_json": [],
            "metrics": [],
            "metrics_json": [],
            "summary": [],
        }

        try:
            # Process each moment separately
            for moment, moment_data in processed_results.items():
                logger.debug(f"Processing moment: {moment}")

                # Extract moment-specific metadata
                moment_metadata = (
                    processing_metadata.get(moment, {}) if processing_metadata else {}
                )

                # Write physio signal files
                physio_file = self._write_physio_file(
                    output_dir, subject_id, session_id, moment, moment_data
                )
                all_file_paths["physio"].append(physio_file)

                physio_json = self._write_physio_metadata(
                    output_dir,
                    subject_id,
                    session_id,
                    moment,
                    moment_data,
                    moment_metadata,
                )
                all_file_paths["physio_json"].append(physio_json)

                # Extract moment-specific metrics if session_metrics provided
                if session_metrics is not None:
                    # Handle both DataFrame and dict formats
                    if (
                        hasattr(session_metrics, "index")
                        and moment in session_metrics.index
                    ):
                        # DataFrame format
                        moment_metrics = session_metrics.loc[moment].to_dict()
                    elif (
                        isinstance(session_metrics, dict) and moment in session_metrics
                    ):
                        # Dict format from extract_session_metrics()
                        moment_metrics = session_metrics[moment]
                    else:
                        # Fallback: extract basic metrics from data
                        moment_metrics = self._extract_basic_metrics(
                            moment_data, moment
                        )
                else:
                    # Fallback: extract basic metrics from data
                    moment_metrics = self._extract_basic_metrics(moment_data, moment)

                # Write metrics files
                metrics_file = self._write_metrics_file(
                    output_dir, subject_id, session_id, moment, moment_metrics
                )
                all_file_paths["metrics"].append(metrics_file)

                metrics_json = self._write_metrics_metadata(
                    output_dir, subject_id, session_id, moment, moment_metrics
                )
                all_file_paths["metrics_json"].append(metrics_json)

                # Write summary file
                summary_file = self._write_summary_file(
                    output_dir,
                    subject_id,
                    session_id,
                    moment,
                    moment_metrics,
                    moment_metadata,
                    {
                        "physio": physio_file,
                        "physio_json": physio_json,
                        "metrics": metrics_file,
                        "metrics_json": metrics_json,
                    },
                )
                all_file_paths["summary"].append(summary_file)

            total_files = sum(len(paths) for paths in all_file_paths.values())
            logger.info(
                f"Temperature results written successfully "
                f"({total_files} files across {len(processed_results)} moments)"
            )
            return all_file_paths

        except Exception as e:
            logger.error(f"Failed to write temperature results: {str(e)}")
            raise

    def _extract_basic_metrics(self, data: pd.DataFrame, moment: str) -> Dict[str, Any]:
        """
        Extract basic temperature metrics from processed data.

        Args:
            data: Processed temperature DataFrame
            moment: Moment identifier

        Returns:
            Dictionary of basic metrics
        """
        temp_clean = data["TEMP_Clean"].dropna()

        if len(temp_clean) == 0:
            return {
                "moment": moment,
                "temp_mean": np.nan,
                "temp_std": np.nan,
                "temp_min": np.nan,
                "temp_max": np.nan,
                "temp_range": np.nan,
            }

        return {
            "moment": moment,
            "temp_mean": float(temp_clean.mean()),
            "temp_std": float(temp_clean.std()),
            "temp_min": float(temp_clean.min()),
            "temp_max": float(temp_clean.max()),
            "temp_range": float(temp_clean.max() - temp_clean.min()),
        }

    def _write_physio_file(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        data: pd.DataFrame,
    ) -> Path:
        """
        Write processed temperature signals to UNCOMPRESSED TSV file.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            data: Cleaned temperature data with columns: time, TEMP_Raw, TEMP_Clean,
                  TEMP_Quality, TEMP_Outliers, TEMP_Interpolated

        Returns:
            Path to written file
        """
        # BIDS filename pattern for processed physio data
        base_filename = (
            f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-temp"
        )

        # Uncompressed TSV file
        signals_tsv = subject_dir / f"{base_filename}.tsv"

        # Select columns in standardized order
        output_columns = [
            "time",
            "TEMP_Raw",
            "TEMP_Clean",
            "TEMP_Quality",
            "TEMP_Outliers",
            "TEMP_Interpolated",
        ]

        # Check if all expected columns exist
        missing_cols = [col for col in output_columns if col not in data.columns]
        if missing_cols:
            logger.warning(
                f"Missing expected columns: {missing_cols}. Using available columns."
            )
            output_columns = [col for col in output_columns if col in data.columns]

        output_data = data[output_columns].copy()

        # Add epoch columns if epoching is enabled in preprocessing mode
        output_data = self._add_epoch_columns(output_data, moment, time_column="time")

        # Convert boolean/flag columns to int
        if "TEMP_Outliers" in output_data.columns:
            output_data["TEMP_Outliers"] = output_data["TEMP_Outliers"].astype(int)
        if "TEMP_Interpolated" in output_data.columns:
            output_data["TEMP_Interpolated"] = output_data["TEMP_Interpolated"].astype(
                int
            )

        # Write UNCOMPRESSED TSV (no .gz)
        output_data.to_csv(signals_tsv, sep="\t", index=False, na_rep="n/a")

        logger.debug(
            f"Saved processed signals: {signals_tsv} "
            f"({len(output_data)} samples, {len(output_columns)} columns)"
        )
        return signals_tsv

    def _write_physio_metadata(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        data: pd.DataFrame,
        cleaning_metadata: Dict[str, Any],
    ) -> Path:
        """
        Write physio signal metadata file.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            data: Cleaned temperature data
            cleaning_metadata: Cleaning process metadata

        Returns:
            Path to written file
        """
        # BIDS filename pattern for processed physio data
        base_filename = (
            f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-temp"
        )

        signals_json = subject_dir / f"{base_filename}.json"

        # Calculate signal characteristics
        sampling_rate = 4.0  # Temperature is 4 Hz for Empatica E4
        duration = (
            data["time"].iloc[-1] - data["time"].iloc[0] if len(data) > 0 else 0.0
        )

        metadata = {
            "TaskName": moment,
            "SamplingFrequency": sampling_rate,
            "StartTime": 0.0,
            "Columns": [
                "time",
                "TEMP_Raw",
                "TEMP_Clean",
                "TEMP_Quality",
                "TEMP_Outliers",
                "TEMP_Interpolated",
            ],
            "Units": ["s", "°C", "°C", "a.u.", "n/a", "n/a"],
            "Descriptions": [
                "Time in seconds from start of recording",
                "Raw peripheral skin temperature in degrees Celsius (before cleaning)",
                "Cleaned peripheral skin temperature in degrees Celsius (after outlier removal and interpolation)",
                "Quality score (0-1, 1=highest quality)",
                "Outlier flag (1=outlier removed, 0=valid)",
                "Interpolation flag (1=interpolated, 0=original)",
            ],
            "ProcessingMetadata": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "Duration": float(duration),
                "ValidSamples": int(cleaning_metadata.get("valid_samples", len(data))),
                "TotalSamples": int(cleaning_metadata.get("total_samples", len(data))),
                "QualityScore": float(cleaning_metadata.get("quality_score", 0)),
                "OutlierThreshold": cleaning_metadata.get(
                    "processing_parameters", {}
                ).get("outlier_threshold_celsius", [25, 40]),
                "JumpThreshold": cleaning_metadata.get("processing_parameters", {}).get(
                    "jump_threshold_celsius", 2.0
                ),
                "InterpolationMaxGap": cleaning_metadata.get(
                    "processing_parameters", {}
                ).get("interpolation_max_gap_seconds", 10),
            },
        }

        # Use base class JSON serialization
        self._save_json_sidecar(signals_json, metadata)

        logger.debug(f"Saved processed signals metadata: {signals_json}")
        return signals_json

    def _write_metrics_file(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        metrics: Dict[str, Any],
    ) -> Path:
        """
        Write temperature metrics to TSV file.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            metrics: Extracted temperature metrics (can be flat dict or nested with categories)

        Returns:
            Path to written file
        """
        # BIDS filename pattern
        base_filename = f"{subject_id}_{session_id}_task-{moment}"
        metrics_tsv = subject_dir / f"{base_filename}_desc-temp-metrics.tsv"

        # Handle both flat and nested dictionaries
        flattened_metrics = {}

        # If metrics is already a flat dictionary (from _extract_basic_metrics)
        if all(not isinstance(v, dict) for v in metrics.values()):
            flattened_metrics = metrics.copy()
        else:
            # Flatten nested metrics dictionary (from TEMPMetricsExtractor)
            for category, category_metrics in metrics.items():
                if category in ["moment", "summary"]:
                    # Keep these at top level
                    if category == "moment":
                        flattened_metrics["moment"] = category_metrics
                    continue
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        flattened_metrics[metric_name] = value

            # Add summary information if present
            if "summary" in metrics:
                flattened_metrics["total_metrics"] = metrics["summary"].get(
                    "total_metrics_extracted", 0
                )
                flattened_metrics["quality_assessment"] = metrics["summary"].get(
                    "overall_quality_assessment", "unknown"
                )

        # Create single-row DataFrame
        metrics_df = pd.DataFrame([flattened_metrics])

        # Write metrics TSV
        metrics_df.to_csv(metrics_tsv, sep="\t", index=False, na_rep="n/a")

        logger.debug(
            f"Saved temperature metrics: {metrics_tsv} ({len(flattened_metrics)} metrics)"
        )
        return metrics_tsv

    def _write_metrics_metadata(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        metrics: Dict[str, Any],
    ) -> Path:
        """
        Write metrics metadata file with descriptions.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            metrics: Extracted temperature metrics

        Returns:
            Path to written file
        """
        # BIDS filename pattern
        base_filename = f"{subject_id}_{session_id}_task-{moment}"
        metrics_json = subject_dir / f"{base_filename}_desc-temp-metrics.json"

        # Try to import TEMPMetricsExtractor to get full descriptions
        try:
            from src.physio.preprocessing.temp_metrics import TEMPMetricsExtractor

            extractor = TEMPMetricsExtractor()
            descriptions = extractor.get_metrics_description()
        except ImportError:
            logger.warning(
                "Could not import TEMPMetricsExtractor, using basic descriptions"
            )
            descriptions = {}

        metadata = {
            "Description": "Peripheral skin temperature metrics extracted from cleaned temperature signals",
            "TaskName": moment,
            "MetricsCategories": {
                "descriptive": "Basic statistical measures of temperature distribution",
                "trend": "Temporal trends and changes in temperature over time",
                "stability": "Measures of temperature variability and stability",
                "contextual": "Recording context and data quality metrics",
            },
            "MetricsDescriptions": descriptions,
            "ProcessingInfo": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "TotalMetrics": metrics.get("summary", {}).get(
                    "total_metrics_extracted",
                    len([k for k in metrics if k not in ["moment", "summary"]]),
                ),
                "QualityAssessment": metrics.get("summary", {}).get(
                    "overall_quality_assessment", "unknown"
                ),
            },
            "Units": {
                "temp_*": "°C (degrees Celsius) unless otherwise specified",
                "*_duration": "seconds",
                "*_quality": "quality score (0-1)",
                "*_samples": "number of samples",
                "*_slope_per_minute": "°C per minute",
            },
            "Interpretation": {
                "temp_change_positive": "Increasing temperature may indicate relaxation (vasodilation)",
                "temp_change_negative": "Decreasing temperature may indicate stress/anxiety (vasoconstriction)",
                "typical_range": "Normal peripheral skin temperature: 25-35°C",
            },
        }

        # Use base class JSON serialization
        self._save_json_sidecar(metrics_json, metadata)

        logger.debug(f"Saved temperature metrics metadata: {metrics_json}")
        return metrics_json

    def _write_summary_file(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        metrics: Dict[str, Any],
        cleaning_metadata: Dict[str, Any],
        file_paths: Dict[str, Path],
    ) -> Path:
        """
        Write processing summary file.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            metrics: Extracted temperature metrics
            cleaning_metadata: Cleaning process metadata
            file_paths: Dictionary of written file paths for this moment

        Returns:
            Path to written file
        """
        # BIDS filename pattern
        base_filename = f"{subject_id}_{session_id}_task-{moment}"
        summary_json = subject_dir / f"{base_filename}_desc-temp-summary.json"

        summary = {
            "ProcessingInfo": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "Subject": subject_id,
                "Session": session_id,
                "Task": moment,
            },
            "DataQuality": {
                "TotalSamples": int(cleaning_metadata.get("total_samples", 0)),
                "ValidSamples": int(cleaning_metadata.get("valid_samples", 0)),
                "DataCompleteness": float(
                    cleaning_metadata.get("data_completeness", 0)
                ),
                "QualityScore": float(cleaning_metadata.get("quality_score", 0)),
                "OutlierPercentage": float(
                    cleaning_metadata.get("outlier_percentage", 0)
                ),
                "InterpolatedPercentage": float(
                    cleaning_metadata.get("interpolated_percentage", 0)
                ),
            },
            "MetricsSummary": {
                "TotalMetricsExtracted": metrics.get("summary", {}).get(
                    "total_metrics_extracted",
                    len([k for k in metrics if k not in ["moment", "summary"]]),
                ),
                "QualityAssessment": metrics.get("summary", {}).get(
                    "overall_quality_assessment", "unknown"
                ),
                "DescriptiveMetrics": metrics.get("summary", {}).get(
                    "descriptive_count", 0
                ),
                "TrendMetrics": metrics.get("summary", {}).get("trend_count", 0),
                "StabilityMetrics": metrics.get("summary", {}).get(
                    "stability_count", 0
                ),
                "ContextualMetrics": metrics.get("summary", {}).get(
                    "contextual_count", 0
                ),
            },
            "KeyResults": {
                "MeanTemp": metrics.get("descriptive", {}).get("temp_mean")
                if "descriptive" in metrics
                else metrics.get("temp_mean"),
                "TempRange": metrics.get("descriptive", {}).get("temp_range")
                if "descriptive" in metrics
                else metrics.get("temp_range"),
                "TempChange": metrics.get("trend", {}).get("temp_change")
                if "trend" in metrics
                else metrics.get("temp_change"),
                "TempSlopePerMinute": metrics.get("trend", {}).get(
                    "temp_slope_per_minute"
                )
                if "trend" in metrics
                else metrics.get("temp_slope_per_minute"),
                "Duration": metrics.get("contextual", {}).get("temp_duration")
                if "contextual" in metrics
                else metrics.get("temp_duration"),
            },
            "OutputFiles": {
                "ProcessedSignals": str(file_paths.get("physio", "")),
                "Metrics": str(file_paths.get("metrics", "")),
                "Summary": str(summary_json),
            },
        }

        # Use base class JSON serialization
        self._save_json_sidecar(summary_json, summary)

        logger.debug(f"Saved temperature processing summary: {summary_json}")
        return summary_json
