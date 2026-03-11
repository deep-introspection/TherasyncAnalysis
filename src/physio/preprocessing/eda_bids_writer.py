"""
EDA BIDS Writer for TherasyncPipeline.

This module provides functionality to save processed EDA data and extracted metrics
in BIDS-compliant format under data/derivatives/.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

from src.physio.preprocessing.base_bids_writer import PhysioBIDSWriter
from src.core.bids_utils import BIDSUtils

logger = logging.getLogger(__name__)


class EDABIDSWriter(PhysioBIDSWriter):
    """
    Save processed EDA data and metrics in BIDS-compliant format.

    This class handles saving processed signals (tonic, phasic, SCR events),
    extracted metrics, and metadata following BIDS derivatives specifications
    for physiological data.

    Inherits from PhysioBIDSWriter to ensure consistent API across modalities.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the EDA BIDS writer with configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        super().__init__(config_path)
        self.bids_utils = BIDSUtils()

        # EDA-specific configuration
        self.modality_subdir = self.config.get("output.modality_subdirs.eda", "eda")
        self.pipeline_name = "therasync-eda"

    def _get_modality_name(self) -> str:
        """Return the modality identifier for EDA."""
        return "eda"

    def save_processed_data(
        self,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, pd.DataFrame],
        session_metrics: Optional[pd.DataFrame] = None,
        processing_metadata: Optional[Dict] = None,
    ) -> Dict[str, List[Path]]:
        """
        Save processed EDA data and metrics in BIDS format.

        Args:
            subject_id: Subject identifier (with or without 'sub-' prefix)
            session_id: Session identifier (with or without 'ses-' prefix)
            processed_results: Dict of processed DataFrames from EDACleaner
                             (keys: moment names, values: processed signals with EDA_Quality)
            session_metrics: DataFrame with extracted metrics from EDAMetricsExtractor
            processing_metadata: Additional metadata about processing

        Returns:
            Dictionary with lists of created file paths (Path objects)
        """
        # Ensure proper prefixes
        subject_id = self._ensure_prefix(subject_id, "sub-")
        session_id = self._ensure_prefix(session_id, "ses-")

        # Get subject/session directory using base class method
        subject_dir = self._get_subject_session_dir(subject_id, session_id)

        created_files = {
            "processed_signals": [],
            "scr_events": [],
            "metrics": [],
            "metadata": [],
            "summary": [],
        }

        logger.info(f"Saving processed EDA data for {subject_id}/{session_id}")

        # Save processed signals for each moment
        for moment, processed_signals in processed_results.items():
            # Save processed signals (tonic, phasic components)
            signal_files = self._save_processed_signals(
                subject_dir, subject_id, session_id, moment, processed_signals
            )
            created_files["processed_signals"].extend(signal_files)

            # Save SCR events if peaks detected
            if "SCR_Peaks" in processed_signals.columns:
                scr_files = self._save_scr_events(
                    subject_dir, subject_id, session_id, moment, processed_signals
                )
                created_files["scr_events"].extend(scr_files)

            # Save moment-specific metadata
            metadata_file = self._save_moment_metadata(
                subject_dir, subject_id, session_id, moment, processed_signals
            )
            if metadata_file:
                created_files["metadata"].append(metadata_file)

        # Save extracted metrics if provided
        if session_metrics is not None and not session_metrics.empty:
            metrics_files = self._save_session_metrics(
                subject_dir, subject_id, session_id, session_metrics
            )
            created_files["metrics"].extend(metrics_files)

        # Save processing summary
        summary_file = self._save_processing_summary(
            subject_dir,
            subject_id,
            session_id,
            processed_results,
            session_metrics,
            processing_metadata,
        )
        if summary_file:
            created_files["summary"].append(summary_file)

        total_files = sum(len(files) for files in created_files.values())
        logger.info(
            f"Created {total_files} BIDS-compliant files for {subject_id}/{session_id}"
        )

        return created_files

    def _save_processed_signals(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        processed_signals: pd.DataFrame,
    ) -> List[Path]:
        """
        Save processed EDA signals (tonic, phasic) in BIDS format.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            processed_signals: Processed signals DataFrame with tonic/phasic and EDA_Quality

        Returns:
            List of created file paths (Path objects)
        """
        created_files = []

        # BIDS filename pattern for processed physio data
        base_filename = (
            f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-eda"
        )

        # Save processed signals as TSV
        signals_tsv = subject_dir / f"{base_filename}.tsv"

        # Prepare signals data for saving
        output_data = processed_signals.copy()

        # Add time column if not present
        if "time" not in output_data.columns:
            # Get sampling rate from config or use default 4 Hz
            sampling_rate = self.config.get("physio.eda.sampling_rate", 4)
            time_values = np.arange(len(output_data)) / sampling_rate
            output_data.insert(0, "time", time_values)

        # Add epoch columns if epoching is enabled in preprocessing mode
        output_data = self._add_epoch_columns(output_data, moment, time_column="time")

        # Select columns to save (include EDA_Quality, exclude SCR_Peaks binary if present)
        columns_to_save = [
            "time",
            "EDA_Raw",
            "EDA_Clean",
            "EDA_Tonic",
            "EDA_Phasic",
            "EDA_Quality",
        ]
        # Also include any epoch columns
        epoch_cols = [col for col in output_data.columns if col.startswith("epoch_")]
        columns_to_save.extend(epoch_cols)

        columns_available = [
            col for col in columns_to_save if col in output_data.columns
        ]
        output_data_filtered = output_data[columns_available]

        # Save TSV file
        output_data_filtered.to_csv(signals_tsv, sep="\t", index=False, na_rep="n/a")
        created_files.append(signals_tsv)

        # Create JSON sidecar for processed signals using base class method
        signals_metadata = self._create_processed_signals_metadata(
            processed_signals, moment
        )
        signals_json = subject_dir / f"{base_filename}.json"
        self._save_json_sidecar(signals_json, signals_metadata)
        created_files.append(signals_json)

        logger.debug(f"Saved processed signals: {signals_tsv}")

        return created_files

    def _save_scr_events(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        processed_signals: pd.DataFrame,
    ) -> List[Path]:
        """
        Save SCR (Skin Conductance Response) events in BIDS format.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            processed_signals: Processed signals with SCR peak information

        Returns:
            List of created file paths (Path objects)
        """
        created_files = []

        # Check if SCR peaks detected
        if "SCR_Peaks" not in processed_signals.columns:
            return created_files

        # Get SCR peak indices
        scr_peaks = processed_signals[processed_signals["SCR_Peaks"] == 1]

        if len(scr_peaks) == 0:
            logger.debug(f"No SCR peaks detected for {moment}")
            return created_files

        # BIDS filename for events
        base_filename = f"{subject_id}_{session_id}_task-{moment}_desc-scr_events"

        # Create events DataFrame
        sampling_rate = self.config.get("physio.eda.sampling_rate", 4)

        events_data = []
        for idx, row in scr_peaks.iterrows():
            event = {
                "onset": row.get("time", idx / sampling_rate),
                "duration": row.get("SCR_RecoveryTime", 0),
                "amplitude": row.get("SCR_Amplitude", np.nan),
                "rise_time": row.get("SCR_RiseTime", np.nan),
                "recovery_time": row.get("SCR_RecoveryTime", np.nan),
            }
            events_data.append(event)

        events_df = pd.DataFrame(events_data)

        # Save events as TSV
        events_tsv = subject_dir / f"{base_filename}.tsv"
        events_df.to_csv(events_tsv, sep="\t", index=False, na_rep="n/a")
        created_files.append(events_tsv)

        # Create JSON sidecar for events
        events_metadata = {
            "Description": "Skin Conductance Response (SCR) events detected in EDA signal",
            "Columns": {
                "onset": "Event onset time in seconds",
                "duration": "SCR recovery time in seconds",
                "amplitude": "SCR amplitude in microsiemens",
                "rise_time": "SCR rise time in seconds",
                "recovery_time": "SCR recovery time in seconds",
            },
            "Units": {
                "onset": "s",
                "duration": "s",
                "amplitude": "μS",
                "rise_time": "s",
                "recovery_time": "s",
            },
            "NumberOfEvents": len(events_df),
            "ProcessingPipeline": "therasync-eda",
            "ProcessingVersion": self.pipeline_version,
        }

        events_json = subject_dir / f"{base_filename}.json"
        self._save_json_sidecar(events_json, events_metadata)
        created_files.append(events_json)

        logger.debug(f"Saved SCR events: {events_tsv} ({len(events_df)} events)")

        return created_files

    def _save_session_metrics(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        session_metrics: pd.DataFrame,
    ) -> List[Path]:
        """
        Save extracted EDA metrics in BIDS format.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            session_metrics: Extracted metrics DataFrame

        Returns:
            List of created file paths (Path objects)
        """
        created_files = []

        if session_metrics.empty:
            logger.warning("No session metrics to save")
            return created_files

        # BIDS filename for metrics
        base_filename = f"{subject_id}_{session_id}_desc-eda-metrics_physio"

        # Save metrics as TSV
        metrics_tsv = subject_dir / f"{base_filename}.tsv"
        session_metrics.to_csv(metrics_tsv, sep="\t", index=False, na_rep="n/a")
        created_files.append(metrics_tsv)

        # Create JSON sidecar for metrics
        metrics_metadata = self._create_metrics_metadata(session_metrics)
        metrics_json = subject_dir / f"{base_filename}.json"
        self._save_json_sidecar(metrics_json, metrics_metadata)
        created_files.append(metrics_json)

        logger.debug(f"Saved EDA metrics: {metrics_tsv}")

        return created_files

    def _save_moment_metadata(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        processed_signals: pd.DataFrame,
    ) -> Optional[Path]:
        """
        Save moment-specific processing metadata.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            processed_signals: Processed signals DataFrame

        Returns:
            Path to created metadata file, or None if failed
        """
        try:
            # BIDS filename for moment metadata
            metadata_filename = f"{subject_id}_{session_id}_task-{moment}_desc-processing_recording-eda.json"
            metadata_file = subject_dir / metadata_filename

            # Get processing parameters
            eda_config = self.config.get("physio.eda", {})
            processing_config = eda_config.get("processing", {})

            # Count SCR peaks if present
            num_scr_peaks = (
                int(processed_signals["SCR_Peaks"].sum())
                if "SCR_Peaks" in processed_signals.columns
                else 0
            )

            # Create comprehensive metadata
            metadata = {
                "TaskName": moment,
                "ProcessingMethod": processing_config.get("method", "cvxEDA"),
                "SamplingFrequency": eda_config.get("sampling_rate", 4),
                "NumberOfSamples": len(processed_signals),
                "Duration": len(processed_signals) / eda_config.get("sampling_rate", 4),
                "NumberOfSCRPeaks": num_scr_peaks,
                "SCRDetectionThreshold": processing_config.get("scr_threshold", 0.01),
                "FilteringApplied": processing_config.get("filter", True),
                "ProcessingTimestamp": datetime.now().isoformat(),
                "ProcessingPipeline": "therasync-eda",
                "ProcessingVersion": self.pipeline_version,
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=self._json_serializer)

            logger.debug(f"Saved moment metadata: {metadata_file}")
            return metadata_file

        except Exception as e:
            logger.error(f"Failed to save moment metadata for {moment}: {e}")
            return None

    def _save_processing_summary(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, pd.DataFrame],
        session_metrics: Optional[pd.DataFrame],
        processing_metadata: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Save overall processing summary.

        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            processed_results: Processed results dictionary
            session_metrics: Session metrics DataFrame (optional)
            processing_metadata: Additional processing metadata

        Returns:
            Path to created summary file, or None if failed
        """
        try:
            # BIDS filename for summary
            summary_filename = (
                f"{subject_id}_{session_id}_desc-eda-summary_recording-eda.json"
            )
            summary_file = subject_dir / summary_filename

            # Calculate total SCR peaks across all moments
            total_scr_peaks = 0
            for processed_signals in processed_results.values():
                if "SCR_Peaks" in processed_signals.columns:
                    total_scr_peaks += int(processed_signals["SCR_Peaks"].sum())

            # Calculate total duration
            sampling_rate = self.config.get("physio.eda.sampling_rate", 4)
            total_duration = sum(
                len(signals) / sampling_rate for signals in processed_results.values()
            )

            # Create processing summary
            summary = {
                "SubjectID": subject_id,
                "SessionID": session_id,
                "ProcessingDate": datetime.now().isoformat(),
                "ProcessingPipeline": "therasync-eda",
                "ProcessingVersion": self.pipeline_version,
                "MomentsProcessed": list(processed_results.keys()),
                "TotalSignalDuration": total_duration,
                "TotalSCRPeaks": total_scr_peaks,
            }

            # Add metrics info if available
            if session_metrics is not None and not session_metrics.empty:
                summary["MetricsExtracted"] = (
                    len(session_metrics.columns) - 1
                    if "moment" in session_metrics.columns
                    else len(session_metrics.columns)
                )
                summary["QualityAssessment"] = self._assess_overall_quality(
                    processed_results, session_metrics
                )

            # Add custom metadata if provided
            if processing_metadata:
                summary["AdditionalMetadata"] = processing_metadata

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=self._json_serializer)

            logger.debug(f"Saved processing summary: {summary_file}")
            return summary_file

        except Exception as e:
            logger.error(f"Failed to save processing summary: {e}")
            return None

    def _create_processed_signals_metadata(
        self, processed_signals: pd.DataFrame, moment: str
    ) -> Dict:
        """Create metadata for processed signals."""
        sampling_rate = self.config.get("physio.eda.sampling_rate", 4)
        eda_config = self.config.get("physio.eda", {})
        processing_config = eda_config.get("processing", {})

        # Include EDA_Quality if present
        available_columns = [
            col
            for col in processed_signals.columns
            if col
            in [
                "time",
                "EDA_Raw",
                "EDA_Clean",
                "EDA_Tonic",
                "EDA_Phasic",
                "EDA_Quality",
            ]
        ]

        metadata = {
            "Description": "Processed EDA signals from TherasyncPipeline with tonic-phasic decomposition and quality assessment",
            "TaskName": moment,
            "SamplingFrequency": sampling_rate,
            "StartTime": 0,
            "ProcessingMethod": processing_config.get("method", "cvxEDA"),
            "Columns": available_columns,
            "Units": {
                "time": "s",
                "EDA_Raw": "μS",
                "EDA_Clean": "μS",
                "EDA_Tonic": "μS",
                "EDA_Phasic": "μS",
                "EDA_Quality": "0-1 (quality score)",
            },
            "ProcessingPipeline": "therasync-eda",
            "ProcessingVersion": self.pipeline_version,
            "DecompositionNote": "Tonic = slowly varying baseline, Phasic = rapid SCR responses",
            "QualityNote": "Quality score based on signal stability and physiological plausibility (1=highest quality)",
        }

        return metadata

    def _create_metrics_metadata(self, session_metrics: pd.DataFrame) -> Dict:
        """Create metadata for extracted metrics."""
        # Get all metric column names (exclude 'moment' column)
        metric_columns = [col for col in session_metrics.columns if col != "moment"]

        metadata = {
            "Description": "EDA-derived SCR and autonomic nervous system metrics from TherasyncPipeline",
            "ProcessingPipeline": "therasync-eda",
            "ProcessingVersion": self.pipeline_version,
            "MetricsExtracted": metric_columns,
            "Columns": {
                "moment": "Task/moment identifier",
            },
            "Units": {
                "moment": "categorical",
                # SCR metrics
                "SCR_Peaks_N": "count",
                "SCR_Peaks_Rate": "per minute",
                "SCR_Peaks_Amplitude_Mean": "μS",
                "SCR_Peaks_Amplitude_Max": "μS",
                "SCR_Peaks_Amplitude_SD": "μS",
                "SCR_Peaks_RiseTime_Mean": "s",
                "SCR_Peaks_RiseTime_SD": "s",
                "SCR_Peaks_RecoveryTime_Mean": "s",
                "SCR_Peaks_RecoveryTime_SD": "s",
                # Tonic metrics
                "EDA_Tonic_Mean": "μS",
                "EDA_Tonic_SD": "μS",
                "EDA_Tonic_Min": "μS",
                "EDA_Tonic_Max": "μS",
                "EDA_Tonic_Range": "μS",
                # Phasic metrics
                "EDA_Phasic_Mean": "μS",
                "EDA_Phasic_SD": "μS",
                "EDA_Phasic_Min": "μS",
                "EDA_Phasic_Max": "μS",
                "EDA_Phasic_Range": "μS",
                "EDA_Phasic_Rate": "per minute",
                # Metadata
                "EDA_Duration": "s",
                "EDA_SamplingRate": "Hz",
            },
            "MetricCategories": {
                "SCR": "Skin Conductance Response (phasic event) metrics",
                "Tonic": "Slowly varying baseline skin conductance level",
                "Phasic": "Rapidly changing skin conductance responses",
            },
        }

        return metadata

    def _assess_overall_quality(
        self, processed_results: Dict[str, pd.DataFrame], session_metrics: pd.DataFrame
    ) -> Dict:
        """Assess overall quality of processing."""
        # Count total SCR peaks
        total_scr_peaks = 0
        for processed_signals in processed_results.values():
            if "SCR_Peaks" in processed_signals.columns:
                total_scr_peaks += int(processed_signals["SCR_Peaks"].sum())

        # Calculate mean SCR rate if available
        if "SCR_Peaks_Rate" in session_metrics.columns:
            scr_rates = session_metrics["SCR_Peaks_Rate"].dropna()
            mean_scr_rate = float(scr_rates.mean()) if len(scr_rates) > 0 else None
        else:
            mean_scr_rate = None

        # Calculate mean tonic level if available
        if "EDA_Tonic_Mean" in session_metrics.columns:
            tonic_means = session_metrics["EDA_Tonic_Mean"].dropna()
            mean_tonic_level = (
                float(tonic_means.mean()) if len(tonic_means) > 0 else None
            )
        else:
            mean_tonic_level = None

        quality_assessment = {
            "moments_processed": len(processed_results),
            "moments_with_metrics": len(session_metrics),
            "total_scr_peaks_detected": total_scr_peaks,
            "mean_scr_rate": mean_scr_rate,
            "mean_tonic_level": mean_tonic_level,
        }

        return quality_assessment

    # Note: _create_dataset_description and _json_serializer are inherited from PhysioBIDSWriter

    def create_group_summary(
        self,
        subjects_data: Dict[str, Dict[str, pd.DataFrame]],
        output_filename: str = "group_eda_metrics.tsv",
    ) -> str:
        """
        Create group-level summary of EDA metrics across subjects.

        Args:
            subjects_data: Nested dict {subject_id: {session_id: metrics_df}}
            output_filename: Name of output file

        Returns:
            Path to created group summary file
        """
        group_data = []

        for subject_id, sessions in subjects_data.items():
            for session_id, metrics_df in sessions.items():
                for _, row in metrics_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict["subject_id"] = subject_id
                    row_dict["session_id"] = session_id
                    group_data.append(row_dict)

        # Create DataFrame and save
        group_df = pd.DataFrame(group_data)

        # Reorder columns to put identifiers first
        id_cols = ["subject_id", "session_id", "moment"]
        other_cols = [col for col in group_df.columns if col not in id_cols]
        group_df = group_df[id_cols + other_cols]

        # Save to preprocessing directory
        preprocessing_base = self.derivatives_base / self.preprocessing_dir
        group_file = preprocessing_base / output_filename
        group_df.to_csv(group_file, sep="\t", index=False, na_rep="n/a")

        # Create accompanying JSON
        group_json = preprocessing_base / f"{output_filename.replace('.tsv', '.json')}"
        group_metadata = {
            "Description": "Group-level EDA metrics summary",
            "ProcessingPipeline": "therasync-eda",
            "ProcessingVersion": self.pipeline_version,
            "NumberOfSubjects": len(subjects_data),
            "TotalSessions": sum(len(sessions) for sessions in subjects_data.values()),
            "CreationDate": datetime.now().isoformat(),
        }

        with open(group_json, "w") as f:
            json.dump(group_metadata, f, indent=2)

        logger.info(f"Created group summary: {group_file}")
        return str(group_file)
