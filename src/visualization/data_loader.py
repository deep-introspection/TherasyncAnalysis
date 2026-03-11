"""
Data Loader for Visualization Module.

This module loads preprocessed physiological data from the BIDS derivatives
structure for visualization purposes.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import logging

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def discover_moments(subject: str, session: str, derivatives_path: Path) -> List[str]:
    """
    Automatically discover available moments/tasks for a subject/session.

    Scans the preprocessing directory and extracts moment names from BIDS
    filename patterns (task-{moment}_).

    Args:
        subject: Subject ID (e.g., 'g01p01')
        session: Session ID (e.g., '01')
        derivatives_path: Path to derivatives/preprocessing directory

    Returns:
        Sorted list of moment names found (e.g., ['restingstate', 'therapy'])
        Returns empty list if no moments found

    Examples:
        >>> moments = discover_moments('g01p01', '01', Path('data/derivatives/preprocessing'))
        >>> print(moments)  # ['restingstate', 'therapy']
    """
    subject_id = f"sub-{subject}" if not subject.startswith("sub-") else subject
    session_id = f"ses-{session}" if not session.startswith("ses-") else session

    subject_session_path = derivatives_path / subject_id / session_id

    if not subject_session_path.exists():
        logger.warning(f"Subject/session directory not found: {subject_session_path}")
        return []

    moments = set()

    # Scan all modality directories
    for modality in ["bvp", "eda", "hr", "temp"]:
        modality_path = subject_session_path / modality

        if not modality_path.exists():
            continue

        # Extract task-{moment} from all TSV files
        for file in modality_path.glob("*task-*.tsv"):
            match = re.search(r"task-(\w+)_", file.name)
            if match:
                moment = match.group(1)
                moments.add(moment)
                logger.debug(f"Found moment '{moment}' in {file.name}")

    result = sorted(moments)
    logger.info(
        f"Discovered {len(result)} moments for {subject_id}/{session_id}: {result}"
    )

    return result


class VisualizationDataLoader:
    """
    Loads preprocessed BVP, EDA, and HR data for visualization.

    This class handles loading from the BIDS derivatives structure:
    data/derivatives/preprocessing/sub-{subject}/ses-{session}/{modality}/

    Examples:
        >>> loader = VisualizationDataLoader()
        >>> data = loader.load_subject_session('g01p01', '01')
        >>> bvp_signals = data['bvp']['signals']
        >>> eda_metrics = data['eda']['metrics']
    """

    def __init__(
        self,
        derivatives_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the data loader.

        Args:
            derivatives_path: Path to derivatives directory
                Default: Loaded from config YAML
            config_path: Path to configuration YAML file
                Default: config/config.yaml
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config

        # Use derivatives_path from config if not provided
        if derivatives_path is None:
            base_path = Path(self.config["paths"]["derivatives"])
            preprocessing_dir = self.config["output"]["preprocessing_dir"]
            derivatives_path = base_path / preprocessing_dir

        self.derivatives_path = Path(derivatives_path)

        if not self.derivatives_path.exists():
            raise FileNotFoundError(
                f"Derivatives directory not found: {self.derivatives_path}"
            )

    def load_subject_session(
        self, subject: str, session: str, modalities: Optional[List[str]] = None
    ) -> Dict:
        """
        Load all data for a subject/session.

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., '01')
            modalities: List of modalities to load ['bvp', 'eda', 'hr']
                Default: Load all available

        Returns:
            Dictionary with structure:
            {
                'bvp': {'signals': {...}, 'metrics': {...}, 'metadata': {...}},
                'eda': {'signals': {...}, 'metrics': {...}, 'events': {...}, 'metadata': {...}},
                'hr': {'signals': {...}, 'metrics': {...}, 'metadata': {...}},
                'subject': 'g01p01',
                'session': '01'
            }
        """
        if modalities is None:
            modalities = ["bvp", "eda", "hr", "temp"]

        # Build paths
        subject_id = f"sub-{subject}" if not subject.startswith("sub-") else subject
        session_id = f"ses-{session}" if not session.startswith("ses-") else session

        subject_session_path = self.derivatives_path / subject_id / session_id

        if not subject_session_path.exists():
            raise FileNotFoundError(
                f"Subject/session directory not found: {subject_session_path}"
            )

        logger.info(f"Loading data for {subject_id}/{session_id}")

        data = {
            "subject": subject,
            "session": session,
            "subject_id": subject_id,
            "session_id": session_id,
            "config": self.config,  # Include config for plotters
        }

        # Load each modality
        for modality in modalities:
            modality_path = subject_session_path / modality

            if not modality_path.exists():
                logger.warning(
                    f"Modality {modality} not found for {subject_id}/{session_id}"
                )
                continue

            logger.info(f"Loading {modality} data...")
            data[modality] = self._load_modality_data(
                modality, modality_path, subject_id, session_id
            )

        return data

    def _discover_moments_in_modality(
        self, modality_path: Path, modality: str
    ) -> List[str]:
        """
        Discover moments available in a specific modality directory.

        Args:
            modality_path: Path to modality directory (e.g., .../bvp/)
            modality: Modality name ('bvp', 'eda', 'hr')

        Returns:
            Sorted list of moment names found
        """
        moments = set()

        # Look for files with task-{moment} pattern
        for file in modality_path.glob("*task-*_*.tsv"):
            match = re.search(r"task-(\w+)_", file.name)
            if match:
                moments.add(match.group(1))

        return sorted(moments)

    def _load_modality_data(
        self, modality: str, modality_path: Path, subject_id: str, session_id: str
    ) -> Dict:
        """
        Load processed signals, metadata, and metrics for a single modality.

        Args:
            modality: Modality name ('bvp', 'eda', 'hr', 'temp')
            modality_path: Path to modality directory
            subject_id: BIDS subject ID (e.g., 'sub-g01p01')
            session_id: BIDS session ID (e.g., 'ses-01')

        Returns:
            Dictionary with 'signals', 'metrics', 'metadata' (and 'events' for EDA)
        """
        prefix = f"{subject_id}_{session_id}"
        mod_upper = modality.upper()
        data: Dict = {"signals": {}, "metrics": None, "metadata": {}}

        if modality == "eda":
            data["events"] = {}

        moments = self._discover_moments_in_modality(modality_path, modality)

        for moment in moments:
            # Signals
            signal_file = (
                modality_path
                / f"{prefix}_task-{moment}_desc-processed_recording-{modality}.tsv"
            )
            if signal_file.exists():
                data["signals"][moment] = pd.read_csv(signal_file, sep="\t")
                logger.info(
                    f"  Loaded {mod_upper} signals for {moment}: {len(data['signals'][moment])} samples"
                )
            else:
                logger.warning(
                    f"  {mod_upper} signal file not found for moment '{moment}': {signal_file}"
                )

            # Metadata
            metadata_file = signal_file.with_suffix(".json")
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    data["metadata"][moment] = json.load(f)

            # EDA-specific: SCR events
            if modality == "eda":
                events_file = (
                    modality_path / f"{prefix}_task-{moment}_desc-scr_events.tsv"
                )
                if events_file.exists():
                    data["events"][moment] = pd.read_csv(events_file, sep="\t")
                    logger.info(
                        f"  Loaded SCR events for {moment}: {len(data['events'][moment])} events"
                    )

        # Load metrics (modality-specific patterns)
        data["metrics"] = self._load_modality_metrics(
            modality, modality_path, prefix, moments
        )

        return data

    def _load_modality_metrics(
        self, modality: str, modality_path: Path, prefix: str, moments: list[str]
    ) -> pd.DataFrame | None:
        """Load metrics for a modality, handling different file formats per modality."""
        mod_upper = modality.upper()

        if modality == "hr":
            # HR uses a JSON summary
            metrics_file = modality_path / f"{prefix}_desc-hr-summary.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    logger.info("  Loaded HR summary metrics")
                    return pd.DataFrame([json.load(f)])
            return None

        # TSV metrics (bvp, eda, temp)
        metrics_file = modality_path / f"{prefix}_desc-{modality}-metrics_physio.tsv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file, sep="\t")
            logger.info(f"  Loaded {mod_upper} metrics: {len(df)} rows")
            return df

        # TEMP fallback: per-moment metrics files
        if modality == "temp":
            metrics_list = []
            for moment in moments:
                moment_file = (
                    modality_path / f"{prefix}_task-{moment}_desc-temp-metrics.tsv"
                )
                if moment_file.exists():
                    moment_df = pd.read_csv(moment_file, sep="\t")
                    moment_df["moment"] = moment
                    metrics_list.append(moment_df)
            if metrics_list:
                combined = pd.concat(metrics_list, ignore_index=True)
                logger.info(
                    f"  Loaded TEMP metrics from {len(metrics_list)} moments: {len(combined)} rows"
                )
                return combined

        return None

    def list_available_subjects(self) -> List[Tuple[str, str]]:
        """
        List all available subject/session combinations.

        Returns:
            List of (subject, session) tuples
        """
        subjects_sessions = []

        for subject_dir in sorted(self.derivatives_path.glob("sub-*")):
            subject = subject_dir.name.replace("sub-", "")

            for session_dir in sorted(subject_dir.glob("ses-*")):
                session = session_dir.name.replace("ses-", "")

                # Check if at least one modality exists
                has_data = any(
                    [
                        (session_dir / "bvp").exists(),
                        (session_dir / "eda").exists(),
                        (session_dir / "hr").exists(),
                        (session_dir / "temp").exists(),
                    ]
                )

                if has_data:
                    subjects_sessions.append((subject, session))

        logger.info(f"Found {len(subjects_sessions)} subject/session combinations")
        return subjects_sessions

    def get_available_modalities(self, subject: str, session: str) -> List[str]:
        """
        Get list of available modalities for a subject/session.

        Args:
            subject: Subject ID
            session: Session ID

        Returns:
            List of available modalities ['bvp', 'eda', 'hr']
        """
        subject_id = f"sub-{subject}" if not subject.startswith("sub-") else subject
        session_id = f"ses-{session}" if not session.startswith("ses-") else session

        subject_session_path = self.derivatives_path / subject_id / session_id

        modalities = []
        for modality in ["bvp", "eda", "hr", "temp"]:
            if (subject_session_path / modality).exists():
                modalities.append(modality)

        return modalities
