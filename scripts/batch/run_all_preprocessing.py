#!/usr/bin/env python
"""
Batch Preprocessing Script for TherasyncPipeline.

This script runs the complete preprocessing pipeline (BVP → EDA → HR) for all
available subjects and sessions in the raw data directory.

Usage:
    # Process all subjects/sessions
    python scripts/batch/run_all_preprocessing.py

    # Dry run to see what would be processed
    python scripts/batch/run_all_preprocessing.py --dry-run

    # Process specific subjects only
    python scripts/batch/run_all_preprocessing.py --subjects g01p01 g01p02

    # Skip subjects that already have outputs
    python scripts/batch/run_all_preprocessing.py --skip-existing

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class BatchPreprocessor:
    """Handles batch preprocessing of multiple subjects/sessions."""

    def __init__(self, config_path: Path = None, dry_run: bool = False):
        """
        Initialize batch preprocessor.

        Args:
            config_path: Path to config YAML file
            dry_run: If True, only show what would be processed without executing
        """
        self.config = ConfigLoader(config_path).config
        self.dry_run = dry_run

        # Paths from config
        self.rawdata_path = Path(self.config["paths"]["rawdata"])
        self.derivatives_path = Path(self.config["paths"]["derivatives"])
        self.preprocessing_path = (
            self.derivatives_path / self.config["output"]["preprocessing_dir"]
        )

        # Batch processing settings from config
        batch_config = self.config.get("batch", {})
        self.timeout = batch_config.get("timeout", 600)  # Default 10 minutes
        self.validate_durations = batch_config.get("validate_durations", True)
        self.duration_tolerance = batch_config.get("duration_tolerance", 5.0)  # seconds

        # Statistics
        self.stats = {
            "total_subjects": 0,
            "total_sessions": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "duration_warnings": [],
            "errors": [],
        }

    def find_subjects_sessions(self) -> List[Tuple[str, str]]:
        """
        Find all subject/session combinations in raw data.

        Returns:
            List of (subject_id, session_id) tuples
        """
        subjects_sessions = []

        if not self.rawdata_path.exists():
            logger.error(f"Raw data path not found: {self.rawdata_path}")
            return subjects_sessions

        # Find all subject directories
        subject_dirs = sorted(
            [
                d
                for d in self.rawdata_path.iterdir()
                if d.is_dir() and d.name.startswith("sub-")
            ]
        )

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name.replace("sub-", "")

            # Find all session directories for this subject
            session_dirs = sorted(
                [
                    d
                    for d in subject_dir.iterdir()
                    if d.is_dir() and d.name.startswith("ses-")
                ]
            )

            for session_dir in session_dirs:
                session_id = session_dir.name.replace("ses-", "")
                subjects_sessions.append((subject_id, session_id))

        return subjects_sessions

    def check_already_processed(
        self, subject_id: str, session_id: str
    ) -> Dict[str, bool]:
        """
        Check which modalities have already been processed.

        Args:
            subject_id: Subject ID (e.g., 'g01p01')
            session_id: Session ID (e.g., '01')

        Returns:
            Dictionary with modality: processed status
        """
        subject_session_path = (
            self.preprocessing_path / f"sub-{subject_id}" / f"ses-{session_id}"
        )

        status = {"bvp": False, "eda": False, "hr": False, "temp": False}

        if not subject_session_path.exists():
            return status

        # Check for key output files
        bvp_path = subject_session_path / "bvp"
        eda_path = subject_session_path / "eda"
        hr_path = subject_session_path / "hr"
        temp_path = subject_session_path / "temp"

        status["bvp"] = (bvp_path / "metrics.tsv").exists()
        status["eda"] = (eda_path / "metrics.tsv").exists()
        status["hr"] = (hr_path / "metrics.tsv").exists()
        status["temp"] = (
            len(list(temp_path.glob("*_desc-temp-metrics.tsv"))) > 0
            if temp_path.exists()
            else False
        )

        return status

    def validate_modality_durations(
        self, subject_id: str, session_id: str
    ) -> Tuple[bool, str]:
        """
        Validate that all processed modalities have consistent durations.

        This checks that BVP, EDA, and HR signals have similar durations,
        which indicates proper temporal alignment of the recordings.

        Args:
            subject_id: Subject ID (e.g., 'g01p01')
            session_id: Session ID (e.g., '01')

        Returns:
            Tuple of (is_valid, message)
        """

        subject_session_path = (
            self.preprocessing_path / f"sub-{subject_id}" / f"ses-{session_id}"
        )

        durations = {}

        # Check each modality for duration information
        for modality in ["bvp", "eda", "hr", "temp"]:
            modality_path = subject_session_path / modality
            if not modality_path.exists():
                continue

            # Look for summary JSON files to get duration
            summary_files = list(modality_path.glob("*_desc-*-summary.json"))
            if not summary_files:
                # Try alternative patterns
                summary_files = list(modality_path.glob("*_summary.json"))

            for summary_file in summary_files:
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)

                    # Extract duration from various possible locations
                    duration = None
                    if "DataQuality" in summary:
                        duration = summary["DataQuality"].get("Duration")
                    elif "KeyResults" in summary:
                        duration = summary["KeyResults"].get("Duration")
                    elif "ProcessingMetadata" in summary:
                        duration = summary["ProcessingMetadata"].get("Duration")

                    if duration is not None:
                        if modality not in durations:
                            durations[modality] = []
                        durations[modality].append(float(duration))

                except Exception as e:
                    logger.debug(f"Could not read duration from {summary_file}: {e}")

        # If we don't have durations for at least 2 modalities, skip validation
        if len(durations) < 2:
            return True, "Insufficient modalities for duration validation"

        # Calculate average duration per modality
        avg_durations = {mod: sum(durs) / len(durs) for mod, durs in durations.items()}

        # Check for significant differences
        all_durations = list(avg_durations.values())
        max_diff = max(all_durations) - min(all_durations)

        if max_diff > self.duration_tolerance:
            # Build detailed message
            details = ", ".join(
                [f"{mod}={dur:.1f}s" for mod, dur in avg_durations.items()]
            )
            message = f"Duration mismatch (diff={max_diff:.1f}s): {details}"
            return False, message

        return True, f"Durations consistent (diff={max_diff:.1f}s)"

    def run_preprocessing_script(
        self, script_name: str, subject_id: str, session_id: str
    ) -> bool:
        """
        Run a preprocessing script for a subject/session.

        Args:
            script_name: Name of the script (e.g., 'preprocess_bvp.py')
            subject_id: Subject ID
            session_id: Session ID

        Returns:
            True if successful, False otherwise
        """
        script_path = Path("scripts/physio/preprocessing") / script_name

        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--subject",
            subject_id,
            "--session",
            session_id,
        ]

        cmd_str = " ".join(cmd)
        logger.info(f"  Running: {cmd_str}")
        logger.debug(f"  Working directory: {Path.cwd()}")
        logger.debug(f"  Script exists: {script_path.exists()}")

        if self.dry_run:
            logger.info("  [DRY RUN] Would execute command")
            return True

        start_time = datetime.now()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,  # Use configurable timeout
                cwd=Path.cwd(),
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                logger.info(
                    f"  ✓ {script_name} completed successfully in {elapsed:.1f}s"
                )
                logger.debug(f"  STDOUT (last 200 chars): {result.stdout[-200:]}")
                return True
            else:
                logger.error(
                    f"  ✗ {script_name} failed with exit code {result.returncode} after {elapsed:.1f}s"
                )
                logger.error(f"  Command: {cmd_str}")

                # Log stderr
                if result.stderr:
                    logger.error("  STDERR (last 1000 chars):")
                    for line in result.stderr[-1000:].split("\n"):
                        if line.strip():
                            logger.error(f"    {line}")
                else:
                    logger.error("  No stderr output")

                # Log stdout (might contain useful error info)
                if result.stdout:
                    logger.error("  STDOUT (last 1000 chars):")
                    for line in result.stdout[-1000:].split("\n"):
                        if line.strip():
                            logger.error(f"    {line}")
                else:
                    logger.error("  No stdout output")

                return False

        except subprocess.TimeoutExpired as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"  ✗ {script_name} timed out after {elapsed:.1f}s (limit: {self.timeout}s)"
            )
            logger.error(f"  Command: {cmd_str}")

            # Log partial output if available
            if hasattr(e, "stderr") and e.stderr:
                logger.error("  Partial STDERR before timeout:")
                for line in str(e.stderr)[-1000:].split("\n"):
                    if line.strip():
                        logger.error(f"    {line}")

            if hasattr(e, "stdout") and e.stdout:
                logger.error("  Partial STDOUT before timeout:")
                for line in str(e.stdout)[-1000:].split("\n"):
                    if line.strip():
                        logger.error(f"    {line}")

            return False

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"  ✗ {script_name} failed with exception after {elapsed:.1f}s: {type(e).__name__}"
            )
            logger.error(f"  Command: {cmd_str}")
            logger.error(f"  Exception details: {str(e)}", exc_info=True)
            return False

    def process_subject_session(
        self, subject_id: str, session_id: str, skip_existing: bool = False
    ) -> bool:
        """
        Process one subject/session through the complete pipeline.

        Args:
            subject_id: Subject ID
            session_id: Session ID
            skip_existing: If True, skip modalities that are already processed

        Returns:
            True if all steps successful, False otherwise
        """
        logger.info(f"Processing sub-{subject_id}/ses-{session_id}")
        logger.debug(f"  Skip existing: {skip_existing}")

        # Check what's already processed
        processed = self.check_already_processed(subject_id, session_id)
        logger.debug(
            f"  Already processed: BVP={processed['bvp']}, EDA={processed['eda']}, HR={processed['hr']}, TEMP={processed['temp']}"
        )

        if skip_existing and all(processed.values()):
            logger.info("  Skipping: Already fully processed")
            self.stats["skipped"] += 1
            return True

        success = True

        # Step 1: BVP preprocessing
        if skip_existing and processed["bvp"]:
            logger.info("  BVP: Already processed, skipping")
        else:
            logger.info("  BVP: Starting preprocessing...")
            if not self.run_preprocessing_script(
                "preprocess_bvp.py", subject_id, session_id
            ):
                success = False
                error_msg = (
                    f"sub-{subject_id}/ses-{session_id}: BVP preprocessing failed"
                )
                self.stats["errors"].append(error_msg)
                logger.error(f"  {error_msg}")

        # Step 2: EDA preprocessing (only if BVP succeeded or was already done)
        if success or processed["bvp"]:
            if skip_existing and processed["eda"]:
                logger.info("  EDA: Already processed, skipping")
            else:
                logger.info("  EDA: Starting preprocessing...")
                if not self.run_preprocessing_script(
                    "preprocess_eda.py", subject_id, session_id
                ):
                    success = False
                    error_msg = (
                        f"sub-{subject_id}/ses-{session_id}: EDA preprocessing failed"
                    )
                    self.stats["errors"].append(error_msg)
                    logger.error(f"  {error_msg}")
        else:
            logger.warning("  EDA: Skipped due to BVP failure")

        # Step 3: HR preprocessing (only if previous steps succeeded or were already done)
        if success or (processed["bvp"] and processed["eda"]):
            if skip_existing and processed["hr"]:
                logger.info("  HR: Already processed, skipping")
            else:
                logger.info("  HR: Starting preprocessing...")
                if not self.run_preprocessing_script(
                    "preprocess_hr.py", subject_id, session_id
                ):
                    success = False
                    error_msg = (
                        f"sub-{subject_id}/ses-{session_id}: HR preprocessing failed"
                    )
                    self.stats["errors"].append(error_msg)
                    logger.error(f"  {error_msg}")
        else:
            logger.warning("  HR: Skipped due to previous failures")

        # Step 4: TEMP preprocessing (independent of other modalities, can always run)
        if skip_existing and processed["temp"]:
            logger.info("  TEMP: Already processed, skipping")
        else:
            logger.info("  TEMP: Starting preprocessing...")
            if not self.run_preprocessing_script(
                "preprocess_temp.py", subject_id, session_id
            ):
                # TEMP failure is not critical - log warning but don't fail the session
                logger.warning(
                    f"  ⚠ TEMP preprocessing failed for sub-{subject_id}/ses-{session_id}"
                )
                self.stats["errors"].append(
                    f"sub-{subject_id}/ses-{session_id}: TEMP preprocessing failed (non-critical)"
                )

        if success:
            logger.info(
                f"  ✓ All preprocessing steps completed for sub-{subject_id}/ses-{session_id}"
            )

            # Validate duration consistency between modalities
            if self.validate_durations and not self.dry_run:
                is_valid, message = self.validate_modality_durations(
                    subject_id, session_id
                )
                if not is_valid:
                    logger.warning(f"  ⚠ Duration validation warning: {message}")
                    self.stats["duration_warnings"].append(
                        f"sub-{subject_id}/ses-{session_id}: {message}"
                    )
                else:
                    logger.debug(f"  Duration validation: {message}")
        else:
            logger.error(
                f"  ✗ Some preprocessing steps failed for sub-{subject_id}/ses-{session_id}"
            )

        return success

    def run_batch(self, subjects_filter: List[str] = None, skip_existing: bool = False):
        """
        Run batch preprocessing on all or filtered subjects.

        Args:
            subjects_filter: If provided, only process these subjects
            skip_existing: If True, skip subjects that are already processed
        """
        logger.info("=" * 80)
        logger.info("BATCH PREPROCESSING STARTED")
        logger.info("=" * 80)
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Skip existing: {skip_existing}")
        logger.info(f"Config: {self.config.get('study', {}).get('name', 'unknown')}")
        logger.info(f"Raw data path: {self.rawdata_path}")
        logger.info(f"Output path: {self.preprocessing_path}")

        # Find all subjects/sessions
        logger.info("Scanning for subjects/sessions...")
        subjects_sessions = self.find_subjects_sessions()

        if not subjects_sessions:
            logger.error("No subjects/sessions found!")
            logger.error(f"  Checked path: {self.rawdata_path}")
            logger.error(f"  Path exists: {self.rawdata_path.exists()}")
            return

        # Filter if requested
        if subjects_filter:
            logger.info(f"Filtering for subjects: {subjects_filter}")
            subjects_sessions = [
                (s, ses) for s, ses in subjects_sessions if s in subjects_filter
            ]
            logger.info(f"  After filtering: {len(subjects_sessions)} sessions")

        self.stats["total_subjects"] = len(set(s for s, _ in subjects_sessions))
        self.stats["total_sessions"] = len(subjects_sessions)

        logger.info(
            f"Found {self.stats['total_subjects']} subjects, {self.stats['total_sessions']} sessions"
        )
        logger.info("")

        # Process each subject/session
        for i, (subject_id, session_id) in enumerate(subjects_sessions, 1):
            logger.info(f"[{i}/{len(subjects_sessions)}] " + "=" * 60)

            try:
                success = self.process_subject_session(
                    subject_id, session_id, skip_existing
                )

                if success:
                    self.stats["successful"] += 1
                else:
                    self.stats["failed"] += 1

            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                logger.info("Partial results:")
                self.print_summary()
                raise

            except Exception as e:
                logger.error(
                    f"Unexpected error processing sub-{subject_id}/ses-{session_id}:"
                )
                logger.error(f"  Exception type: {type(e).__name__}")
                logger.error(f"  Exception message: {str(e)}", exc_info=True)
                self.stats["failed"] += 1
                self.stats["errors"].append(
                    f"sub-{subject_id}/ses-{session_id}: {type(e).__name__}: {str(e)}"
                )

            logger.info("")

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final processing summary."""
        logger.info("=" * 80)
        logger.info("BATCH PREPROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total subjects:  {self.stats['total_subjects']}")
        logger.info(f"Total sessions:  {self.stats['total_sessions']}")
        logger.info(f"Successful:      {self.stats['successful']}")
        logger.info(f"Failed:          {self.stats['failed']}")
        logger.info(f"Skipped:         {self.stats['skipped']}")

        if self.stats["duration_warnings"]:
            logger.info("")
            logger.info(f"DURATION WARNINGS ({len(self.stats['duration_warnings'])}):")
            for warning in self.stats["duration_warnings"]:
                logger.warning(f"  - {warning}")

        if self.stats["errors"]:
            logger.info("")
            logger.info("ERRORS:")
            for error in self.stats["errors"]:
                logger.error(f"  - {error}")

        logger.info("=" * 80)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_preprocessing_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    logger.info(f"Logging to: {log_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch preprocessing for all subjects/sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python scripts/batch/run_all_preprocessing.py
  
  # Dry run to see what would be processed
  python scripts/batch/run_all_preprocessing.py --dry-run
  
  # Process specific subjects only
  python scripts/batch/run_all_preprocessing.py --subjects g01p01 g01p02
  
  # Skip already processed subjects
  python scripts/batch/run_all_preprocessing.py --skip-existing
        """,
    )

    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        help="Process only these subjects (e.g., g01p01 g02p01)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects/sessions that are already processed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Parse config path
    config_path = Path(args.config) if args.config else None

    # Run batch processing
    try:
        processor = BatchPreprocessor(config_path, args.dry_run)
        processor.run_batch(args.subjects, args.skip_existing)

        # Exit with error code if any processing failed
        sys.exit(0 if processor.stats["failed"] == 0 else 1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
