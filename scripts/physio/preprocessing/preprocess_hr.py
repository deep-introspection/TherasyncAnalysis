#!/usr/bin/env python
"""
HR Preprocessing CLI Script for TherasyncPipeline.

This script provides command-line interface for processing Heart Rate (HR) data
from Empatica devices. It integrates all HR pipeline components:
HR Loader → HR Cleaner → HR Metrics Extractor → HR BIDS Writer

Usage:
    # Process single subject/session
    python scripts/preprocess_hr.py --subject g01p01 --session 01

    # Process with specific moment
    python scripts/preprocess_hr.py --subject g01p01 --session 01 --moment therapy

    # Batch process multiple subjects
    python scripts/preprocess_hr.py --batch --config-file config/hr_batch.yaml

    # Process with custom config
    python scripts/preprocess_hr.py --subject g01p01 --session 01 --config config/custom_hr.yaml

Authors: Lena Adel, Remy Ramadour
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.config_loader import ConfigLoader
from src.physio.preprocessing.hr_loader import HRLoader
from src.physio.preprocessing.hr_cleaner import HRCleaner
from src.physio.preprocessing.hr_metrics import HRMetricsExtractor
from src.physio.preprocessing.hr_bids_writer import HRBIDSWriter


def setup_logging(config_path: Optional[Path] = None) -> None:
    """
    Setup logging configuration.

    Args:
        config_path: Path to configuration file
    """
    config = ConfigLoader(config_path)
    log_level = config.get("logging.level", "INFO")

    # Create logs directory if it doesn't exist
    log_dir = Path(config.get("paths.log", "log"))
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "hr_preprocessing.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger(__name__)


class HRPreprocessor:
    """
    Complete HR preprocessing pipeline.

    This class orchestrates the entire HR processing workflow:
    1. Load HR data from Empatica files
    2. Clean signals (outlier removal, interpolation)
    3. Extract comprehensive HR metrics (25 metrics)
    4. Write BIDS-compliant output (7 files)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize HR preprocessing pipeline.

        Args:
            config_path: Optional path to custom configuration file
        """
        # Load configuration
        self.config = ConfigLoader(config_path)

        # Initialize pipeline components with config_path (uniform pattern)
        self.hr_loader = HRLoader(config_path)
        self.hr_cleaner = HRCleaner(config_path)
        self.hr_metrics = HRMetricsExtractor(config_path)
        self.hr_writer = HRBIDSWriter(config_path)

        logger.info("HR Preprocessor initialized")

    def process_subject_session(
        self, subject: str, session: str, moment: Optional[str] = None
    ) -> bool:
        """
        Process HR data for a single subject/session.

        Args:
            subject: Subject identifier (e.g., 'g01p01')
            session: Session identifier (e.g., '01')
            moment: Optional specific moment to process (if None, processes all available moments)

        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing HR data for sub-{subject} ses-{session}")

        try:
            # Determine which moments to process
            if moment:
                moments_to_process = [moment]
            else:
                # Default: process both restingstate and therapy
                moments_to_process = ["restingstate", "therapy"]

            # Dictionary to store processed results for all moments
            processed_results = {}
            all_moments_metadata = {}

            # Process each moment
            for current_moment in moments_to_process:
                logger.info(f"Processing moment: {current_moment}")

                # Step 1: Load HR data for this moment
                logger.debug(f"Step 1: Loading HR data for {current_moment}...")
                load_result = self.hr_loader.load_subject_session(
                    subject, session, current_moment
                )

                if load_result is None:
                    logger.warning(
                        f"No HR data found for {current_moment}, skipping..."
                    )
                    continue

                # Handle both tuple return (data, metadata) and DataFrame return
                if isinstance(load_result, tuple):
                    hr_data, load_metadata = load_result
                else:
                    hr_data = load_result
                    load_metadata = {}  # noqa: F841

                if hr_data is None or len(hr_data) == 0:
                    logger.warning(f"Empty HR data for {current_moment}, skipping...")
                    continue

                logger.debug(f"Loaded {len(hr_data)} HR samples for {current_moment}")

                # Step 2: Clean HR signals
                logger.debug(f"Step 2: Cleaning HR signals for {current_moment}...")
                cleaned_data, cleaning_metadata = self.hr_cleaner.clean_signal(
                    hr_data, current_moment
                )

                # Validate cleaning quality
                is_valid, quality_message = self.hr_cleaner.validate_cleaning_quality(
                    cleaning_metadata
                )
                if not is_valid:
                    logger.warning(
                        f"HR cleaning quality issues for {current_moment}: {quality_message}"
                    )
                else:
                    logger.debug(
                        f"HR cleaning successful for {current_moment}: {quality_message}"
                    )

                # Verify all expected columns are present (now standardized in HRCleaner)
                expected_columns = [
                    "time",
                    "HR_Raw",
                    "HR_Clean",
                    "HR_Quality",
                    "HR_Outliers",
                    "HR_Interpolated",
                ]
                missing_columns = [
                    col for col in expected_columns if col not in cleaned_data.columns
                ]
                if missing_columns:
                    logger.error(f"Missing expected columns: {missing_columns}")
                    return False

                # Store processed data for this moment
                processed_results[current_moment] = cleaned_data
                all_moments_metadata[current_moment] = cleaning_metadata

                logger.info(
                    f"✓ Successfully processed moment: {current_moment} ({len(cleaned_data)} samples)"
                )

            # Check if we have any processed data
            if not processed_results:
                logger.error(
                    f"No moments could be processed for sub-{subject} ses-{session}"
                )
                return False

            # Step 4: Extract metrics for all moments
            logger.info(
                f"Step 4: Extracting HR metrics for {len(processed_results)} moment(s)..."
            )

            session_metrics = self.hr_metrics.extract_session_metrics(processed_results)

            for moment_name, metrics in session_metrics.items():
                valid_count = sum(
                    1
                    for v in metrics.values()
                    if not (isinstance(v, float) and np.isnan(v))
                )
                logger.info(f"Extracted {valid_count} metrics for {moment_name}")

            # Step 5: Write BIDS output
            logger.info(
                f"Step 5: Writing BIDS output for {len(processed_results)} moment(s)..."
            )
            output_files = self.hr_writer.save_processed_data(
                subject_id=f"sub-{subject}",  # Ensure prefix
                session_id=f"ses-{session}",  # Ensure prefix
                processed_results=processed_results,
                session_metrics=session_metrics,
                processing_metadata=all_moments_metadata,
            )

            # Log output files
            logger.info("HR processing complete! Files written:")
            for file_type, file_paths in output_files.items():
                if file_paths:
                    logger.info(f"  {file_type}: {len(file_paths)} file(s)")
                    for path in file_paths:
                        logger.debug(f"    - {path}")

            return True

        except Exception as e:
            logger.error(
                f"HR processing failed for sub-{subject} ses-{session}: {str(e)}"
            )
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def process_batch(self, subjects_sessions: List[tuple]) -> dict:
        """
        Process multiple subjects/sessions in batch.

        Args:
            subjects_sessions: List of (subject, session) tuples

        Returns:
            Dictionary with processing results
        """
        logger.info(
            f"Starting batch HR processing ({len(subjects_sessions)} subjects/sessions)"
        )

        results = {"successful": [], "failed": [], "total": len(subjects_sessions)}

        for subject, session in subjects_sessions:
            logger.info(f"Processing {subject} ses-{session}...")

            success = self.process_subject_session(subject, session)

            if success:
                results["successful"].append((subject, session))
                logger.info(f"✓ Successfully processed {subject} ses-{session}")
            else:
                results["failed"].append((subject, session))
                logger.error(f"✗ Failed to process {subject} ses-{session}")

        # Log batch summary
        success_rate = len(results["successful"]) / results["total"] * 100
        logger.info(
            f"Batch HR processing complete: "
            f"{len(results['successful'])}/{results['total']} successful "
            f"({success_rate:.1f}% success rate)"
        )

        if results["failed"]:
            logger.warning(f"Failed subjects/sessions: {results['failed']}")

        return results


def discover_subjects_sessions(data_dir: Path) -> List[tuple]:
    """
    Auto-discover available subjects and sessions from data directory.

    Args:
        data_dir: Path to data directory

    Returns:
        List of (subject, session) tuples
    """
    subjects_sessions = []

    # Look for subject directories in data/raw
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return subjects_sessions

    for subject_dir in data_dir.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue

        subject = subject_dir.name.replace("sub-", "")

        # Look for session directories
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith("ses-"):
                continue

            session = session_dir.name.replace("ses-", "")

            # Check if physio directory exists
            physio_dir = session_dir / "physio"
            if physio_dir.exists():
                # Check if HR files exist
                hr_files = list(physio_dir.glob("*recording-hr.tsv"))
                if hr_files:
                    subjects_sessions.append((subject, session))

    logger.info(f"Discovered {len(subjects_sessions)} subjects/sessions with HR data")
    return subjects_sessions


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HR Preprocessing Pipeline for TherasyncPipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject/session
  python scripts/preprocess_hr.py --subject g01p01 --session 01
  
  # Process specific moment
  python scripts/preprocess_hr.py --subject g01p01 --session 01 --moment therapy
  
  # Auto-discover and process all subjects
  python scripts/preprocess_hr.py --batch
  
  # Use custom configuration
  python scripts/preprocess_hr.py --subject g01p01 --session 01 --config config/custom.yaml
        """,
    )

    # Subject/session selection
    parser.add_argument("--subject", type=str, help="Subject identifier (e.g., g01p01)")
    parser.add_argument("--session", type=str, help="Session identifier (e.g., 01)")
    parser.add_argument(
        "--moment",
        type=str,
        help="Specific moment to process (e.g., therapy, restingstate)",
    )

    # Batch processing
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Auto-discover and process all available subjects/sessions",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        help="YAML file with list of subjects/sessions to process",
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    parser.add_argument(
        "--data-dir", type=str, help="Path to data directory (default: data/raw)"
    )

    # Processing options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )

    args = parser.parse_args()

    # Setup logging with proper log directory
    setup_logging(args.config)

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Initialize preprocessor
    try:
        preprocessor = HRPreprocessor(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize HR preprocessor: {str(e)}")
        sys.exit(1)

    # Determine processing mode
    if args.batch or args.batch_file:
        # Batch processing
        if args.batch_file:
            # Load subjects/sessions from file
            try:
                with open(args.batch_file, "r") as f:
                    batch_config = yaml.safe_load(f)
                subjects_sessions = [
                    (item["subject"], item["session"])
                    for item in batch_config.get("subjects_sessions", [])
                ]
            except Exception as e:
                logger.error(f"Failed to load batch file {args.batch_file}: {str(e)}")
                sys.exit(1)
        else:
            # Auto-discover subjects/sessions
            data_dir = Path(args.data_dir) if args.data_dir else Path("data/raw")
            subjects_sessions = discover_subjects_sessions(data_dir)

            if not subjects_sessions:
                logger.error("No subjects/sessions found for batch processing")
                sys.exit(1)

        # Process batch
        results = preprocessor.process_batch(subjects_sessions)

        # Exit with error code if any failures
        if results["failed"]:
            sys.exit(1)

    elif args.subject and args.session:
        # Single subject/session processing
        success = preprocessor.process_subject_session(
            args.subject, args.session, args.moment
        )

        if not success:
            sys.exit(1)

    else:
        # Invalid arguments
        logger.error("Must specify either --subject/--session or --batch/--batch-file")
        parser.print_help()
        sys.exit(1)

    logger.info("HR preprocessing completed successfully!")


if __name__ == "__main__":
    main()
