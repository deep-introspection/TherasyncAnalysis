#!/usr/bin/env python3
"""
EDA Preprocessing Pipeline Script for TherasyncPipeline.

This script orchestrates the complete EDA preprocessing pipeline:
Load → Clean → Extract Metrics → Save in BIDS format

Usage:
    python scripts/preprocess_eda.py --subject sub-g01p01 --session ses-01
    python scripts/preprocess_eda.py --config config/config.yaml --batch
    python scripts/preprocess_eda.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict


# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.config_loader import ConfigLoader
from src.physio.preprocessing.eda_loader import EDALoader
from src.physio.preprocessing.eda_cleaner import EDACleaner
from src.physio.preprocessing.eda_metrics import EDAMetricsExtractor
from src.physio.preprocessing.eda_bids_writer import EDABIDSWriter


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
            logging.FileHandler(log_dir / "eda_preprocessing.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def process_single_subject(
    subject_id: str,
    session_id: str,
    config_path: Optional[Path] = None,
    moments: Optional[List[str]] = None,
    verbose: bool = True,
) -> bool:
    """
    Process EDA data for a single subject and session.

    Args:
        subject_id: Subject identifier (e.g., 'sub-g01p01')
        session_id: Session identifier (e.g., 'ses-01')
        config_path: Path to configuration file
        moments: List of moments to process (if None, processes all)
        verbose: Enable verbose logging

    Returns:
        True if processing succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting EDA processing for {subject_id}/{session_id}")

        # Initialize pipeline components
        loader = EDALoader(config_path)
        cleaner = EDACleaner(config_path)
        metrics_extractor = EDAMetricsExtractor(config_path)
        bids_writer = EDABIDSWriter(config_path)

        # Determine which moments to process
        if moments is None:
            # Try to detect available moments
            available_moments = []
            for moment in ["restingstate", "therapy"]:
                try:
                    data, _ = loader.load_subject_session(
                        subject_id, session_id, moment=moment
                    )
                    if data is not None and not data.empty:
                        available_moments.append(moment)
                except Exception:
                    continue

            if not available_moments:
                logger.error(f"No EDA data found for {subject_id}/{session_id}")
                return False

            moments = available_moments
            logger.info(f"Auto-detected moments: {moments}")

        # Step 1: Load and process EDA data for each moment
        logger.info(f"Step 1: Loading EDA data for {len(moments)} moments...")
        processed_results = {}
        moments_data = {}

        for moment in moments:
            try:
                # Load data
                data, metadata = loader.load_subject_session(
                    subject_id, session_id, moment=moment
                )

                if data is None or data.empty:
                    logger.warning(f"No data for {moment}, skipping")
                    continue

                logger.info(
                    f"Loaded {moment}: {len(data)} samples, {len(data) / 4:.1f}s"
                )

                # Step 2: Clean and process EDA signal
                logger.info(f"Step 2: Processing {moment} signal...")
                processed = cleaner.clean_signal(data, moment=moment)

                if processed is None or processed.empty:
                    logger.warning(f"Failed to process {moment}, skipping")
                    continue

                # Count SCR peaks
                num_scr_peaks = (
                    int(processed["SCR_Peaks"].sum())
                    if "SCR_Peaks" in processed.columns
                    else 0
                )
                logger.info(f"Processed {moment}: {num_scr_peaks} SCR peaks detected")

                # Calculate EDA_Quality using the cleaner method
                if "EDA_Quality" not in processed.columns:
                    processed = cleaner.calculate_quality(processed)
                    logger.debug(
                        f"Added EDA_Quality column (mean: {processed['EDA_Quality'].mean():.3f})"
                    )

                processed_results[moment] = processed
                moments_data[moment] = processed

            except Exception as e:
                logger.error(f"Failed to process {moment}: {e}")
                if verbose:
                    logger.exception("Detailed error:")
                continue

        if not processed_results:
            logger.error("No moments were successfully processed")
            return False

        logger.info(f"Successfully processed {len(processed_results)} moments")

        # Step 3: Extract EDA metrics
        logger.info("Step 3: Extracting EDA metrics...")
        try:
            session_metrics = metrics_extractor.extract_multiple_moments(moments_data)

            if session_metrics.empty:
                logger.error("No metrics were successfully extracted")
                return False

            # Log extracted metrics summary
            for _, row in session_metrics.iterrows():
                moment = row["moment"]
                scr_peaks = row.get("SCR_Peaks_N", 0)
                scr_rate = row.get("SCR_Peaks_Rate", 0)
                logger.info(
                    f"Extracted metrics for {moment}: {scr_peaks} SCRs ({scr_rate:.2f}/min)"
                )

        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
            if verbose:
                logger.exception("Detailed error:")
            return False

        # Step 4: Save results in BIDS format
        logger.info("Step 4: Saving results in BIDS format...")
        try:
            processing_metadata = {
                "script_version": "1.0.0",
                "processing_arguments": {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "moments": moments,
                    "config_path": str(config_path) if config_path else None,
                },
            }

            created_files = bids_writer.save_processed_data(
                subject_id,
                session_id,
                processed_results,
                session_metrics,
                processing_metadata,
            )

            total_files = sum(len(files) for files in created_files.values())
            logger.info(f"Created {total_files} BIDS-compliant output files")

            # Log file counts by category
            for category, files in created_files.items():
                if files:
                    logger.info(f"  {category}: {len(files)} files")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            if verbose:
                logger.exception("Detailed error:")
            return False

        logger.info(
            f"Successfully completed EDA processing for {subject_id}/{session_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False


def process_batch(
    config_path: Optional[Path] = None,
    subject_pattern: str = "sub-*",
    moments: Optional[List[str]] = None,
    continue_on_error: bool = True,
) -> Dict[str, bool]:
    """
    Process EDA data for multiple subjects in batch mode.

    Args:
        config_path: Path to configuration file
        subject_pattern: Pattern to match subject directories
        moments: List of moments to process (if None, processes all)
        continue_on_error: Continue processing other subjects if one fails

    Returns:
        Dictionary mapping subject/session to success status
    """
    logger = logging.getLogger(__name__)

    # Load configuration to get data paths
    config = ConfigLoader(config_path)
    data_path = Path(config.get("paths.raw", "data/raw"))

    # Find all matching subjects
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return {}

    # Scan for subjects matching pattern

    available_data = {}
    for subject_dir in sorted(data_path.glob(subject_pattern)):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        # Find sessions for this subject
        for session_dir in sorted(subject_dir.glob("ses-*")):
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name

            # Check if physio directory exists
            physio_dir = session_dir / "physio"
            if physio_dir.exists() and any(
                physio_dir.glob("*_recording-eda_physio.tsv")
            ):
                if subject_id not in available_data:
                    available_data[subject_id] = []
                available_data[subject_id].append(session_id)

    if not available_data:
        logger.error(f"No EDA data found matching pattern: {subject_pattern}")
        return {}

    results = {}
    total_sessions = sum(len(sessions) for sessions in available_data.values())
    processed = 0
    successful = 0

    logger.info(
        f"Starting batch processing for {len(available_data)} subjects, {total_sessions} sessions"
    )

    for subject_id, sessions in available_data.items():
        for session_id in sessions:
            processed += 1
            key = f"{subject_id}/{session_id}"

            logger.info(f"Processing {key} ({processed}/{total_sessions})")

            try:
                success = process_single_subject(
                    subject_id, session_id, config_path, moments, verbose=False
                )
                results[key] = success
                if success:
                    successful += 1

            except Exception as e:
                logger.error(f"Failed to process {key}: {e}")
                results[key] = False
                if not continue_on_error:
                    logger.error("Stopping batch processing due to error")
                    break

    logger.info(f"Batch processing completed: {successful}/{processed} successful")
    return results


def main():
    """Main entry point for the EDA preprocessing script."""
    parser = argparse.ArgumentParser(
        description="EDA Preprocessing Pipeline for TherasyncPipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject/session
  python scripts/preprocess_eda.py --subject sub-g01p01 --session ses-01
  
  # Process with custom config
  python scripts/preprocess_eda.py --subject sub-g01p01 --session ses-01 --config custom_config.yaml
  
  # Process specific moments only
  python scripts/preprocess_eda.py --subject sub-g01p01 --session ses-01 --moments restingstate therapy
  
  # Batch process all subjects
  python scripts/preprocess_eda.py --batch
  
  # Batch process with pattern matching
  python scripts/preprocess_eda.py --batch --subject-pattern "sub-g01*"
        """,
    )

    # Main arguments
    parser.add_argument(
        "--subject", "-s", type=str, help="Subject identifier (e.g., sub-g01p01)"
    )

    parser.add_argument(
        "--session", "-e", type=str, help="Session identifier (e.g., ses-01)"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to configuration file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--moments",
        "-m",
        nargs="+",
        help="Specific moments to process (default: all available)",
    )

    # Batch processing options
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all available subjects in batch mode",
    )

    parser.add_argument(
        "--subject-pattern",
        type=str,
        default="sub-*",
        help="Pattern for batch subject selection (default: sub-*)",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue batch processing if individual subjects fail",
    )

    # Logging options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (log to file only)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if not args.batch and (not args.subject or not args.session):
        parser.error("Either --batch or both --subject and --session must be specified")

    # Setup logging
    setup_logging(args.config)
    logger = logging.getLogger(__name__)

    if args.quiet:
        logging.getLogger().handlers = [
            h
            for h in logging.getLogger().handlers
            if not isinstance(h, logging.StreamHandler)
        ]

    # Execute processing
    try:
        if args.batch:
            logger.info("Starting batch EDA processing")
            results = process_batch(
                config_path=args.config,
                subject_pattern=args.subject_pattern,
                moments=args.moments,
                continue_on_error=args.continue_on_error,
            )

            # Print summary
            successful = sum(results.values())
            total = len(results)
            print(f"\nBatch processing completed: {successful}/{total} successful")

            # Print failed subjects
            failed = [key for key, success in results.items() if not success]
            if failed:
                print(f"Failed subjects: {failed}")

            return 0 if successful == total else 1

        else:
            # Ensure subject and session have BIDS prefixes
            subject_id = (
                args.subject
                if args.subject.startswith("sub-")
                else f"sub-{args.subject}"
            )
            session_id = (
                args.session
                if args.session.startswith("ses-")
                else f"ses-{args.session}"
            )

            logger.info(
                f"Starting single subject EDA processing: {subject_id}/{session_id}"
            )
            success = process_single_subject(
                subject_id,
                session_id,
                config_path=args.config,
                moments=args.moments,
                verbose=args.verbose,
            )

            if success:
                print(f"Successfully processed {subject_id}/{session_id}")
                return 0
            else:
                print(f"Failed to process {subject_id}/{session_id}")
                return 1

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
