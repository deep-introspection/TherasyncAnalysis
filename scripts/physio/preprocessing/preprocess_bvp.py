#!/usr/bin/env python3
"""
BVP Preprocessing Pipeline Script for TherasyncPipeline.

This script orchestrates the complete BVP preprocessing pipeline:
Load → Clean → Extract Metrics → Save in BIDS format

Usage:
    python scripts/preprocess_bvp.py --subject sub-g01p01 --session ses-01
    python scripts/preprocess_bvp.py --config config/config.yaml --batch
    python scripts/preprocess_bvp.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.config_loader import ConfigLoader
from src.physio.preprocessing.bvp_loader import BVPLoader
from src.physio.preprocessing.bvp_cleaner import BVPCleaner
from src.physio.preprocessing.bvp_metrics import BVPMetricsExtractor
from src.physio.preprocessing.bvp_bids_writer import BVPBIDSWriter


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
            logging.FileHandler(log_dir / "bvp_preprocessing.log"),
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
    Process BVP data for a single subject and session.

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
        logger.info(f"Starting BVP processing for {subject_id}/{session_id}")

        # Initialize pipeline components
        loader = BVPLoader(config_path)
        cleaner = BVPCleaner(config_path)
        metrics_extractor = BVPMetricsExtractor(config_path)
        bids_writer = BVPBIDSWriter(config_path)

        # Step 1: Load BVP data
        logger.info("Step 1: Loading BVP data...")
        try:
            loaded_data = loader.load_subject_session_data(
                subject_id, session_id, moments
            )
            if not loaded_data:
                logger.error(f"No BVP data found for {subject_id}/{session_id}")
                return False

            logger.info(
                f"Loaded BVP data for {len(loaded_data)} moments: {list(loaded_data.keys())}"
            )

        except Exception as e:
            logger.error(f"Failed to load BVP data: {e}")
            return False

        # Step 2: Clean and process BVP signals
        logger.info("Step 2: Cleaning BVP signals...")
        try:
            # Type cast to satisfy type checker
            processed_results = cleaner.process_moment_signals(loaded_data)  # type: ignore
            if not processed_results:
                logger.error("No signals were successfully processed")
                return False

            logger.info(f"Successfully processed {len(processed_results)} moments")

        except Exception as e:
            logger.error(f"Failed to clean BVP signals: {e}")
            return False

        # Step 3: Extract HRV metrics
        logger.info("Step 3: Extracting HRV metrics...")
        try:
            session_metrics = metrics_extractor.extract_session_metrics(
                processed_results
            )
            if not session_metrics:
                logger.error("No metrics were successfully extracted")
                return False

            # Log extracted metrics summary
            for moment, metrics in session_metrics.items():
                valid_metrics = sum(1 for v in metrics.values() if not pd.isna(v))
                logger.info(
                    f"Extracted {valid_metrics}/{len(metrics)} valid metrics for {moment}"
                )

        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
            return False

        # Step 3.5: Extract RR intervals (if enabled in config)
        rr_intervals_data = {}
        rr_config = metrics_extractor.bvp_config.get("rr_intervals", {})
        if rr_config.get("enabled", False):
            logger.info("Step 3.5: Extracting RR intervals...")
            try:
                for moment, (
                    processed_signals,
                    processing_info,
                ) in processed_results.items():
                    peaks = processing_info.get("PPG_Peaks", [])
                    sampling_rate = processing_info.get("sampling_rate", 64)

                    if len(peaks) >= 2:
                        rr_df = metrics_extractor.extract_rr_intervals(
                            peaks, sampling_rate, moment
                        )
                        rr_intervals_data[moment] = rr_df
                        logger.info(
                            f"Extracted {len(rr_df)} RR intervals for {moment} "
                            f"({rr_df['is_valid'].sum()} valid)"
                        )
                    else:
                        logger.warning(
                            f"Insufficient peaks for RR intervals in {moment}"
                        )

            except Exception as e:
                logger.warning(f"Failed to extract RR intervals: {e}")
                # Don't fail the whole process if RR extraction fails
                rr_intervals_data = {}

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

            # Save RR intervals if extracted
            if rr_intervals_data:
                logger.info("Saving RR intervals...")
                rr_files_created = []
                for moment, rr_df in rr_intervals_data.items():
                    try:
                        # Get expected number of peaks for validation
                        expected_peaks = None
                        if moment in processed_results:
                            _, processing_info = processed_results[moment]
                            expected_peaks = len(processing_info.get("PPG_Peaks", []))

                        tsv_path, json_path = bids_writer.save_rr_intervals(
                            subject_id,
                            session_id,
                            moment,
                            rr_df,
                            expected_peaks=expected_peaks,
                        )
                        rr_files_created.extend([tsv_path, json_path])
                    except ValueError as e:
                        # Critical data quality issue - fail the process
                        logger.error(f"Data quality check failed for {moment}: {e}")
                        return False
                    except Exception as e:
                        logger.warning(f"Failed to save RR intervals for {moment}: {e}")

                if rr_files_created:
                    logger.info(f"Created {len(rr_files_created)} RR interval files")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

        logger.info(
            f"Successfully completed BVP processing for {subject_id}/{session_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        return False


def process_batch(
    config_path: Optional[Path] = None,
    subject_pattern: str = "sub-*",
    session_pattern: str = "ses-*",
    moments: Optional[List[str]] = None,
    continue_on_error: bool = True,
) -> Dict[str, bool]:
    """
    Process BVP data for multiple subjects in batch mode.

    Args:
        config_path: Path to configuration file
        subject_pattern: Pattern to match subject directories
        session_pattern: Pattern to match session directories
        moments: List of moments to process (if None, processes all)
        continue_on_error: Continue processing other subjects if one fails

    Returns:
        Dictionary mapping subject/session to success status
    """
    logger = logging.getLogger(__name__)

    # Initialize loader to scan for available data
    loader = BVPLoader(config_path)
    available_data = loader.get_available_data(subject_pattern)

    if not available_data:
        logger.error(f"No BVP data found matching pattern: {subject_pattern}")
        return {}

    results = {}
    total_subjects = len(available_data)
    processed = 0
    successful = 0

    logger.info(f"Starting batch processing for {total_subjects} subjects")

    for subject_id, sessions in available_data.items():
        for session_id, available_moments in sessions.items():
            processed += 1
            key = f"{subject_id}/{session_id}"

            # Filter moments if specified
            process_moments = moments if moments else available_moments
            process_moments = [m for m in process_moments if m in available_moments]

            if not process_moments:
                logger.warning(f"No matching moments for {key}, skipping")
                results[key] = False
                continue

            logger.info(f"Processing {key} ({processed}/{total_subjects})")

            try:
                success = process_single_subject(
                    subject_id, session_id, config_path, process_moments, verbose=False
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
    """Main entry point for the BVP preprocessing script."""
    parser = argparse.ArgumentParser(
        description="BVP Preprocessing Pipeline for TherasyncPipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject/session
  python scripts/preprocess_bvp.py --subject sub-g01p01 --session ses-01
  
  # Process with custom config
  python scripts/preprocess_bvp.py --subject sub-g01p01 --session ses-01 --config custom_config.yaml
  
  # Process specific moments only
  python scripts/preprocess_bvp.py --subject sub-g01p01 --session ses-01 --moments restingstate therapy
  
  # Batch process all subjects
  python scripts/preprocess_bvp.py --batch
  
  # Batch process with pattern matching
  python scripts/preprocess_bvp.py --batch --subject-pattern "sub-g01*"
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
        "--session-pattern",
        type=str,
        default="ses-*",
        help="Pattern for batch session selection (default: ses-*)",
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
            logger.info("Starting batch BVP processing")
            results = process_batch(
                config_path=args.config,
                subject_pattern=args.subject_pattern,
                session_pattern=args.session_pattern,
                moments=args.moments,
                continue_on_error=args.continue_on_error,
            )

            # Print summary
            successful = sum(results.values())
            total = len(results)
            print(f"\\nBatch processing completed: {successful}/{total} successful")

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
                f"Starting single subject BVP/HRV processing: {subject_id}/{session_id}"
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
