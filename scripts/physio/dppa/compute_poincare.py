#!/usr/bin/env python3
"""
Compute Poincaré plot centroids from epoched RR intervals.

This script processes epoched RR interval files and computes Poincaré plot
metrics (centroids, SD1, SD2) for each epoch. Results are saved in BIDS format.

Usage:
    # Single session
    python scripts/physio/dppa/compute_poincare.py --subject g01p01 --session 01

    # Batch process all sessions
    python scripts/physio/dppa/compute_poincare.py --batch

    # Dry run (show what would be processed)
    python scripts/physio/dppa/compute_poincare.py --batch --dry-run

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.physio.dppa import PoincareCalculator

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, verbose: bool = False):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"compute_poincare_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger.info(f"Logging to: {log_file}")


def find_all_sessions(preprocessing_dir: Path) -> List[Tuple[str, str]]:
    """
    Find all subject/session pairs in preprocessing data.

    Returns:
        List of (subject, session) tuples
    """
    sessions = []

    for sub_dir in sorted(preprocessing_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        subject = sub_dir.name.replace("sub-", "")

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue

            session = ses_dir.name

            # Check if BVP directory exists and has RR interval files
            bvp_dir = ses_dir / "bvp"
            if bvp_dir.exists():
                rr_files = list(bvp_dir.glob("*_desc-rrintervals_physio.tsv"))
                if rr_files:
                    sessions.append((subject, session))

    return sessions


def save_poincare_results(
    results: Dict, output_dir: Path, subject: str, session: str, dry_run: bool = False
) -> int:
    """
    Save Poincaré centroid results to BIDS-compliant files.

    Args:
        results: Dictionary {task: {method: DataFrame}}
        output_dir: Base output directory
        subject: Subject ID
        session: Session ID
        dry_run: If True, don't save files

    Returns:
        Number of files saved
    """
    if dry_run:
        return 0

    # Create output directory structure
    sub_ses_dir = output_dir / f"sub-{subject}" / session / "poincare"
    sub_ses_dir.mkdir(parents=True, exist_ok=True)

    files_saved = 0

    for task, methods in results.items():
        for method, df in methods.items():
            # Generate filename
            filename = f"sub-{subject}_{session}_task-{task}_method-{method}_desc-poincare_physio.tsv"
            output_file = sub_ses_dir / filename

            # Save DataFrame
            df.to_csv(output_file, sep="\t", index=False)

            # Create JSON sidecar
            json_file = output_file.with_suffix(".json")
            metadata = {
                "Description": "Poincaré plot centroids and metrics computed from RR intervals",
                "TaskName": task,
                "EpochingMethod": method,
                "Columns": {
                    "epoch_id": "Epoch identifier",
                    "centroid_x": "Mean of current RR intervals (RRₙ) in ms",
                    "centroid_y": "Mean of next RR intervals (RRₙ₊₁) in ms",
                    "sd1": "Short-term variability (SD1) in ms",
                    "sd2": "Long-term variability (SD2) in ms",
                    "sd_ratio": "Ratio SD1/SD2 (sympatho-vagal balance)",
                    "n_intervals": "Number of RR interval pairs in epoch",
                },
                "Formula": {
                    "centroid_x": "mean(RRₙ)",
                    "centroid_y": "mean(RRₙ₊₁)",
                    "SD1": "sqrt(var(RRₙ - RRₙ₊₁) / 2)",
                    "SD2": "sqrt(var(RRₙ + RRₙ₊₁) / 2)",
                },
                "CreationDate": datetime.now().isoformat(),
                "ValidEpochs": int(df["centroid_x"].notna().sum()),
                "TotalEpochs": len(df),
                "MeanIntervalsPerEpoch": float(df["n_intervals"].mean()),
            }

            with open(json_file, "w") as f:
                json.dump(metadata, f, indent=2)

            files_saved += 2  # TSV + JSON
            logger.info(
                f"  Saved: {filename} ({metadata['ValidEpochs']}/{metadata['TotalEpochs']} valid epochs)"
            )

    return files_saved


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute Poincaré plot centroids from RR intervals (with epoch columns)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single session
  python scripts/physio/dppa/compute_poincare.py --subject g01p01 --session ses-01
  
  # Batch process all sessions
  python scripts/physio/dppa/compute_poincare.py --batch
  
  # Dry run to see what would be processed
  python scripts/physio/dppa/compute_poincare.py --batch --dry-run
  
  # Process specific subjects only
  python scripts/physio/dppa/compute_poincare.py --batch --subjects g01p01 g01p02
        """,
    )

    parser.add_argument("--subject", type=str, help="Subject ID (e.g., g01p01)")
    parser.add_argument("--session", type=str, help="Session ID (e.g., ses-01)")
    parser.add_argument(
        "--batch", action="store_true", help="Process all subjects/sessions"
    )
    parser.add_argument("--subjects", nargs="+", help="Process only these subjects")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load config
    config = ConfigLoader(args.config)
    paths = config.get("paths", {})

    # Setup logging
    log_dir = Path(paths.get("logs", "log"))
    setup_logging(log_dir, args.verbose)

    logger.info("=" * 80)
    logger.info("COMPUTE POINCARÉ CENTROIDS")
    logger.info("=" * 80)
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Config: {args.config}")

    # Initialize calculator
    calculator = PoincareCalculator(args.config)

    # Output directory
    output_dir = Path(paths.get("dppa", "data/derivatives/dppa"))

    # Determine sessions to process
    if args.batch:
        # Get preprocessing directory
        paths = config.get("paths", {})
        derivatives_dir = Path(paths.get("derivatives", "data/derivatives"))
        preprocessing_subdir = config.get("output", {}).get(
            "preprocessing_dir", "preprocessing"
        )
        preprocessing_dir = derivatives_dir / preprocessing_subdir

        sessions = find_all_sessions(preprocessing_dir)

        # Filter by subjects if specified
        if args.subjects:
            sessions = [(s, ses) for s, ses in sessions if s in args.subjects]

        logger.info(f"Found {len(sessions)} sessions to process")
    elif args.subject and args.session:
        # Normalize session format
        session = (
            args.session if args.session.startswith("ses-") else f"ses-{args.session}"
        )
        sessions = [(args.subject, session)]
        logger.info(f"Processing single session: {args.subject}/{session}")
    else:
        parser.error("Must specify either --batch or both --subject and --session")
        return 1

    # Process sessions
    total_files = 0
    successful_sessions = 0
    failed_sessions = []

    for i, (subject, session) in enumerate(sessions, 1):
        logger.info(f"\n[{i}/{len(sessions)}] Processing sub-{subject}/{session}")

        if args.dry_run:
            logger.info("  DRY RUN - would process this session")
            continue

        try:
            # Compute Poincaré centroids for all tasks and methods
            results = calculator.compute_subject_session(subject, session)

            if not results:
                logger.warning(
                    f"  No RR interval files found for sub-{subject}/{session}"
                )
                continue

            # Save results
            files_saved = save_poincare_results(
                results, output_dir, subject, session, args.dry_run
            )

            total_files += files_saved
            successful_sessions += 1

        except Exception as e:
            logger.error(f"Failed to process sub-{subject}/{session}: {e}")
            if args.verbose:
                logger.exception("Detailed traceback:")
            failed_sessions.append(f"sub-{subject}/{session}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Sessions processed:  {successful_sessions}/{len(sessions)}")
    logger.info(f"Files created:       {total_files}")

    if failed_sessions:
        logger.warning(f"\nFailed sessions ({len(failed_sessions)}):")
        for session in failed_sessions:
            logger.warning(f"  - {session}")

    logger.info("=" * 80)

    return 0 if not failed_sessions else 1


if __name__ == "__main__":
    sys.exit(main())
