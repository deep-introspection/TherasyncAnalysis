#!/usr/bin/env python
"""
Batch Visualization Script for TherasyncPipeline.

This script generates all 6 core visualizations for all preprocessed subjects/sessions.

Usage:
    # Generate visualizations for all preprocessed subjects
    python scripts/batch/run_all_visualizations.py

    # Dry run to see what would be processed
    python scripts/batch/run_all_visualizations.py --dry-run

    # Generate for specific subjects only
    python scripts/batch/run_all_visualizations.py --subjects g01p01 g01p02

    # Generate specific visualizations only
    python scripts/batch/run_all_visualizations.py --plots 1 2 3

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import ConfigLoader
from src.visualization.data_loader import VisualizationDataLoader
from src.visualization.plotters.signal_plots import (
    plot_multisignal_dashboard,
    plot_hr_dynamics_timeline,
)
from src.visualization.plotters.hrv_plots import (
    plot_poincare_hrv,
    plot_autonomic_balance,
)
from src.visualization.plotters.eda_plots import (
    plot_eda_arousal_profile,
    plot_scr_distribution,
)
from src.visualization.config import OUTPUT_CONFIG

logger = logging.getLogger(__name__)


# Mapping of plot numbers to (filename, function) pairs
PLOT_FUNCTIONS = {
    1: ("01_dashboard_multisignals.png", plot_multisignal_dashboard),
    2: ("02_poincare_hrv.png", plot_poincare_hrv),
    3: ("03_autonomic_balance.png", plot_autonomic_balance),
    4: ("04_eda_arousal_profile.png", plot_eda_arousal_profile),
    5: ("05_scr_distribution.png", plot_scr_distribution),
    6: ("06_hr_dynamics_timeline.png", plot_hr_dynamics_timeline),
}


class BatchVisualizer:
    """Handles batch visualization generation for multiple subjects/sessions."""

    def __init__(self, config_path: Path = None, dry_run: bool = False):
        """
        Initialize batch visualizer.

        Args:
            config_path: Path to config YAML file
            dry_run: If True, only show what would be processed without executing
        """
        self.config = ConfigLoader(config_path).config
        self.dry_run = dry_run
        self.data_loader = VisualizationDataLoader(config_path=config_path)

        # Output path
        self.output_base = Path(OUTPUT_CONFIG["base_path"])

        # Statistics
        self.stats = {
            "total_subjects": 0,
            "total_sessions": 0,
            "total_visualizations": 0,
            "successful_subjects": 0,
            "successful_visualizations": 0,
            "failed_subjects": 0,
            "failed_visualizations": 0,
            "errors": [],
        }

    def find_preprocessed_subjects(self) -> List[tuple]:
        """
        Find all preprocessed subject/session combinations.

        Returns:
            List of (subject_id, session_id) tuples
        """
        return self.data_loader.list_available_subjects()

    def generate_visualizations_for_subject(
        self, subject: str, session: str, plots: Optional[List[int]] = None
    ) -> Dict[str, int]:
        """
        Generate visualizations for one subject/session.

        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., '01')
            plots: List of plot numbers to generate (None = all)

        Returns:
            Dictionary with 'success' and 'failed' counts
        """
        logger.info(f"Processing sub-{subject}/ses-{session}")
        logger.debug(f"  Plots to generate: {plots if plots else 'all (1-6)'}")

        result = {"success": 0, "failed": 0}

        if self.dry_run:
            logger.info("  [DRY RUN] Would generate visualizations")
            plots_to_generate = plots if plots else list(PLOT_FUNCTIONS.keys())
            result["success"] = len(plots_to_generate)
            return result

        start_time = datetime.now()

        try:
            # Load data
            logger.debug("  Loading data from derivatives...")
            data = self.data_loader.load_subject_session(subject, session)

            if not data:
                logger.error(f"  ✗ No data loaded for sub-{subject}/ses-{session}")
                logger.error("    Check that preprocessing outputs exist")
                plots_count = len(plots) if plots else len(PLOT_FUNCTIONS)
                result["failed"] = plots_count
                self.stats["errors"].append(
                    f"sub-{subject}/ses-{session}: No data loaded"
                )
                return result

            # Log data availability
            logger.debug("  Data loaded successfully:")
            for modality in ["bvp", "eda", "hr"]:
                if modality in data:
                    logger.debug(f"    {modality.upper()}: ✓")
                    if "signals" in data[modality]:
                        logger.debug(
                            f"      signals: {list(data[modality]['signals'].keys())}"
                        )
                    if (
                        "metrics" in data[modality]
                        and data[modality]["metrics"] is not None
                    ):
                        logger.debug(
                            f"      metrics: {len(data[modality]['metrics'])} rows"
                        )
                else:
                    logger.debug(f"    {modality.upper()}: ✗")

            # Setup output directory
            subject_id = data["subject_id"]
            session_id = data["session_id"]

            figures_dir = (
                self.output_base
                / subject_id
                / session_id
                / OUTPUT_CONFIG["figures_subdir"]
            )
            figures_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Output: {figures_dir}")

            # Determine which plots to generate
            plots_to_generate = plots if plots else list(PLOT_FUNCTIONS.keys())
            logger.debug(f"  Generating {len(plots_to_generate)} visualizations")

            # Generate each visualization
            for plot_num in plots_to_generate:
                if plot_num not in PLOT_FUNCTIONS:
                    logger.warning(f"  Plot #{plot_num} not implemented, skipping")
                    continue

                filename, plot_func = PLOT_FUNCTIONS[plot_num]
                output_path = figures_dir / filename

                logger.info(f"  Generating #{plot_num}: {filename}")

                plot_start = datetime.now()

                try:
                    plot_func(data, output_path=output_path, show=False)
                    plot_elapsed = (datetime.now() - plot_start).total_seconds()
                    logger.info(f"    ✓ Saved in {plot_elapsed:.1f}s")
                    result["success"] += 1

                except Exception as e:
                    plot_elapsed = (datetime.now() - plot_start).total_seconds()
                    logger.error(f"    ✗ Failed after {plot_elapsed:.1f}s")
                    logger.error(f"    Exception type: {type(e).__name__}")
                    logger.error(f"    Exception message: {str(e)}")
                    logger.debug("    Full traceback:", exc_info=True)
                    result["failed"] += 1
                    self.stats["errors"].append(
                        f"sub-{subject}/ses-{session} plot #{plot_num} ({filename}): {type(e).__name__}: {str(e)}"
                    )

            total_elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"  Completed in {total_elapsed:.1f}s ({result['success']} success, {result['failed']} failed)"
            )

            return result

        except Exception as e:
            total_elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"  ✗ Error after {total_elapsed:.1f}s")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception message: {str(e)}", exc_info=True)
            plots_count = len(plots) if plots else len(PLOT_FUNCTIONS)
            result["failed"] = plots_count
            self.stats["errors"].append(
                f"sub-{subject}/ses-{session}: {type(e).__name__}: {str(e)}"
            )
            return result

    def run_batch(
        self, subjects_filter: List[str] = None, plots: Optional[List[int]] = None
    ):
        """
        Run batch visualization on all or filtered subjects.

        Args:
            subjects_filter: If provided, only process these subjects
            plots: List of plot numbers to generate (None = all)
        """
        logger.info("=" * 80)
        logger.info("BATCH VISUALIZATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Config: {self.config.get('study', {}).get('name', 'unknown')}")
        logger.info(f"Output base: {self.output_base}")

        # Find all preprocessed subjects/sessions
        logger.info("Scanning for preprocessed subjects/sessions...")
        subjects_sessions = self.find_preprocessed_subjects()

        if not subjects_sessions:
            logger.warning("No preprocessed subjects found")
            logger.warning(f"  Checked path: {self.data_loader.derivatives_path}")
            logger.warning(
                f"  Path exists: {self.data_loader.derivatives_path.exists()}"
            )
            logger.warning("  Make sure to run preprocessing first!")
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

        plots_to_generate = plots if plots else list(PLOT_FUNCTIONS.keys())
        self.stats["total_visualizations"] = len(subjects_sessions) * len(
            plots_to_generate
        )

        logger.info(
            f"Found {self.stats['total_subjects']} subjects, {self.stats['total_sessions']} sessions"
        )
        logger.info(
            f"Generating {len(plots_to_generate)} visualizations per session: {plots_to_generate}"
        )
        logger.info(
            f"Total visualizations to generate: {self.stats['total_visualizations']}"
        )
        logger.info("")

        # Process each subject/session
        for i, (subject, session) in enumerate(subjects_sessions, 1):
            logger.info(f"[{i}/{len(subjects_sessions)}] " + "=" * 60)

            try:
                result = self.generate_visualizations_for_subject(
                    subject, session, plots
                )

                if result["failed"] == 0:
                    self.stats["successful_subjects"] += 1
                else:
                    self.stats["failed_subjects"] += 1

                self.stats["successful_visualizations"] += result["success"]
                self.stats["failed_visualizations"] += result["failed"]

            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                logger.info("Partial results:")
                self.print_summary()
                raise

            except Exception as e:
                logger.error(
                    f"Unexpected error processing sub-{subject}/ses-{session}:"
                )
                logger.error(f"  Exception type: {type(e).__name__}")
                logger.error(f"  Exception message: {str(e)}", exc_info=True)
                self.stats["failed_subjects"] += 1
                plots_count = len(plots_to_generate)
                self.stats["failed_visualizations"] += plots_count
                self.stats["errors"].append(
                    f"sub-{subject}/ses-{session}: {type(e).__name__}: {str(e)}"
                )

            logger.info("")

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final processing summary."""
        logger.info("=" * 80)
        logger.info("BATCH VISUALIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total subjects:             {self.stats['total_subjects']}")
        logger.info(f"Total sessions:             {self.stats['total_sessions']}")
        logger.info(f"Total visualizations:       {self.stats['total_visualizations']}")
        logger.info(f"Successful subjects:        {self.stats['successful_subjects']}")
        logger.info(f"Failed subjects:            {self.stats['failed_subjects']}")
        logger.info(
            f"Successful visualizations:  {self.stats['successful_visualizations']}"
        )
        logger.info(
            f"Failed visualizations:      {self.stats['failed_visualizations']}"
        )

        if self.stats["errors"]:
            logger.info("")
            logger.info("ERRORS:")
            for error in self.stats["errors"][:20]:  # Show first 20 errors
                logger.error(f"  - {error}")
            if len(self.stats["errors"]) > 20:
                logger.error(f"  ... and {len(self.stats['errors']) - 20} more errors")

        logger.info("=" * 80)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_visualization_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    logger.info(f"Logging to: {log_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch visualization generation for all preprocessed subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations for all subjects
  python scripts/batch/run_all_visualizations.py
  
  # Dry run to see what would be generated
  python scripts/batch/run_all_visualizations.py --dry-run
  
  # Generate for specific subjects only
  python scripts/batch/run_all_visualizations.py --subjects g01p01 g01p02
  
  # Generate specific visualizations only
  python scripts/batch/run_all_visualizations.py --plots 1 2 3
        """,
    )

    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        help="Process only these subjects (e.g., g01p01 g02p01)",
    )
    parser.add_argument(
        "--plots",
        type=int,
        nargs="+",
        help="Generate only these plot numbers (e.g., 1 2 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without executing",
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

    # Run batch visualization
    try:
        visualizer = BatchVisualizer(config_path, args.dry_run)
        visualizer.run_batch(args.subjects, args.plots)

        # Exit with error code if any visualization failed
        sys.exit(0 if visualizer.stats["failed_visualizations"] == 0 else 1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
