#!/usr/bin/env python3
"""
Generate ICD Statistical Visualizations.

Creates publication-ready visualizations comparing real vs pseudo dyad ICDs:
- Distribution comparison (violin + box + points)
- Epoch-by-epoch evolution with confidence bands
- ICD heatmap matrix

Usage:
    # Inter-session therapy ICD
    python scripts/physio/dppa/plot_icd_stats.py --mode inter --task therapy

    # Both tasks
    python scripts/physio/dppa/plot_icd_stats.py --mode inter --all-tasks

    # Intra-family
    python scripts/physio/dppa/plot_icd_stats.py --mode intra --task therapy

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.physio.dppa import ICDStatsPlotter

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, verbose: bool = False):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"plot_icd_stats_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def find_icd_file(mode: str, task: str, dppa_dir: Path) -> Path:
    """
    Find the ICD file for the given mode and task.

    Args:
        mode: 'inter' or 'intra'
        task: 'therapy' or 'restingstate'
        dppa_dir: Base DPPA derivatives directory

    Returns:
        Path to ICD CSV file

    Raises:
        FileNotFoundError: If no matching file found
    """
    # Determine subdirectory
    if mode == "inter":
        subdir = dppa_dir / "inter_session"
    else:
        subdir = dppa_dir / "intra_family"

    # Find matching file
    pattern = f"*_task-{task}_*.csv"
    matches = list(subdir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No ICD file found for mode={mode}, task={task} in {subdir}"
        )

    if len(matches) > 1:
        logger.warning(f"Multiple ICD files found, using: {matches[0].name}")

    return matches[0]


def process_task(
    plotter: ICDStatsPlotter, mode: str, task: str, dppa_dir: Path, output_dir: Path
) -> bool:
    """
    Generate all visualizations for a single mode/task combination.

    Args:
        plotter: ICDStatsPlotter instance
        mode: 'inter' or 'intra'
        task: Task name
        dppa_dir: Base DPPA derivatives directory
        output_dir: Output directory for figures

    Returns:
        True if successful
    """
    try:
        # Find ICD file
        icd_file = find_icd_file(mode, task, dppa_dir)
        logger.info(f"Found ICD file: {icd_file.name}")

        # Create task-specific output directory
        task_output_dir = output_dir / mode / task

        # Generate report
        result = plotter.generate_full_report(icd_file, task, task_output_dir)

        # Log summary
        stats = result["statistics"]
        meta = result["metadata"]

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"SUMMARY: {mode.upper()} - {task.upper()}")
        logger.info("=" * 60)

        # Check if we have pseudo dyads (inter vs intra mode)
        if stats.get("pseudo") is not None:
            # Inter-session mode with comparison
            logger.info(
                f"Real dyads:   n={meta['n_real_dyads']}, "
                f"mean={stats['real']['mean']:.1f} ms"
            )
            logger.info(
                f"Pseudo dyads: n={meta['n_pseudo_dyads']}, "
                f"mean={stats['pseudo']['mean']:.1f} ms"
            )
            logger.info(f"Difference:   {stats['difference']:.1f} ms")
            logger.info(f"Mann-Whitney: p = {stats['mann_whitney']['p']:.2e}")
            logger.info(f"Cohen's d:    {stats['cohens_d']:.3f}")
        else:
            # Intra-family mode (real dyads only)
            logger.info("Mode: Intra-family (real dyads only)")
            logger.info(f"Real dyads: n={meta['n_real_dyads']}")
            logger.info(f"Epochs:     {meta['n_epochs']}")
            logger.info(f"Mean ICD:   {stats['real']['mean']:.1f} ms")
            logger.info(f"Std ICD:    {stats['real']['std']:.1f} ms")
            logger.info(f"Median:     {stats['real']['median']:.1f} ms")

        logger.info("")
        logger.info("Files generated:")
        for name, path in result["files"].items():
            logger.info(f"  - {name}: {path.name}")
        logger.info("=" * 60)

        return True

    except FileNotFoundError as e:
        logger.warning(f"Skipping {mode}/{task}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing {mode}/{task}: {e}", exc_info=True)
        return False
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ICD statistical visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inter-session therapy
  python scripts/physio/dppa/plot_icd_stats.py --mode inter --task therapy
  
  # All tasks for inter-session
  python scripts/physio/dppa/plot_icd_stats.py --mode inter --all-tasks
  
  # Both modes and all tasks
  python scripts/physio/dppa/plot_icd_stats.py --all

Output:
  Figures are saved to data/derivatives/dppa/figures/{mode}/{task}/
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inter", "intra"],
        help="Dyad mode (inter-session or intra-family)",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["therapy", "restingstate"],
        help="Task to visualize",
    )

    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Process all available tasks for the specified mode",
    )

    parser.add_argument(
        "--all", action="store_true", help="Process all modes and tasks"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: data/derivatives/dppa/figures)",
    )

    # General options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validation
    if not args.all and not args.mode:
        parser.error("Either --mode or --all must be specified")

    if args.mode and not args.all_tasks and not args.task and not args.all:
        parser.error("Either --task, --all-tasks, or --all must be specified")

    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    log_dir = project_root / "log"
    setup_logging(log_dir, args.verbose)

    logger.info("=" * 80)
    logger.info("PLOT ICD STATISTICS")
    logger.info("=" * 80)

    # Determine base directories
    ConfigLoader()
    dppa_dir = project_root / "data" / "derivatives" / "dppa"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dppa_dir / "figures"

    logger.info(f"DPPA directory: {dppa_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize plotter
    plotter = ICDStatsPlotter()

    # Determine what to process
    if args.all:
        modes = ["inter", "intra"]
        tasks = ["therapy", "restingstate"]
    elif args.all_tasks:
        modes = [args.mode]
        tasks = ["therapy", "restingstate"]
    else:
        modes = [args.mode]
        tasks = [args.task]

    # Process each combination
    results = {"success": 0, "failed": 0}

    for mode in modes:
        for task in tasks:
            logger.info(f"\nProcessing: {mode} / {task}")

            success = process_task(plotter, mode, task, dppa_dir, output_dir)

            if success:
                results["success"] += 1
            else:
                results["failed"] += 1

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Failed:     {results['failed']}")
    logger.info(f"Output:     {output_dir}")
    logger.info("=" * 80)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
