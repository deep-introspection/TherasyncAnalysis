#!/usr/bin/env python3
"""
Generate DPPA dyadic visualizations.

This script creates 4-subplot visualizations for dyad pairs:
- Subplot 1 (full width): ICD time series with trendline and resting baseline
- Subplot 2-4: SD1, SD2, SD1/SD2 ratio for both subjects

Usage:
    # Single dyad
    python scripts/physio/dppa/plot_dyad.py --dyad g01p01_ses-01_vs_g01p02_ses-01 --method nsplit120

    # Batch mode (all dyads from config)
    python scripts/physio/dppa/plot_dyad.py --batch --method nsplit120 --mode inter

    # Batch mode with specific task
    python scripts/physio/dppa/plot_dyad.py --batch --method nsplit120 --mode inter --task therapy

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.physio.dppa import (
    DyadICDLoader,
    DyadCentroidLoader,
    DyadPlotter,
    DyadConfigLoader,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, verbose: bool = False):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"plot_dyad_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def plot_single_dyad(
    dyad_pair: str,
    method: str,
    output_dir: Path,
    icd_loader: DyadICDLoader,
    centroid_loader: DyadCentroidLoader,
    plotter: DyadPlotter,
) -> bool:
    """
    Plot a single dyad visualization.

    Args:
        dyad_pair: Dyad identifier (e.g., "g01p01_ses-01_vs_g01p02_ses-01")
        method: Poincaré method (e.g., "nsplit120")
        output_dir: Output directory for figures
        icd_loader: DyadICDLoader instance
        centroid_loader: DyadCentroidLoader instance
        plotter: DyadPlotter instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing dyad: {dyad_pair}")

        # Parse dyad info
        dyad_info = icd_loader.parse_dyad_info(dyad_pair)

        # Load ICD data
        logger.debug(f"Loading ICD data for {dyad_pair}")
        icd_data = icd_loader.load_both_tasks(dyad_pair, method)

        # Load centroid data
        logger.debug(f"Loading centroid data for {dyad_pair}")
        centroid_data = centroid_loader.load_both_tasks(dyad_info, method)

        # Create method-specific subdirectory
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        output_file = method_dir / (
            f"sub-{dyad_info['sub1']}_ses-{dyad_info['ses1']}_vs_"
            f"sub-{dyad_info['sub2']}_ses-{dyad_info['ses2']}_"
            f"method-{method}_desc-dyad_viz.png"
        )

        # Generate plot
        logger.debug(f"Generating plot: {output_file.name}")
        plotter.plot_dyad(
            icd_data=icd_data,
            centroid_data=centroid_data,
            dyad_info=dyad_info,
            method=method,
            output_path=output_file,
        )

        logger.info(f"✓ Successfully plotted: {output_file.name}")
        return True

    except FileNotFoundError as e:
        logger.warning(f"✗ Missing data for {dyad_pair}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error plotting {dyad_pair}: {e}", exc_info=True)
        return False


def plot_batch_dyads(
    method: str,
    mode: str,
    task: Optional[str],
    output_dir: Path,
    dyad_config_loader: DyadConfigLoader,
    icd_loader: DyadICDLoader,
    centroid_loader: DyadCentroidLoader,
    plotter: DyadPlotter,
) -> dict:
    """
    Plot visualizations for multiple dyads in batch mode.

    Args:
        method: Poincaré method (e.g., "nsplit120")
        mode: Dyad mode ("inter" or "intra")
        task: Optional task filter ("therapy" or "restingstate")
        output_dir: Output directory for figures
        dyad_config_loader: DyadConfigLoader instance
        icd_loader: DyadICDLoader instance
        centroid_loader: DyadCentroidLoader instance
        plotter: DyadPlotter instance

    Returns:
        Dictionary with statistics (total, success, failed)
    """
    logger.info(f"Batch mode: {mode}")
    logger.info(f"Method: {method}")
    logger.info(f"Task filter: {task if task else 'all'}")

    # Get dyad pairs
    if mode == "inter":
        raw_pairs = dyad_config_loader.get_inter_session_pairs()
        logger.info(f"Found {len(raw_pairs)} inter-session pairs")
    elif mode == "intra":
        raw_pairs = dyad_config_loader.get_intra_family_pairs()
        logger.info(f"Found {len(raw_pairs)} intra-family pairs")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'inter' or 'intra'.")

    # Convert tuples to string format
    # Inter-session: (('g01p01', 'ses-01'), ('g01p02', 'ses-01')) -> "g01p01_ses-01_vs_g01p02_ses-01"
    # Intra-family: ('g01p01', 'g01p02', 'ses-01') -> "g01p01_ses-01_vs_g01p02_ses-01"
    dyad_pairs = []
    for pair in raw_pairs:
        if mode == "inter":
            # Inter-session format: (('sub1', 'ses1'), ('sub2', 'ses2'))
            sub1, ses1 = pair[0]  # type: ignore
            sub2, ses2 = pair[1]  # type: ignore
            dyad_string = f"{sub1}_{ses1}_vs_{sub2}_{ses2}"
        else:  # intra
            # Intra-family format: ('sub1', 'sub2', 'ses')
            sub1 = pair[0]  # type: ignore
            sub2 = pair[1]  # type: ignore
            ses = pair[2]  # type: ignore
            dyad_string = f"{sub1}_{ses}_vs_{sub2}_{ses}"
        dyad_pairs.append(dyad_string)

    # Statistics
    stats = {"total": 0, "success": 0, "failed": 0}
    failed_dyads = []

    # Process each dyad
    for i, dyad_pair in enumerate(dyad_pairs, 1):
        stats["total"] += 1

        success = plot_single_dyad(
            dyad_pair=dyad_pair,
            method=method,
            output_dir=output_dir,
            icd_loader=icd_loader,
            centroid_loader=centroid_loader,
            plotter=plotter,
        )

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            failed_dyads.append(dyad_pair)

        # Progress update every 10 dyads
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(dyad_pairs)} dyads processed")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total dyads:      {stats['total']}")
    logger.info(f"Successful:       {stats['success']}")
    logger.info(f"Failed:           {stats['failed']}")
    logger.info(f"Success rate:     {stats['success'] / stats['total'] * 100:.1f}%")

    if failed_dyads:
        logger.warning(f"\nFailed dyads ({len(failed_dyads)}):")
        for dyad in failed_dyads[:10]:  # Show first 10
            logger.warning(f"  - {dyad}")
        if len(failed_dyads) > 10:
            logger.warning(f"  ... and {len(failed_dyads) - 10} more")

    logger.info("=" * 80)

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate DPPA dyadic visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dyad
  python scripts/physio/dppa/plot_dyad.py --dyad g01p01_ses-01_vs_g01p02_ses-01 --method nsplit120
  
  # Batch inter-session
  python scripts/physio/dppa/plot_dyad.py --batch --method nsplit120 --mode inter
  
  # Batch intra-family
  python scripts/physio/dppa/plot_dyad.py --batch --method nsplit120 --mode intra
        """,
    )

    # Required arguments
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Poincaré method (e.g., nsplit120, sliding_duration30s_step5s)",
    )

    # Single dyad mode
    parser.add_argument(
        "--dyad",
        type=str,
        help="Dyad identifier (e.g., g01p01_ses-01_vs_g01p02_ses-01)",
    )

    # Batch mode
    parser.add_argument(
        "--batch", action="store_true", help="Batch mode: process multiple dyads"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["inter", "intra"],
        help="Dyad mode for batch processing (inter-session or intra-family)",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["therapy", "restingstate"],
        help="Optional task filter for batch mode",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for figures (default: from config)",
    )

    # General options
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validation
    if not args.batch and not args.dyad:
        parser.error("Either --dyad or --batch must be specified")

    if args.batch and not args.mode:
        parser.error("--mode is required for batch processing")

    if args.dyad and args.batch:
        parser.error("Cannot use both --dyad and --batch")

    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / args.config
    log_dir = project_root / "log"

    setup_logging(log_dir, args.verbose)

    logger.info("=" * 80)
    logger.info("PLOT DPPA DYADIC VISUALIZATIONS")
    logger.info("=" * 80)
    logger.info(f"Method: {args.method}")
    logger.info(f"Mode: {'Batch' if args.batch else 'Single dyad'}")

    try:
        # Load configuration
        config = ConfigLoader(config_path)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            viz_config = config.get("visualization.dppa", {})
            output_base = viz_config.get("output", {}).get("base_dir", "dppa/figures")
            output_dir = project_root / "data" / "derivatives" / output_base

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Initialize loaders and plotter
        logger.info("Initializing components...")
        icd_loader = DyadICDLoader()
        centroid_loader = DyadCentroidLoader()
        plotter = DyadPlotter()

        if args.batch:
            # Batch mode
            dyad_config_loader = DyadConfigLoader()

            stats = plot_batch_dyads(
                method=args.method,
                mode=args.mode,
                task=args.task,
                output_dir=output_dir,
                dyad_config_loader=dyad_config_loader,
                icd_loader=icd_loader,
                centroid_loader=centroid_loader,
                plotter=plotter,
            )

            # Exit code based on success rate
            if stats["failed"] == 0:
                logger.info("All dyads processed successfully!")
                return 0
            elif stats["success"] > 0:
                logger.warning(f"Partial success: {stats['failed']} dyads failed")
                return 1
            else:
                logger.error("All dyads failed!")
                return 2

        else:
            # Single dyad mode
            success = plot_single_dyad(
                dyad_pair=args.dyad,
                method=args.method,
                output_dir=output_dir,
                icd_loader=icd_loader,
                centroid_loader=centroid_loader,
                plotter=plotter,
            )

            if success:
                logger.info("Plot generated successfully!")
                return 0
            else:
                logger.error("Failed to generate plot")
                return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
