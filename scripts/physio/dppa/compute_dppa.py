#!/usr/bin/env python3
"""
Compute DPPA (Dyadic Poincaré Plot Analysis) Inter-Centroid Distances.

This script orchestrates the full DPPA pipeline:
1. Load pre-computed Poincaré centroids
2. Identify dyad pairs (inter-session or intra-family)
3. Calculate Inter-Centroid Distances
4. Export results to BIDS-compliant CSV files

Usage:
    # Single dyad (intra-family)
    python scripts/physio/dppa/compute_dppa.py --mode intra --family g01 --session 01 --task therapy

    # Batch inter-session
    python scripts/physio/dppa/compute_dppa.py --mode inter --task all --batch

    # Batch intra-family
    python scripts/physio/dppa/compute_dppa.py --mode intra --task all --batch

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.physio.dppa import CentroidLoader, DyadConfigLoader, ICDCalculator, DPPAWriter

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, verbose: bool = False):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"compute_dppa_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger.info(f"Logging to: {log_file}")


def compute_inter_session(
    loader: CentroidLoader,
    dyad_loader: DyadConfigLoader,
    calculator: ICDCalculator,
    writer: DPPAWriter,
    task: str,
    dry_run: bool = False,
) -> Dict:
    """
    Compute inter-session ICD for all dyad pairs.

    Returns:
        Dictionary with statistics
    """
    logger.info(f"Computing inter-session ICD for task: {task}")

    # Get dyad pairs and method for this specific task
    pairs = dyad_loader.get_inter_session_pairs(task=task)
    method = dyad_loader.get_inter_session_method(task=task)

    logger.info(f"Processing {len(pairs)} inter-session pairs with method: {method}")

    if dry_run:
        logger.info("DRY RUN - would process dyad pairs")
        return {"pairs": len(pairs), "processed": 0, "failed": 0}

    # Compute ICDs for all pairs
    icd_results = {}
    failed = 0

    for i, ((subj1, ses1), (subj2, ses2)) in enumerate(pairs, 1):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(pairs)} pairs processed")

        try:
            # Load centroids
            df1 = loader.load_centroid(subj1, ses1, task, method)
            df2 = loader.load_centroid(subj2, ses2, task, method)

            if df1 is None or df2 is None:
                logger.debug(f"Missing centroids for {subj1}/{ses1} or {subj2}/{ses2}")
                failed += 1
                continue

            # Compute ICD
            icd_df = calculator.compute_icd(df1, df2)
            icd_results[(subj1, ses1, subj2, ses2)] = icd_df

        except Exception as e:
            logger.error(
                f"Failed to compute ICD for {subj1}/{ses1} vs {subj2}/{ses2}: {e}"
            )
            failed += 1

    # Write results
    if icd_results:
        csv_path = writer.write_inter_session(icd_results, task=task, method=method)
        logger.info(f"Wrote inter-session results: {csv_path}")
    else:
        logger.warning("No inter-session ICDs computed")

    return {"pairs": len(pairs), "processed": len(icd_results), "failed": failed}


def compute_intra_family(
    loader: CentroidLoader,
    dyad_loader: DyadConfigLoader,
    calculator: ICDCalculator,
    writer: DPPAWriter,
    task: str,
    family: str = None,
    session: str = None,
    dry_run: bool = False,
) -> Dict:
    """
    Compute intra-family ICD for same-session dyad pairs.

    Returns:
        Dictionary with statistics
    """
    logger.info(f"Computing intra-family ICD for task: {task}")
    if family:
        logger.info(f"  Family filter: {family}")
    if session:
        logger.info(f"  Session filter: {session}")

    # Get dyad pairs and method
    pairs = dyad_loader.get_intra_family_pairs(
        family=family, session=session, task=task
    )
    method = dyad_loader.get_intra_family_method()

    logger.info(f"Processing {len(pairs)} intra-family pairs with method: {method}")

    if dry_run:
        logger.info("DRY RUN - would process dyad pairs")
        return {"pairs": len(pairs), "processed": 0, "failed": 0}

    # Compute ICDs for all pairs
    icd_results = {}
    failed = 0

    for i, ((fam, subj1, ses), (_, subj2, _)) in enumerate(pairs, 1):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(pairs)} pairs processed")

        try:
            # Load centroids
            df1 = loader.load_centroid(subj1, ses, task, method)
            df2 = loader.load_centroid(subj2, ses, task, method)

            if df1 is None or df2 is None:
                logger.debug(f"Missing centroids for {subj1} or {subj2} ({ses})")
                failed += 1
                continue

            # Compute ICD
            icd_df = calculator.compute_icd(df1, df2)
            icd_results[(fam, subj1, subj2, ses, task)] = icd_df

        except Exception as e:
            logger.error(f"Failed to compute ICD for {fam} {subj1} vs {subj2}: {e}")
            failed += 1

    # Write results
    if icd_results:
        csv_path = writer.write_intra_family(icd_results, task=task, method=method)
        logger.info(f"Wrote intra-family results: {csv_path}")
    else:
        logger.warning("No intra-family ICDs computed")

    return {"pairs": len(pairs), "processed": len(icd_results), "failed": failed}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute DPPA Inter-Centroid Distances between dyads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single intra-family analysis
  python scripts/physio/dppa/compute_dppa.py --mode intra --family g01 --session 01 --task therapy
  
  # Batch inter-session (all pairs)
  python scripts/physio/dppa/compute_dppa.py --mode inter --task therapy --batch
  
  # Batch intra-family (all families)
  python scripts/physio/dppa/compute_dppa.py --mode intra --task therapy --batch
  
  # Both modes, all tasks
  python scripts/physio/dppa/compute_dppa.py --mode both --task all --batch
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["inter", "intra", "both"],
        help="Analysis mode: inter-session, intra-family, or both",
    )
    parser.add_argument(
        "--task", type=str, default="all", help="Task name (therapy, restingstate, all)"
    )
    parser.add_argument(
        "--family", type=str, help="Family filter for intra-family mode (e.g., g01)"
    )
    parser.add_argument(
        "--session", type=str, help="Session filter for intra-family mode (e.g., 01)"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process all available dyads"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--dyad-config",
        type=str,
        default="config/dppa_dyads_real.yaml",
        help="Path to dyad config file for real/pseudo dyad distinction",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Normalize session format (add ses- prefix if needed)
    if args.session and not args.session.startswith("ses-"):
        args.session = f"ses-{args.session}"

    # Load config
    config = ConfigLoader(args.config)
    paths = config.get("paths", {})

    # Setup logging
    log_dir = Path(paths.get("logs", "log"))
    setup_logging(log_dir, args.verbose)

    logger.info("=" * 80)
    logger.info("COMPUTE DPPA INTER-CENTROID DISTANCES")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Batch: {args.batch}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Dyad config: {args.dyad_config}")

    # Initialize components
    loader = CentroidLoader(args.config)
    dyad_loader = DyadConfigLoader(args.dyad_config)
    calculator = ICDCalculator()
    writer = DPPAWriter(args.config, dyad_config_path=args.dyad_config)

    # Determine tasks to process
    if args.task == "all":
        if args.mode == "inter" or args.mode == "both":
            tasks = dyad_loader.get_inter_session_tasks()
        else:
            tasks = dyad_loader.get_intra_family_tasks()
    else:
        tasks = [args.task]

    logger.info(f"Tasks to process: {tasks}")

    # Process each task
    total_stats = {"pairs": 0, "processed": 0, "failed": 0}

    for task in tasks:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing task: {task}")
        logger.info(f"{'=' * 80}")

        # Inter-session mode
        if args.mode == "inter" or args.mode == "both":
            logger.info("\n--- INTER-SESSION MODE ---")
            stats = compute_inter_session(
                loader, dyad_loader, calculator, writer, task=task, dry_run=args.dry_run
            )
            logger.info(
                f"Inter-session stats: {stats['processed']}/{stats['pairs']} pairs, {stats['failed']} failed"
            )
            for key in total_stats:
                total_stats[key] += stats[key]

        # Intra-family mode
        if args.mode == "intra" or args.mode == "both":
            logger.info("\n--- INTRA-FAMILY MODE ---")
            stats = compute_intra_family(
                loader,
                dyad_loader,
                calculator,
                writer,
                task=task,
                family=args.family,
                session=args.session,
                dry_run=args.dry_run,
            )
            logger.info(
                f"Intra-family stats: {stats['processed']}/{stats['pairs']} pairs, {stats['failed']} failed"
            )
            for key in total_stats:
                total_stats[key] += stats[key]

    # Clear cache to free memory
    cache_info = loader.get_cache_info()
    logger.info(f"\nCache before clear: {cache_info['entries']} entries")
    loader.clear_cache()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total pairs:      {total_stats['pairs']}")
    logger.info(f"ICDs computed:    {total_stats['processed']}")
    logger.info(f"Failed:           {total_stats['failed']}")

    if total_stats["pairs"] > 0:
        success_rate = (total_stats["processed"] / total_stats["pairs"]) * 100
        logger.info(f"Success rate:     {success_rate:.1f}%")

    logger.info("=" * 80)

    return 0 if total_stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
