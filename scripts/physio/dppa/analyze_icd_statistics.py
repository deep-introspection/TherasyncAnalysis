#!/usr/bin/env python3
"""
Analyze ICD Statistics with Corrected Statistical Tests.

This script compares naive vs corrected statistical analyses for ICD data,
addressing the non-independence issue where participants appear in multiple dyads.

Three approaches are computed:
1. Naive: Mann-Whitney U treating each dyad as independent (INFLATED significance)
2. Participant-aggregated: Mean ICD per participant, then paired test
3. Mixed model: Linear mixed model with participant as random effect

Usage:
    uv run python scripts/physio/dppa/analyze_icd_statistics.py --task therapy
    uv run python scripts/physio/dppa/analyze_icd_statistics.py --task restingstate
    uv run python scripts/physio/dppa/analyze_icd_statistics.py --all

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.core.logger_setup import setup_logging
from src.physio.dppa.icd_stats_plotter import ICDStatsPlotter

logger = logging.getLogger(__name__)


def analyze_task(
    plotter: ICDStatsPlotter,
    icd_file: Path,
    task: str,
    output_dir: Path
) -> dict:
    """
    Run corrected statistical analysis for a single task.
    
    Args:
        plotter: ICDStatsPlotter instance
        icd_file: Path to ICD CSV file
        task: Task name
        output_dir: Output directory
    
    Returns:
        Results dictionary
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing {task} task")
    logger.info(f"{'='*70}")
    
    # Load data
    df, metadata = plotter.load_icd_data(icd_file, task)
    
    # Check if we have pseudo dyads
    if metadata['n_pseudo_dyads'] == 0:
        logger.warning(f"No pseudo dyads found for {task}. Skipping corrected analysis.")
        return None
    
    # Run corrected analysis
    output_path = output_dir / f"icd_corrected_stats_{task}.txt"
    results = plotter.generate_corrected_statistics_report(df, metadata, output_path)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze ICD statistics with corrected tests"
    )
    parser.add_argument(
        "--task",
        choices=["therapy", "restingstate"],
        help="Task to analyze"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all tasks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir=Path("log"))
    
    # Initialize
    config = ConfigLoader(args.config)
    plotter = ICDStatsPlotter(args.config)
    
    # Paths
    derivatives_dir = Path(config.get("paths.derivatives_dir", "data/derivatives"))
    inter_session_dir = derivatives_dir / "dppa" / "inter_session"
    output_dir = inter_session_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine tasks to analyze
    if args.all:
        tasks = ["therapy", "restingstate"]
    elif args.task:
        tasks = [args.task]
    else:
        parser.error("Please specify --task or --all")
        return
    
    # Analyze each task
    all_results = {}
    for task in tasks:
        # Find ICD file
        icd_files = list(inter_session_dir.glob(f"*_task-{task}_*.csv"))
        
        if not icd_files:
            logger.warning(f"No ICD file found for task {task}")
            continue
        
        icd_file = icd_files[0]
        logger.info(f"Using ICD file: {icd_file.name}")
        
        results = analyze_task(plotter, icd_file, task, output_dir)
        if results:
            all_results[task] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Tasks analyzed: {list(all_results.keys())}")
    
    for task, results in all_results.items():
        naive_p = results['naive']['mann_whitney']['p']
        corrected_p = results['participant_level']['wilcoxon']['p']
        print(f"\n{task.upper()}:")
        print(f"  Naive p-value:     {naive_p:.2e}")
        print(f"  Corrected p-value: {corrected_p:.4f}")
        if naive_p < 0.05 and corrected_p >= 0.05:
            print(f"  ⚠️  Significance LOST after correction!")


if __name__ == "__main__":
    main()
