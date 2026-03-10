#!/usr/bin/env python3
"""
Alliance-ICD Analysis CLI Script.

Runs the complete alliance-ICD correlation analysis, generating
statistical reports and visualizations.

Usage:
    uv run python scripts/analysis/run_alliance_icd_analysis.py
    uv run python scripts/analysis/run_alliance_icd_analysis.py --no-viz
    uv run python scripts/analysis/run_alliance_icd_analysis.py --output-dir custom/path

Authors: Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.logger_setup import LoggerSetup
from src.alliance.alliance_icd_loader import AllianceICDLoader
from src.alliance.alliance_icd_analyzer import AllianceICDAnalyzer
from src.alliance.alliance_icd_plotter import AllianceICDStatsPlotter

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Alliance-ICD correlation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full analysis with visualizations
    uv run python scripts/analysis/run_alliance_icd_analysis.py
    
    # Run without visualizations (stats only)
    uv run python scripts/analysis/run_alliance_icd_analysis.py --no-viz
    
    # Specify output directory
    uv run python scripts/analysis/run_alliance_icd_analysis.py --output-dir data/derivatives/analysis
        """
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Custom output directory for reports (default: derivatives/visualization/alliance_icd)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger_setup = LoggerSetup(log_level=log_level)
    logger_setup.setup_root_logger(console_level=log_level)
    
    print("=" * 70)
    print("ALLIANCE-ICD CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Initialize components
    loader = AllianceICDLoader(args.config)
    analyzer = AllianceICDAnalyzer(args.config)
    
    # Step 1: Discover MOI sessions
    print("Step 1: Discovering MOI annotations...")
    sessions_with_moi = loader.get_sessions_with_moi()
    
    print(f"  Found {len(sessions_with_moi)} families with MOI:")
    total_sessions = 0
    for family, sessions in sorted(sessions_with_moi.items()):
        print(f"    {family}: {sessions}")
        total_sessions += len(sessions)
    print(f"  Total sessions with MOI: {total_sessions}")
    print()
    
    # Step 2: Get valid dyads
    print("Step 2: Identifying valid dyads...")
    valid_real, valid_pseudo = loader.get_valid_dyads(sessions_with_moi)
    print(f"  Valid real dyads: {len(valid_real)}")
    print(f"  Valid pseudo-dyads: {len(valid_pseudo)}")
    print(f"  Total: {len(valid_real) + len(valid_pseudo)}")
    print()
    
    # Step 3: Load merged data
    print("Step 3: Loading and merging data...")
    data = loader.load_all_merged_data(sessions_with_moi)
    
    if data.empty:
        print("ERROR: No data could be loaded. Check ICD files and MOI annotations.")
        return 1
    
    print(f"  Loaded {len(data)} epoch-dyad observations")
    print(f"  Real dyad observations: {len(data[data['dyad_type'] == 'real'])}")
    print(f"  Pseudo-dyad observations: {len(data[data['dyad_type'] == 'pseudo'])}")
    print()
    
    # Step 4: Generate statistical report
    print("Step 4: Computing statistics...")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = loader.derivatives_path / 'visualization' / 'alliance_icd'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save report
    report_path = output_dir / 'reports' / 'alliance_icd_stats_report.txt'
    report = analyzer.generate_report(data, report_path)
    
    print()
    print(report)
    print()
    
    # Step 5: Generate visualizations
    if not args.no_viz:
        print("Step 5: Generating visualizations...")
        plotter = AllianceICDStatsPlotter(args.config)
        
        outputs = plotter.generate_all_visualizations(data)
        
        print(f"  Generated {len(outputs)} figures:")
        for name, path in outputs.items():
            print(f"    - {name}: {path.name}")
        print()
    else:
        print("Step 5: Skipping visualizations (--no-viz)")
        print()
    
    # Summary
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Report saved: {report_path}")
    if not args.no_viz:
        print(f"Figures saved: {output_dir / 'figures'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
