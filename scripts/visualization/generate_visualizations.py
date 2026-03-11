#!/usr/bin/env python
"""
Visualization Generation Script for TherasyncPipeline.

This script generates publication-ready visualizations in two modes:
- COMPOSITE (default): 5 comprehensive multi-panel figures for reports
- INDIVIDUAL: 8 individual plots for detailed analysis

Composite Figures (default):
    #1: Multi-Signal Dashboard (Overview of all modalities)
    #2: HRV Analysis (Poincaré + Autonomic Balance + Frequency Domain)
    #3: EDA Analysis (Arousal Profile + SCR Distribution + SCR Timeline)
    #4: Temperature Analysis (Timeline + Metrics + Correlation with EDA)
    #5: Quality Report (Signal quality heatmap + Coverage statistics)

Individual Figures (--individual mode):
    #1: Multi-Signal Dashboard
    #2: Poincaré Plot (HRV)
    #3: Autonomic Balance
    #4: EDA Arousal Profile
    #5: SCR Distribution
    #6: HR Dynamics Timeline
    #7: Temperature Timeline
    #8: Temperature Metrics

Usage:
    # Default: Generate composite figures
    python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01

    # Generate individual plots
    python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01 --individual

    # All available subjects (composite mode)
    python scripts/visualization/generate_visualizations.py --all

    # Specific composite figures only
    python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01 --plots 1 2 3

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
from src.visualization.plotters.temp_plots import (
    plot_temp_timeline,
    plot_temp_metrics_comparison,
)
from src.visualization.plotters.composite_plots import (
    plot_hrv_analysis,
    plot_eda_analysis,
    plot_temp_analysis,
    plot_quality_report,
)
from src.visualization.config import OUTPUT_CONFIG

logger = logging.getLogger(__name__)


# COMPOSITE MODE: 5 comprehensive multi-panel figures
COMPOSITE_FUNCTIONS = {
    1: ("01_overview_physiological.png", plot_multisignal_dashboard),
    2: ("02_hrv_analysis.png", plot_hrv_analysis),
    3: ("03_eda_analysis.png", plot_eda_analysis),
    4: ("04_temp_analysis.png", plot_temp_analysis),
    5: ("05_quality_report.png", plot_quality_report),
}

# INDIVIDUAL MODE: 8 separate detailed figures
INDIVIDUAL_FUNCTIONS = {
    1: ("01_dashboard_multisignals.png", plot_multisignal_dashboard),
    2: ("02_poincare_hrv.png", plot_poincare_hrv),
    3: ("03_autonomic_balance.png", plot_autonomic_balance),
    4: ("04_eda_arousal_profile.png", plot_eda_arousal_profile),
    5: ("05_scr_distribution.png", plot_scr_distribution),
    6: ("06_hr_dynamics_timeline.png", plot_hr_dynamics_timeline),
    7: ("07_temp_timeline.png", plot_temp_timeline),
    8: ("08_temp_metrics_comparison.png", plot_temp_metrics_comparison),
}

# Default mode
PLOT_FUNCTIONS = COMPOSITE_FUNCTIONS


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def generate_visualizations_for_subject(
    subject: str,
    session: str,
    plots: Optional[List[int]] = None,
    output_base: Optional[Path] = None,
    individual_mode: bool = False,
) -> bool:
    """
    Generate all visualizations for a single subject/session.

    Args:
        subject: Subject ID (e.g., 'g01p01')
        session: Session ID (e.g., '01')
        plots: List of plot numbers to generate (None = all)
        output_base: Base output directory
        individual_mode: If True, generate individual plots instead of composite

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing subject {subject}, session {session}")
    mode_str = "individual" if individual_mode else "composite"
    logger.info(f"Mode: {mode_str}")

    # Select appropriate plot functions
    plot_functions = INDIVIDUAL_FUNCTIONS if individual_mode else COMPOSITE_FUNCTIONS

    try:
        # Load data
        loader = VisualizationDataLoader()
        data = loader.load_subject_session(subject, session)

        if not data:
            logger.error(f"No data loaded for {subject}/{session}")
            return False

        # Setup output directory
        if output_base is None:
            output_base = Path(OUTPUT_CONFIG["base_path"])

        subject_id = data["subject_id"]
        session_id = data["session_id"]

        figures_dir = (
            output_base / subject_id / session_id / OUTPUT_CONFIG["figures_subdir"]
        )
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {figures_dir}")

        # Determine which plots to generate
        plots_to_generate = plots if plots else list(plot_functions.keys())

        # Generate each visualization
        success_count = 0
        for plot_num in plots_to_generate:
            if plot_num not in plot_functions:
                logger.warning(
                    f"Plot #{plot_num} not available in {mode_str} mode, skipping"
                )
                continue

            filename, plot_func = plot_functions[plot_num]
            output_path = figures_dir / filename

            logger.info(f"Generating visualization #{plot_num}: {filename}")

            try:
                plot_func(data, output_path=output_path, show=False)
                logger.info(f"  ✓ Saved to {output_path}")
                success_count += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to generate plot #{plot_num}: {str(e)}")
                logger.debug("Error details:", exc_info=True)

        logger.info(
            f"Successfully generated {success_count}/{len(plots_to_generate)} visualizations"
        )

        return success_count > 0

    except Exception as e:
        logger.error(f"Error processing {subject}/{session}: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        return False


def batch_process_all_subjects(
    plots: Optional[List[int]] = None,
    output_base: Optional[Path] = None,
    individual_mode: bool = False,
) -> dict:
    """
    Process all available subjects/sessions.

    Args:
        plots: List of plot numbers to generate (None = all)
        output_base: Base output directory
        individual_mode: If True, generate individual plots instead of composite

    Returns:
        Dictionary with processing statistics
    """
    mode_str = "individual" if individual_mode else "composite"
    logger.info(f"Starting batch processing of all subjects/sessions ({mode_str} mode)")

    loader = VisualizationDataLoader()
    subjects_sessions = loader.list_available_subjects()

    if not subjects_sessions:
        logger.warning("No subjects/sessions found")
        return {"total": 0, "success": 0, "failed": 0}

    logger.info(f"Found {len(subjects_sessions)} subject/session combinations")

    stats = {"total": len(subjects_sessions), "success": 0, "failed": 0}

    for i, (subject, session) in enumerate(subjects_sessions, 1):
        logger.info(f"[{i}/{len(subjects_sessions)}] Processing {subject}/{session}")

        success = generate_visualizations_for_subject(
            subject, session, plots, output_base, individual_mode
        )

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    logger.info(
        f"Batch processing complete: {stats['success']}/{stats['total']} successful"
    )

    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for TherasyncPipeline preprocessed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization Modes:
  COMPOSITE (default): 5 comprehensive multi-panel figures for reports
    #1: Overview - Multi-signal dashboard
    #2: HRV Analysis - Poincaré + Autonomic Balance + Frequency Domain
    #3: EDA Analysis - Arousal + SCR Distribution + SCR Timeline
    #4: Temperature Analysis - Timeline + Metrics + Correlation
    #5: Quality Report - Signal quality overview
  
  INDIVIDUAL (--individual): 8 separate detailed figures

Examples:
  # Generate composite figures (default)
  python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01
  
  # Generate individual plots
  python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01 --individual
  
  # Generate specific composite figures only
  python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01 --plots 1 2 3
  
  # Process all subjects (composite mode)
  python scripts/visualization/generate_visualizations.py --all
  
  # Custom output directory
  python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01 --output custom/path
        """,
    )

    # Subject/session selection
    parser.add_argument("--subject", type=str, help="Subject ID (e.g., g01p01)")
    parser.add_argument("--session", type=str, help="Session ID (e.g., 01)")
    parser.add_argument(
        "--all", action="store_true", help="Process all available subjects/sessions"
    )

    # Visualization mode
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Generate individual plots instead of composite figures (default: composite)",
    )

    # Visualization options
    parser.add_argument(
        "--plots",
        type=int,
        nargs="+",
        help="Specific plot numbers to generate (default: all available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output base directory (default: data/derivatives/visualization)",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if not args.all and (not args.subject or not args.session):
        parser.error("Either --all or both --subject and --session must be specified")

    # Parse output path
    output_base = Path(args.output) if args.output else None

    # Execute
    if args.all:
        stats = batch_process_all_subjects(args.plots, output_base, args.individual)
        sys.exit(0 if stats["success"] > 0 else 1)
    else:
        success = generate_visualizations_for_subject(
            args.subject, args.session, args.plots, output_base, args.individual
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
