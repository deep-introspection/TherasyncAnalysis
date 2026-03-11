#!/usr/bin/env python
"""
Visualize MOI (Moments of Interest) Annotations.

Creates timeseries and distribution visualizations for epoched MOI annotations.

Usage:
    # Visualize specific group/session
    python scripts/alliance/visualize_moi_annotations.py --group g01 --session 01

    # Visualize all available sessions
    python scripts/alliance/visualize_moi_annotations.py --all

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.alliance.moi_visualizer import MOIVisualizer
from src.alliance.moi_loader import MOILoader

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def visualize_session(
    visualizer: MOIVisualizer, group_id: str, session_id: str
) -> bool:
    """
    Create visualizations for a single session.

    Args:
        visualizer: MOI visualizer instance
        group_id: Group ID (e.g., 'g01')
        session_id: Session ID (e.g., '01')

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Visualizing: {group_id}/ses-{session_id}")
        logger.info(f"{'=' * 80}")

        results = visualizer.visualize_session(group_id, session_id)

        logger.info(f"Created {len(results['timeseries'])} timeseries plots")
        logger.info(f"Created {len(results['distribution'])} distribution plots")

        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(
            f"Error visualizing {group_id}/ses-{session_id}: {e}", exc_info=True
        )
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize MOI annotations")
    parser.add_argument("--group", "-g", type=str, help="Group ID (e.g., 'g01')")
    parser.add_argument("--session", "-s", type=str, help="Session ID (e.g., '01')")
    parser.add_argument(
        "--all", action="store_true", help="Visualize all available sessions"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("MOI Annotation Visualization Script")
    logger.info("=" * 80)

    # Initialize components
    visualizer = MOIVisualizer()
    loader = MOILoader()

    # Determine sessions to process
    if args.all:
        sessions = loader.get_available_sessions()
        logger.info(f"\nFound {len(sessions)} sessions to visualize")
    elif args.group and args.session:
        sessions = [(args.group, args.session)]
    else:
        logger.error("Error: Must specify either --all or both --group and --session")
        return 1

    # Process sessions
    stats = {"total": len(sessions), "success": 0, "failed": 0}

    for group_id, session_id in sessions:
        success = visualize_session(visualizer, group_id, session_id)

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary:")
    logger.info(f"  Total sessions:         {stats['total']}")
    logger.info(f"  Successfully visualized: {stats['success']}")
    logger.info(f"  Failed:                 {stats['failed']}")
    logger.info("=" * 80)

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
