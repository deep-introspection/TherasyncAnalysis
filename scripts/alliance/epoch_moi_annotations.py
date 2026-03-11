#!/usr/bin/env python
"""
Epoch MOI (Moments of Interest) Annotations.

This script adds epoch columns to MOI alliance/emotion annotation files
using the same epoching methods as physiological data.

Usage:
    # Process specific group/session
    python scripts/alliance/epoch_moi_annotations.py --group g01 --session 01

    # Process all available sessions
    python scripts/alliance/epoch_moi_annotations.py --all

    # Dry run
    python scripts/alliance/epoch_moi_annotations.py --all --dry-run

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.alliance.moi_loader import MOILoader
from src.alliance.moi_epocher import MOIEpocher
from src.alliance.moi_writer import MOIWriter

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def process_moi_session(
    group_id: str,
    session_id: str,
    loader: MOILoader,
    epocher: MOIEpocher,
    writer: MOIWriter,
    dry_run: bool = False,
) -> bool:
    """
    Process a single MOI session: load, epoch, and save.

    Args:
        group_id: Group ID (e.g., 'g01')
        session_id: Session ID (e.g., '01')
        loader: MOI loader instance
        epocher: MOI epocher instance
        writer: MOI writer instance
        dry_run: If True, don't save files

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load MOI file
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {group_id}/ses-{session_id}")
        logger.info(f"{'=' * 80}")

        data = loader.load_moi_file(group_id, session_id)
        df = data["annotations"]
        metadata = data["metadata"]

        logger.info(f"Loaded {len(df)} annotations")
        logger.info(
            f"Session duration: {metadata.get('Duration', 0):.1f}s ({metadata.get('Duration', 0) / 60:.1f}min)"
        )

        # Add epoch columns
        df_epoched = epocher.add_epoch_columns(df, metadata)

        # Show epoch statistics
        for method in ["fixed", "nsplit", "sliding"]:
            col = f"epoch_{method}"
            if col in df_epoched.columns:
                # Now each cell contains a list of epoch IDs
                all_epochs = [
                    eid for epoch_list in df_epoched[col] for eid in epoch_list
                ]
                if all_epochs:
                    min_epoch = min(all_epochs)
                    max_epoch = max(all_epochs)
                    n_unique_epochs = len(set(all_epochs))
                    logger.info(
                        f"  {method}: {n_unique_epochs} unique epochs (range: {min_epoch}-{max_epoch})"
                    )
                else:
                    logger.info(f"  {method}: No epochs assigned")

        if dry_run:
            logger.info("[DRY RUN] Would save epoched file")
            logger.info("  Preview of epoched data:")
            # Show first few rows with a sample of data
            preview_cols = [
                "start",
                "end",
                "start_seconds",
                "end_seconds",
                "epoch_fixed",
                "epoch_nsplit",
                "epoch_sliding",
            ]
            logger.info(f"\n{df_epoched[preview_cols].head(10)}")
        else:
            # Save epoched file
            output_file = writer.save_epoched_moi(
                df_epoched, metadata, group_id, session_id
            )
            logger.info(f"✓ Saved to: {output_file}")

        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(
            f"Error processing {group_id}/ses-{session_id}: {e}", exc_info=True
        )
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Epoch MOI annotations")
    parser.add_argument("--group", "-g", type=str, help="Group ID (e.g., 'g01')")
    parser.add_argument("--session", "-s", type=str, help="Session ID (e.g., '01')")
    parser.add_argument(
        "--all", action="store_true", help="Process all available sessions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("MOI Annotation Epoching Script")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be saved")

    # Initialize components
    loader = MOILoader()
    epocher = MOIEpocher()
    writer = MOIWriter()

    # Determine sessions to process
    if args.all:
        sessions = loader.get_available_sessions()
        logger.info(f"\nFound {len(sessions)} sessions to process")
    elif args.group and args.session:
        sessions = [(args.group, args.session)]
    else:
        logger.error("Error: Must specify either --all or both --group and --session")
        return 1

    # Process sessions
    stats = {"total": len(sessions), "success": 0, "failed": 0}

    for group_id, session_id in sessions:
        success = process_moi_session(
            group_id, session_id, loader, epocher, writer, dry_run=args.dry_run
        )

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary:")
    logger.info(f"  Total sessions:    {stats['total']}")
    logger.info(f"  Successfully processed: {stats['success']}")
    logger.info(f"  Failed:           {stats['failed']}")
    logger.info("=" * 80)

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
