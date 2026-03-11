#!/usr/bin/env python3
"""
Clean physiological signal processing outputs for fresh pipeline runs.

This utility script safely removes processed derivatives (BVP, EDA, HR) and logs,
allowing for clean re-runs during testing and development.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import ConfigLoader
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def confirm_deletion(path: Path, force: bool = False) -> bool:
    """
    Confirm deletion with user unless force flag is set.

    Args:
        path: Path to be deleted
        force: Skip confirmation if True

    Returns:
        True if deletion should proceed, False otherwise
    """
    if force:
        return True

    response = input(f"Delete {path}? [y/N]: ").strip().lower()
    return response in ["y", "yes"]


def clean_derivatives(
    config: ConfigLoader,
    modality: Optional[str] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Clean physiological processing derivatives.

    Args:
        config: Configuration loader instance
        modality: Specific modality to clean (bvp, eda, hr, or None = all)
        subject: Specific subject to clean (None = all subjects)
        session: Specific session to clean (None = all sessions)
        force: Skip confirmation prompts
        dry_run: Show what would be deleted without actually deleting

    Returns:
        Number of items deleted
    """
    derivatives_path = Path(config.get("paths.derivatives"))
    preprocessing_dir = config.get("output.preprocessing_dir", "preprocessing")
    preprocessing_path = derivatives_path / preprocessing_dir

    if not preprocessing_path.exists():
        logger.info(f"No preprocessing derivatives found at {preprocessing_path}")
        return 0

    deleted_count = 0

    # Determine which modalities to clean
    if modality:
        modalities = [modality]
    else:
        modality_subdirs = config.get("output.modality_subdirs", {})
        modalities = list(modality_subdirs.values())  # ['bvp', 'eda', 'hr']

    # Clean specific subject/session
    if subject:
        # Add 'sub-' prefix if not present
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        subject_path = preprocessing_path / subject

        if not subject_path.exists():
            logger.warning(f"Subject {subject} not found in derivatives")
            return 0

        if session:
            # Add 'ses-' prefix if not present
            if not session.startswith("ses-"):
                session = f"ses-{session}"

            session_path = subject_path / session

            if not session_path.exists():
                logger.warning(f"Session {session} not found for {subject}")
                return 0

            # Clean specific modalities within session
            for mod in modalities:
                modality_path = session_path / mod
                if modality_path.exists():
                    logger.info(
                        f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {modality_path}"
                    )

                    if not dry_run and confirm_deletion(modality_path, force):
                        shutil.rmtree(modality_path)
                        deleted_count += 1
                        logger.info(f"Deleted modality: {modality_path}")

            # If session is empty after cleaning, remove it
            if (
                not dry_run
                and session_path.exists()
                and not any(session_path.iterdir())
            ):
                session_path.rmdir()
                logger.info(f"Removed empty session directory: {session_path}")
        else:
            # Clean all sessions for subject
            logger.info(
                f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} all sessions for {subject}"
            )

            if not dry_run and confirm_deletion(subject_path, force):
                shutil.rmtree(subject_path)
                deleted_count += 1
                logger.info(f"Deleted subject: {subject_path}")
    else:
        # Clean all derivatives (or specific modality across all subjects)
        if modality:
            logger.info(
                f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} all {modality} derivatives"
            )
        else:
            logger.info(
                f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} all preprocessing derivatives"
            )

        if modality:
            # Clean specific modality across all subjects/sessions
            for subject_dir in preprocessing_path.glob("sub-*"):
                for session_dir in subject_dir.glob("ses-*"):
                    modality_path = session_dir / modality
                    if modality_path.exists():
                        logger.info(
                            f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {modality_path}"
                        )

                        if not dry_run and confirm_deletion(modality_path, force):
                            shutil.rmtree(modality_path)
                            deleted_count += 1
                            logger.info(f"Deleted: {modality_path}")

                        # Clean up empty session directories
                        if (
                            not dry_run
                            and session_dir.exists()
                            and not any(session_dir.iterdir())
                        ):
                            session_dir.rmdir()
                            logger.info(f"Removed empty session: {session_dir}")

                # Clean up empty subject directories
                if (
                    not dry_run
                    and subject_dir.exists()
                    and not any(subject_dir.iterdir())
                ):
                    subject_dir.rmdir()
                    logger.info(f"Removed empty subject: {subject_dir}")
        else:
            # Clean entire preprocessing directory
            if not dry_run and confirm_deletion(preprocessing_path, force):
                shutil.rmtree(preprocessing_path)
                deleted_count += 1
                logger.info(
                    f"Deleted all preprocessing derivatives: {preprocessing_path}"
                )

    return deleted_count


def clean_logs(config: ConfigLoader, force: bool = False, dry_run: bool = False) -> int:
    """
    Clean log files.

    Args:
        config: Configuration loader instance
        force: Skip confirmation prompts
        dry_run: Show what would be deleted without actually deleting

    Returns:
        Number of log files deleted
    """
    log_path = Path(config.get("paths.logs", "log"))

    if not log_path.exists():
        logger.info(f"No log directory found at {log_path}")
        return 0

    log_files = list(log_path.glob("*.log*"))

    if not log_files:
        logger.info("No log files found")
        return 0

    logger.info(f"Found {len(log_files)} log file(s)")

    deleted_count = 0

    if dry_run:
        logger.info("[DRY RUN] Would delete the following log files:")
        for log_file in log_files:
            logger.info(f"  - {log_file}")
        return len(log_files)

    if not force:
        response = (
            input(f"Delete {len(log_files)} log file(s)? [y/N]: ").strip().lower()
        )
        if response not in ["y", "yes"]:
            logger.info("Log deletion cancelled")
            return 0

    for log_file in log_files:
        try:
            log_file.unlink()
            deleted_count += 1
            logger.debug(f"Deleted log file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to delete {log_file}: {e}")

    logger.info(f"Deleted {deleted_count} log file(s)")
    return deleted_count


def clean_cache(force: bool = False, dry_run: bool = False) -> int:
    """
    Clean Python cache files (__pycache__ directories).

    Args:
        force: Skip confirmation prompts
        dry_run: Show what would be deleted without actually deleting

    Returns:
        Number of cache directories deleted
    """
    project_root = Path(__file__).parent.parent
    cache_dirs = list(project_root.rglob("__pycache__"))

    if not cache_dirs:
        logger.info("No cache directories found")
        return 0

    logger.info(f"Found {len(cache_dirs)} cache director(y/ies)")

    if dry_run:
        logger.info("[DRY RUN] Would delete the following cache directories:")
        for cache_dir in cache_dirs:
            logger.info(f"  - {cache_dir}")
        return len(cache_dirs)

    if not force:
        response = (
            input(f"Delete {len(cache_dirs)} cache director(y/ies)? [y/N]: ")
            .strip()
            .lower()
        )
        if response not in ["y", "yes"]:
            logger.info("Cache deletion cancelled")
            return 0

    deleted_count = 0
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            deleted_count += 1
            logger.debug(f"Deleted cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to delete {cache_dir}: {e}")

    logger.info(f"Deleted {deleted_count} cache director(y/ies)")
    return deleted_count


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Clean physiological signal processing outputs for fresh pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be deleted
  python scripts/clean_outputs.py --dry-run
  
  # Clean all derivatives and logs (with confirmation)
  python scripts/clean_outputs.py --derivatives --logs
  
  # Clean specific modality
  python scripts/clean_outputs.py --derivatives --modality hr
  
  # Clean specific subject/session
  python scripts/clean_outputs.py --derivatives --subject g01p01 --session 01
  
  # Force clean without confirmation
  python scripts/clean_outputs.py --all --force
  
  # Clean only Python cache files
  python scripts/clean_outputs.py --cache
        """,
    )

    # What to clean
    parser.add_argument(
        "--derivatives", "-d", action="store_true", help="Clean processing derivatives"
    )

    parser.add_argument(
        "--modality",
        "-m",
        choices=["bvp", "eda", "hr"],
        help="Clean specific modality only (bvp, eda, or hr)",
    )

    parser.add_argument("--logs", "-l", action="store_true", help="Clean log files")

    parser.add_argument(
        "--cache",
        "-c",
        action="store_true",
        help="Clean Python cache files (__pycache__)",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Clean everything (derivatives, logs, and cache)",
    )

    # Scope filters
    parser.add_argument(
        "--subject",
        "-s",
        help='Clean specific subject only (for derivatives). Can be with or without "sub-" prefix.',
    )

    parser.add_argument(
        "--session",
        "-e",
        help='Clean specific session only (requires --subject). Can be with or without "ses-" prefix.',
    )

    # Options
    parser.add_argument(
        "--config",
        "-cfg",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--force", "-f", action="store_true", help="Skip confirmation prompts"
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel("DEBUG")

    # Validate arguments
    if args.session and not args.subject:
        parser.error("--session requires --subject")

    if args.modality and not args.derivatives:
        parser.error("--modality requires --derivatives")

    if not any([args.derivatives, args.logs, args.cache, args.all]):
        parser.error(
            "Must specify at least one of: --derivatives, --logs, --cache, or --all"
        )

    # Load configuration
    try:
        config = ConfigLoader(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Perform cleaning
    total_deleted = 0

    try:
        if args.dry_run:
            logger.info("=== DRY RUN MODE - No files will be deleted ===")

        # Clean derivatives
        if args.all or args.derivatives:
            logger.info("Cleaning preprocessing derivatives...")
            count = clean_derivatives(
                config,
                modality=args.modality,
                subject=args.subject,
                session=args.session,
                force=args.force,
                dry_run=args.dry_run,
            )
            total_deleted += count

        # Clean logs
        if args.all or args.logs:
            logger.info("Cleaning log files...")
            count = clean_logs(config, force=args.force, dry_run=args.dry_run)
            total_deleted += count

        # Clean cache
        if args.all or args.cache:
            logger.info("Cleaning Python cache...")
            count = clean_cache(force=args.force, dry_run=args.dry_run)
            total_deleted += count

        # Summary
        if args.dry_run:
            logger.info(
                f"=== DRY RUN COMPLETE: Would delete {total_deleted} item(s) ==="
            )
        else:
            logger.info(f"=== CLEANING COMPLETE: Deleted {total_deleted} item(s) ===")

        return 0

    except KeyboardInterrupt:
        logger.warning("Cleaning interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Cleaning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
