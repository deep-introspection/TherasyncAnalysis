#!/usr/bin/env python
"""
Generate JSON sidecars for MOI (Moments of Interest) annotation files.

This script creates JSON sidecars for each alliance annotations TSV file
in the sourcedata directory. It retrieves the session duration from the
corresponding physiological data (participant 01 of the same family/session).

Usage:
    python scripts/batch/generate_moi_sidecars.py
    python scripts/batch/generate_moi_sidecars.py --dry-run

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_session_duration(
    family_id: str, session_id: str, data_root: Path
) -> Optional[float]:
    """
    Get session duration from physiological data of participant 01.

    Args:
        family_id: Family ID (e.g., 'g01')
        session_id: Session ID (e.g., '01')
        data_root: Root directory of raw data

    Returns:
        Session duration in seconds, or None if not found
    """
    # Construct path to participant 01's therapy BVP file (highest sampling rate)
    subject_id = f"{family_id}p01"
    bvp_file = (
        data_root
        / f"sub-{subject_id}"
        / f"ses-{session_id}"
        / "physio"
        / f"sub-{subject_id}_ses-{session_id}_task-therapy_recording-bvp.tsv"
    )

    if not bvp_file.exists():
        logger.warning(
            f"BVP file not found for {subject_id}/ses-{session_id}: {bvp_file}"
        )
        return None

    try:
        # Read last line to get final timestamp
        df = pd.read_csv(bvp_file, sep="\t")
        if len(df) == 0:
            logger.warning(f"Empty BVP file: {bvp_file}")
            return None

        # Duration is the last timestamp (assuming starts at 0)
        duration = float(df.iloc[-1, 0])  # First column is time
        logger.info(
            f"Found duration for {family_id}/ses-{session_id}: {duration:.2f}s ({duration / 60:.1f}min)"
        )
        return duration

    except Exception as e:
        logger.error(f"Error reading BVP file {bvp_file}: {e}")
        return None


def create_moi_sidecar(
    tsv_file: Path, family_id: str, session_id: str, duration: float
) -> Dict:
    """
    Create sidecar JSON content for MOI annotation file.

    Args:
        tsv_file: Path to the MOI TSV file
        family_id: Family ID (e.g., 'g01')
        session_id: Session ID (e.g., '01')
        duration: Session duration in seconds

    Returns:
        Dictionary with sidecar metadata
    """
    sidecar = {
        "Description": "Alliance and emotion annotations during therapy session",
        "AnnotationType": "moments_of_interest",
        "TaskName": "therapy",
        "GroupID": family_id,
        "SessionID": session_id,
        "Duration": duration,
        "DurationUnit": "seconds",
        "Columns": ["start", "end", "source", "target", "alliance", "emotion"],
        "ColumnDescriptions": {
            "start": "Start timestamp of the interaction (HH:MM:SS)",
            "end": "End timestamp of the interaction (HH:MM:SS)",
            "source": "Person(s) initiating the interaction",
            "target": "Person(s) receiving the interaction",
            "alliance": "Alliance rating: -1 (negative), 0 (neutral), 1 (positive), or empty",
            "emotion": "Emotion rating: -1 (negative), 0 (neutral), 1 (positive), or empty",
        },
        "AllianceScale": {
            "-1": "Negative alliance",
            "0": "Neutral alliance",
            "1": "Positive alliance",
        },
        "EmotionScale": {
            "-1": "Negative emotion",
            "0": "Neutral emotion",
            "1": "Positive emotion",
        },
        "Participants": [
            "Therapist (Thérapeute)",
            "Mother (Mère)",
            "Father (Père)",
            "Daughter (Fille)",
            "Son (Garçon)",
        ],
        "Notes": "Multiple participants can be listed in source/target fields",
    }

    return sidecar


def process_moi_files(
    data_root: Path, dry_run: bool = False, force: bool = False
) -> Dict[str, int]:
    """
    Process all MOI annotation files and create sidecars.

    Args:
        data_root: Root directory of raw data
        dry_run: If True, only show what would be done
        force: If True, overwrite existing sidecars

    Returns:
        Dictionary with processing statistics
    """
    stats = {"found": 0, "created": 0, "skipped": 0, "errors": 0}

    sourcedata_dir = data_root / "sourcedata"
    if not sourcedata_dir.exists():
        logger.error(f"Sourcedata directory not found: {sourcedata_dir}")
        return stats

    # Find all MOI TSV files
    moi_files = list(
        sourcedata_dir.glob("*/*/moi_tables/*_desc-alliance_annotations.tsv")
    )
    stats["found"] = len(moi_files)

    logger.info(f"Found {stats['found']} MOI annotation files")

    for tsv_file in sorted(moi_files):
        # Parse family and session from path
        # Path structure: sourcedata/sub-g01shared/ses-01/moi_tables/sub-g01_ses-01_desc-alliance_annotations.tsv
        parts = tsv_file.parts
        family_shared = parts[-4]  # sub-g01shared
        session_dir = parts[-3]  # ses-01

        family_id = family_shared.replace("sub-", "").replace("shared", "")  # g01
        session_id = session_dir.replace("ses-", "")  # 01

        logger.info(f"\nProcessing: {tsv_file.name}")
        logger.info(f"  Family: {family_id}, Session: {session_id}")

        # Check if sidecar already exists
        json_file = tsv_file.with_suffix(".json")
        if json_file.exists() and not dry_run and not force:
            logger.info(f"  Sidecar already exists: {json_file.name}")
            stats["skipped"] += 1
            continue

        # Get session duration
        duration = get_session_duration(family_id, session_id, data_root)
        if duration is None:
            logger.error("  Could not determine duration, skipping")
            stats["errors"] += 1
            continue

        # Create sidecar content
        sidecar = create_moi_sidecar(tsv_file, family_id, session_id, duration)

        if dry_run:
            logger.info(f"  [DRY RUN] Would create: {json_file.name}")
            logger.info(f"  Duration: {duration:.2f}s ({duration / 60:.1f}min)")
        else:
            # Write JSON file
            try:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=4, ensure_ascii=False)
                logger.info(f"  ✓ Created: {json_file.name}")
                stats["created"] += 1
            except Exception as e:
                logger.error(f"  Error writing {json_file}: {e}")
                stats["errors"] += 1

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate JSON sidecars for MOI annotation files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating files",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing sidecar files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory of raw data (default: data/raw)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("MOI Sidecar Generation Script")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be created")

    if args.force:
        logger.info("FORCE MODE - Overwriting existing sidecars")

    # Process files
    stats = process_moi_files(args.data_root, args.dry_run, args.force)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary:")
    logger.info(f"  Files found:    {stats['found']}")
    logger.info(f"  Sidecars created: {stats['created']}")
    logger.info(f"  Skipped (exist):  {stats['skipped']}")
    logger.info(f"  Errors:          {stats['errors']}")
    logger.info("=" * 80)

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
