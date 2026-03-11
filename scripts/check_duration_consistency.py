#!/usr/bin/env python3
"""
Check duration consistency between MOI annotations and physiological data.

This script verifies that physiological recordings are long enough to cover
all MOI annotations timestamps.

Author: Remy Ramadour
Date: 2025-11-19
"""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS timestamp to seconds."""
    parts = ts.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def get_moi_max_timestamp(moi_file: Path) -> float:
    """Get maximum timestamp from MOI annotation file."""
    df = pd.read_csv(moi_file, sep="\t")

    # Get max from both start and end columns
    max_start = df["start"].apply(parse_timestamp).max()
    max_end = df["end"].apply(parse_timestamp).max()

    return max(max_start, max_end)


def get_physio_duration(
    subject: str, session: str, data_root: Path
) -> Dict[str, float]:
    """Get physiological recording durations for all participants in a group."""
    durations = {}

    # Extract group ID from subject (e.g., 'g01' from 'g01shared')
    group_id = subject.replace("shared", "").replace("sub-", "")

    # Find all participants for this group/session
    participant_pattern = f"sub-{group_id}p*"
    participant_dirs = list(data_root.glob(participant_pattern))

    for participant_dir in participant_dirs:
        participant_id = participant_dir.name.replace("sub-", "")
        session_dir = participant_dir / session / "physio"

        if not session_dir.exists():
            continue

        # Look for BVP TSV files (prefer therapy task, fallback to restingstate)
        tsv_files = list(session_dir.glob("*_task-therapy_recording-bvp.tsv"))
        if not tsv_files:
            tsv_files = list(session_dir.glob("*_task-restingstate_recording-bvp.tsv"))

        if tsv_files:
            # Calculate duration from TSV file (max time value)
            tsv_file = tsv_files[0]
            try:
                df = pd.read_csv(tsv_file, sep="\t")
                if "time" in df.columns:
                    duration = df["time"].max()
                    durations[participant_id] = duration
            except Exception as e:
                logger.debug(f"Could not read {tsv_file}: {e}")
                continue

    return durations


def check_all_sessions(data_root: Path) -> List[Dict]:
    """Check all sessions for duration consistency."""
    results = []

    # Find all MOI annotation files in raw/sourcedata
    sourcedata_dir = data_root / "raw" / "sourcedata"
    moi_files = list(
        sourcedata_dir.glob(
            "sub-*shared/ses-*/moi_tables/*_desc-alliance_annotations.tsv"
        )
    )

    logger.info(f"Found {len(moi_files)} MOI annotation files")

    for moi_file in moi_files:
        # Parse path to extract group and session
        parts = moi_file.parts
        subject = [p for p in parts if p.startswith("sub-")][0]
        session = [p for p in parts if p.startswith("ses-")][0]

        try:
            # Get MOI max timestamp
            moi_max_ts = get_moi_max_timestamp(moi_file)

            # Get physiological durations (look in data/raw, not data/raw/raw)
            physio_durations = get_physio_duration(subject, session, data_root / "raw")

            if not physio_durations:
                logger.warning(f"{subject}/{session}: No physiological data found")
                results.append(
                    {
                        "subject": subject,
                        "session": session,
                        "moi_max_ts": moi_max_ts,
                        "physio_durations": {},
                        "status": "NO_PHYSIO_DATA",
                    }
                )
                continue

            # Check if all participants have sufficient duration
            all_sufficient = True
            insufficient_participants = []

            for participant_id, physio_duration in physio_durations.items():
                if physio_duration < moi_max_ts:
                    all_sufficient = False
                    insufficient_participants.append(participant_id)
                    logger.error(
                        f"{subject}/{session}/{participant_id}: "
                        f"INSUFFICIENT - MOI max: {moi_max_ts:.1f}s ({moi_max_ts / 60:.1f}min), "
                        f"Physio: {physio_duration:.1f}s ({physio_duration / 60:.1f}min), "
                        f"Gap: {moi_max_ts - physio_duration:.1f}s"
                    )

            if all_sufficient:
                logger.info(
                    f"{subject}/{session}: OK - MOI max: {moi_max_ts:.1f}s, "
                    f"Physio range: [{min(physio_durations.values()):.1f}s - {max(physio_durations.values()):.1f}s]"
                )
                status = "OK"
            else:
                status = "INSUFFICIENT"

            results.append(
                {
                    "subject": subject,
                    "session": session,
                    "moi_max_ts": moi_max_ts,
                    "physio_durations": physio_durations,
                    "insufficient_participants": insufficient_participants,
                    "status": status,
                }
            )

        except Exception as e:
            logger.error(f"{subject}/{session}: Error - {e}")
            results.append(
                {
                    "subject": subject,
                    "session": session,
                    "status": "ERROR",
                    "error": str(e),
                }
            )

    return results


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    total = len(results)
    ok_count = sum(1 for r in results if r["status"] == "OK")
    insufficient_count = sum(1 for r in results if r["status"] == "INSUFFICIENT")
    no_data_count = sum(1 for r in results if r["status"] == "NO_PHYSIO_DATA")
    error_count = sum(1 for r in results if r["status"] == "ERROR")

    logger.info(f"Total sessions checked: {total}")
    logger.info(f"✓ OK (physio >= MOI): {ok_count}")
    logger.info(f"✗ INSUFFICIENT (physio < MOI): {insufficient_count}")
    logger.info(f"⚠ No physio data: {no_data_count}")
    logger.info(f"⚠ Errors: {error_count}")

    if insufficient_count > 0:
        logger.info("\nSessions with insufficient physiological data:")
        for result in results:
            if result["status"] == "INSUFFICIENT":
                logger.info(
                    f"  - {result['subject']}/{result['session']}: "
                    f"MOI max: {result['moi_max_ts']:.1f}s, "
                    f"Insufficient participants: {', '.join(result['insufficient_participants'])}"
                )


def main():
    """Main entry point."""
    # Get data root from script location
    script_dir = Path(__file__).parent
    data_root = script_dir.parent / "data"

    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        sys.exit(1)

    logger.info(f"Checking duration consistency in: {data_root}")
    logger.info("=" * 80)

    results = check_all_sessions(data_root)
    print_summary(results)

    # Exit with error code if any issues found
    if any(r["status"] in ["INSUFFICIENT", "ERROR"] for r in results):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
