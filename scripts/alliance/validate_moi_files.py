#!/usr/bin/env python
"""
Validate MOI (Moments of Interest) Annotation Files.

This script validates the format and content of MOI alliance/emotion annotation
files, detecting potential issues before processing.

Note: Overlapping time intervals are NOT flagged as errors - they represent
"split alliance" situations where different alliance states occur simultaneously
between different participant pairs.

Usage:
    python scripts/alliance/validate_moi_files.py --all
    python scripts/alliance/validate_moi_files.py --group g01 --session 01

Authors: Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class MOIValidator:
    """Validates MOI annotation files for format and content issues."""

    REQUIRED_COLUMNS = ["start", "end", "source", "target", "alliance", "emotion"]
    VALID_ALLIANCE_VALUES = [-1, 0, 1, "", None]
    VALID_EMOTION_VALUES = [-1, 0, 1, "", None]
    TIMESTAMP_FORMAT = r"^\d{2}:\d{2}:\d{2}$"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize validator."""
        self.config = ConfigLoader(config_path).config
        self.rawdata_path = Path(self.config["paths"]["rawdata"])
        self.sourcedata_path = self.rawdata_path / "sourcedata"

    def validate_file(self, group_id: str, session_id: str) -> Dict:
        """
        Validate a single MOI annotation file.

        Args:
            group_id: Group ID (e.g., 'g01')
            session_id: Session ID (e.g., '01')

        Returns:
            Dictionary with validation results:
                - valid: bool
                - errors: list of error messages
                - warnings: list of warning messages
                - stats: dict with file statistics
        """
        result = {
            "group_id": group_id,
            "session_id": session_id,
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        # Construct paths
        subject_dir = f"sub-{group_id}shared"
        session_dir = f"ses-{session_id}"

        tsv_filename = f"sub-{group_id}_ses-{session_id}_desc-alliance_annotations.tsv"
        json_filename = (
            f"sub-{group_id}_ses-{session_id}_desc-alliance_annotations.json"
        )

        moi_dir = self.sourcedata_path / subject_dir / session_dir / "moi_tables"
        tsv_file = moi_dir / tsv_filename
        json_file = moi_dir / json_filename

        # Check files exist
        if not tsv_file.exists():
            result["errors"].append(f"TSV file not found: {tsv_file}")
            result["valid"] = False
            return result

        if not json_file.exists():
            result["warnings"].append(f"JSON sidecar not found: {json_file}")

        # Load and validate TSV
        try:
            df = pd.read_csv(tsv_file, sep="\t")
        except Exception as e:
            result["errors"].append(f"Failed to parse TSV: {e}")
            result["valid"] = False
            return result

        result["stats"]["n_annotations"] = len(df)

        # Validate columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            result["errors"].append(f"Missing columns: {missing_cols}")
            result["valid"] = False

        extra_cols = set(df.columns) - set(self.REQUIRED_COLUMNS)
        if extra_cols:
            result["warnings"].append(f"Extra columns found: {extra_cols}")

        # Validate timestamps
        timestamp_errors = self._validate_timestamps(df)
        result["errors"].extend(timestamp_errors)
        if timestamp_errors:
            result["valid"] = False

        # Validate alliance/emotion values
        value_errors = self._validate_values(df)
        result["errors"].extend(value_errors)
        if value_errors:
            result["valid"] = False

        # Check for newlines in fields (common data entry issue)
        newline_issues = self._check_newlines(df)
        if newline_issues:
            result["errors"].extend(newline_issues)
            result["valid"] = False

        # Validate JSON sidecar if present
        if json_file.exists():
            json_errors, json_warnings = self._validate_json(json_file, df)
            result["errors"].extend(json_errors)
            result["warnings"].extend(json_warnings)
            if json_errors:
                result["valid"] = False

        # Collect statistics
        result["stats"].update(self._compute_stats(df))

        return result

    def _validate_timestamps(self, df: pd.DataFrame) -> List[str]:
        """Validate timestamp format and logic."""
        errors = []
        import re

        for idx, row in df.iterrows():
            # Check format
            for col in ["start", "end"]:
                if col in df.columns:
                    val = str(row[col])
                    if not re.match(self.TIMESTAMP_FORMAT, val):
                        errors.append(
                            f"Row {idx + 2}: Invalid {col} timestamp format: '{val}'"
                        )

            # Check start < end
            if "start" in df.columns and "end" in df.columns:
                start = self._timestamp_to_seconds(str(row["start"]))
                end = self._timestamp_to_seconds(str(row["end"]))
                if start > end:
                    errors.append(
                        f"Row {idx + 2}: start ({row['start']}) > end ({row['end']})"
                    )

        return errors

    def _validate_values(self, df: pd.DataFrame) -> List[str]:
        """Validate alliance and emotion values."""
        errors = []

        for idx, row in df.iterrows():
            # Alliance validation
            if "alliance" in df.columns:
                val = row["alliance"]
                if pd.notna(val) and val != "":
                    try:
                        int_val = int(val)
                        if int_val not in [-1, 0, 1]:
                            errors.append(
                                f"Row {idx + 2}: Invalid alliance value: {val}"
                            )
                    except (ValueError, TypeError):
                        errors.append(
                            f"Row {idx + 2}: Non-numeric alliance value: {val}"
                        )

            # Emotion validation
            if "emotion" in df.columns:
                val = row["emotion"]
                if pd.notna(val) and val != "":
                    try:
                        int_val = int(val)
                        if int_val not in [-1, 0, 1]:
                            errors.append(
                                f"Row {idx + 2}: Invalid emotion value: {val}"
                            )
                    except (ValueError, TypeError):
                        errors.append(
                            f"Row {idx + 2}: Non-numeric emotion value: {val}"
                        )

            # Check that alliance has source AND target (emotion only needs source)
            if (
                "alliance" in df.columns
                and pd.notna(row.get("alliance"))
                and row.get("alliance") != ""
            ):
                if pd.isna(row.get("source")) or row.get("source") == "":
                    errors.append(f"Row {idx + 2}: Alliance annotation missing source")
                if pd.isna(row.get("target")) or row.get("target") == "":
                    # Only warning - might be intentional
                    pass  # Target can be empty for emotion-only rows

        return errors

    def _check_newlines(self, df: pd.DataFrame) -> List[str]:
        """Check for newlines in text fields."""
        errors = []

        for col in ["source", "target"]:
            if col in df.columns:
                for idx, val in df[col].items():
                    if pd.notna(val) and "\n" in str(val):
                        errors.append(
                            f"Row {idx + 2}: Newline found in {col}: '{val[:30]}...'"
                        )

        return errors

    def _validate_json(
        self, json_file: Path, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Validate JSON sidecar."""
        errors = []
        warnings = []

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            errors.append(f"Failed to parse JSON: {e}")
            return errors, warnings

        # Check required fields
        required_fields = ["Duration", "GroupID", "SessionID"]
        for field in required_fields:
            if field not in metadata:
                warnings.append(f"JSON missing recommended field: {field}")

        # Check duration makes sense
        if "Duration" in metadata:
            duration = metadata["Duration"]
            if duration <= 0:
                errors.append(f"Invalid duration in JSON: {duration}")

            # Check if last annotation is within session duration
            if "end" in df.columns and len(df) > 0:
                max_end = (
                    df["end"].apply(lambda x: self._timestamp_to_seconds(str(x))).max()
                )
                if max_end > duration:
                    warnings.append(
                        f"Annotation end time ({max_end:.0f}s) exceeds "
                        f"session duration ({duration:.0f}s)"
                    )

        return errors, warnings

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistics about the annotations."""
        stats = {}

        # Alliance distribution
        if "alliance" in df.columns:
            alliance_counts = df["alliance"].value_counts(dropna=False)
            stats["alliance_positive"] = int(alliance_counts.get(1, 0))
            stats["alliance_neutral"] = int(alliance_counts.get(0, 0))
            stats["alliance_negative"] = int(alliance_counts.get(-1, 0))
            stats["alliance_empty"] = int(df["alliance"].isna().sum())

        # Emotion distribution
        if "emotion" in df.columns:
            emotion_counts = df["emotion"].value_counts(dropna=False)
            stats["emotion_positive"] = int(emotion_counts.get(1, 0))
            stats["emotion_neutral"] = int(emotion_counts.get(0, 0))
            stats["emotion_negative"] = int(emotion_counts.get(-1, 0))
            stats["emotion_empty"] = int(df["emotion"].isna().sum())

        # Unique participants
        participants = set()
        for col in ["source", "target"]:
            if col in df.columns:
                for val in df[col].dropna():
                    # Split by common delimiters
                    for p in str(val).replace(",", " ").split():
                        if p and p not in ['"', "'", "(", ")"]:
                            participants.add(p)
        stats["unique_participants"] = len(participants)

        return stats

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert HH:MM:SS to seconds."""
        if pd.isna(timestamp) or timestamp == "":
            return 0.0

        parts = timestamp.strip().split(":")
        if len(parts) != 3:
            return 0.0

        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            return 0.0

    def get_available_sessions(self) -> List[Tuple[str, str]]:
        """Get list of available MOI annotation sessions."""
        sessions = []

        if not self.sourcedata_path.exists():
            return sessions

        tsv_files = list(
            self.sourcedata_path.glob("*/*/moi_tables/*_desc-alliance_annotations.tsv")
        )

        for tsv_file in tsv_files:
            filename = tsv_file.stem
            parts = filename.split("_")

            if len(parts) >= 2:
                group_part = parts[0].replace("sub-", "")
                session_part = parts[1].replace("ses-", "")
                sessions.append((group_part, session_part))

        return sorted(sessions)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate MOI annotation files")
    parser.add_argument("--group", "-g", type=str, help="Group ID (e.g., 'g01')")
    parser.add_argument("--session", "-s", type=str, help="Session ID (e.g., '01')")
    parser.add_argument(
        "--all", action="store_true", help="Validate all available sessions"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("MOI Annotation File Validator")
    logger.info("=" * 80)

    validator = MOIValidator()

    # Determine sessions to validate
    if args.all:
        sessions = validator.get_available_sessions()
        logger.info(f"\nFound {len(sessions)} sessions to validate")
    elif args.group and args.session:
        sessions = [(args.group, args.session)]
    else:
        logger.error("Error: Must specify either --all or both --group and --session")
        return 1

    # Validate sessions
    all_valid = True
    total_errors = 0
    total_warnings = 0

    for group_id, session_id in sessions:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Validating: {group_id}/ses-{session_id}")
        logger.info(f"{'=' * 60}")

        result = validator.validate_file(group_id, session_id)

        if result["valid"]:
            logger.info("✓ VALID")
        else:
            logger.error("✗ INVALID")
            all_valid = False

        # Show errors
        for error in result["errors"]:
            logger.error(f"  ERROR: {error}")
            total_errors += 1

        # Show warnings
        for warning in result["warnings"]:
            logger.warning(f"  WARNING: {warning}")
            total_warnings += 1

        # Show stats
        if result["stats"]:
            logger.info(f"  Stats: {result['stats']['n_annotations']} annotations")
            if "alliance_positive" in result["stats"]:
                logger.info(
                    f"    Alliance: +{result['stats']['alliance_positive']} / "
                    f"0:{result['stats']['alliance_neutral']} / "
                    f"-{result['stats']['alliance_negative']}"
                )
            if "emotion_positive" in result["stats"]:
                logger.info(
                    f"    Emotion:  +{result['stats']['emotion_positive']} / "
                    f"0:{result['stats']['emotion_neutral']} / "
                    f"-{result['stats']['emotion_negative']}"
                )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Sessions validated: {len(sessions)}")
    logger.info(f"Total errors:       {total_errors}")
    logger.info(f"Total warnings:     {total_warnings}")
    logger.info(
        f"Overall status:     {'✓ ALL VALID' if all_valid else '✗ SOME INVALID'}"
    )

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
