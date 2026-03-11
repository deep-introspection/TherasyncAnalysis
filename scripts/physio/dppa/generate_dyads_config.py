#!/usr/bin/env python3
"""
Generate DPPA Dyads Configuration from participants.tsv.

This script automatically generates the dppa_dyads_real.yaml configuration
file by reading participant information from the BIDS participants.tsv file
and detecting available sessions from the preprocessing derivatives.

Usage:
    python scripts/physio/dppa/generate_dyads_config.py
    python scripts/physio/dppa/generate_dyads_config.py --output config/dppa_dyads_real.yaml

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from itertools import combinations
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import yaml

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_participants(participants_file: Path) -> pd.DataFrame:
    """
    Load participants from BIDS participants.tsv.

    Args:
        participants_file: Path to participants.tsv

    Returns:
        DataFrame with participant_id, family_id, role, device_id
    """
    if not participants_file.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_file}")

    df = pd.read_csv(participants_file, sep="\t")
    logger.info(f"Loaded {len(df)} participants from {participants_file}")
    return df


def find_available_sessions(derivatives_dir: Path) -> dict:
    """
    Find available sessions for each participant in preprocessing derivatives.

    Args:
        derivatives_dir: Path to derivatives/preprocessing directory

    Returns:
        Dict mapping participant_id -> list of session IDs
    """
    sessions_per_subject = {}

    for sub_dir in sorted(derivatives_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        subject = sub_dir.name  # Keep 'sub-' prefix
        sessions = sorted([s.name for s in sub_dir.glob("ses-*") if s.is_dir()])

        if sessions:
            sessions_per_subject[subject] = sessions

    logger.info(f"Found sessions for {len(sessions_per_subject)} participants")
    return sessions_per_subject


def build_dyads_config(
    participants: pd.DataFrame,
    sessions_per_subject: dict,
    intra_session_method: str = "sliding_duration30s_step5s",
    inter_session_methods: dict = None,
    tasks: list = None,
) -> dict:
    """
    Build the complete dyads configuration.

    Args:
        participants: DataFrame with participant info
        sessions_per_subject: Dict mapping subject -> sessions
        intra_session_method: Epoching method for intra-session
        inter_session_methods: Dict mapping task -> method (e.g., {'restingstate': 'nsplit1', 'therapy': 'nsplit120'})
        tasks: List of task names to include

    Returns:
        Complete configuration dictionary
    """
    if inter_session_methods is None:
        inter_session_methods = {"restingstate": "nsplit1", "therapy": "nsplit120"}
    if tasks is None:
        tasks = list(inter_session_methods.keys())

    config = {
        "description": "DPPA Dyad Configuration - Auto-generated from participants.tsv",
        "generated_from": "data/raw/participants.tsv",
        "generated_at": datetime.now().isoformat(),
        # Roles reference
        "roles": {
            "therapist": "The therapist conducting the session",
            "mother": "Mother of the family",
            "father": "Father of the family",
            "child1": "First child",
            "child2": "Second child",
            "child3": "Third child",
        },
        # Epoching methods - now per-task for inter_session
        "epoching": {
            "intra_session": {
                "method": intra_session_method,
                "description": "Sliding window for comparing dyads within same session (same epoch count)",
                "tasks": tasks,
            },
            "inter_session": {
                "methods": inter_session_methods,
                "description": "Per-task epoching for comparing across all sessions",
                "tasks": tasks,
            },
        },
        # Families
        "families": {},
        # Real dyads
        "real_dyads": [],
    }

    # Build families and real dyads
    for family_id, group in participants.groupby("family_id"):
        family_data = {"therapist": None, "patients": {}, "sessions": {}}

        # Get therapist and patients
        for _, row in group.iterrows():
            subject = row["participant_id"]  # Already has 'sub-' prefix
            role = row["role"]

            if role == "therapist":
                family_data["therapist"] = subject
            else:
                family_data["patients"][role] = subject

        # Find sessions where family members were present
        family_members = [row["participant_id"] for _, row in group.iterrows()]

        all_sessions = set()
        for member in family_members:
            if member in sessions_per_subject:
                all_sessions.update(sessions_per_subject[member])

        # For each session, find who was present and generate dyads
        for session in sorted(all_sessions):
            present = [
                m
                for m in family_members
                if m in sessions_per_subject and session in sessions_per_subject[m]
            ]

            if len(present) >= 2:  # Need at least 2 for a dyad
                family_data["sessions"][session] = present

                # Generate all real dyad pairs for this session
                for p1, p2 in combinations(present, 2):
                    # Get roles
                    role1 = participants[participants["participant_id"] == p1][
                        "role"
                    ].values[0]
                    role2 = participants[participants["participant_id"] == p2][
                        "role"
                    ].values[0]

                    # Determine dyad type
                    if role1 == "therapist" or role2 == "therapist":
                        dyad_type = "therapist-patient"
                    else:
                        dyad_type = "patient-patient"

                    dyad = {
                        "dyad_id": f"{family_id}_{p1.replace('sub-', '')}_{p2.replace('sub-', '')}_{session}",
                        "family": family_id,
                        "subject1": p1,
                        "role1": role1,
                        "subject2": p2,
                        "role2": role2,
                        "session": session,
                        "dyad_type": dyad_type,
                    }
                    config["real_dyads"].append(dyad)

        config["families"][family_id] = family_data

    return config


def print_statistics(config: dict):
    """Print statistics about the generated configuration."""
    real_dyads = config["real_dyads"]

    n_therapist_patient = sum(
        1 for d in real_dyads if d["dyad_type"] == "therapist-patient"
    )
    n_patient_patient = sum(
        1 for d in real_dyads if d["dyad_type"] == "patient-patient"
    )

    print("\n" + "=" * 60)
    print("DPPA DYADS CONFIGURATION GENERATED")
    print("=" * 60)
    print(f"\nTotal real dyads: {len(real_dyads)}")
    print(f"  - Therapist-Patient: {n_therapist_patient}")
    print(f"  - Patient-Patient: {n_patient_patient}")
    print(f"\nFamilies: {len(config['families'])}")

    print("\nDyads per family:")
    for family_id in sorted(config["families"].keys()):
        family_dyads = [d for d in real_dyads if d["family"] == family_id]
        sessions = set(d["session"] for d in family_dyads)
        print(
            f"  {family_id}: {len(family_dyads)} dyads across {len(sessions)} sessions"
        )

    print("\nEpoching methods:")
    print(f"  Intra-session: {config['epoching']['intra_session']['method']}")
    print("  Inter-session (per task):")
    for task, method in config["epoching"]["inter_session"]["methods"].items():
        print(f"    - {task}: {method}")
    print("=" * 60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate DPPA dyads configuration from participants.tsv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with defaults
  python scripts/physio/dppa/generate_dyads_config.py
  
  # Specify output file
  python scripts/physio/dppa/generate_dyads_config.py --output config/my_dyads.yaml
  
  # Verbose mode
  python scripts/physio/dppa/generate_dyads_config.py -v
        """,
    )

    parser.add_argument(
        "--participants",
        type=str,
        default="data/raw/participants.tsv",
        help="Path to participants.tsv file",
    )
    parser.add_argument(
        "--derivatives",
        type=str,
        default="data/derivatives/preprocessing",
        help="Path to preprocessing derivatives directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="config/dppa_dyads_real.yaml",
        help="Output YAML file path",
    )
    parser.add_argument(
        "--intra-method",
        type=str,
        default="sliding_duration30s_step5s",
        help="Epoching method for intra-session comparison",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to main config.yaml to read epoching parameters",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load main config to get epoching parameters per task
    config_loader = ConfigLoader(args.config)
    nsplit_config = config_loader.get("epoching.methods.nsplit", {})

    # Build per-task inter-session methods from config.yaml
    inter_session_methods = {}
    tasks = []

    for task_name, task_params in nsplit_config.items():
        if task_name == "enabled":
            continue
        if isinstance(task_params, dict) and "n_epochs" in task_params:
            n_epochs = task_params["n_epochs"]
            inter_session_methods[task_name] = f"nsplit{n_epochs}"
            tasks.append(task_name)
            logger.info(
                f"Task '{task_name}': n_epochs={n_epochs} -> method=nsplit{n_epochs}"
            )

    if not inter_session_methods:
        logger.warning("No nsplit config found in config.yaml, using defaults")
        inter_session_methods = {"restingstate": "nsplit1", "therapy": "nsplit120"}
        tasks = ["restingstate", "therapy"]

    # Load data
    participants_file = Path(args.participants)
    derivatives_dir = Path(args.derivatives)
    output_file = Path(args.output)

    logger.info("Loading participant data...")
    participants = load_participants(participants_file)

    logger.info("Finding available sessions...")
    sessions_per_subject = find_available_sessions(derivatives_dir)

    logger.info("Building dyads configuration...")
    config = build_dyads_config(
        participants,
        sessions_per_subject,
        intra_session_method=args.intra_method,
        inter_session_methods=inter_session_methods,
        tasks=tasks,
    )

    # Save configuration
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    logger.info(f"Configuration saved to: {output_file}")

    # Print statistics
    print_statistics(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
