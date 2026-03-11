#!/usr/bin/env python3
"""
Count valid dyads and pseudo-dyads for alliance-ICD analysis.

This script identifies which dyads have valid MOI annotations
and can be used for alliance-ICD correlation analysis.

Only dyads where BOTH participants have MOI data are valid.
For pseudo-dyads (cross-family), both families must have MOI for the session.

Author: Therasync Team
Date: November 2025
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd


def discover_moi_sessions(alliance_dir: Path) -> dict[str, set[str]]:
    """
    Discover which families and sessions have MOI annotations.

    Args:
        alliance_dir: Path to alliance derivatives directory

    Returns:
        Dictionary mapping family ID to set of session IDs with MOI
    """
    sessions_with_moi = {}

    for family_dir in sorted(alliance_dir.glob("sub-*shared")):
        family_id = family_dir.name.replace("sub-", "").replace("shared", "")
        sessions = set()

        for session_dir in family_dir.glob("ses-*"):
            session_id = session_dir.name
            # Check if annotations exist
            annotations_dir = session_dir / "annotations"
            if annotations_dir.exists() and list(annotations_dir.glob("*.tsv")):
                sessions.add(session_id)

        if sessions:
            sessions_with_moi[family_id] = sessions

    return sessions_with_moi


def parse_intra_family_dyads(icd_path: Path) -> list[dict]:
    """
    Parse intra-family ICD file to extract dyad information.

    Args:
        icd_path: Path to intra-family ICD CSV file

    Returns:
        List of dyad info dicts with keys: column, p1, p2, family, session
    """
    df = pd.read_csv(icd_path, nrows=1)
    cols = [c for c in df.columns if c != "epoch_id"]

    dyads = []
    for col in cols:
        # Format: g01p02_vs_g01p01_ses-01
        parts = col.split("_vs_")
        p1 = parts[0]  # g01p02
        rest = parts[1]  # g01p01_ses-01
        p2_parts = rest.rsplit("_", 1)
        p2 = p2_parts[0]  # g01p01
        session = p2_parts[1]  # ses-01
        family = p1[:3]  # g01

        dyads.append(
            {
                "column": col,
                "p1": p1,
                "p2": p2,
                "family": family,
                "session": session,
                "type": "real",
            }
        )

    return dyads


def parse_inter_session_dyads(icd_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Parse inter-session ICD file to extract pseudo-dyad information.

    Args:
        icd_path: Path to inter-session ICD CSV file

    Returns:
        Tuple of (cross_family_dyads, same_family_dyads)
    """
    df = pd.read_csv(icd_path, nrows=1)
    cols = [c for c in df.columns if c != "epoch_id"]

    cross_family = []
    same_family = []

    for col in cols:
        # Format: g01p02_ses-01_vs_g03p02_ses-01
        parts = col.split("_vs_")
        p1_full = parts[0]  # g01p02_ses-01
        p2_full = parts[1]  # g03p02_ses-01

        p1_parts = p1_full.rsplit("_", 1)
        p1 = p1_parts[0]  # g01p02
        s1 = p1_parts[1]  # ses-01

        p2_parts = p2_full.rsplit("_", 1)
        p2 = p2_parts[0]  # g03p02
        s2 = p2_parts[1]  # ses-01

        f1 = p1[:3]  # g01
        f2 = p2[:3]  # g03

        dyad_info = {
            "column": col,
            "p1": p1,
            "p2": p2,
            "family1": f1,
            "family2": f2,
            "session1": s1,
            "session2": s2,
            "type": "pseudo" if f1 != f2 else "same_family_cross_session",
        }

        if f1 != f2:
            cross_family.append(dyad_info)
        else:
            same_family.append(dyad_info)

    return cross_family, same_family


def main():
    """Main function to count valid dyads."""
    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    alliance_dir = base_dir / "data" / "derivatives" / "alliance"
    dppa_dir = base_dir / "data" / "derivatives" / "dppa"

    # Note: intra-family uses sliding method, inter-session uses nsplit
    # For alliance analysis, we'll use the sliding method for real dyads
    intra_path = (
        dppa_dir
        / "intra_family"
        / "intra_family_icd_task-therapy_method-sliding_duration30s_step5s.csv"
    )
    inter_path = (
        dppa_dir
        / "inter_session"
        / "inter_session_icd_task-therapy_method-nsplit120.csv"
    )

    # Discover MOI sessions
    print("=" * 70)
    print("DISCOVERING MOI ANNOTATIONS")
    print("=" * 70)

    sessions_with_moi = discover_moi_sessions(alliance_dir)

    print(f"\nFamilies with MOI: {len(sessions_with_moi)}")
    total_sessions = 0
    for family, sessions in sorted(sessions_with_moi.items()):
        print(f"  {family}: {sorted(sessions)}")
        total_sessions += len(sessions)
    print(f"\nTotal sessions with MOI: {total_sessions}")

    # Parse real dyads
    print("\n" + "=" * 70)
    print("REAL DYADS (intra-family)")
    print("=" * 70)

    if not intra_path.exists():
        print(f"ERROR: File not found: {intra_path}")
        return

    real_dyads = parse_intra_family_dyads(intra_path)

    valid_real = []
    invalid_real = []

    for dyad in real_dyads:
        family = dyad["family"]
        session = dyad["session"]

        if family in sessions_with_moi and session in sessions_with_moi[family]:
            valid_real.append(dyad)
        else:
            invalid_real.append(dyad)

    print(f"\nTotal real dyad columns: {len(real_dyads)}")
    print(f"Valid (family+session with MOI): {len(valid_real)}")
    print(f"Invalid (no MOI): {len(invalid_real)}")

    # Group valid by family/session
    by_family_session = defaultdict(list)
    for dyad in valid_real:
        key = (dyad["family"], dyad["session"])
        by_family_session[key].append(dyad)

    print("\nValid real dyads by family/session:")
    for (family, session), dyads in sorted(by_family_session.items()):
        print(f"  {family} {session}: {len(dyads)} dyads")

    # Parse pseudo-dyads
    print("\n" + "=" * 70)
    print("PSEUDO-DYADS (cross-family)")
    print("=" * 70)

    if not inter_path.exists():
        print(f"ERROR: File not found: {inter_path}")
        return

    cross_family, same_family = parse_inter_session_dyads(inter_path)

    print(f"\nTotal inter-session columns: {len(cross_family) + len(same_family)}")
    print(f"  - Same family (different sessions): {len(same_family)}")
    print(f"  - Cross-family (pseudo-dyads): {len(cross_family)}")

    valid_pseudo = []
    invalid_pseudo = []

    for dyad in cross_family:
        f1, s1 = dyad["family1"], dyad["session1"]
        f2, s2 = dyad["family2"], dyad["session2"]

        f1_valid = f1 in sessions_with_moi and s1 in sessions_with_moi[f1]
        f2_valid = f2 in sessions_with_moi and s2 in sessions_with_moi[f2]

        if f1_valid and f2_valid:
            valid_pseudo.append(dyad)
        else:
            invalid_pseudo.append({**dyad, "f1_valid": f1_valid, "f2_valid": f2_valid})

    print("\nCross-family pseudo-dyads:")
    print(f"  - Valid (both families+sessions with MOI): {len(valid_pseudo)}")
    print(f"  - Invalid (at least one without MOI): {len(invalid_pseudo)}")

    # Group valid pseudo by family pair
    by_family_pair = defaultdict(list)
    for dyad in valid_pseudo:
        key = tuple(sorted([dyad["family1"], dyad["family2"]]))
        by_family_pair[key].append(dyad)

    print("\nValid pseudo-dyads by family pair:")
    for (f1, f2), dyads in sorted(by_family_pair.items()):
        print(f"  {f1} x {f2}: {len(dyads)} pseudo-dyads")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR ALLIANCE-ICD ANALYSIS")
    print("=" * 70)
    print(
        f"\nFamilies with MOI: {len(sessions_with_moi)} ({', '.join(sorted(sessions_with_moi.keys()))})"
    )
    print(f"Sessions with MOI: {total_sessions}")
    print(f"\nValid real dyads: {len(valid_real)}")
    print(f"Valid pseudo-dyads: {len(valid_pseudo)}")
    print(
        f"\nTOTAL valid dyad-session comparisons: {len(valid_real) + len(valid_pseudo)}"
    )

    # Ratio
    if len(valid_pseudo) > 0:
        ratio = len(valid_pseudo) / len(valid_real)
        print(f"\nRatio pseudo/real: {ratio:.2f}x")

    # Show sample of valid dyads
    print("\n" + "-" * 70)
    print("Sample valid real dyads (first 5):")
    for dyad in valid_real[:5]:
        print(f"  {dyad['column']}")

    print("\nSample valid pseudo-dyads (first 5):")
    for dyad in valid_pseudo[:5]:
        print(f"  {dyad['column']}")


if __name__ == "__main__":
    main()
