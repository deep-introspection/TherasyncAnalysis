#!/usr/bin/env python3
"""
Generate Epoch-by-Epoch Animation Frames

Create PNG frames for Poincaré plot animations showing temporal evolution
of dyadic synchrony. Each frame shows current epoch with cumulative traces.

Usage:
    uv run python scripts/physio/dppa/generate_epoch_frames.py \\
        --dyad g01p01_ses-01_vs_g01p02_ses-01 \\
        --method sliding_duration30s_step5s \\
        --task therapy

Author: Lena Adel, Remy Ramadour
Date: 2025-11-12
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.physio.dppa.dyad_centroid_loader import DyadCentroidLoader  # noqa: E402
from src.physio.dppa.epoch_animator import EpochAnimator  # noqa: E402
from src.physio.dppa.poincare_plotter import PoincarePlotter  # noqa: E402


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console."""
    log_dir = project_root / "log"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generate_epoch_frames_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def parse_dyad_info(dyad_pair: str) -> tuple:
    """
    Parse dyad identifier into components.

    Supports two formats:
        - Format A: 'sub1_ses1_vs_sub2_ses2' (session per subject)
        - Format B: 'sub1_vs_sub2_ses' (shared session at end)

    Args:
        dyad_pair: Dyad identifier string in either format

    Returns:
        Tuple (subject1, session1, subject2, session2)
    """
    parts = dyad_pair.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Invalid dyad format: {dyad_pair}")

    left_parts = parts[0].split("_")
    right_parts = parts[1].split("_")

    # Format A: sub1_ses1_vs_sub2_ses2 (2 parts each side)
    if len(left_parts) == 2 and len(right_parts) == 2:
        subject1, session1 = left_parts
        subject2, session2 = right_parts
        return subject1, session1, subject2, session2

    # Format B: sub1_vs_sub2_ses (1 part left, 2 parts right with shared session)
    if len(left_parts) == 1 and len(right_parts) == 2:
        subject1 = left_parts[0]
        subject2, session = right_parts
        return subject1, session, subject2, session

    raise ValueError(f"Invalid dyad format: {dyad_pair}")


def create_frame(
    epoch_id: int,
    time_min: float,
    dyad_info: tuple,
    poincare_data_1: dict,
    poincare_data_2: dict,
    centroid_params_1: dict,
    centroid_params_2: dict,
    icd: float,
    cumulative_data: dict,
    plotter: PoincarePlotter,
    config: dict,
) -> plt.Figure:
    """
    Create single animation frame with Poincaré plot and cumulative traces.

    Args:
        epoch_id: Current epoch ID
        time_min: Current time in minutes
        dyad_info: (subject1, session1, subject2, session2)
        poincare_data_1: Subject 1 Poincaré points
        poincare_data_2: Subject 2 Poincaré points
        centroid_params_1: Subject 1 ellipse parameters
        centroid_params_2: Subject 2 ellipse parameters
        icd: Inter-centroid distance (ms)
        cumulative_data: Historical data for traces
        plotter: PoincarePlotter instance
        config: Visualization config

    Returns:
        Matplotlib figure
    """
    subject1, _, subject2, _ = dyad_info

    # Create figure with 2x2 subplots layout
    # Top row: Poincaré plot (left) | ICD trace (right)
    # Bottom row: Centroid traces (left) | SD ratio traces (right)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_poincare = fig.add_subplot(gs[0, 0])  # Top-left
    ax_icd = fig.add_subplot(gs[0, 1])  # Top-right
    ax_centroids = fig.add_subplot(gs[1, 0])  # Bottom-left
    ax_ratios = fig.add_subplot(gs[1, 1])  # Bottom-right

    # === TOP: Poincaré Plot ===
    ax_poincare.set_title(
        f"Poincaré Plot - Epoch {epoch_id} (Time: {time_min:.1f} min)",
        fontsize=14,
        fontweight="bold",
    )
    ax_poincare.set_xlabel("RRn (ms)", fontsize=12)
    ax_poincare.set_ylabel("RRn+1 (ms)", fontsize=12)

    # Plot points for both subjects
    colors = config.get("colors", {})
    if poincare_data_1["n_points"] > 0:
        ax_poincare.scatter(
            poincare_data_1["rr_n"],
            poincare_data_1["rr_n_plus_1"],
            c=colors.get("subject1", "#1f77b4"),
            s=20,
            alpha=0.6,
            label=subject1,
        )

    if poincare_data_2["n_points"] > 0:
        ax_poincare.scatter(
            poincare_data_2["rr_n"],
            poincare_data_2["rr_n_plus_1"],
            c=colors.get("subject2", "#ff7f0e"),
            s=20,
            alpha=0.6,
            marker="s",
            label=subject2,
        )

    # Draw ellipses
    if not np.isnan(centroid_params_1["sd1"]):
        plotter.draw_ellipse(
            ax_poincare,
            centroid_params_1["centroid_x"],
            centroid_params_1["centroid_y"],
            centroid_params_1["sd1"],
            centroid_params_1["sd2"],
            centroid_params_1["angle"],
            color=colors.get("subject1", "#1f77b4"),
        )

    if not np.isnan(centroid_params_2["sd1"]):
        plotter.draw_ellipse(
            ax_poincare,
            centroid_params_2["centroid_x"],
            centroid_params_2["centroid_y"],
            centroid_params_2["sd1"],
            centroid_params_2["sd2"],
            centroid_params_2["angle"],
            color=colors.get("subject2", "#ff7f0e"),
        )

    # Draw ICD line
    if not np.isnan(icd):
        plotter.draw_icd_line(
            ax_poincare,
            centroid_params_1["centroid_x"],
            centroid_params_1["centroid_y"],
            centroid_params_2["centroid_x"],
            centroid_params_2["centroid_y"],
        )

    # Mark centroids with crosses
    if not np.isnan(centroid_params_1["centroid_x"]):
        ax_poincare.scatter(
            [centroid_params_1["centroid_x"]],
            [centroid_params_1["centroid_y"]],
            marker="x",
            s=200,
            linewidths=3,
            color=colors.get("subject1", "#1f77b4"),
            zorder=10,
            label=f"{subject1} centroid",
        )

    if not np.isnan(centroid_params_2["centroid_x"]):
        ax_poincare.scatter(
            [centroid_params_2["centroid_x"]],
            [centroid_params_2["centroid_y"]],
            marker="x",
            s=200,
            linewidths=3,
            color=colors.get("subject2", "#ff7f0e"),
            zorder=10,
            label=f"{subject2} centroid",
        )

    # Add metrics annotation
    plotter.annotate_metrics(
        ax_poincare,
        icd,
        centroid_params_1["sd1"],
        centroid_params_1["sd2"],
        centroid_params_2["sd1"],
        centroid_params_2["sd2"],
        subject1,
        subject2,
    )

    # Identity line
    all_rr = []
    if poincare_data_1["n_points"] > 0:
        all_rr.extend(poincare_data_1["rr_n"])
        all_rr.extend(poincare_data_1["rr_n_plus_1"])
    if poincare_data_2["n_points"] > 0:
        all_rr.extend(poincare_data_2["rr_n"])
        all_rr.extend(poincare_data_2["rr_n_plus_1"])

    if all_rr:
        rr_min, rr_max = min(all_rr), max(all_rr)
        ax_poincare.plot(
            [rr_min, rr_max], [rr_min, rr_max], "k--", alpha=0.3, linewidth=0.5
        )

    ax_poincare.grid(True, alpha=0.2)
    ax_poincare.legend(loc="upper left", fontsize=9)

    # === TOP-RIGHT: ICD Evolution ===
    ax_icd.set_title("ICD Evolution", fontsize=12, fontweight="bold")
    ax_icd.set_xlabel("Time (minutes)", fontsize=10)
    ax_icd.set_ylabel("ICD (ms)", fontsize=10)

    times = cumulative_data["times"][: epoch_id + 1]
    icd_values = cumulative_data["icd"][: epoch_id + 1]

    # Set fixed x-axis limits (full session duration)
    all_times = cumulative_data["all_times"]
    ax_icd.set_xlim(0, all_times[-1])

    ax_icd.plot(
        times,
        icd_values,
        color=colors.get("therapy", "#d62728"),
        linewidth=2,
        alpha=0.8,
    )
    ax_icd.scatter(
        [time_min],
        [icd_values[-1]],
        color="red",
        s=80,
        zorder=10,
        edgecolors="darkred",
        linewidths=2,
    )
    ax_icd.grid(True, alpha=0.2)

    # === BOTTOM-LEFT: Centroid Averages ===
    ax_centroids.set_title(
        "Centroid Averages ((X+Y)/2)", fontsize=12, fontweight="bold"
    )
    ax_centroids.set_xlabel("Time (minutes)", fontsize=10)
    ax_centroids.set_ylabel("Centroid (ms)", fontsize=10)

    centroid_avg_1 = cumulative_data["centroid_avg_1"][: epoch_id + 1]
    centroid_avg_2 = cumulative_data["centroid_avg_2"][: epoch_id + 1]

    # Set fixed x-axis limits (full session duration)
    ax_centroids.set_xlim(0, all_times[-1])

    ax_centroids.plot(
        times,
        centroid_avg_1,
        color=colors.get("subject1", "#1f77b4"),
        linewidth=2,
        label=subject1,
        alpha=0.8,
    )
    ax_centroids.plot(
        times,
        centroid_avg_2,
        color=colors.get("subject2", "#ff7f0e"),
        linewidth=2,
        label=subject2,
        alpha=0.8,
    )
    ax_centroids.scatter(
        [time_min],
        [centroid_avg_1[-1]],
        color=colors.get("subject1", "#1f77b4"),
        s=80,
        zorder=10,
        edgecolors="darkblue",
        linewidths=2,
    )
    ax_centroids.scatter(
        [time_min],
        [centroid_avg_2[-1]],
        color=colors.get("subject2", "#ff7f0e"),
        s=80,
        zorder=10,
        edgecolors="darkorange",
        linewidths=2,
    )
    ax_centroids.grid(True, alpha=0.2)
    ax_centroids.legend(loc="best", fontsize=9)

    # === BOTTOM-RIGHT: SD1/SD2 Ratios ===
    ax_ratios.set_title("SD1/SD2 Ratios", fontsize=12, fontweight="bold")
    ax_ratios.set_xlabel("Time (minutes)", fontsize=10)
    ax_ratios.set_ylabel("SD1/SD2 Ratio", fontsize=10)

    ratio_1 = cumulative_data["ratio_1"][: epoch_id + 1]
    ratio_2 = cumulative_data["ratio_2"][: epoch_id + 1]

    # Set fixed x-axis limits (full session duration)
    ax_ratios.set_xlim(0, all_times[-1])

    ax_ratios.plot(
        times,
        ratio_1,
        color=colors.get("subject1", "#1f77b4"),
        linewidth=2,
        label=subject1,
        alpha=0.8,
    )
    ax_ratios.plot(
        times,
        ratio_2,
        color=colors.get("subject2", "#ff7f0e"),
        linewidth=2,
        label=subject2,
        alpha=0.8,
    )
    ax_ratios.scatter(
        [time_min],
        [ratio_1[-1]],
        color=colors.get("subject1", "#1f77b4"),
        s=80,
        zorder=10,
        edgecolors="darkblue",
        linewidths=2,
    )
    ax_ratios.scatter(
        [time_min],
        [ratio_2[-1]],
        color=colors.get("subject2", "#ff7f0e"),
        s=80,
        zorder=10,
        edgecolors="darkorange",
        linewidths=2,
    )
    ax_ratios.grid(True, alpha=0.2)
    ax_ratios.legend(loc="best", fontsize=9)

    plt.tight_layout()
    return fig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate epoch-by-epoch animation frames for DPPA analysis"
    )
    parser.add_argument(
        "--dyad",
        required=True,
        help="Dyad identifier (e.g., 'g01p01_ses-01_vs_g01p02_ses-01')",
    )
    parser.add_argument(
        "--method",
        default="sliding_duration30s_step5s",
        help="Epoching method (default: sliding_duration30s_step5s)",
    )
    parser.add_argument(
        "--task",
        default="therapy",
        choices=["therapy", "restingstate"],
        help="Task name (default: therapy)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for frames (default: data/derivatives/dppa/frames/{dyad})",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Maximum number of epochs to process (for testing)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=project_root / "config" / "config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = (
            project_root / "data" / "derivatives" / "dppa" / "frames" / args.dyad
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("=" * 80)
    logger.info("GENERATE EPOCH FRAMES")
    logger.info("=" * 80)
    logger.info(f"Dyad: {args.dyad}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output: {args.output_dir}")

    # Parse dyad info
    try:
        dyad_info = parse_dyad_info(args.dyad)
        subject1, session1, subject2, session2 = dyad_info
        logger.info(f"Subjects: {subject1}/{session1} vs {subject2}/{session2}")
    except ValueError as e:
        logger.error(f"Error parsing dyad: {e}")
        return 1

    # Initialize modules
    logger.info("\nInitializing modules...")
    centroid_loader = DyadCentroidLoader(args.config)
    animator = EpochAnimator(args.config)
    plotter = PoincarePlotter(args.config)

    # Load config
    from src.core.config_loader import ConfigLoader

    config = ConfigLoader(args.config).config
    viz_config = config.get("visualization", {}).get("dppa", {})

    # Load centroid data
    logger.info("\nLoading centroid data...")
    try:
        # Convert dyad_info tuple to dict with expected keys
        dyad_dict = {
            "sub1": subject1,
            "ses1": session1.replace("ses-", ""),  # Remove 'ses-' prefix
            "sub2": subject2,
            "ses2": session2.replace("ses-", ""),
        }
        centroid_data = centroid_loader.load_both_tasks(dyad_dict, args.method)
        centroids_1_therapy, centroids_2_therapy = centroid_data["therapy"]

        if centroids_1_therapy is None or centroids_2_therapy is None:
            logger.error("Failed to load centroid data for therapy task")
            return 1

        n_epochs = len(centroids_1_therapy)
        logger.info(f"Loaded centroids: {n_epochs} epochs")
    except Exception as e:
        logger.error(f"Error loading centroids: {e}")
        return 1

    # Load RR intervals
    logger.info("\nLoading RR intervals...")
    try:
        rr_df_1 = animator.load_rr_intervals(subject1, session1, args.task, args.method)
        rr_df_2 = animator.load_rr_intervals(subject2, session2, args.task, args.method)
        logger.info(
            f"Loaded RR intervals: {len(rr_df_1)} ({subject1}), {len(rr_df_2)} ({subject2})"
        )
    except Exception as e:
        logger.error(f"Error loading RR intervals: {e}")
        return 1

    # Prepare cumulative data storage
    cumulative_data = {
        "times": [],
        "icd": [],
        "centroid_avg_1": [],
        "centroid_avg_2": [],
        "ratio_1": [],
        "ratio_2": [],
        "all_times": [],  # Store all possible times for fixed x-axis
    }

    # Determine number of epochs to process
    max_epochs = min(args.max_epochs, n_epochs) if args.max_epochs else n_epochs
    logger.info(f"\nGenerating {max_epochs} frames...")

    # Pre-calculate all times for fixed x-axis
    step_size_sec = 5  # For sliding_duration30s_step5s
    for i in range(max_epochs):
        cumulative_data["all_times"].append(i * step_size_sec / 60)

    # Generate frames
    success_count = 0

    for epoch_id in tqdm(range(max_epochs), desc="Generating frames"):
        try:
            # Compute time in minutes
            time_min = epoch_id * step_size_sec / 60

            # Get Poincaré points for current epoch
            poincare_data_1 = animator.compute_poincare_for_epoch(
                rr_df_1, epoch_id, args.method
            )
            poincare_data_2 = animator.compute_poincare_for_epoch(
                rr_df_2, epoch_id, args.method
            )

            # Calculate ellipse parameters
            centroid_params_1 = plotter.calculate_ellipse_parameters(
                poincare_data_1["rr_n"], poincare_data_1["rr_n_plus_1"]
            )
            centroid_params_2 = plotter.calculate_ellipse_parameters(
                poincare_data_2["rr_n"], poincare_data_2["rr_n_plus_1"]
            )

            # Get ICD from centroid data
            centroid_1 = centroids_1_therapy.iloc[epoch_id]
            centroid_2 = centroids_2_therapy.iloc[epoch_id]
            icd = np.sqrt(
                (centroid_1["centroid_x"] - centroid_2["centroid_x"]) ** 2
                + (centroid_1["centroid_y"] - centroid_2["centroid_y"]) ** 2
            )

            # Store cumulative data
            cumulative_data["times"].append(time_min)
            cumulative_data["icd"].append(icd)
            cumulative_data["centroid_avg_1"].append(
                (centroid_1["centroid_x"] + centroid_1["centroid_y"]) / 2
            )
            cumulative_data["centroid_avg_2"].append(
                (centroid_2["centroid_x"] + centroid_2["centroid_y"]) / 2
            )
            cumulative_data["ratio_1"].append(centroid_1["sd_ratio"])
            cumulative_data["ratio_2"].append(centroid_2["sd_ratio"])

            # Create frame
            fig = create_frame(
                epoch_id,
                time_min,
                dyad_info,
                poincare_data_1,
                poincare_data_2,
                centroid_params_1,
                centroid_params_2,
                icd,
                cumulative_data,
                plotter,
                viz_config,
            )

            # Save frame
            frame_path = args.output_dir / f"frame_{epoch_id:04d}.png"
            fig.savefig(frame_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            success_count += 1

        except Exception as e:
            logger.error(f"Error generating frame {epoch_id}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total frames:    {max_epochs}")
    logger.info(f"Generated:       {success_count}")
    logger.info(f"Failed:          {max_epochs - success_count}")
    logger.info(f"Success rate:    {100 * success_count / max_epochs:.1f}%")
    logger.info(f"Output:          {args.output_dir}")
    logger.info("=" * 80)

    return 0 if success_count == max_epochs else 1


if __name__ == "__main__":
    sys.exit(main())
