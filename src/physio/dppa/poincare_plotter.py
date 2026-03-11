"""
Poincaré Plotter Module

Generate Poincaré plot visualizations with ellipses, ICD lines, and metrics.
Handles SD1/SD2 ellipse calculation and rendering for epoch-by-epoch animations.

Author: Lena Adel, Remy Ramadour
Date: 2025-11-12
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class PoincarePlotter:
    """
    Create Poincaré plot visualizations for DPPA analysis.

    Draws points, SD1/SD2 ellipses, ICD lines, and metric annotations
    for epoch-by-epoch animations.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Poincaré Plotter.

        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        self.config = ConfigLoader(config_path).config

        # Load visualization config
        viz_config = self.config.get("visualization", {}).get("dppa", {})
        self.colors = viz_config.get("colors", {})
        self.styles = viz_config.get("styles", {})

        logger.info("Poincaré Plotter initialized")

    def calculate_ellipse_parameters(
        self, rr_n: np.ndarray, rr_n_plus_1: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Poincaré ellipse parameters (centroid, SD1, SD2, angle).

        Args:
            rr_n: RRn values (current intervals, ms)
            rr_n_plus_1: RRn+1 values (next intervals, ms)

        Returns:
            Dictionary with keys:
                - centroid_x: Mean RRn (ms)
                - centroid_y: Mean RRn+1 (ms)
                - sd1: Short-term variability (ms)
                - sd2: Long-term variability (ms)
                - angle: Ellipse rotation angle (degrees)

        Notes:
            SD1 = std(RRn - RRn+1) / √2  (perpendicular to identity line)
            SD2 = std(RRn + RRn+1) / √2  (parallel to identity line)
            Angle = 45° (ellipse aligned with Poincaré axes)
        """
        if len(rr_n) == 0 or len(rr_n_plus_1) == 0:
            return {
                "centroid_x": np.nan,
                "centroid_y": np.nan,
                "sd1": np.nan,
                "sd2": np.nan,
                "angle": 45.0,
            }

        # Centroid (mean of points)
        centroid_x = np.mean(rr_n)
        centroid_y = np.mean(rr_n_plus_1)

        # SD1 and SD2 calculation
        diff = rr_n - rr_n_plus_1  # Perpendicular to identity
        sum_vals = rr_n + rr_n_plus_1  # Parallel to identity

        sd1 = np.std(diff, ddof=1) / np.sqrt(2) if len(diff) > 1 else 0.0
        sd2 = np.std(sum_vals, ddof=1) / np.sqrt(2) if len(sum_vals) > 1 else 0.0

        return {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "sd1": sd1,
            "sd2": sd2,
            "angle": 45.0,  # Ellipse at 45° for Poincaré plot
        }

    def draw_ellipse(
        self,
        ax: Axes,
        centroid_x: float,
        centroid_y: float,
        sd1: float,
        sd2: float,
        angle: float = 45.0,
        color: str = "blue",
        label: Optional[str] = None,
    ) -> mpatches.Ellipse:
        """
        Draw SD1/SD2 ellipse on Poincaré plot.

        Args:
            ax: Matplotlib axes
            centroid_x: Ellipse center X (ms)
            centroid_y: Ellipse center Y (ms)
            sd1: Semi-minor axis (SD1, ms)
            sd2: Semi-major axis (SD2, ms)
            angle: Rotation angle (degrees, default 45° for Poincaré)
            color: Edge color
            label: Legend label

        Returns:
            Matplotlib Ellipse patch

        Notes:
            - Width = 2 * SD1 (perpendicular to identity line)
            - Height = 2 * SD2 (parallel to identity line)
            - Ellipse covers ~68% of points (1 SD)
        """
        ellipse = mpatches.Ellipse(
            xy=(centroid_x, centroid_y),
            width=2 * sd1,  # Full width (2 * semi-axis)
            height=2 * sd2,  # Full height
            angle=angle,
            edgecolor=color,
            facecolor="none",  # Outline only (as specified)
            linewidth=self.styles.get("baseline_linewidth", 1.0),
            label=label,
        )

        ax.add_patch(ellipse)
        return ellipse

    def draw_icd_line(
        self,
        ax: Axes,
        centroid1_x: float,
        centroid1_y: float,
        centroid2_x: float,
        centroid2_y: float,
        color: Optional[str] = None,
    ) -> Line2D:
        """
        Draw ICD line connecting two centroids.

        Args:
            ax: Matplotlib axes
            centroid1_x: First centroid X (ms)
            centroid1_y: First centroid Y (ms)
            centroid2_x: Second centroid X (ms)
            centroid2_y: Second centroid Y (ms)
            color: Line color (default: black)

        Returns:
            Matplotlib Line2D object
        """
        if color is None:
            color = self.colors.get("trendline", "#000000")

        line = ax.plot(
            [centroid1_x, centroid2_x],
            [centroid1_y, centroid2_y],
            color=color,
            linestyle="--",
            linewidth=self.styles.get("therapy_linewidth", 1.5),
            label="ICD",
            zorder=5,  # Draw on top
        )[0]

        return line

    def annotate_metrics(
        self,
        ax: Axes,
        icd: float,
        sd1_1: float,
        sd2_1: float,
        sd1_2: float,
        sd2_2: float,
        subject1_name: str = "S1",
        subject2_name: str = "S2",
    ) -> None:
        """
        Add legend box with ICD and SD metrics.

        Args:
            ax: Matplotlib axes
            icd: Inter-centroid distance (ms)
            sd1_1: SD1 for subject 1 (ms)
            sd2_1: SD2 for subject 1 (ms)
            sd1_2: SD1 for subject 2 (ms)
            sd2_2: SD2 for subject 2 (ms)
            subject1_name: Label for subject 1 (default: "S1")
            subject2_name: Label for subject 2 (default: "S2")

        Notes:
            - Legend box positioned in plot area (as specified)
            - Format: ICD: XXX.X ms, S1: SD1/SD2, S2: SD1/SD2
        """
        # Format text
        text = (
            f"ICD: {icd:.1f} ms\n"
            f"{subject1_name}: {sd1_1:.1f}/{sd2_1:.1f}\n"
            f"{subject2_name}: {sd1_2:.1f}/{sd2_2:.1f}"
        )

        # Add text box
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")

        ax.text(
            0.95,
            0.95,  # Top-right corner
            text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )
