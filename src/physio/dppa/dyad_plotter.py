"""
Module for generating dyadic DPPA visualizations.

This module provides functionality to create 4-subplot figures showing:
- ICD evolution with trendline
- SD1, SD2, and SD1/SD2 ratio comparisons
- Resting state baselines for all metrics
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class DyadPlotter:
    """
    Generate 4-subplot visualizations for dyadic DPPA analysis.

    This class creates publication-ready figures showing ICD and Poincaré
    metrics (SD1, SD2, SD1/SD2) for dyad pairs across therapy sessions,
    with resting state baselines.

    Attributes:
        config: Configuration object containing plot settings.
        viz_config: Visualization-specific configuration.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DyadPlotter.

        Args:
            config_path: Optional path to configuration file.
                        If None, uses default config.yaml.
        """
        self.config = ConfigLoader(config_path)
        self.viz_config = self.config.get("visualization.dppa", {})
        logger.info("DyadPlotter initialized")

    def plot_dyad(
        self,
        icd_data: Dict[str, pd.DataFrame],
        centroid_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        dyad_info: Dict[str, str],
        method: str,
        output_path: Path,
    ) -> None:
        """
        Generate 4-subplot dyad visualization.

        Creates a figure with:
        - Row 1: ICD time series with resting baseline and trendline
        - Row 2: SD1, SD2, SD1/SD2 comparisons with baselines

        Args:
            icd_data: Dict with 'restingstate' and 'therapy' ICD DataFrames
            centroid_data: Dict with 'restingstate' and 'therapy' centroid tuples
            dyad_info: Dict with 'sub1', 'ses1', 'sub2', 'ses2'
            method: Epoching method name
            output_path: Path to save figure

        Example:
            >>> plotter = DyadPlotter()
            >>> plotter.plot_dyad(icd_data, centroid_data, dyad_info, "nsplit120", Path("output.png"))
        """
        logger.info(
            f"Generating plot for {dyad_info['sub1']}_ses-{dyad_info['ses1']} "
            f"vs {dyad_info['sub2']}_ses-{dyad_info['ses2']}"
        )

        # Load plot settings from config
        fig_width = self.viz_config.get("figure", {}).get("width", 12)
        fig_height = self.viz_config.get("figure", {}).get("height", 8)
        dpi = self.viz_config.get("figure", {}).get("dpi", 150)

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Row 1: ICD subplot (spans all columns)
        ax_icd = fig.add_subplot(gs[0, :])
        self._plot_icd_subplot(ax_icd, icd_data)

        # Row 2: SD1, SD2, SD1/SD2 subplots
        ax_sd1 = fig.add_subplot(gs[1, 0])
        ax_sd2 = fig.add_subplot(gs[1, 1])
        ax_ratio = fig.add_subplot(gs[1, 2])

        self._plot_sd1_subplot(ax_sd1, centroid_data, dyad_info)
        self._plot_sd2_subplot(ax_sd2, centroid_data, dyad_info)
        self._plot_ratio_subplot(ax_ratio, centroid_data, dyad_info)

        # Add title
        title = (
            f"Dyadic Analysis: {dyad_info['sub1']}/ses-{dyad_info['ses1']} "
            f"vs {dyad_info['sub2']}/ses-{dyad_info['ses2']} ({method})"
        )
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Plot saved to: {output_path}")

    def _calculate_trendline(
        self, therapy_icd: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate linear trendline for therapy ICD data.

        Args:
            therapy_icd: DataFrame with 'epoch_id' and 'icd_value' columns

        Returns:
            Tuple of (fitted_values, slope_coefficient)

        Example:
            >>> plotter = DyadPlotter()
            >>> df = pd.DataFrame({"epoch_id": [0,1,2], "icd_value": [50,48,46]})
            >>> fitted, slope = plotter._calculate_trendline(df)
            >>> print(f"Slope: {slope:.3f}")
            Slope: -2.000
        """
        # Remove NaN values before regression
        clean_data = therapy_icd.dropna(subset=["icd_value"])
        
        if len(clean_data) < 2:
            logger.warning(
                f"Insufficient data for trendline: {len(clean_data)} valid points"
            )
            # Return zeros if not enough data
            x_all = therapy_icd["epoch_id"].values
            return np.zeros_like(x_all, dtype=float), 0.0
        
        x = clean_data["epoch_id"].values
        y = clean_data["icd_value"].values

        # Linear regression on clean data
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Compute fitted values for all epochs (including those with NaN)
        x_all = therapy_icd["epoch_id"].values
        fitted_values = slope * x_all + intercept

        logger.debug(
            f"Trendline: slope={slope:.4f}, r²={r_value**2:.4f}, p={p_value:.4f} "
            f"(using {len(clean_data)}/{len(therapy_icd)} valid points)"
        )

        return fitted_values, slope

    def _plot_icd_subplot(
        self, ax: plt.Axes, icd_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Plot ICD time series with resting baseline and trendline."""
        # Get color configuration
        colors = self.viz_config.get("colors", {})
        therapy_color = colors.get("therapy", "#d62728")
        resting_color = colors.get("resting", "#2ca02c")
        trendline_color = colors.get("trendline", "#000000")

        styles = self.viz_config.get("styles", {})
        therapy_lw = styles.get("therapy_linewidth", 1.5)
        baseline_lw = styles.get("baseline_linewidth", 1.0)
        baseline_ls = styles.get("baseline_linestyle", "--")
        trendline_lw = styles.get("trendline_linewidth", 1.0)
        trendline_ls = styles.get("trendline_linestyle", "--")

        # Plot therapy ICD
        therapy_df = icd_data["therapy"]
        ax.plot(
            therapy_df["epoch_id"],
            therapy_df["icd_value"],
            color=therapy_color,
            linewidth=therapy_lw,
            label="Therapy ICD",
            marker='o',
            markersize=3
        )

        # Calculate and plot trendline
        fitted_values, slope = self._calculate_trendline(therapy_df)
        ax.plot(
            therapy_df["epoch_id"],
            fitted_values,
            color=trendline_color,
            linewidth=trendline_lw,
            linestyle=trendline_ls,
            label=f"Trendline (slope={slope:.3f})"
        )

        # Plot resting state baseline
        resting_value = icd_data["restingstate"]["icd_value"].iloc[0]
        ax.axhline(
            y=resting_value,
            color=resting_color,
            linewidth=baseline_lw,
            linestyle=baseline_ls,
            label="Resting Baseline"
        )

        # Labels and legend
        labels = self.viz_config.get("labels", {})
        ax.set_xlabel(labels.get("epoch", "Epoch ID"))
        ax.set_ylabel(labels.get("icd", "Inter-Centroid Distance (ms)"))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set Y-axis limits
        ylimits = self.viz_config.get("ylimits", {})
        if "icd" in ylimits:
            ax.set_ylim(0, ylimits["icd"])

    def _plot_sd1_subplot(
        self,
        ax: plt.Axes,
        centroid_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        dyad_info: Dict[str, str],
    ) -> None:
        """Plot SD1 time series for both subjects with baselines."""
        self._plot_metric_subplot(
            ax, centroid_data, dyad_info, "sd1", "SD1 (ms)"
        )

    def _plot_sd2_subplot(
        self,
        ax: plt.Axes,
        centroid_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        dyad_info: Dict[str, str],
    ) -> None:
        """Plot SD2 time series for both subjects with baselines."""
        self._plot_metric_subplot(
            ax, centroid_data, dyad_info, "sd2", "SD2 (ms)"
        )

    def _plot_ratio_subplot(
        self,
        ax: plt.Axes,
        centroid_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        dyad_info: Dict[str, str],
    ) -> None:
        """Plot SD1/SD2 ratio time series for both subjects with baselines."""
        self._plot_metric_subplot(
            ax, centroid_data, dyad_info, "sd_ratio", "SD1/SD2 Ratio"
        )

    def _plot_metric_subplot(
        self,
        ax: plt.Axes,
        centroid_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        dyad_info: Dict[str, str],
        metric: str,
        ylabel: str,
    ) -> None:
        """
        Generic method to plot a metric for both subjects with baselines.

        Args:
            ax: Matplotlib axis
            centroid_data: Centroid data dict
            dyad_info: Dyad information
            metric: Column name to plot ('sd1', 'sd2', or 'sd_ratio')
            ylabel: Y-axis label
        """
        # Get color configuration
        colors = self.viz_config.get("colors", {})
        sub1_color = colors.get("subject1", "#1f77b4")
        sub2_color = colors.get("subject2", "#ff7f0e")
        resting_color = colors.get("resting", "#2ca02c")

        styles = self.viz_config.get("styles", {})
        therapy_lw = styles.get("therapy_linewidth", 1.5)
        baseline_lw = styles.get("baseline_linewidth", 1.0)
        baseline_ls = styles.get("baseline_linestyle", "--")

        # Get therapy data
        df1_therapy, df2_therapy = centroid_data["therapy"]
        df1_rest, df2_rest = centroid_data["restingstate"]

        # Plot therapy data for both subjects
        ax.plot(
            df1_therapy["epoch_id"],
            df1_therapy[metric],
            color=sub1_color,
            linewidth=therapy_lw,
            label=f"{dyad_info['sub1']}",
            marker='o',
            markersize=2
        )
        ax.plot(
            df2_therapy["epoch_id"],
            df2_therapy[metric],
            color=sub2_color,
            linewidth=therapy_lw,
            label=f"{dyad_info['sub2']}",
            marker='s',
            markersize=2
        )

        # Plot resting baselines
        rest_val1 = df1_rest[metric].iloc[0]
        rest_val2 = df2_rest[metric].iloc[0]

        ax.axhline(
            y=rest_val1,
            color=sub1_color,
            linewidth=baseline_lw,
            linestyle=baseline_ls,
            alpha=0.7
        )
        ax.axhline(
            y=rest_val2,
            color=sub2_color,
            linewidth=baseline_lw,
            linestyle=baseline_ls,
            alpha=0.7
        )

        # Labels and legend
        labels = self.viz_config.get("labels", {})
        ax.set_xlabel(labels.get("epoch", "Epoch ID"))
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set Y-axis limits based on metric
        ylimits = self.viz_config.get("ylimits", {})
        if metric in ylimits:
            ax.set_ylim(0, ylimits[metric])
        elif metric == "sd_ratio" and "ratio" in ylimits:
            ax.set_ylim(0, ylimits["ratio"])

