"""
Alliance-ICD Statistics Plotter.

Creates visualizations for alliance-ICD correlation analysis.

Authors: Remy Ramadour
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.core.config_loader import ConfigLoader
from src.visualization.config import (
    COLORS,
    ALPHA,
    FIGSIZE,
    FONTSIZE,
    apply_plot_style,
    DPI,
)
from src.alliance.alliance_icd_loader import AllianceICDLoader
from src.alliance.alliance_icd_analyzer import AllianceICDAnalyzer

logger = logging.getLogger(__name__)


class AllianceICDStatsPlotter:
    """Creates visualizations for alliance-ICD analysis."""

    # Alliance colors
    ALLIANCE_COLORS = {
        "Neutral": COLORS["gray"],
        "Positive": COLORS["positive"],
        "Negative": COLORS["negative"],
        "Split": COLORS["medium"],  # Amber for mixed
    }

    # Dyad type colors
    DYAD_COLORS = {
        "real": "#7EB8DA",  # Sky Blue
        "pseudo": "#F4A4B8",  # Rose Pink
    }

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize plotter.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.derivatives_path = Path(self.config["paths"]["derivatives"])
        self.loader = AllianceICDLoader(config_path)
        self.analyzer = AllianceICDAnalyzer(config_path)

        # Apply project-wide plot style
        apply_plot_style()

        # Output directory
        self.output_dir = self.derivatives_path / "visualization" / "alliance_icd"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_icd_by_alliance(
        self,
        data: Optional[pd.DataFrame] = None,
        plot_type: str = "violin",
        split_by_dyad_type: bool = False,
        save: bool = True,
    ) -> Tuple[plt.Figure, Path]:
        """
        Plot ICD distribution by alliance state.

        Args:
            data: DataFrame with merged data
            plot_type: 'violin', 'box', or 'strip'
            split_by_dyad_type: If True, split violins by dyad type
            save: Whether to save the figure

        Returns:
            Tuple of (figure, output_path)
        """
        if data is None:
            data = self.analyzer.load_data()

        fig, ax = plt.subplots(figsize=FIGSIZE["medium"])

        alliance_order = ["Neutral", "Positive", "Negative", "Split"]
        positions = np.arange(len(alliance_order))

        if split_by_dyad_type:
            width = 0.35

            for i, alliance in enumerate(alliance_order):
                for j, dtype in enumerate(["real", "pseudo"]):
                    subset = data[
                        (data["alliance_label"] == alliance.lower())
                        & (data["dyad_type"] == dtype)
                    ]["icd"]

                    if len(subset) > 0:
                        pos = positions[i] + (j - 0.5) * width

                        if plot_type == "violin":
                            parts = ax.violinplot(
                                [subset],
                                positions=[pos],
                                widths=width * 0.9,
                                showmeans=True,
                                showextrema=True,
                                showmedians=True,
                            )
                            for pc in parts["bodies"]:
                                pc.set_facecolor(self.DYAD_COLORS[dtype])
                                pc.set_alpha(ALPHA["high"])
                        else:
                            bp = ax.boxplot(
                                [subset],
                                positions=[pos],
                                widths=width * 0.9,
                                patch_artist=True,
                            )
                            for patch in bp["boxes"]:
                                patch.set_facecolor(self.DYAD_COLORS[dtype])
                                patch.set_alpha(ALPHA["high"])

            # Legend for dyad types
            legend_patches = [
                mpatches.Patch(
                    color=self.DYAD_COLORS["real"],
                    label="Real Dyads",
                    alpha=ALPHA["high"],
                ),
                mpatches.Patch(
                    color=self.DYAD_COLORS["pseudo"],
                    label="Pseudo-Dyads",
                    alpha=ALPHA["high"],
                ),
            ]
            ax.legend(
                handles=legend_patches, loc="upper right", fontsize=FONTSIZE["legend"]
            )

        else:
            for i, alliance in enumerate(alliance_order):
                subset = data[data["alliance_label"] == alliance.lower()]["icd"]

                if len(subset) > 0:
                    color = self.ALLIANCE_COLORS[alliance]

                    if plot_type == "violin":
                        parts = ax.violinplot(
                            [subset],
                            positions=[positions[i]],
                            widths=0.7,
                            showmeans=True,
                            showextrema=True,
                            showmedians=True,
                        )
                        for pc in parts["bodies"]:
                            pc.set_facecolor(color)
                            pc.set_alpha(ALPHA["high"])
                    else:
                        bp = ax.boxplot(
                            [subset],
                            positions=[positions[i]],
                            widths=0.6,
                            patch_artist=True,
                        )
                        for patch in bp["boxes"]:
                            patch.set_facecolor(color)
                            patch.set_alpha(ALPHA["high"])

        ax.set_xticks(positions)
        ax.set_xticklabels(alliance_order, fontsize=FONTSIZE["label"])
        ax.set_xlabel("Alliance State", fontsize=FONTSIZE["label"])
        ax.set_ylabel("ICD (Inter-Centroid Distance)", fontsize=FONTSIZE["label"])
        ax.set_title(
            "ICD Distribution by Alliance State",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()

        output_path = self.output_dir / "figures" / "icd_by_alliance_violin.png"
        if save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=DPI["print"], bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        return fig, output_path

    def plot_icd_comparison_heatmap(
        self, data: Optional[pd.DataFrame] = None, save: bool = True
    ) -> Tuple[plt.Figure, Path]:
        """
        Create heatmap showing mean ICD by alliance state and dyad type.

        Args:
            data: DataFrame with merged data
            save: Whether to save the figure

        Returns:
            Tuple of (figure, output_path)
        """
        if data is None:
            data = self.analyzer.load_data()

        # Create pivot table with explicit column order
        pivot = data.groupby(["alliance_label", "dyad_type"])["icd"].mean().unstack()
        pivot = pivot.reindex(["neutral", "positive", "negative", "split"])
        # Only keep columns that exist in the data
        col_order = [c for c in ["real", "pseudo"] if c in pivot.columns]
        if not col_order:
            logger.warning("No dyad type columns found in pivot table")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig, None
        pivot = pivot[col_order]

        col_labels = {"real": "Real Dyads", "pseudo": "Pseudo-Dyads"}

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(
            [col_labels[c] for c in pivot.columns], fontsize=FONTSIZE["label"]
        )
        ax.set_yticklabels(
            [s.capitalize() for s in pivot.index], fontsize=FONTSIZE["label"]
        )

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    text_color = "white" if value > pivot.values.mean() else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=FONTSIZE["label"],
                        fontweight="bold",
                    )

        plt.colorbar(im, ax=ax, label="Mean ICD")
        ax.set_title(
            "Mean ICD by Alliance State and Dyad Type",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )

        plt.tight_layout()

        output_path = self.output_dir / "figures" / "icd_alliance_heatmap.png"
        if save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=DPI["print"], bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        return fig, output_path

    def plot_alliance_distribution(
        self, data: Optional[pd.DataFrame] = None, save: bool = True
    ) -> Tuple[plt.Figure, Path]:
        """
        Plot distribution of alliance states (pie + bar chart).

        Args:
            data: DataFrame with merged data
            save: Whether to save the figure

        Returns:
            Tuple of (figure, output_path)
        """
        if data is None:
            data = self.analyzer.load_data()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Count by alliance
        counts = data["alliance_label"].value_counts()
        labels = [s.capitalize() for s in counts.index]
        colors = [
            self.ALLIANCE_COLORS.get(s.capitalize(), COLORS["gray"])
            for s in counts.index
        ]

        # Pie chart
        wedges, texts, autotexts = axes[0].pie(
            counts.values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": FONTSIZE["label"]},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        axes[0].set_title(
            "Alliance State Distribution", fontsize=FONTSIZE["title"], fontweight="bold"
        )

        # Bar chart by dyad type
        cross_tab = pd.crosstab(data["alliance_label"], data["dyad_type"])
        cross_tab = cross_tab.reindex(["neutral", "positive", "negative", "split"])

        x = np.arange(len(cross_tab.index))
        width = 0.35

        axes[1].bar(
            x - width / 2,
            cross_tab.get("real", [0] * 4),
            width,
            label="Real Dyads",
            color=self.DYAD_COLORS["real"],
        )
        axes[1].bar(
            x + width / 2,
            cross_tab.get("pseudo", [0] * 4),
            width,
            label="Pseudo-Dyads",
            color=self.DYAD_COLORS["pseudo"],
        )

        axes[1].set_xlabel("Alliance State", fontsize=FONTSIZE["label"])
        axes[1].set_ylabel("Count (epochs)", fontsize=FONTSIZE["label"])
        axes[1].set_title(
            "Epoch Count by Alliance and Dyad Type",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([s.capitalize() for s in cross_tab.index])
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()

        output_path = self.output_dir / "figures" / "alliance_distribution.png"
        if save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=DPI["print"], bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        return fig, output_path

    def plot_summary_dashboard(
        self, data: Optional[pd.DataFrame] = None, save: bool = True
    ) -> Tuple[plt.Figure, Path]:
        """
        Create comprehensive summary dashboard.

        Args:
            data: DataFrame with merged data
            save: Whether to save the figure

        Returns:
            Tuple of (figure, output_path)
        """
        if data is None:
            data = self.analyzer.load_data()

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # --- Panel 1: Violin plot (top left) ---
        ax1 = fig.add_subplot(gs[0, 0])
        alliance_order = ["neutral", "positive", "negative", "split"]
        positions = np.arange(len(alliance_order))

        for i, alliance in enumerate(alliance_order):
            subset = data[data["alliance_label"] == alliance]["icd"]
            if len(subset) > 0:
                color = self.ALLIANCE_COLORS[alliance.capitalize()]
                parts = ax1.violinplot(
                    [subset],
                    positions=[i],
                    widths=0.7,
                    showmeans=True,
                    showmedians=True,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(ALPHA["high"])

        ax1.set_xticks(positions)
        ax1.set_xticklabels([s.capitalize() for s in alliance_order])
        ax1.set_ylabel("ICD")
        ax1.set_title("ICD by Alliance State", fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.3)

        # --- Panel 2: Real vs Pseudo comparison (top middle) ---
        ax2 = fig.add_subplot(gs[0, 1])

        for i, alliance in enumerate(alliance_order):
            for j, dtype in enumerate(["real", "pseudo"]):
                subset = data[
                    (data["alliance_label"] == alliance) & (data["dyad_type"] == dtype)
                ]["icd"]
                if len(subset) > 0:
                    pos = i + (j - 0.5) * 0.3
                    bp = ax2.boxplot(
                        [subset], positions=[pos], widths=0.25, patch_artist=True
                    )
                    for patch in bp["boxes"]:
                        patch.set_facecolor(self.DYAD_COLORS[dtype])
                        patch.set_alpha(ALPHA["high"])

        ax2.set_xticks(positions)
        ax2.set_xticklabels([s.capitalize() for s in alliance_order])
        ax2.set_ylabel("ICD")
        ax2.set_title("Real vs Pseudo by Alliance", fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)

        # Legend
        legend_patches = [
            mpatches.Patch(
                color=self.DYAD_COLORS["real"], label="Real", alpha=ALPHA["high"]
            ),
            mpatches.Patch(
                color=self.DYAD_COLORS["pseudo"], label="Pseudo", alpha=ALPHA["high"]
            ),
        ]
        ax2.legend(
            handles=legend_patches, loc="upper right", fontsize=FONTSIZE["small"]
        )

        # --- Panel 3: Descriptive stats table (top right) ---
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")

        stats_df = self.analyzer.compute_descriptive_stats(data)

        # Create table
        cell_text = []
        for idx in stats_df.index:
            row = stats_df.loc[idx]
            cell_text.append(
                [
                    idx.capitalize(),
                    f"{row['mean']:.4f}",
                    f"{row['std']:.4f}",
                    f"{row['median']:.4f}",
                    f"{int(row['n'])}",
                ]
            )

        table = ax3.table(
            cellText=cell_text,
            colLabels=["Alliance", "Mean", "Std", "Median", "N"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE["small"])
        table.scale(1.2, 1.5)
        ax3.set_title("Descriptive Statistics", fontweight="bold", pad=20)

        # --- Panel 4: Alliance distribution pie (bottom left) ---
        ax4 = fig.add_subplot(gs[1, 0])

        counts = data["alliance_label"].value_counts()
        labels = [s.capitalize() for s in counts.index]
        colors = [
            self.ALLIANCE_COLORS.get(s.capitalize(), COLORS["gray"])
            for s in counts.index
        ]

        ax4.pie(
            counts.values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax4.set_title("Alliance Distribution", fontweight="bold")

        # --- Panel 5: Statistical test results (bottom middle) ---
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis("off")

        test_result = self.analyzer.test_alliance_effect(data)

        text_lines = [
            "Kruskal-Wallis H-test",
            "─" * 25,
            f"Statistic: {test_result.get('statistic', 'N/A'):.3f}",
            f"p-value: {test_result.get('p_value', 'N/A'):.6f}",
            "",
            f"Significant (α=0.05): {'Yes ✓' if test_result.get('significant') else 'No ✗'}",
            "",
            f"Groups: {', '.join(test_result.get('groups', []))}",
            f"N per group: {test_result.get('n_per_group', [])}",
        ]

        ax5.text(
            0.5,
            0.5,
            "\n".join(text_lines),
            transform=ax5.transAxes,
            ha="center",
            va="center",
            fontsize=FONTSIZE["label"],
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax5.set_title("Statistical Test", fontweight="bold")

        # --- Panel 6: Data summary (bottom right) ---
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")

        n_real = len(data[data["dyad_type"] == "real"])
        n_pseudo = len(data[data["dyad_type"] == "pseudo"])

        summary_lines = [
            "DATA SUMMARY",
            "─" * 25,
            f"Total observations: {len(data):,}",
            f"Real dyad obs: {n_real:,}",
            f"Pseudo-dyad obs: {n_pseudo:,}",
            "",
            f"Unique dyads: {data['dyad'].nunique()}",
            f"Families: {', '.join(sorted(data['family'].unique()))}",
            "",
            f"Ratio pseudo/real: {n_pseudo / n_real:.2f}x" if n_real > 0 else "",
        ]

        ax6.text(
            0.5,
            0.5,
            "\n".join(summary_lines),
            transform=ax6.transAxes,
            ha="center",
            va="center",
            fontsize=FONTSIZE["label"],
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )
        ax6.set_title("Data Summary", fontweight="bold")

        fig.suptitle(
            "Alliance-ICD Analysis Dashboard",
            fontsize=FONTSIZE["title"] + 2,
            fontweight="bold",
            y=1.02,
        )

        output_path = self.output_dir / "figures" / "alliance_icd_dashboard.png"
        if save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=DPI["print"], bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        return fig, output_path

    def generate_all_visualizations(
        self, data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Path]:
        """
        Generate all alliance-ICD visualizations.

        Args:
            data: DataFrame with merged data

        Returns:
            Dict mapping figure name to output path
        """
        if data is None:
            data = self.analyzer.load_data()

        logger.info("Generating alliance-ICD visualizations...")

        outputs = {}

        # Generate each visualization
        _, path = self.plot_icd_by_alliance(data)
        outputs["violin"] = path
        plt.close()

        _, path = self.plot_icd_comparison_heatmap(data)
        outputs["heatmap"] = path
        plt.close()

        _, path = self.plot_alliance_distribution(data)
        outputs["distribution"] = path
        plt.close()

        _, path = self.plot_summary_dashboard(data)
        outputs["dashboard"] = path
        plt.close()

        _, path = self.plot_icd_distributions_by_alliance(data)
        outputs["distributions"] = path
        plt.close()

        logger.info(f"Generated {len(outputs)} visualizations")

        return outputs

    def plot_icd_distributions_by_alliance(
        self, data: Optional[pd.DataFrame] = None, bins: int = 50, save: bool = True
    ) -> Tuple[plt.Figure, Path]:
        """
        Plot overlapping histograms of ICD for real vs pseudo dyads by alliance state.

        Creates a 2x2 grid with one subplot per alliance state, showing
        the distribution of ICD values for real dyads vs pseudo-dyads.

        Args:
            data: DataFrame with merged data
            bins: Number of histogram bins
            save: Whether to save the figure

        Returns:
            Tuple of (figure, output_path)
        """
        if data is None:
            data = self.analyzer.load_data()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        alliance_order = ["Neutral", "Positive", "Negative", "Split"]

        for i, alliance in enumerate(alliance_order):
            ax = axes[i]
            alliance_lower = alliance.lower()

            # Get data for each dyad type
            real_data = data[
                (data["alliance_label"] == alliance_lower)
                & (data["dyad_type"] == "real")
            ]["icd"]

            pseudo_data = data[
                (data["alliance_label"] == alliance_lower)
                & (data["dyad_type"] == "pseudo")
            ]["icd"]

            # Compute common bin edges for both distributions
            all_values = pd.concat([real_data, pseudo_data])
            if len(all_values) > 0:
                bin_edges = np.histogram_bin_edges(all_values, bins=bins)
            else:
                bin_edges = np.linspace(0, 500, bins + 1)

            # Plot histograms with density normalization
            if len(real_data) > 0:
                ax.hist(
                    real_data,
                    bins=bin_edges,
                    alpha=0.6,
                    label=f"Real (n={len(real_data):,})",
                    color=self.DYAD_COLORS["real"],
                    density=True,
                    edgecolor="white",
                    linewidth=0.5,
                )

            if len(pseudo_data) > 0:
                ax.hist(
                    pseudo_data,
                    bins=bin_edges,
                    alpha=0.6,
                    label=f"Pseudo (n={len(pseudo_data):,})",
                    color=self.DYAD_COLORS["pseudo"],
                    density=True,
                    edgecolor="white",
                    linewidth=0.5,
                )

            # Add vertical lines for means
            if len(real_data) > 0:
                ax.axvline(
                    real_data.mean(),
                    color=self.DYAD_COLORS["real"],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                    label=f"Real mean: {real_data.mean():.1f}",
                )

            if len(pseudo_data) > 0:
                ax.axvline(
                    pseudo_data.mean(),
                    color=self.DYAD_COLORS["pseudo"],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                    label=f"Pseudo mean: {pseudo_data.mean():.1f}",
                )

            # Styling
            ax.set_xlabel("ICD (Inter-Centroid Distance)", fontsize=FONTSIZE["label"])
            ax.set_ylabel("Density", fontsize=FONTSIZE["label"])
            ax.set_title(
                f"{alliance} Alliance",
                fontsize=FONTSIZE["title"],
                fontweight="bold",
                color=self.ALLIANCE_COLORS[alliance],
            )
            ax.legend(loc="upper right", fontsize=FONTSIZE["small"])
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(0, None)

        fig.suptitle(
            "ICD Distributions: Real vs Pseudo-Dyads by Alliance State",
            fontsize=FONTSIZE["title"] + 2,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        output_path = self.output_dir / "figures" / "icd_distributions_by_alliance.png"
        if save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=DPI["print"], bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        return fig, output_path
