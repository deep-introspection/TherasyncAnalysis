"""
Composite Plots Module.

Creates publication-ready composite figures combining multiple related plots.
These figures provide comprehensive views of physiological data for reports.

Figures:
    - HRV Analysis: Poincaré + Autonomic Balance + Frequency Domain
    - EDA Analysis: Arousal Profile + SCR Distribution + SCR Timeline
    - Temperature Analysis: Timeline + Metrics + Correlations
    - Quality Report: Signal quality heatmap + Statistics

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from typing import Dict, Optional
from pathlib import Path
import logging

from ..config import (
    COLORS,
    ALPHA,
    FONTSIZE,
    LINEWIDTH,
    MARKERSIZE,
    apply_plot_style,
    get_moment_color,
    get_moment_label,
    get_modality_color,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FIGURE 2: HRV ANALYSIS (Composite)
# =============================================================================


def plot_hrv_analysis(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Create comprehensive HRV analysis figure.

    Layout:
    ┌─────────────────┬─────────────────┐
    │   Poincaré      │   Autonomic     │
    │   Plot          │   Balance       │
    ├─────────────────┴─────────────────┤
    │   HRV Frequency Domain            │
    │   (LF/HF Power Spectrum)          │
    └───────────────────────────────────┘

    Args:
        data: Dictionary containing BVP/HRV data
        output_path: Where to save the figure
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    fig = plt.figure(figsize=(16, 11))  # Wider for external legends
    gs = GridSpec(2, 2, height_ratios=[1, 0.9], hspace=0.4, wspace=0.45)  # More spacing

    # Get metrics
    bvp_data = data.get("bvp", {})
    metrics = bvp_data.get("metrics")

    # ========== Panel 1: Poincaré Plot ==========
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_poincare_panel(ax1, bvp_data, data)

    # ========== Panel 2: Autonomic Balance ==========
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_autonomic_balance_panel(ax2, metrics)

    # ========== Panel 3: Frequency Domain ==========
    ax3 = fig.add_subplot(gs[1, :])
    _plot_frequency_domain_panel(ax3, metrics)

    # Main title
    subject = data.get("subject", "Unknown")
    session = data.get("session", "Unknown")
    fig.suptitle(
        f"Heart Rate Variability Analysis\nSubject {subject} • Session {session}",
        fontsize=FONTSIZE["title"] + 2,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved HRV analysis figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_poincare_panel(ax: plt.Axes, bvp_data: Dict, data: Dict) -> None:
    """
    Plot Poincaré plots as side-by-side facets for each moment.

    Uses inset axes to create a grid of small multiples, which scales
    well to any number of moments and avoids overlapping data.
    """
    from src.visualization.config import AXIS_LIMITS

    ax.set_title(
        "Poincaré Plot", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )
    ax.axis("off")  # Turn off the main axes - we'll use insets

    signals = bvp_data.get("signals", {})
    if not signals:
        ax.text(
            0.5,
            0.5,
            "No BVP signals",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Collect data for all moments
    moment_data = {}
    all_rr = []

    for moment_name, moment_signals in sorted(signals.items()):
        rr = None

        # Try peaks first for true RR intervals
        if "PPG_Peaks" in moment_signals.columns and "time" in moment_signals.columns:
            peaks_mask = moment_signals["PPG_Peaks"] == 1
            peak_times = moment_signals.loc[peaks_mask, "time"].values
            if len(peak_times) > 10:
                rr = np.diff(peak_times) * 1000  # Convert to ms
                rr = rr[(rr > 300) & (rr < 2000)]  # Filter physiological range

        # Fallback to PPG_Rate
        if rr is None or len(rr) < 10:
            if "PPG_Rate" in moment_signals.columns:
                hr = moment_signals["PPG_Rate"].dropna().values
                if len(hr) >= 10:
                    rr = 60000 / hr
                    rr = rr[(rr > 300) & (rr < 2000)]

        if rr is not None and len(rr) >= 10:
            moment_data[moment_name] = rr
            all_rr.extend(rr)

    if not moment_data:
        ax.text(
            0.5,
            0.5,
            "Insufficient RR data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Calculate global limits for consistent comparison across sessions
    limits = AXIS_LIMITS.get("poincare_rr")
    if limits:
        rr_min, rr_max = limits
    else:
        rr_min, rr_max = min(all_rr), max(all_rr)
        margin = (rr_max - rr_min) * 0.1
        rr_min, rr_max = rr_min - margin, rr_max + margin

    # Create faceted subplots - one per moment
    n_moments = len(moment_data)
    n_cols = min(n_moments, 3)  # Max 3 columns
    n_rows = (n_moments + n_cols - 1) // n_cols

    # Calculate subplot dimensions to fit in axes area
    margin_left, margin_right = 0.05, 0.05
    margin_top, margin_bottom = 0.12, 0.08
    h_gap, v_gap = 0.08, 0.15

    available_width = 1.0 - margin_left - margin_right - (n_cols - 1) * h_gap
    available_height = 1.0 - margin_top - margin_bottom - (n_rows - 1) * v_gap
    cell_width = available_width / n_cols
    cell_height = available_height / n_rows

    for idx, (moment_name, rr) in enumerate(sorted(moment_data.items())):
        row = idx // n_cols
        col = idx % n_cols

        # Position for this facet
        left = margin_left + col * (cell_width + h_gap)
        bottom = 1.0 - margin_top - (row + 1) * cell_height - row * v_gap

        # Create inset axes
        inset_ax = ax.inset_axes([left, bottom, cell_width, cell_height])

        color = get_moment_color(moment_name)
        label = get_moment_label(moment_name)

        rr_n = rr[:-1]
        rr_n1 = rr[1:]

        # Scatter plot
        inset_ax.scatter(
            rr_n, rr_n1, c=color, alpha=0.5, s=MARKERSIZE["small"], edgecolors="none"
        )

        # Identity line
        inset_ax.plot(
            [rr_min, rr_max],
            [rr_min, rr_max],
            "--",
            color=COLORS["gray"],
            alpha=0.5,
            linewidth=LINEWIDTH["thin"],
        )

        # Calculate SD1/SD2
        rr_mean = np.mean(rr)
        sd1 = np.std(np.subtract(rr_n1, rr_n) / np.sqrt(2))
        sd2 = np.std(np.add(rr_n1, rr_n) / np.sqrt(2))

        # Draw SD1/SD2 ellipse
        ellipse = Ellipse(
            (rr_mean, rr_mean),
            width=2 * sd2,
            height=2 * sd1,
            angle=45,
            facecolor=color,
            alpha=0.15,
            edgecolor=color,
            linewidth=LINEWIDTH["medium"],
        )
        inset_ax.add_patch(ellipse)

        # Set consistent limits for cross-session comparison
        inset_ax.set_xlim(rr_min, rr_max)
        inset_ax.set_ylim(rr_min, rr_max)
        inset_ax.set_aspect("equal")

        # Title with moment name and SD values
        inset_ax.set_title(
            f"{label}\nSD1={sd1:.0f}ms  SD2={sd2:.0f}ms",
            fontsize=FONTSIZE["small"],
            fontweight="bold",
            color=color,
            pad=3,
        )

        # Axis labels only on edge facets
        if row == n_rows - 1:
            inset_ax.set_xlabel("RR(n) ms", fontsize=FONTSIZE["small"])
        else:
            inset_ax.set_xticklabels([])

        if col == 0:
            inset_ax.set_ylabel("RR(n+1) ms", fontsize=FONTSIZE["small"])
        else:
            inset_ax.set_yticklabels([])

        inset_ax.tick_params(labelsize=FONTSIZE["small"] - 1)
        inset_ax.grid(True, alpha=0.3, linestyle="--")

    # Add legend explanation at the bottom of the main axes
    ax.text(
        0.5,
        -0.02,
        "Ellipse: SD1 (short-term) × SD2 (long-term) variability  •  Dashed line: Identity (RR(n) = RR(n+1))",
        transform=ax.transAxes,
        fontsize=FONTSIZE["small"],
        color=COLORS["dark_gray"],
        ha="center",
        va="top",
        fontweight="bold",
    )


def _plot_autonomic_balance_panel(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    """Plot SDNN vs RMSSD comparison across moments."""
    from src.visualization.config import AXIS_LIMITS

    ax.set_title(
        "Autonomic Balance", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )

    if metrics is None or metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No HRV metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    moments = metrics["moment"].tolist() if "moment" in metrics.columns else []
    if not moments:
        return

    x = np.arange(len(moments))
    width = 0.35

    sdnn_values = metrics["HRV_SDNN"].values if "HRV_SDNN" in metrics.columns else []
    rmssd_values = metrics["HRV_RMSSD"].values if "HRV_RMSSD" in metrics.columns else []

    if len(sdnn_values) == 0 or len(rmssd_values) == 0:
        ax.text(
            0.5,
            0.5,
            "Missing SDNN/RMSSD data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Bar plot
    bars1 = ax.bar(
        x - width / 2,
        sdnn_values,
        width,
        label="SDNN",
        color=COLORS["lf"],
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        rmssd_values,
        width,
        label="RMSSD",
        color=COLORS["hf"],
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
    )

    # Value labels on bars
    for bar, val in zip(bars1, sdnn_values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["small"],
                fontweight="bold",
                color=COLORS["text"],
            )

    for bar, val in zip(bars2, rmssd_values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["small"],
                fontweight="bold",
                color=COLORS["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_moment_label(m) for m in moments], fontsize=FONTSIZE["tick"]
    )
    ax.set_ylabel("Duration (ms)", fontsize=FONTSIZE["label"])

    # Apply fixed limits for cross-session comparison
    sdnn_limits = AXIS_LIMITS.get("sdnn")
    rmssd_limits = AXIS_LIMITS.get("rmssd")
    if sdnn_limits and rmssd_limits:
        ax.set_ylim(0, max(sdnn_limits[1], rmssd_limits[1]))

    ax.legend(loc="upper right", fontsize=FONTSIZE["legend"], framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Add interpretation note below x-axis
    ax.text(
        0.5,
        -0.12,
        "SDNN: Overall variability  •  RMSSD: Parasympathetic activity",
        transform=ax.transAxes,
        fontsize=FONTSIZE["small"],
        color=COLORS["dark_gray"],
        ha="center",
        va="top",
        fontweight="bold",
    )


def _plot_frequency_domain_panel(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    """Plot LF/HF power distribution across moments."""
    ax.set_title(
        "Frequency Domain Analysis",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
        pad=10,
    )

    if metrics is None or metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No frequency domain data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    moments = metrics["moment"].tolist() if "moment" in metrics.columns else []
    if not moments:
        return

    # Check for LF/HF data
    has_lf = "HRV_LF" in metrics.columns
    has_hf = "HRV_HF" in metrics.columns
    has_lfhf = "HRV_LFHF" in metrics.columns

    if not (has_lf or has_hf):
        ax.text(
            0.5,
            0.5,
            "No LF/HF data available\n(Frequency analysis requires longer recordings)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    x = np.arange(len(moments))
    width = 0.3

    # Get values (handle NaN and string 'n/a')
    def safe_float(val):
        if pd.isna(val) or val == "n/a":
            return np.nan
        try:
            return float(val)
        except Exception:
            return np.nan

    lf_values = (
        [safe_float(v) for v in metrics["HRV_LF"].values]
        if has_lf
        else [np.nan] * len(moments)
    )
    hf_values = (
        [safe_float(v) for v in metrics["HRV_HF"].values]
        if has_hf
        else [np.nan] * len(moments)
    )
    lfhf_values = (
        [safe_float(v) for v in metrics["HRV_LFHF"].values]
        if has_lfhf
        else [np.nan] * len(moments)
    )

    # Create grouped bar chart - only plot non-NaN values
    bars1 = ax.bar(
        x - width / 2,
        [v if not np.isnan(v) else 0 for v in lf_values],
        width,
        label="LF (0.04-0.15 Hz)",
        color=COLORS["lf"],
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        [v if not np.isnan(v) else 0 for v in hf_values],
        width,
        label="HF (0.15-0.40 Hz)",
        color=COLORS["hf"],
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
    )

    # Add value labels on bars
    for i, (bar1, bar2, lf, hf) in enumerate(zip(bars1, bars2, lf_values, hf_values)):
        if not np.isnan(lf) and lf > 0:
            ax.text(
                bar1.get_x() + bar1.get_width() / 2,
                bar1.get_height() + 0.002,
                f"{lf:.3f}",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["small"],
                fontweight="bold",
                color=COLORS["lf"],
            )
        elif np.isnan(lf) or lf == 0:
            ax.text(
                bar1.get_x() + bar1.get_width() / 2,
                0.001,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["small"],
                color=COLORS["gray"],
                style="italic",
            )

        if not np.isnan(hf) and hf > 0:
            ax.text(
                bar2.get_x() + bar2.get_width() / 2,
                bar2.get_height() + 0.002,
                f"{hf:.3f}",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["small"],
                fontweight="bold",
                color=COLORS["hf"],
            )

    # Secondary y-axis for LF/HF ratio - properly centered on each moment
    valid_lfhf = [
        (i, v) for i, v in enumerate(lfhf_values) if not np.isnan(v) and v > 0
    ]
    if valid_lfhf:
        ax2 = ax.twinx()
        x_ratio = [x[i] for i, v in valid_lfhf]
        y_ratio = [v for i, v in valid_lfhf]

        # Plot ratio as line connecting valid points
        ax2.plot(
            x_ratio,
            y_ratio,
            "s-",
            color=COLORS["scr"],
            linewidth=LINEWIDTH["thick"],
            markersize=MARKERSIZE["large"],
            label="LF/HF Ratio",
            markeredgecolor="white",
            markeredgewidth=1,
        )

        # Add ratio values next to markers
        for xi, yi in zip(x_ratio, y_ratio):
            ax2.annotate(
                f"{yi:.2f}",
                xy=(xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=FONTSIZE["small"],
                fontweight="bold",
                color=COLORS["scr"],
            )

        ax2.set_ylabel("LF/HF Ratio", fontsize=FONTSIZE["label"], color=COLORS["scr"])
        ax2.tick_params(axis="y", labelcolor=COLORS["scr"])
        ax2.set_ylim(0, max(y_ratio) * 1.5)  # Give headroom for annotations
        ax2.legend(loc="upper right", fontsize=FONTSIZE["legend"])

    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_moment_label(m) for m in moments], fontsize=FONTSIZE["tick"]
    )
    ax.set_ylabel("Power (normalized)", fontsize=FONTSIZE["label"])
    ax.legend(loc="upper left", fontsize=FONTSIZE["legend"], framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Interpretation guide below the plot
    ax.text(
        0.5,
        -0.1,
        "LF: Sympathetic + Parasympathetic  •  HF: Parasympathetic  •  LF/HF > 1: Sympathetic dominance",
        transform=ax.transAxes,
        fontsize=FONTSIZE["small"],
        color=COLORS["dark_gray"],
        ha="center",
        va="top",
        fontweight="bold",
    )


# =============================================================================
# FIGURE 3: EDA ANALYSIS (Composite)
# =============================================================================


def plot_eda_analysis(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Create comprehensive EDA analysis figure.

    Layout (3 rows for clarity and scalability):
    ┌───────────────────────────────────┐
    │   Arousal Profile                 │
    ├───────────────────────────────────┤
    │   SCR Amplitude Distribution      │
    ├───────────────────────────────────┤
    │   SCR Events Timeline             │
    └───────────────────────────────────┘

    Args:
        data: Dictionary containing EDA data
        output_path: Where to save the figure
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

    eda_data = data.get("eda", {})

    # ========== Panel 1: Arousal Profile ==========
    ax1 = fig.add_subplot(gs[0])
    _plot_arousal_profile_panel(ax1, eda_data)

    # ========== Panel 2: SCR Distribution ==========
    ax2 = fig.add_subplot(gs[1])
    _plot_scr_distribution_panel(ax2, eda_data)

    # ========== Panel 3: SCR Timeline ==========
    ax3 = fig.add_subplot(gs[2])
    _plot_scr_timeline_panel(ax3, eda_data, data)

    # Main title
    subject = data.get("subject", "Unknown")
    session = data.get("session", "Unknown")
    fig.suptitle(
        f"Electrodermal Activity Analysis\nSubject {subject} • Session {session}",
        fontsize=FONTSIZE["title"] + 2,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved EDA analysis figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_arousal_profile_panel(ax: plt.Axes, eda_data: Dict) -> None:
    """
    Plot arousal metrics comparison across moments.
    Shows Tonic EDA (baseline arousal) and SCR Rate (reactivity).
    """
    ax.set_title(
        "Arousal Profile", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )

    metrics = eda_data.get("metrics")
    if metrics is None or metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No EDA metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    moments = metrics["moment"].tolist() if "moment" in metrics.columns else []
    if not moments:
        return

    x = np.arange(len(moments))
    width = 0.35

    tonic_values = (
        metrics["EDA_Tonic_Mean"].values if "EDA_Tonic_Mean" in metrics.columns else []
    )
    scr_rate_values = (
        metrics["SCR_Peaks_Rate"].values if "SCR_Peaks_Rate" in metrics.columns else []
    )

    if len(tonic_values) == 0:
        ax.text(
            0.5,
            0.5,
            "No tonic data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Plot Tonic EDA bars (primary y-axis)
    for i, moment in enumerate(moments):
        moment_color = get_moment_color(moment)
        ax.bar(
            x[i] - width / 2,
            tonic_values[i],
            width,
            color=moment_color,
            alpha=ALPHA["high"],
            edgecolor="white",
            linewidth=1,
            label="Tonic EDA (µS)" if i == 0 else "",
        )
        # Value label
        ax.text(
            x[i] - width / 2,
            tonic_values[i] + 0.02,
            f"{tonic_values[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE["small"],
            fontweight="bold",
        )

    ax.set_ylabel("Tonic EDA (µS)", fontsize=FONTSIZE["label"])
    ax.set_ylim(0, max(tonic_values) * 1.3)

    # Secondary y-axis for SCR Rate
    if len(scr_rate_values) > 0:
        ax2 = ax.twinx()
        for i, moment in enumerate(moments):
            moment_color = get_moment_color(moment)
            ax2.bar(
                x[i] + width / 2,
                scr_rate_values[i],
                width,
                color=moment_color,
                alpha=ALPHA["medium"],
                edgecolor=moment_color,
                linewidth=1.5,
                hatch="///",
                label="SCR Rate (/min)" if i == 0 else "",
            )
            # Value label
            ax2.text(
                x[i] + width / 2,
                scr_rate_values[i] + 0.5,
                f"{scr_rate_values[i]:.1f}",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["small"],
                fontweight="bold",
            )

        ax2.set_ylabel("SCR Rate (per min)", fontsize=FONTSIZE["label"])
        ax2.set_ylim(0, max(scr_rate_values) * 1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_moment_label(m) for m in moments], fontsize=FONTSIZE["tick"]
    )
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Combined legend
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=FONTSIZE["legend"])
    if len(scr_rate_values) > 0:
        ax2.legend(
            loc="upper left", bbox_to_anchor=(1.15, 0.85), fontsize=FONTSIZE["legend"]
        )

    # Interpretation note
    ax.text(
        0.5,
        -0.12,
        "Solid: Tonic (baseline arousal)  •  Hatched: SCR Rate (reactivity)",
        transform=ax.transAxes,
        fontsize=FONTSIZE["small"],
        color=COLORS["dark_gray"],
        ha="center",
        va="top",
        fontweight="bold",
    )


def _plot_scr_distribution_panel(ax: plt.Axes, eda_data: Dict) -> None:
    """
    Plot SCR amplitude distribution using boxplots.
    Boxplots scale well to any number of moments without overlap.
    """
    from src.visualization.config import AXIS_LIMITS

    ax.set_title(
        "SCR Amplitude Distribution",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
        pad=10,
    )

    events = eda_data.get("events", {})
    if not events:
        ax.text(
            0.5,
            0.5,
            "No SCR events",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Collect amplitudes by moment
    moment_amplitudes = {}
    for moment_name, moment_events in sorted(events.items()):
        if "amplitude" in moment_events.columns:
            amps = moment_events["amplitude"].dropna().values
            if len(amps) > 0:
                moment_amplitudes[moment_name] = amps

    if not moment_amplitudes:
        ax.text(
            0.5,
            0.5,
            "No SCR amplitudes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Create boxplot data
    moments = list(moment_amplitudes.keys())
    box_data = [moment_amplitudes[m] for m in moments]
    positions = np.arange(len(moments))

    # Draw boxplots with moment colors
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops={"markersize": 3, "alpha": 0.5},
    )

    # Color each boxplot by moment
    for i, (patch, moment) in enumerate(zip(bp["boxes"], moments)):
        color = get_moment_color(moment)
        patch.set_facecolor(color)
        patch.set_alpha(ALPHA["high"])
        patch.set_edgecolor("white")
        patch.set_linewidth(1.5)

        # Color median line
        bp["medians"][i].set_color("white")
        bp["medians"][i].set_linewidth(2)

        # Add count annotation above each box
        n = len(moment_amplitudes[moment])
        median = np.median(moment_amplitudes[moment])
        ax.text(
            positions[i],
            ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else median * 1.5,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE["small"],
            fontweight="bold",
            color=color,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [get_moment_label(m) for m in moments], fontsize=FONTSIZE["tick"]
    )
    ax.set_ylabel("SCR Amplitude (µS)", fontsize=FONTSIZE["label"])
    ax.set_xlabel("")

    # Apply fixed limits if configured
    limits = AXIS_LIMITS.get("scr_amplitude")
    if limits:
        ax.set_ylim(limits)

    ax.grid(True, alpha=0.3, axis="y", linestyle="--")


def _plot_scr_timeline_panel(ax: plt.Axes, eda_data: Dict, data: Dict) -> None:
    """Plot SCR events over time with density indication."""
    ax.set_title(
        "SCR Events Timeline", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )

    events = eda_data.get("events", {})
    if not events:
        ax.text(
            0.5,
            0.5,
            "No SCR events",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Calculate moment boundaries
    moment_boundaries = {}
    offset = 0
    for moment_name in sorted(events.keys()):
        moment_events = events[moment_name]
        if len(moment_events) > 0 and "onset" in moment_events.columns:
            duration = moment_events["onset"].max() + 60  # Add buffer
            moment_boundaries[moment_name] = {"start": offset, "end": offset + duration}
            offset += duration + 10  # Gap between moments

    # Plot SCR events as stems
    for moment_name, moment_events in events.items():
        if moment_name not in moment_boundaries:
            continue

        if (
            "onset" not in moment_events.columns
            or "amplitude" not in moment_events.columns
        ):
            continue

        valid_events = moment_events.dropna(subset=["onset", "amplitude"])
        if len(valid_events) == 0:
            continue

        color = get_moment_color(moment_name)
        start_offset = moment_boundaries[moment_name]["start"]

        times = valid_events["onset"].values + start_offset
        amps = valid_events["amplitude"].values

        markerline, stemlines, baseline = ax.stem(
            times, amps, linefmt=color, markerfmt="o", basefmt=" "
        )
        markerline.set_markersize(MARKERSIZE["small"])
        markerline.set_color(color)
        stemlines.set_linewidth(LINEWIDTH["normal"])
        stemlines.set_alpha(ALPHA["high"])

        # Add moment label
        mid = (
            moment_boundaries[moment_name]["start"]
            + moment_boundaries[moment_name]["end"]
        ) / 2
        ax.text(
            mid,
            ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
            get_moment_label(moment_name),
            ha="center",
            va="top",
            fontsize=FONTSIZE["annotation"],
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=color, alpha=0.3, edgecolor="none"
            ),
        )

    ax.set_xlabel("Time (s)", fontsize=FONTSIZE["label"])
    ax.set_ylabel("SCR Amplitude (µS)", fontsize=FONTSIZE["label"])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")


# =============================================================================
# FIGURE 4: TEMPERATURE ANALYSIS (Composite)
# =============================================================================


def plot_temp_analysis(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Create comprehensive temperature analysis figure.

    Layout:
    ┌───────────────────────────────────┐
    │   Temperature Timeline            │
    ├─────────────────┬─────────────────┤
    │   Temp Metrics  │   Temp vs EDA   │
    │   Comparison    │   Correlation   │
    └─────────────────┴─────────────────┘

    Args:
        data: Dictionary containing TEMP and EDA data
        output_path: Where to save the figure
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.35, wspace=0.3)

    temp_data = data.get("temp", {})
    eda_data = data.get("eda", {})

    # ========== Panel 1: Temperature Timeline (full width) ==========
    ax1 = fig.add_subplot(gs[0, :])
    _plot_temp_timeline_panel(ax1, temp_data)

    # ========== Panel 2: Metrics Comparison ==========
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_temp_metrics_panel(ax2, temp_data)

    # ========== Panel 3: Temp-EDA Correlation ==========
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_temp_eda_correlation_panel(ax3, temp_data, eda_data)

    # Main title
    subject = data.get("subject", "Unknown")
    session = data.get("session", "Unknown")
    fig.suptitle(
        f"Peripheral Temperature Analysis\nSubject {subject} • Session {session}",
        fontsize=FONTSIZE["title"] + 2,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved temperature analysis figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_temp_timeline_panel(ax: plt.Axes, temp_data: Dict) -> None:
    """Plot temperature over time with physiological zones."""
    from src.visualization.config import AXIS_LIMITS

    ax.set_title(
        "Temperature Timeline", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )

    signals = temp_data.get("signals", {})
    if not signals:
        ax.text(
            0.5,
            0.5,
            "No temperature data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    all_temps = []
    offset = 0
    moment_labels = []

    for moment_name in sorted(signals.keys()):
        moment_signals = signals[moment_name]

        if "time" not in moment_signals.columns:
            continue

        temp_col = None
        for col in ["TEMP_Clean", "TEMP_Raw", "temperature"]:
            if col in moment_signals.columns:
                temp_col = col
                break

        if temp_col is None:
            continue

        time = moment_signals["time"].values + offset
        temp = moment_signals[temp_col].values
        all_temps.extend(temp)

        color = get_moment_color(moment_name)
        label = get_moment_label(moment_name)
        ax.plot(
            time,
            temp,
            color=color,
            linewidth=LINEWIDTH["medium"],
            label=label,
            alpha=ALPHA["line"],
        )
        moment_labels.append(label)

        offset = time[-1] + 10  # Gap between moments

    # Add comfort zone bands (without labels to keep legend clean)
    if all_temps:
        ax.axhspan(32, 35, color=COLORS["good"], alpha=0.1)
        ax.axhspan(28, 32, color=COLORS["medium"], alpha=0.1)
        ax.axhspan(35, 37, color=COLORS["poor"], alpha=0.1)

        # Apply fixed limits if configured, otherwise auto
        limits = AXIS_LIMITS.get("temp")
        if limits:
            ax.set_ylim(limits)
        else:
            y_min = min(all_temps) - 0.5
            y_max = max(all_temps) + 0.5
            ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Time (s)", fontsize=FONTSIZE["label"])
    ax.set_ylabel("Temperature (°C)", fontsize=FONTSIZE["label"])

    # Legend outside for clarity with many moments
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=FONTSIZE["legend"])
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add zone legend as text annotation
    ax.text(
        1.01,
        0.3,
        "Zones:\n32-35°C: Comfort\n28-32°C: Cool\n35-37°C: Warm",
        transform=ax.transAxes,
        fontsize=FONTSIZE["small"],
        color=COLORS["dark_gray"],
        va="top",
        fontweight="bold",
    )


def _plot_temp_metrics_panel(ax: plt.Axes, temp_data: Dict) -> None:
    """Plot temperature summary statistics across moments."""
    from src.visualization.config import AXIS_LIMITS

    ax.set_title(
        "Temperature Summary", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )

    metrics = temp_data.get("metrics")
    if metrics is None or metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No temperature metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    moments = metrics["moment"].tolist() if "moment" in metrics.columns else []
    if not moments:
        ax.text(
            0.5,
            0.5,
            "No moment data in metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    x = np.arange(len(moments))
    width = 0.6

    # Support both naming conventions (TEMP_Mean or temp_mean)
    mean_col = None
    sd_col = None
    for col in ["TEMP_Mean", "temp_mean"]:
        if col in metrics.columns:
            mean_col = col
            break
    for col in ["TEMP_SD", "temp_std"]:
        if col in metrics.columns:
            sd_col = col
            break

    if mean_col is None:
        ax.text(
            0.5,
            0.5,
            f"No temp mean column found\nAvailable: {list(metrics.columns)[:5]}...",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    means = metrics[mean_col].values
    sds = metrics[sd_col].values if sd_col else np.zeros_like(means)

    colors = [get_moment_color(m) for m in moments]

    bars = ax.bar(
        x,
        means,
        width,
        yerr=sds,
        capsize=5,
        color=colors,
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
        error_kw={"elinewidth": LINEWIDTH["normal"], "capthick": LINEWIDTH["normal"]},
    )

    # Value labels
    for bar, mean, sd in zip(bars, means, sds):
        if not np.isnan(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sd + 0.1,
                f"{mean:.1f}°C",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["annotation"],
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_moment_label(m) for m in moments], fontsize=FONTSIZE["tick"]
    )
    ax.set_ylabel("Temperature (°C)", fontsize=FONTSIZE["label"])

    # Apply fixed limits if configured
    limits = AXIS_LIMITS.get("temp")
    if limits:
        ax.set_ylim(limits)

    ax.grid(True, alpha=0.3, axis="y", linestyle="--")


def _plot_temp_eda_correlation_panel(
    ax: plt.Axes, temp_data: Dict, eda_data: Dict
) -> None:
    """
    Plot Temperature-EDA correlation coefficients as bar chart.

    Shows the Pearson correlation between temperature and tonic EDA
    for each moment. Negative correlation suggests stress response
    (peripheral vasoconstriction).
    """
    ax.set_title(
        "Temp-EDA Correlation", fontsize=FONTSIZE["subtitle"], fontweight="bold", pad=10
    )

    temp_signals = temp_data.get("signals", {})
    eda_signals = eda_data.get("signals", {})

    if not temp_signals or not eda_signals:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Find common moments
    common_moments = sorted(set(temp_signals.keys()) & set(eda_signals.keys()))

    if not common_moments:
        ax.text(
            0.5,
            0.5,
            "No matching moments",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    correlations = []
    valid_moments = []

    for moment_name in common_moments:
        temp_df = temp_signals[moment_name]
        eda_df = eda_signals[moment_name]

        # Find temperature column
        temp_col = None
        for col in ["TEMP_Clean", "TEMP_Raw", "temperature"]:
            if col in temp_df.columns:
                temp_col = col
                break

        # Find EDA column (prefer tonic for baseline correlation)
        eda_col = None
        for col in ["EDA_Tonic", "EDA_Clean", "eda"]:
            if col in eda_df.columns:
                eda_col = col
                break

        if temp_col is None or eda_col is None:
            continue

        # Get values and resample to match
        temp_vals = temp_df[temp_col].dropna().values
        eda_vals = eda_df[eda_col].dropna().values

        min_len = min(len(temp_vals), len(eda_vals))
        if min_len < 30:  # Need enough points for meaningful correlation
            continue

        # Resample to same length
        temp_resampled = np.interp(
            np.linspace(0, 1, min_len), np.linspace(0, 1, len(temp_vals)), temp_vals
        )
        eda_resampled = np.interp(
            np.linspace(0, 1, min_len), np.linspace(0, 1, len(eda_vals)), eda_vals
        )

        # Calculate Pearson correlation
        corr = np.corrcoef(temp_resampled, eda_resampled)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
            valid_moments.append(moment_name)

    if not correlations:
        ax.text(
            0.5,
            0.5,
            "Could not compute correlations",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Plot horizontal bar chart
    x = np.arange(len(valid_moments))
    colors = [get_moment_color(m) for m in valid_moments]

    bars = ax.barh(
        x,
        correlations,
        color=colors,
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
        height=0.6,
    )

    # Add value labels
    for bar, corr in zip(bars, correlations):
        # Position label based on sign
        if corr >= 0:
            ax.text(
                corr + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"r = {corr:.2f}",
                ha="left",
                va="center",
                fontsize=FONTSIZE["annotation"],
                fontweight="bold",
            )
        else:
            ax.text(
                corr - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"r = {corr:.2f}",
                ha="right",
                va="center",
                fontsize=FONTSIZE["annotation"],
                fontweight="bold",
            )

    # Reference line at 0
    ax.axvline(x=0, color=COLORS["gray"], linestyle="-", linewidth=1)

    # Y-axis labels
    ax.set_yticks(x)
    ax.set_yticklabels(
        [get_moment_label(m) for m in valid_moments], fontsize=FONTSIZE["tick"]
    )

    # X-axis
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Correlation (r)", fontsize=FONTSIZE["label"])
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    # Add interpretation note
    ax.text(
        0.5,
        -0.15,
        "Negative r: Stress → ↓Temp (vasoconstriction)  •  Positive r: Relaxation → ↑Temp",
        transform=ax.transAxes,
        fontsize=FONTSIZE["small"],
        color=COLORS["dark_gray"],
        ha="center",
        va="top",
        fontweight="bold",
    )


# =============================================================================
# FIGURE 5: QUALITY REPORT (Composite)
# =============================================================================


def plot_quality_report(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Create signal quality report figure.

    Layout:
    ┌───────────────────────────────────┬────────┐
    │   Signal Quality Overview         │ Legend │
    │   (Bar chart per modality)        │        │
    ├─────────────────┬─────────────────┼────────┤
    │   Data Coverage │   Quality       │ Legend │
    │   (% valid)     │   Distribution  │        │
    └─────────────────┴─────────────────┴────────┘

    Args:
        data: Dictionary containing all modality data
        output_path: Where to save the figure
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    # Wider figure to accommodate legends on right
    fig = plt.figure(figsize=(16, 10))
    # Leave space on right for legends
    gs = GridSpec(
        2,
        2,
        height_ratios=[1, 0.8],
        hspace=0.35,
        wspace=0.3,
        left=0.06,
        right=0.82,
        top=0.92,
        bottom=0.08,
    )

    # ========== Panel 1: Quality Overview ==========
    ax1 = fig.add_subplot(gs[0, :])
    _plot_quality_overview_panel(ax1, data, fig)

    # ========== Panel 2: Data Coverage ==========
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_data_coverage_panel(ax2, data, fig)

    # ========== Panel 3: Quality Distribution ==========
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_quality_distribution_panel(ax3, data, fig)

    # Main title - single line to avoid overlap with panel title
    subject = data.get("subject", "Unknown")
    session = data.get("session", "Unknown")
    fig.suptitle(
        f"Signal Quality Report — Subject {subject} • Session {session}",
        fontsize=FONTSIZE["title"] + 2,
        fontweight="bold",
        y=0.99,
    )

    # Note: No tight_layout - we use GridSpec with explicit margins to accommodate legends

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved quality report figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_quality_overview_panel(
    ax: plt.Axes, data: Dict, fig: plt.Figure = None
) -> None:
    """Plot mean quality score per modality and moment."""
    ax.set_title(
        "Signal Quality by Modality",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
        pad=10,
    )

    modalities = ["bvp", "eda", "hr", "temp"]
    quality_data = {}

    for modality in modalities:
        mod_data = data.get(modality, {})
        signals = mod_data.get("signals", {})

        for moment_name, moment_signals in signals.items():
            # Look for quality column
            quality_col = None
            for col in [
                f"{modality.upper()}_Quality",
                "PPG_Quality",
                "EDA_Quality",
                "HR_Quality",
                "TEMP_Quality",
                "quality",
            ]:
                if col in moment_signals.columns:
                    quality_col = col
                    break

            if quality_col:
                quality = moment_signals[quality_col].dropna().mean()
            else:
                quality = np.nan

            if modality not in quality_data:
                quality_data[modality] = {}
            quality_data[modality][moment_name] = quality

    if not quality_data:
        ax.text(
            0.5,
            0.5,
            "No quality data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
            color=COLORS["gray"],
        )
        return

    # Get all moments
    all_moments = set()
    for mod_moments in quality_data.values():
        all_moments.update(mod_moments.keys())
    moments = sorted(all_moments)

    x = np.arange(len(modalities))
    width = 0.8 / len(moments) if moments else 0.8

    handles = []
    for i, moment in enumerate(moments):
        values = [quality_data.get(mod, {}).get(moment, np.nan) for mod in modalities]
        offset = (i - len(moments) / 2 + 0.5) * width
        color = get_moment_color(moment)

        bars = ax.bar(
            x + offset,
            values,
            width * 0.9,
            label=get_moment_label(moment),
            color=color,
            alpha=ALPHA["high"],
            edgecolor="white",
        )
        handles.append(bars[0])

    # Add quality threshold lines
    line_acceptable = ax.axhline(
        0.7,
        color=COLORS["medium"],
        linestyle="--",
        linewidth=LINEWIDTH["normal"],
        alpha=0.7,
    )
    line_good = ax.axhline(
        0.9,
        color=COLORS["good"],
        linestyle="--",
        linewidth=LINEWIDTH["normal"],
        alpha=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modalities], fontsize=FONTSIZE["tick"])
    ax.set_ylabel("Mean Quality Score", fontsize=FONTSIZE["label"])
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Legend outside on right
    legend_handles = handles + [line_acceptable, line_good]
    legend_labels = [get_moment_label(m) for m in moments] + [
        "Acceptable (70%)",
        "Good (90%)",
    ]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=FONTSIZE["legend"],
        frameon=True,
        fancybox=True,
    )


def _plot_data_coverage_panel(ax: plt.Axes, data: Dict, fig: plt.Figure = None) -> None:
    """
    Plot data coverage showing % of valid and good quality samples per modality.

    Two bars per modality:
    - Valid samples: non-NaN quality values
    - Good quality: samples with quality >= 0.7
    """
    ax.set_title(
        "Data Quality Coverage",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
        pad=10,
    )

    modalities = ["bvp", "eda", "hr", "temp"]
    coverage_good = {}  # Quality >= 0.7
    coverage_valid = {}  # Non-NaN samples

    for modality in modalities:
        mod_data = data.get(modality, {})
        signals = mod_data.get("signals", {})

        total_samples = 0
        good_samples = 0
        valid_samples = 0

        for moment_signals in signals.values():
            if len(moment_signals) == 0:
                continue

            total_samples += len(moment_signals)

            # Find quality column
            quality_col = None
            for col in [
                f"{modality.upper()}_Quality",
                "PPG_Quality",
                "EDA_Quality",
                "HR_Quality",
                "TEMP_Quality",
                "quality",
            ]:
                if col in moment_signals.columns:
                    quality_col = col
                    break

            if quality_col:
                quality_vals = moment_signals[quality_col]
                valid_samples += quality_vals.notna().sum()
                good_samples += (quality_vals >= 0.7).sum()
            else:
                # No quality column - count non-NaN in main signal
                for col in [
                    f"{modality.upper()}_Clean",
                    f"{modality.upper()}_Raw",
                    "PPG_Clean",
                    "EDA_Clean",
                    "TEMP_Clean",
                ]:
                    if col in moment_signals.columns:
                        valid_samples += moment_signals[col].notna().sum()
                        good_samples += moment_signals[col].notna().sum()
                        break

        coverage_good[modality] = (
            good_samples / total_samples * 100 if total_samples > 0 else 0
        )
        coverage_valid[modality] = (
            valid_samples / total_samples * 100 if total_samples > 0 else 0
        )

    # Grouped bar chart: valid vs good quality
    x = np.arange(len(modalities))
    width = 0.35

    colors = [get_modality_color(m) for m in modalities]

    # Valid samples (lighter)
    ax.bar(
        x - width / 2,
        [coverage_valid[m] for m in modalities],
        width,
        color=colors,
        alpha=ALPHA["medium"],
        edgecolor="white",
        linewidth=1,
    )

    # Good quality samples (darker with hatch)
    bars2 = ax.bar(
        x + width / 2,
        [coverage_good[m] for m in modalities],
        width,
        color=colors,
        alpha=ALPHA["high"],
        edgecolor="white",
        linewidth=1,
        hatch="///",
    )

    # Add value labels on good quality bars
    for bar, val in zip(bars2, [coverage_good[m] for m in modalities]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE["small"],
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modalities], fontsize=FONTSIZE["tick"])
    ax.set_ylabel("Coverage (%)", fontsize=FONTSIZE["label"])
    ax.set_ylim(0, 115)
    ax.axhline(100, color=COLORS["grid"], linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Add legend inside plot area (top left to avoid overlap)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=COLORS["gray"],
            alpha=ALPHA["medium"],
            edgecolor="white",
            label="Valid",
        ),
        Patch(
            facecolor=COLORS["gray"],
            alpha=ALPHA["high"],
            hatch="///",
            edgecolor="white",
            label="Good (≥70%)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=FONTSIZE["small"],
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )


def _plot_quality_distribution_panel(
    ax: plt.Axes, data: Dict, fig: plt.Figure = None
) -> None:
    """
    Plot quality score distribution using boxplots per modality.

    Shows all 4 modalities with boxplots colored by modality.
    For constant values (std=0), shows a horizontal bar instead.
    """
    ax.set_title(
        "Quality Score Distribution",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
        pad=10,
    )

    modalities = ["bvp", "eda", "hr", "temp"]
    modality_quality = {}

    for modality in modalities:
        mod_data = data.get(modality, {})
        signals = mod_data.get("signals", {})

        quality_vals = []
        for moment_signals in signals.values():
            quality_col = None
            for col in [
                f"{modality.upper()}_Quality",
                "PPG_Quality",
                "EDA_Quality",
                "HR_Quality",
                "TEMP_Quality",
                "quality",
            ]:
                if col in moment_signals.columns:
                    quality_col = col
                    break

            if quality_col:
                vals = moment_signals[quality_col].dropna().values
                quality_vals.extend(vals)

        modality_quality[modality] = quality_vals if quality_vals else None

    # Always show all 4 modalities
    positions = np.arange(len(modalities))
    colors = [get_modality_color(m) for m in modalities]

    # Collect legend handles
    from matplotlib.patches import Patch

    legend_handles = []

    # Draw boxplots or bars for all modalities
    for i, modality in enumerate(modalities):
        color = colors[i]

        if (
            modality_quality[modality] is not None
            and len(modality_quality[modality]) > 0
        ):
            vals = modality_quality[modality]
            std_val = np.std(vals)
            mean_val = np.mean(vals)

            if std_val < 0.001:  # Constant values (like HR/TEMP with all 1.0)
                # Draw a thick horizontal bar at the constant value
                ax.barh(
                    mean_val,
                    0.6,
                    height=0.04,
                    left=i - 0.3,
                    color=color,
                    alpha=ALPHA["high"],
                    edgecolor="white",
                    linewidth=1.5,
                )
                # Add diamond marker
                ax.scatter(
                    i,
                    mean_val,
                    s=100,
                    color=color,
                    marker="D",
                    edgecolors="white",
                    linewidths=2,
                    zorder=5,
                )
                # Add value annotation
                ax.text(
                    i,
                    mean_val + 0.06,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONTSIZE["small"],
                    fontweight="bold",
                    color=COLORS["text"],
                )

                legend_handles.append(
                    Patch(
                        facecolor=color,
                        alpha=ALPHA["high"],
                        edgecolor="white",
                        label=f"{modality.upper()} (constant)",
                    )
                )
            else:
                # Draw boxplot for variable data
                bp = ax.boxplot(
                    [vals],
                    positions=[i],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                )

                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(ALPHA["high"])
                bp["boxes"][0].set_edgecolor("white")
                bp["boxes"][0].set_linewidth(1.5)
                bp["medians"][0].set_color("white")
                bp["medians"][0].set_linewidth(2)

                # Add median annotation
                median = np.median(vals)
                ax.text(
                    i,
                    median + 0.04,
                    f"{median:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONTSIZE["small"],
                    fontweight="bold",
                    color=COLORS["text"],
                )

                legend_handles.append(
                    Patch(
                        facecolor=color,
                        alpha=ALPHA["high"],
                        edgecolor="white",
                        label=modality.upper(),
                    )
                )
        else:
            # Draw placeholder bar with "N/A" label
            ax.bar(
                i,
                0.5,
                width=0.6,
                color=color,
                alpha=0.2,
                edgecolor=color,
                linewidth=2,
                linestyle="--",
            )
            ax.text(
                i,
                0.55,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=FONTSIZE["annotation"],
                color=COLORS["gray"],
                fontweight="bold",
                fontstyle="italic",
            )

            legend_handles.append(
                Patch(
                    facecolor=color,
                    alpha=0.2,
                    edgecolor=color,
                    linestyle="--",
                    label=f"{modality.upper()} (N/A)",
                )
            )

    # Add quality threshold lines
    ax.axhline(
        0.7,
        color=COLORS["medium"],
        linestyle="--",
        linewidth=LINEWIDTH["medium"],
        alpha=0.8,
    )
    ax.axhline(
        0.9,
        color=COLORS["good"],
        linestyle="--",
        linewidth=LINEWIDTH["medium"],
        alpha=0.8,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([m.upper() for m in modalities], fontsize=FONTSIZE["tick"])
    ax.set_ylabel("Quality Score", fontsize=FONTSIZE["label"])
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, len(modalities) - 0.5)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Legend outside on right - add threshold lines
    legend_handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                color=COLORS["medium"],
                linestyle="--",
                linewidth=LINEWIDTH["medium"],
                label="Acceptable (70%)",
            ),
            plt.Line2D(
                [0],
                [0],
                color=COLORS["good"],
                linestyle="--",
                linewidth=LINEWIDTH["medium"],
                label="Good (90%)",
            ),
        ]
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=FONTSIZE["legend"],
        frameon=True,
        fancybox=True,
    )
