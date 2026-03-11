"""
Signal Plots Module.

Implements time-series and temporal visualizations:
- Visualization #1: Multi-Signal Dashboard
- Visualization #6: HR Dynamics Timeline

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional
from pathlib import Path

from ..config import (
    COLORS,
    ALPHA,
    FIGSIZE,
    FONTSIZE,
    LINEWIDTH,
    MARKERSIZE,
    apply_plot_style,
    get_moment_color,
    get_moment_label,
)


def plot_multisignal_dashboard(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Visualization #1: Multi-Signal Dashboard.

    Creates a 4-panel synchronized time-series plot showing:
    - Panel 1: BVP signal + detected peaks
    - Panel 2: HR instantaneous (BPM)
    - Panel 3: EDA tonic (line) + phasic (filled area)
    - Panel 4: SCR events (stem plot)

    Moments are displayed sequentially with vertical separators.

    Args:
        data: Dictionary containing 'bvp', 'eda', 'hr' data
        output_path: Where to save the figure
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    # Create figure with 4 subplots - extra width for legends on the right
    fig = plt.figure(figsize=(FIGSIZE["dashboard"][0] + 2, FIGSIZE["dashboard"][1]))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.3)

    # Collect all moments for x-axis alignment
    # Discover moments from available data (any modality)
    moments = []
    for modality in ["bvp", "eda", "hr"]:
        if modality in data and "signals" in data[modality]:
            moments.extend(data[modality]["signals"].keys())
    moments = sorted(list(set(moments)))  # Unique and sorted

    # Calculate moment boundaries for vertical separators
    moment_boundaries = {}
    offset = 0
    for moment in moments:
        # Try to get duration from any available modality
        duration = None
        for modality in ["bvp", "eda", "hr"]:
            if modality in data and "signals" in data[modality]:
                signals = data[modality]["signals"]
                if moment in signals and "time" in signals[moment].columns:
                    duration = signals[moment]["time"].max()
                    break

        if duration is not None:
            moment_boundaries[moment] = {"start": offset, "end": offset + duration}
            offset = offset + duration + 10  # 10s gap

    # ========== Panel 1: BVP Signal ==========
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bvp_signal(ax1, data.get("bvp", {}), moments, moment_boundaries)
    add_moment_separators(ax1, moment_boundaries, label_position="top")
    ax1.set_title(
        "Blood Volume Pulse (BVP)", fontsize=FONTSIZE["title"], fontweight="bold"
    )
    ax1.set_ylabel("BVP (a.u.)", fontsize=FONTSIZE["label"])
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=FONTSIZE["legend"],
        framealpha=0.95,
    )

    # ========== Panel 2: Heart Rate ==========
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    plot_hr_signal(ax2, data.get("hr", {}), moments, moment_boundaries)
    add_moment_separators(ax2, moment_boundaries)
    ax2.set_title("Heart Rate", fontsize=FONTSIZE["title"], fontweight="bold")
    ax2.set_ylabel("HR (BPM)", fontsize=FONTSIZE["label"])
    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=FONTSIZE["legend"],
        framealpha=0.95,
    )

    # ========== Panel 3: EDA Tonic + Phasic ==========
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    plot_eda_signal(ax3, data.get("eda", {}), moments, moment_boundaries)
    add_moment_separators(ax3, moment_boundaries)
    ax3.set_title(
        "Electrodermal Activity (EDA)", fontsize=FONTSIZE["title"], fontweight="bold"
    )
    ax3.set_ylabel("EDA (µS)", fontsize=FONTSIZE["label"])
    ax3.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=FONTSIZE["legend"],
        framealpha=0.95,
    )

    # ========== Panel 4: SCR Events ==========
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    plot_scr_events(ax4, data.get("eda", {}), moments, moment_boundaries)
    add_moment_separators(ax4, moment_boundaries)
    ax4.set_title(
        "Skin Conductance Responses (SCR)",
        fontsize=FONTSIZE["title"],
        fontweight="bold",
    )
    ax4.set_ylabel("Amplitude (µS)", fontsize=FONTSIZE["label"])
    ax4.set_xlabel("Time (seconds)", fontsize=FONTSIZE["label"])

    # Add overall title
    subject = data.get("subject", "Unknown")
    session = data.get("session", "Unknown")
    fig.suptitle(
        f"Physiological Signals Dashboard - Subject {subject}, Session {session}",
        fontsize=FONTSIZE["title"] + 2,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.98])  # Leave space for legends on right

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def add_moment_separators(
    ax: plt.Axes, moment_boundaries: Dict, label_position: str = "none"
):
    """
    Add vertical lines and labels to mark moment transitions.

    Args:
        ax: Matplotlib axes
        moment_boundaries: Dict with moment names and their start/end times
        label_position: 'top', 'bottom', or 'none'
    """
    for moment, bounds in moment_boundaries.items():
        # Vertical line at start
        ax.axvline(
            bounds["start"],
            color=COLORS["dark_gray"],
            linestyle="--",
            linewidth=LINEWIDTH["normal"],
            alpha=0.6,
            zorder=10,
        )

        # Label at top of panel
        if label_position == "top":
            mid_time = (bounds["start"] + bounds["end"]) / 2
            ax.text(
                mid_time,
                0.98,
                get_moment_label(moment),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=FONTSIZE["annotation"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=get_moment_color(moment),
                    alpha=0.3,
                    edgecolor="none",
                ),
            )


def plot_bvp_signal(
    ax: plt.Axes, bvp_data: Dict, moments: list, moment_boundaries: Dict
):
    """Plot BVP signal with detected peaks."""
    if not bvp_data or "signals" not in bvp_data:
        ax.text(
            0.5,
            0.5,
            "No BVP data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        return

    for moment in moments:
        if moment not in bvp_data["signals"] or moment not in moment_boundaries:
            continue

        signals = bvp_data["signals"][moment]
        color = get_moment_color(moment)
        offset = moment_boundaries[moment]["start"]

        # Plot cleaned BVP signal
        time = signals["time"].values + offset
        if "PPG_Clean" in signals.columns:
            ax.plot(
                time,
                signals["PPG_Clean"],
                color=color,
                linewidth=LINEWIDTH["normal"],
                label=get_moment_label(moment),
                alpha=ALPHA["line"],
            )

        # Mark detected peaks - use same color as moment signal
        if "PPG_Peaks" in signals.columns:
            peaks = signals[signals["PPG_Peaks"] == 1]
            if len(peaks) > 0:
                ax.scatter(
                    peaks["time"].values + offset,
                    peaks["PPG_Clean"].values,
                    color=color,
                    s=MARKERSIZE["small"],
                    marker="o",
                    zorder=5,
                    alpha=0.7,
                    edgecolors="white",
                    linewidths=0.3,
                )

    ax.grid(True, alpha=ALPHA["fill"])


def plot_hr_signal(ax: plt.Axes, hr_data: Dict, moments: list, moment_boundaries: Dict):
    """Plot instantaneous heart rate with zones."""
    if not hr_data or "signals" not in hr_data:
        ax.text(
            0.5,
            0.5,
            "No HR data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        return

    all_hr = []

    # HR data is now split by moments (restingstate, therapy)
    # Each moment has its own DataFrame with HR_Clean column
    for moment in moments:
        if moment not in hr_data["signals"] or moment not in moment_boundaries:
            continue

        signals = hr_data["signals"][moment]
        bounds = moment_boundaries[moment]

        if len(signals) == 0:
            continue

        color = get_moment_color(moment)
        time = signals["time"].values + bounds["start"]

        # Use HR_Clean column from new format
        if "HR_Clean" in signals.columns:
            hr_values = signals["HR_Clean"].values
            all_hr.extend(hr_values)
            ax.plot(
                time,
                hr_values,
                color=color,
                linewidth=LINEWIDTH["medium"],
                label=get_moment_label(moment),
                alpha=ALPHA["line"],
            )

    # Add horizontal zones
    if all_hr:
        hr_mean = np.mean(all_hr)
        ax.axhline(
            hr_mean,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=LINEWIDTH["thin"],
            alpha=0.5,
            label="Mean HR",
        )

        # Set reasonable y-limits first
        hr_min, hr_max = np.min(all_hr), np.max(all_hr)
        y_margin = (hr_max - hr_min) * 0.1
        ax.set_ylim(hr_min - y_margin, hr_max + y_margin)

        # Elevated zone (Mean + 20 BPM)
        ax.axhspan(
            hr_mean + 20,
            ax.get_ylim()[1],
            color=COLORS["poor"],
            alpha=0.1,
            label="Elevated",
        )

        # Rest zone (Mean - 20 BPM)
        ax.axhspan(
            ax.get_ylim()[0],
            max(40, hr_mean - 20),
            color=COLORS["good"],
            alpha=0.1,
            label="Rest",
        )

    ax.grid(True, alpha=ALPHA["fill"])


def plot_eda_signal(
    ax: plt.Axes, eda_data: Dict, moments: list, moment_boundaries: Dict
):
    """Plot EDA tonic (line) and phasic (filled area)."""
    if not eda_data or "signals" not in eda_data:
        ax.text(
            0.5,
            0.5,
            "No EDA data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        return

    for moment in moments:
        if moment not in eda_data["signals"] or moment not in moment_boundaries:
            continue

        signals = eda_data["signals"][moment]
        color = get_moment_color(moment)
        offset = moment_boundaries[moment]["start"]

        time = signals["time"].values + offset

        # Plot tonic component (baseline) - use modality color
        if "EDA_Tonic" in signals.columns:
            ax.plot(
                time,
                signals["EDA_Tonic"],
                color=color,
                linewidth=LINEWIDTH["thick"],
                label=f"{get_moment_label(moment)} - Tonic",
                alpha=ALPHA["line"],
            )

        # Plot phasic component (filled area) - lighter version of moment color
        if "EDA_Phasic" in signals.columns:
            ax.fill_between(
                time,
                0,
                signals["EDA_Phasic"],
                color=color,
                alpha=ALPHA["fill"],
                label=f"{get_moment_label(moment)} - Phasic",
            )

    ax.grid(True, alpha=ALPHA["fill"])


def plot_scr_events(
    ax: plt.Axes, eda_data: Dict, moments: list, moment_boundaries: Dict
):
    """Plot SCR events as stem plot."""
    if not eda_data or "events" not in eda_data:
        ax.text(
            0.5,
            0.5,
            "No SCR events available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        return

    for moment in moments:
        if moment not in eda_data["events"] or moment not in moment_boundaries:
            continue

        events = eda_data["events"][moment]
        color = get_moment_color(moment)
        offset = moment_boundaries[moment]["start"]

        if len(events) == 0:
            continue

        # Get SCR onset times and amplitudes (note: columns are 'onset' and 'amplitude')
        if "onset" in events.columns and "amplitude" in events.columns:
            # Filter out NaN amplitudes
            valid_events = events.dropna(subset=["amplitude"])
            if len(valid_events) == 0:
                continue

            onsets = valid_events["onset"].values + offset
            amplitudes = valid_events["amplitude"].values

            # Stem plot
            markerline, stemlines, baseline = ax.stem(
                onsets, amplitudes, linefmt=color, markerfmt="o", basefmt="k-"
            )
            markerline.set_markersize(MARKERSIZE["medium"])
            markerline.set_color(color)
            markerline.set_label(get_moment_label(moment))
            stemlines.set_linewidth(LINEWIDTH["normal"])
            stemlines.set_alpha(ALPHA["overlay"])

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=ALPHA["fill"])
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=FONTSIZE["legend"],
        framealpha=0.95,
    )


def plot_hr_dynamics_timeline(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Visualization #6: HR Dynamics Timeline.

    Shows HR over time with color-coded zones (rest/moderate/elevated) per moment.
    Each moment has its own zones calculated independently.

    Args:
        data: Dictionary containing 'hr' data with 'signals' sub-dict
        output_path: Where to save the figure (optional)
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

    hr_data = data.get("hr", {})

    if not hr_data or "signals" not in hr_data:
        ax.text(
            0.5,
            0.5,
            "No HR data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # HR data is now split by moments (restingstate, therapy)
    # Each moment has its own DataFrame with HR_Clean column
    signals = hr_data["signals"]

    if not signals:
        ax.text(
            0.5,
            0.5,
            "No HR signals available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # Get moment boundaries from BVP data to align timeline
    moment_boundaries = {}
    bvp_data = data.get("bvp", {})

    if "signals" in bvp_data:
        # Use BVP signal lengths to infer moment boundaries
        offset = 0
        for moment in bvp_data["signals"].keys():
            bvp_signals = bvp_data["signals"][moment]
            duration = bvp_signals["time"].max() if "time" in bvp_signals.columns else 0
            moment_boundaries[moment] = {"start": offset, "end": offset + duration}
            offset += duration

    # Build moment data from per-moment HR signals
    moment_data = []

    for moment in signals.keys():
        moment_hr = signals[moment]

        if moment_hr is None or moment_hr.empty:
            continue

        # Get HR column (use HR_Clean from new format)
        hr_col = "HR_Clean" if "HR_Clean" in moment_hr.columns else "hr"
        if hr_col not in moment_hr.columns:
            continue

        hr_values = moment_hr[hr_col].values
        time_values = moment_hr["time"].values

        # Use moment boundaries if available, otherwise use time range
        if moment in moment_boundaries:
            start_time = moment_boundaries[moment]["start"]
            end_time = moment_boundaries[moment]["end"]
            # Adjust time values to global timeline
            adjusted_time = time_values + start_time
        else:
            start_time = time_values.min()
            end_time = time_values.max()
            adjusted_time = time_values

        moment_data.append(
            {
                "moment": moment,
                "time": adjusted_time,
                "hr": hr_values,
                "start": start_time,
                "end": end_time,
                "hr_mean": np.mean(hr_values),
            }
        )

    if not moment_data:
        ax.text(
            0.5,
            0.5,
            "No HR data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE["label"],
        )
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # Calculate global y-axis limits
    all_hr = np.concatenate([m["hr"] for m in moment_data])
    hr_min, hr_max = np.min(all_hr), np.max(all_hr)
    y_margin = (hr_max - hr_min) * 0.1
    y_min = hr_min - y_margin
    y_max = hr_max + y_margin

    # Plot each moment with its own zones
    for idx, moment_info in enumerate(moment_data):
        moment = moment_info["moment"]
        time = moment_info["time"]
        hr = moment_info["hr"]
        hr_mean = moment_info["hr_mean"]
        hr_std = np.std(hr)  # Calculate standard deviation for this moment
        start = moment_info["start"]
        end = moment_info["end"]

        color = get_moment_color(moment)

        # Define zones for this moment (±1 SD around mean)
        zone_rest_max = hr_mean - hr_std
        zone_elevated_min = hr_mean + hr_std

        # Fill background zones for this moment's timespan
        # Rest zone (green)
        ax.fill_between(
            [start, end],
            y_min,
            zone_rest_max,
            color=COLORS["good"],
            alpha=0.15,
            zorder=0,
        )
        # Moderate zone (yellow)
        ax.fill_between(
            [start, end],
            zone_rest_max,
            zone_elevated_min,
            color=COLORS["medium"],
            alpha=0.15,
            zorder=0,
        )
        # Elevated zone (red)
        ax.fill_between(
            [start, end],
            zone_elevated_min,
            y_max,
            color=COLORS["poor"],
            alpha=0.15,
            zorder=0,
        )

        # Plot HR line for this moment
        ax.plot(
            time,
            hr,
            color=color,
            linewidth=LINEWIDTH["thick"],
            label=f"{get_moment_label(moment)} (μ={hr_mean:.1f} ±{hr_std:.1f} BPM)",
            alpha=ALPHA["line"],
        )

        # Add moment separator (vertical line) - except for last moment
        if idx < len(moment_data) - 1:
            ax.axvline(
                end + 5,
                color="black",
                linestyle="--",
                linewidth=LINEWIDTH["normal"],
                alpha=0.5,
                zorder=10,
            )

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=FONTSIZE["label"], fontweight="bold")
    ax.set_ylabel("Heart Rate (BPM)", fontsize=FONTSIZE["label"], fontweight="bold")
    ax.set_title(
        "Heart Rate Dynamics Timeline\n(with Rest/Moderate/Elevated zones per moment: mean ± 1 SD)",
        fontsize=FONTSIZE["title"],
        fontweight="bold",
    )
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right", fontsize=FONTSIZE["legend"], framealpha=0.95)
    ax.grid(True, alpha=ALPHA["fill"], axis="both", linestyle="--")
    ax.tick_params(labelsize=FONTSIZE["tick"])

    # Overall title
    fig.suptitle(
        f"Subject {data.get('subject', 'Unknown')}, Session {data.get('session', 'Unknown')}",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
