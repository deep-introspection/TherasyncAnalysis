"""
TEMP-specific visualization plots.

Implements visualizations for body temperature:
- Visualization #7: Temperature Timeline (time-series with zones)
- Visualization #8: Temperature Metrics Comparison (bar chart by moment)

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from ..config import (
    COLORS,
    FIGSIZE,
    FONTSIZE,
    LINEWIDTH,
    ALPHA,
    apply_plot_style,
    get_moment_color,
    get_moment_label,
)


def plot_temp_timeline(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Plot temperature dynamics over time with physiological zones.

    Visualization #7: Time-series plot showing:
    - Temperature signal over time per moment
    - Horizontal zones for hypothermia/normal/hyperthermia ranges
    - Mean ± SD indicators per moment

    Args:
        data: Dictionary containing 'temp' data with 'signals' sub-dict
              Each moment should have 'time' and 'TEMP_Clean' columns
        output_path: Where to save the figure (optional)
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

    temp_data = data.get("temp", {})

    if not temp_data or "signals" not in temp_data:
        ax.text(
            0.5,
            0.5,
            "No temperature data available",
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

    signals = temp_data["signals"]

    if not signals:
        ax.text(
            0.5,
            0.5,
            "No temperature signals available",
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

    # Collect data from all moments
    all_temps = []
    moment_data = []
    offset = 0

    for moment in sorted(signals.keys()):
        df = signals[moment]

        if df is None or df.empty:
            continue

        # Get temperature column (use TEMP_Clean from processing)
        temp_col = "TEMP_Clean" if "TEMP_Clean" in df.columns else "temperature"
        if temp_col not in df.columns:
            continue

        temp_values = df[temp_col].values
        time_values = df["time"].values

        # Adjust time to create continuous timeline
        adjusted_time = time_values + offset

        all_temps.extend(temp_values)

        moment_data.append(
            {
                "moment": moment,
                "time": adjusted_time,
                "temp": temp_values,
                "start": offset,
                "end": offset + time_values.max(),
                "mean": np.mean(temp_values),
                "std": np.std(temp_values),
            }
        )

        # Update offset for next moment (with small gap)
        offset = offset + time_values.max() + 10

    if not moment_data:
        ax.text(
            0.5,
            0.5,
            "No temperature data available",
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

    # Calculate y-axis limits
    all_temps = np.array(all_temps)
    temp_min, temp_max = np.min(all_temps), np.max(all_temps)

    # Extend limits to show physiological context (normal skin temp: 32-35°C)
    y_min = min(temp_min - 0.5, 30.0)
    y_max = max(temp_max + 0.5, 37.0)

    # Draw physiological zones (skin temperature ranges)
    # Cold zone (below 32°C)
    ax.axhspan(
        y_min, 32.0, color=COLORS["good"], alpha=0.1, label="Cool (<32°C)", zorder=0
    )
    # Normal zone (32-35°C)
    ax.axhspan(
        32.0,
        35.0,
        color=COLORS["medium"],
        alpha=0.1,
        label="Normal (32-35°C)",
        zorder=0,
    )
    # Warm zone (above 35°C)
    ax.axhspan(
        35.0, y_max, color=COLORS["poor"], alpha=0.1, label="Warm (>35°C)", zorder=0
    )

    # Plot each moment
    for idx, m in enumerate(moment_data):
        color = get_moment_color(m["moment"])
        label = f"{get_moment_label(m['moment'])} (μ={m['mean']:.1f}±{m['std']:.2f}°C)"

        ax.plot(
            m["time"],
            m["temp"],
            color=color,
            linewidth=LINEWIDTH["signal"],
            alpha=ALPHA["line"],
            label=label,
        )

        # Add moment separator (vertical line) - except for last
        if idx < len(moment_data) - 1:
            ax.axvline(
                m["end"] + 5,
                color=COLORS["dark_gray"],
                linestyle="--",
                linewidth=LINEWIDTH["thin"],
                alpha=0.5,
                zorder=10,
            )

    # Formatting
    ax.set_xlabel("Time (s)", fontsize=FONTSIZE["label"], fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=FONTSIZE["label"], fontweight="bold")
    ax.set_title(
        "Body Temperature Dynamics Timeline\n(with physiological zones)",
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


def plot_temp_metrics_comparison(
    data: Dict, output_path: Optional[Path] = None, show: bool = False
) -> plt.Figure:
    """
    Plot temperature metrics comparison between moments.

    Visualization #8: Grouped bar chart showing:
    - Mean temperature per moment (with error bars for SD)
    - Min/Max range indicator
    - Trend direction (slope)

    Args:
        data: Dictionary containing 'temp' data with 'signals' sub-dict
        output_path: Where to save the figure (optional)
        show: Whether to display the figure

    Returns:
        Figure object
    """
    apply_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE["wide"])

    temp_data = data.get("temp", {})

    if not temp_data or "signals" not in temp_data:
        ax1.text(
            0.5,
            0.5,
            "No temperature data available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=FONTSIZE["label"],
        )
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    signals = temp_data["signals"]

    # Calculate statistics for each moment
    stats = {}
    for moment in sorted(signals.keys()):
        df = signals[moment]

        if df is None or df.empty:
            continue

        temp_col = "TEMP_Clean" if "TEMP_Clean" in df.columns else "temperature"
        if temp_col not in df.columns:
            continue

        temp_values = df[temp_col].values
        time_values = df["time"].values

        # Calculate slope (°C per minute)
        if len(time_values) > 1:
            slope_per_sec = np.polyfit(time_values, temp_values, 1)[0]
            slope_per_min = slope_per_sec * 60
        else:
            slope_per_min = 0.0

        stats[moment] = {
            "mean": np.mean(temp_values),
            "std": np.std(temp_values),
            "min": np.min(temp_values),
            "max": np.max(temp_values),
            "range": np.max(temp_values) - np.min(temp_values),
            "slope": slope_per_min,
        }

    if not stats:
        ax1.text(
            0.5,
            0.5,
            "Insufficient temperature data",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=FONTSIZE["label"],
        )
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    moments = list(stats.keys())
    x = np.arange(len(moments))

    # ========== Panel 1: Mean Temperature with Error Bars ==========
    means = [stats[m]["mean"] for m in moments]
    stds = [stats[m]["std"] for m in moments]
    mins = [stats[m]["min"] for m in moments]
    maxs = [stats[m]["max"] for m in moments]
    colors = [get_moment_color(m) for m in moments]

    ax1.bar(x, means, 0.6, color=colors, alpha=0.8, edgecolor="white", linewidth=2)

    # Add error bars
    ax1.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor=COLORS["dark_gray"],
        elinewidth=2,
        capsize=5,
        capthick=2,
    )

    # Add min/max range markers
    for i, (moment, mean, min_val, max_val) in enumerate(
        zip(moments, means, mins, maxs)
    ):
        ax1.scatter(
            [i],
            [min_val],
            marker="v",
            color=colors[i],
            s=50,
            zorder=5,
            edgecolors="white",
            linewidths=1,
        )
        ax1.scatter(
            [i],
            [max_val],
            marker="^",
            color=colors[i],
            s=50,
            zorder=5,
            edgecolors="white",
            linewidths=1,
        )

        # Value label on bar
        ax1.text(
            i,
            mean + max(stds) * 0.1,
            f"{mean:.1f}°C",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE["annotation"],
            fontweight="bold",
            color=colors[i],
        )

    ax1.set_xlabel("Moment", fontsize=FONTSIZE["label"], fontweight="bold")
    ax1.set_ylabel("Temperature (°C)", fontsize=FONTSIZE["label"], fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [get_moment_label(m) for m in moments],
        fontsize=FONTSIZE["tick"],
        fontweight="bold",
    )
    ax1.grid(True, alpha=ALPHA["fill"], axis="y", linestyle="--")
    ax1.set_title(
        "Mean Temperature by Moment\n(error bars: ±1 SD, markers: min/max)",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
    )

    # ========== Panel 2: Temperature Trend (Slope) ==========
    slopes = [stats[m]["slope"] for m in moments]

    # Color based on trend direction
    slope_colors = []
    for slope in slopes:
        if slope > 0.01:
            slope_colors.append(COLORS["poor"])  # Warming
        elif slope < -0.01:
            slope_colors.append(COLORS["good"])  # Cooling
        else:
            slope_colors.append(COLORS["medium"])  # Stable

    ax2.bar(
        x, slopes, 0.6, color=slope_colors, alpha=0.8, edgecolor="white", linewidth=2
    )

    # Add zero line
    ax2.axhline(0, color=COLORS["dark_gray"], linestyle="-", linewidth=1.5, zorder=0)

    # Value labels
    for i, (moment, slope) in enumerate(zip(moments, slopes)):
        y_offset = 0.002 if slope >= 0 else -0.002
        va = "bottom" if slope >= 0 else "top"
        ax2.text(
            i,
            slope + y_offset,
            f"{slope:.3f}",
            ha="center",
            va=va,
            fontsize=FONTSIZE["annotation"],
            fontweight="bold",
            color=slope_colors[i],
        )

    ax2.set_xlabel("Moment", fontsize=FONTSIZE["label"], fontweight="bold")
    ax2.set_ylabel("Slope (°C/min)", fontsize=FONTSIZE["label"], fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [get_moment_label(m) for m in moments],
        fontsize=FONTSIZE["tick"],
        fontweight="bold",
    )
    ax2.grid(True, alpha=ALPHA["fill"], axis="y", linestyle="--")
    ax2.set_title(
        "Temperature Trend by Moment\n(positive = warming, negative = cooling)",
        fontsize=FONTSIZE["subtitle"],
        fontweight="bold",
    )

    # Overall title
    fig.suptitle(
        f"Temperature Metrics Comparison\n"
        f"Subject {data.get('subject', 'Unknown')}, Session {data.get('session', 'Unknown')}",
        fontsize=FONTSIZE["title"],
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
