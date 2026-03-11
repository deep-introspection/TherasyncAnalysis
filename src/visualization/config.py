"""
Visualization Configuration Module.

Defines plotting styles, colors, and parameters for consistent visualizations.
Inspired by modern data visualization best practices with soft pastel colors.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
from src.core.config_loader import ConfigLoader

# Load configuration and build moment labels from config.yaml
_config_loader = ConfigLoader()
_config = _config_loader.config

# Build MOMENT_LABELS dictionary from config.yaml moments
MOMENT_LABELS = {
    moment["name"]: moment.get("displayname", moment["name"])
    for moment in _config.get("moments", [])
}

# =============================================================================
# COLOR PALETTE - Soft Pastels for Professional Visualizations
# =============================================================================

COLORS: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # Primary Signal Colors (soft pastels)
    # -------------------------------------------------------------------------
    "bvp": "#C4A8D4",  # Lavender - Blood Volume Pulse
    "eda": "#B8D4A8",  # Sage Green - Electrodermal Activity
    "hr": "#F4A4B8",  # Rose Pink - Heart Rate
    "temp": "#E8C87A",  # Golden - Temperature
    # -------------------------------------------------------------------------
    # Moment Colors (contrasting but harmonious)
    # -------------------------------------------------------------------------
    "restingstate": "#7EB8DA",  # Sky Blue - Calm state
    "therapy": "#F4A4B8",  # Rose Pink - Active state
    # -------------------------------------------------------------------------
    # EDA Components
    # -------------------------------------------------------------------------
    "tonic": "#7EB8DA",  # Sky Blue - Baseline arousal
    "phasic": "#B8D4A8",  # Sage Green - Phasic responses
    "scr": "#E17055",  # Coral - SCR events
    # -------------------------------------------------------------------------
    # HRV Components
    # -------------------------------------------------------------------------
    "lf": "#E8C87A",  # Golden - Low frequency (sympathetic)
    "hf": "#7EB8DA",  # Sky Blue - High frequency (parasympathetic)
    "vlf": "#C4A8D4",  # Lavender - Very low frequency
    # -------------------------------------------------------------------------
    # Quality Indicators
    # -------------------------------------------------------------------------
    "good": "#6DD47E",  # Fresh Green - Good quality
    "medium": "#F8B500",  # Amber - Medium quality
    "poor": "#E17055",  # Coral - Poor quality
    # -------------------------------------------------------------------------
    # Utility Colors
    # -------------------------------------------------------------------------
    "grid": "#CCCCCC",  # Light gray grid
    "text": "#2C3E50",  # Dark blue-gray text
    "background": "#FFFFFF",  # White background
    "gray": "#95A5A6",  # Neutral gray
    "dark_gray": "#2C3E50",  # Dark blue-gray
    "light_gray": "#ECF0F1",  # Very light gray
    # -------------------------------------------------------------------------
    # Diverging Colors (for correlations, differences)
    # -------------------------------------------------------------------------
    "negative": "#E17055",  # Coral - Negative values
    "zero": "#FFFFFF",  # White - Zero/neutral
    "positive": "#00CEC9",  # Teal - Positive values
}

# Moment color palette (supports multiple moments with modulo fallback)
MOMENT_COLORS = [
    "#7EB8DA",  # Sky Blue
    "#F4A4B8",  # Rose Pink
    "#B8D4A8",  # Sage Green
    "#E8C87A",  # Golden
    "#C4A8D4",  # Lavender
    "#A8D4D0",  # Soft Teal
    "#F5B7B1",  # Blush
    "#AED6F1",  # Light Blue
]

# Map known moment names to indices for backward compatibility
MOMENT_NAME_TO_INDEX = {
    "restingstate": 0,
    "therapy": 1,
}

# =============================================================================
# TRANSPARENCY LEVELS
# =============================================================================

ALPHA = {
    "low": 0.2,
    "medium": 0.4,
    "high": 0.7,
    "fill": 0.25,  # Background fills
    "overlay": 0.5,
    "scatter": 0.6,
    "line": 0.85,  # Line plots
    "solid": 1.0,
}

# =============================================================================
# FIGURE SIZES (width, height in inches)
# =============================================================================

FIGSIZE = {
    "small": (8, 6),
    "medium": (12, 8),
    "large": (16, 10),
    "dashboard": (16, 12),
    "wide": (14, 6),
    "square": (10, 10),
}

# =============================================================================
# TYPOGRAPHY
# =============================================================================

FONTSIZE = {
    "title": 16,
    "subtitle": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
    "small": 8,
}

# =============================================================================
# LINE WIDTHS
# =============================================================================

LINEWIDTH = {
    "thin": 0.5,
    "normal": 1.0,
    "signal": 1.5,
    "medium": 1.5,
    "thick": 2.0,
    "extra_thick": 2.5,
}

# =============================================================================
# MARKER SIZES
# =============================================================================

MARKERSIZE = {
    "tiny": 2,
    "small": 4,
    "medium": 6,
    "large": 8,
    "extra_large": 10,
}

# =============================================================================
# DPI SETTINGS
# =============================================================================

DPI = {
    "screen": 100,
    "presentation": 150,
    "print": 300,
}

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

EXPORT_FORMATS = ["png", "pdf", "svg"]

# =============================================================================
# METRIC LABELS AND UNITS
# =============================================================================

METRIC_LABELS = {
    # -------------------------------------------------------------------------
    # BVP/HRV Time-Domain Metrics
    # -------------------------------------------------------------------------
    "HRV_MeanNN": "Mean NN Interval (ms)",
    "HRV_SDNN": "SDNN (ms)",
    "HRV_RMSSD": "RMSSD (ms)",
    "HRV_pNN50": "pNN50 (%)",
    "HRV_CVNN": "CV of NN Intervals",
    "HRV_SD1": "Poincaré SD1 (ms)",
    "HRV_SD2": "Poincaré SD2 (ms)",
    "HRV_SampEn": "Sample Entropy",
    # -------------------------------------------------------------------------
    # BVP/HRV Frequency-Domain Metrics
    # -------------------------------------------------------------------------
    "HRV_LF": "LF Power (ms²)",
    "HRV_HF": "HF Power (ms²)",
    "HRV_VLF": "VLF Power (ms²)",
    "HRV_TP": "Total Power (ms²)",
    "HRV_LFHF": "LF/HF Ratio",
    "HRV_LFn": "Normalized LF (%)",
    "HRV_HFn": "Normalized HF (%)",
    # -------------------------------------------------------------------------
    # EDA Metrics
    # -------------------------------------------------------------------------
    "SCR_Peaks_N": "Number of SCRs",
    "SCR_Peaks_Rate": "SCR Rate (per min)",
    "SCR_Peaks_Amplitude_Mean": "Mean SCR Amplitude (µS)",
    "SCR_Peaks_Amplitude_Max": "Max SCR Amplitude (µS)",
    "SCR_Peaks_Amplitude_SD": "SCR Amplitude SD (µS)",
    "SCR_RiseTime_Mean": "Mean Rise Time (s)",
    "SCR_RecoveryTime_Mean": "Mean Recovery Time (s)",
    "EDA_Tonic_Mean": "Mean Tonic EDA (µS)",
    "EDA_Tonic_SD": "Tonic EDA SD (µS)",
    "EDA_Phasic_Mean": "Mean Phasic EDA (µS)",
    "EDA_Phasic_SD": "Phasic EDA SD (µS)",
    # -------------------------------------------------------------------------
    # HR Metrics
    # -------------------------------------------------------------------------
    "HR_Mean": "Mean Heart Rate (BPM)",
    "HR_SD": "Heart Rate SD (BPM)",
    "HR_Min": "Min Heart Rate (BPM)",
    "HR_Max": "Max Heart Rate (BPM)",
    "HR_Range": "Heart Rate Range (BPM)",
    "HR_Slope": "Heart Rate Slope (BPM/min)",
    "HR_CV": "Heart Rate CV (%)",
    # -------------------------------------------------------------------------
    # TEMP Metrics
    # -------------------------------------------------------------------------
    "TEMP_Mean": "Mean Temperature (°C)",
    "TEMP_SD": "Temperature SD (°C)",
    "TEMP_Min": "Min Temperature (°C)",
    "TEMP_Max": "Max Temperature (°C)",
    "TEMP_Range": "Temperature Range (°C)",
    "TEMP_Slope": "Temperature Slope (°C/min)",
    "TEMP_CV": "Temperature CV (%)",
}


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================


def configure_matplotlib() -> None:
    """
    Configure matplotlib defaults for consistent professional visualizations.

    Sets font sizes, figure aesthetics, and other defaults according
    to the project style guide. Call once at the start of visualization.
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": COLORS["background"],
            "figure.dpi": DPI["presentation"],
            "figure.figsize": FIGSIZE["medium"],
            # Axes
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlesize": FONTSIZE["title"],
            "axes.labelsize": FONTSIZE["label"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Ticks
            "xtick.labelsize": FONTSIZE["tick"],
            "ytick.labelsize": FONTSIZE["tick"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            # Legend
            "legend.fontsize": FONTSIZE["legend"],
            "legend.frameon": False,
            "legend.loc": "best",
            # Grid
            "grid.alpha": 0.3,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            # Lines
            "lines.linewidth": LINEWIDTH["signal"],
            "lines.markersize": MARKERSIZE["medium"],
            # Font
            "font.family": "sans-serif",
            "font.size": FONTSIZE["label"],
            # Saving
            "savefig.dpi": DPI["print"],
            "savefig.bbox": "tight",
            "savefig.facecolor": COLORS["background"],
            "savefig.edgecolor": "none",
        }
    )


def apply_plot_style() -> None:
    """
    Apply consistent plotting style to matplotlib.

    This is the main function to call before creating visualizations.
    It configures all matplotlib settings for professional output.
    """
    configure_matplotlib()


def get_moment_color(moment) -> str:
    """
    Get color for a specific moment.

    Supports both string names and integer indices.
    Uses modulo fallback for indices beyond the palette size.
    For unknown string names, generates a stable color based on hash.

    Args:
        moment: Either a moment name (str) or index (int)

    Returns:
        Hex color code

    Examples:
        >>> get_moment_color('restingstate')
        '#7EB8DA'
        >>> get_moment_color('therapy')
        '#F4A4B8'
        >>> get_moment_color(0)
        '#7EB8DA'
    """
    if isinstance(moment, int):
        return MOMENT_COLORS[moment % len(MOMENT_COLORS)]
    elif isinstance(moment, str):
        if moment in MOMENT_NAME_TO_INDEX:
            return MOMENT_COLORS[MOMENT_NAME_TO_INDEX[moment]]
        moment_hash = hash(moment)
        color_index = moment_hash % len(MOMENT_COLORS)
        return MOMENT_COLORS[color_index]
    else:
        return COLORS["gray"]


def get_moment_label(moment: str, config: Optional[dict] = None) -> str:
    """
    Get display label for a moment.

    Reads from MOMENT_LABELS dict built from config.yaml at module initialization.
    Falls back to moment name if not found.

    Args:
        moment: Moment name (e.g., 'restingstate', 'therapy')
        config: Unused (kept for backward compatibility)

    Returns:
        Display label from config's displayname field, or moment name if not found
    """
    return MOMENT_LABELS.get(moment, moment)


def get_moment_order(moment: str, moments_list: list) -> int:
    """
    Get the index/order of a moment in a list.

    Args:
        moment: Moment name
        moments_list: List of all available moments (sorted)

    Returns:
        Index of the moment in the list (0-based), or -1 if not found
    """
    try:
        return moments_list.index(moment)
    except ValueError:
        return -1


def get_modality_color(modality: str) -> str:
    """
    Get color for a specific physiological modality.

    Args:
        modality: One of 'bvp', 'eda', 'hr', 'temp'

    Returns:
        Hex color code
    """
    return COLORS.get(modality, COLORS["gray"])


def get_metric_label(metric: str) -> str:
    """
    Get display label for a metric.

    Args:
        metric: Metric name (e.g., 'HRV_SDNN', 'SCR_Peaks_N')

    Returns:
        Human-readable label with units, or metric name if not found
    """
    return METRIC_LABELS.get(metric, metric)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "5m 30s", "1h 15m")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


# =============================================================================
# AXIS LIMITS FOR CROSS-SESSION COMPARISON
# Set to None for auto-scaling, or (min, max) for fixed limits
# =============================================================================

AXIS_LIMITS = {
    # HRV Figure
    "poincare_rr": (400, 1200),  # RR intervals in ms (typical: 500-1200)
    "sdnn": (0, 500),  # SDNN in ms
    "rmssd": (0, 600),  # RMSSD in ms
    "lf_hf_power": (0, 0.3),  # Normalized power
    "lfhf_ratio": (0, 3),  # LF/HF ratio
    # Signal Figures
    "hr": (40, 120),  # Heart rate in BPM
    "eda": None,  # Auto-scale (varies widely)
    "temp": (28, 38),  # Temperature in °C
    # EDA Figure
    "scr_amplitude": (0, 1.5),  # SCR amplitude in µS
    "tonic_eda": None,  # Auto-scale
}


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_CONFIG = {
    "base_path": "data/derivatives/visualization/preprocessing",
    "figures_subdir": "figures",
    "report_subdir": "report",
    "summary_subdir": "summary",
    "dpi": DPI["print"],
    "format": "png",
    "bbox_inches": "tight",
}
