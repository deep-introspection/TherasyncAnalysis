#!/usr/bin/env python3
"""
Detailed investigation of preprocessing artifact outliers.

This script performs Phase 2 of the preprocessing artifacts investigation:
- Loads raw and preprocessed signals for outlier sessions
- Creates comparative visualizations
- Analyzes preprocessing parameters and quality metrics
- Generates detailed diagnostic reports

Phase 2 of preprocessing artifacts investigation (PREPROCESSING_ISSUES.md).

Usage:
    uv run python scripts/analysis/investigate_outliers.py
    uv run python scripts/analysis/investigate_outliers.py --sessions sub-g03p03/ses-01 sub-g04p04/ses-03
    uv run python scripts/analysis/investigate_outliers.py --top-n 5 --verbose

Author: TherasyncPipeline Team
Date: November 11, 2025
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the investigation script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def identify_worst_cases(
    stats_csv: Path, top_n: int, logger: logging.Logger
) -> List[Dict]:
    """Identify worst outlier cases from statistics."""
    df = pd.read_csv(stats_csv)

    cases = []

    # Top N worst HRV CVNN (coefficient of variation) — proxy for HRV quality
    logger.info(f"Finding top {top_n} worst HRV CVNN values...")
    top_cvnn = df.nlargest(top_n, "BVP_HRV_CVNN_max")
    for _, row in top_cvnn.iterrows():
        cases.append(
            {
                "subject": row["subject"],
                "session": row["session"],
                "issue_type": "HIGH_CVNN",
                "cvnn_max": row["BVP_HRV_CVNN_max"],
                "cvnn_mean": row["BVP_HRV_CVNN_mean"],
                "priority": "HIGH",
            }
        )

    # Top N worst negative EDA
    logger.info(f"Finding top {top_n} worst negative EDA values...")
    top_eda = df.nsmallest(top_n, "EDA_EDA_Phasic_Min_min")
    for idx, row in top_eda.iterrows():
        # Check if already in list
        existing = [
            c
            for c in cases
            if c["subject"] == row["subject"] and c["session"] == row["session"]
        ]
        if existing:
            existing[0]["issue_type"] = "BOTH"
            existing[0]["eda_phasic_min"] = row["EDA_EDA_Phasic_Min_min"]
            existing[0]["eda_tonic_min"] = row.get("EDA_EDA_Tonic_Min_min", 0)
        else:
            cases.append(
                {
                    "subject": row["subject"],
                    "session": row["session"],
                    "issue_type": "NEGATIVE_EDA",
                    "eda_phasic_min": row["EDA_EDA_Phasic_Min_min"],
                    "eda_tonic_min": row.get("EDA_EDA_Tonic_Min_min", 0),
                    "priority": "HIGH",
                }
            )

    logger.info(f"Identified {len(cases)} outlier cases for investigation")
    return cases


def load_bvp_signals(
    subject: str, session: str, config: Dict, logger: logging.Logger
) -> Dict:
    """Load BVP raw and preprocessed signals."""
    preprocessing_dir = Path(config["paths"]["derivatives"]) / "preprocessing"
    bvp_dir = preprocessing_dir / subject / session / "bvp"

    signals = {}

    # Load preprocessed BVP for each moment
    for moment in ["restingstate", "therapy"]:
        processed_file = (
            bvp_dir
            / f"{subject}_{session}_task-{moment}_desc-processed_recording-bvp.tsv"
        )
        if processed_file.exists():
            df = pd.read_csv(processed_file, sep="\t")
            signals[f"{moment}_processed"] = df
            logger.debug(f"Loaded {moment} processed BVP: {len(df)} samples")

    # Load BVP metrics
    metrics_file = bvp_dir / f"{subject}_{session}_desc-bvp-metrics_physio.tsv"
    if metrics_file.exists():
        signals["metrics"] = pd.read_csv(metrics_file, sep="\t")
        logger.debug(f"Loaded BVP metrics: {len(signals['metrics'])} rows")

    # Load processing metadata
    for moment in ["restingstate", "therapy"]:
        metadata_file = (
            bvp_dir
            / f"{subject}_{session}_task-{moment}_desc-processing_recording-bvp.json"
        )
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                signals[f"{moment}_metadata"] = json.load(f)

    return signals


def load_eda_signals(
    subject: str, session: str, config: Dict, logger: logging.Logger
) -> Dict:
    """Load EDA raw and preprocessed signals."""
    preprocessing_dir = Path(config["paths"]["derivatives"]) / "preprocessing"
    eda_dir = preprocessing_dir / subject / session / "eda"

    signals = {}

    # Load preprocessed EDA for each moment
    for moment in ["restingstate", "therapy"]:
        processed_file = (
            eda_dir
            / f"{subject}_{session}_task-{moment}_desc-processed_recording-eda.tsv"
        )
        if processed_file.exists():
            df = pd.read_csv(processed_file, sep="\t")
            signals[f"{moment}_processed"] = df
            logger.debug(f"Loaded {moment} processed EDA: {len(df)} samples")

    # Load EDA metrics
    metrics_file = eda_dir / f"{subject}_{session}_desc-eda-metrics_physio.tsv"
    if metrics_file.exists():
        signals["metrics"] = pd.read_csv(metrics_file, sep="\t")
        logger.debug(f"Loaded EDA metrics: {len(signals['metrics'])} rows")

    # Load processing metadata
    for moment in ["restingstate", "therapy"]:
        metadata_file = (
            eda_dir
            / f"{subject}_{session}_task-{moment}_desc-processing_recording-eda.json"
        )
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                signals[f"{moment}_metadata"] = json.load(f)

    return signals


def create_bvp_diagnostic_plot(
    subject: str,
    session: str,
    bvp_signals: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    """Create diagnostic plots for BVP preprocessing artifacts."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"BVP Preprocessing Diagnostics: {subject}/{session}",
        fontsize=14,
        fontweight="bold",
    )

    moments = ["restingstate", "therapy"]
    colors = {"restingstate": "steelblue", "therapy": "coral"}

    for col_idx, moment in enumerate(moments):
        # Plot 1: Processed BVP signal
        ax = axes[0, col_idx]
        if f"{moment}_processed" in bvp_signals:
            df = bvp_signals[f"{moment}_processed"]
            # Check for PPG_Clean (new naming) or BVP_Filtered (legacy)
            bvp_col = (
                "PPG_Clean"
                if "PPG_Clean" in df.columns
                else "BVP_Filtered"
                if "BVP_Filtered" in df.columns
                else None
            )
            if bvp_col:
                ax.plot(df[bvp_col], color=colors[moment], linewidth=0.5, alpha=0.7)
                ax.set_ylabel("BVP Filtered (a.u.)", fontsize=10)
                ax.set_title(
                    f"{moment.capitalize()}: Filtered BVP Signal",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Sample", fontsize=10)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No BVP column found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{moment.capitalize()}: No BVP data", fontsize=11)

        # Plot 2: Heart Rate
        ax = axes[1, col_idx]
        if f"{moment}_processed" in bvp_signals:
            df = bvp_signals[f"{moment}_processed"]
            # Check for PPG_Rate (new naming) or Heart_Rate (legacy)
            hr_col = (
                "PPG_Rate"
                if "PPG_Rate" in df.columns
                else "Heart_Rate"
                if "Heart_Rate" in df.columns
                else None
            )
            if hr_col:
                hr = df[hr_col].dropna()
                if len(hr) > 0:
                    ax.plot(hr, color=colors[moment], linewidth=1)
                    ax.axhline(
                        y=60,
                        color="green",
                        linestyle="--",
                        alpha=0.5,
                        label="Normal range",
                    )
                    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5)
                    ax.fill_between(range(len(hr)), 60, 100, color="green", alpha=0.1)
                    ax.set_ylabel("Heart Rate (bpm)", fontsize=10)
                    ax.set_title(
                        f"{moment.capitalize()}: Heart Rate",
                        fontsize=11,
                        fontweight="bold",
                    )
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("Sample", fontsize=10)
                    ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No HR column found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        # Plot 3: HRV Frequency Analysis (if available)
        ax = axes[2, col_idx]
        if "metrics" in bvp_signals:
            metrics = bvp_signals["metrics"]
            moment_metrics = metrics[metrics["moment"] == moment]
            if len(moment_metrics) > 0:
                row = moment_metrics.iloc[0]
                if "HRV_LF" in row and "HRV_HF" in row and "HRV_LFHF" in row:
                    lf = row["HRV_LF"]
                    hf = row["HRV_HF"]
                    lfhf = row["HRV_LFHF"]

                    bars = ax.bar(
                        ["LF\nPower", "HF\nPower"],
                        [lf, hf],
                        color=[colors[moment], colors[moment]],
                        alpha=0.6,
                    )
                    ax.set_ylabel("Power (ms²)", fontsize=10)
                    ax.set_title(
                        f"{moment.capitalize()}: HRV Frequency\nLF/HF = {lfhf:.2f}",
                        fontsize=11,
                        fontweight="bold",
                    )
                    ax.grid(True, alpha=0.3, axis="y")

                    # Add values on bars
                    for bar, val in zip(bars, [lf, hf]):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{val:.4f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )

                    # Highlight if aberrant
                    if lfhf > 10:
                        ax.text(
                            0.5,
                            0.95,
                            "⚠️ ABERRANT RATIO",
                            transform=ax.transAxes,
                            ha="center",
                            va="top",
                            fontsize=10,
                            color="red",
                            fontweight="bold",
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
                        )

    plt.tight_layout()
    output_file = output_dir / f"{subject}_{session}_bvp_diagnostics.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved BVP diagnostic plot: {output_file}")
    plt.close()


def create_eda_diagnostic_plot(
    subject: str,
    session: str,
    eda_signals: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    """Create diagnostic plots for EDA preprocessing artifacts."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"EDA Preprocessing Diagnostics: {subject}/{session}",
        fontsize=14,
        fontweight="bold",
    )

    moments = ["restingstate", "therapy"]
    colors = {"restingstate": "mediumseagreen", "therapy": "orangered"}

    for col_idx, moment in enumerate(moments):
        # Plot 1: EDA Raw
        ax = axes[0, col_idx]
        if f"{moment}_processed" in eda_signals:
            df = eda_signals[f"{moment}_processed"]
            if "EDA_Raw" in df.columns:
                ax.plot(
                    df["EDA_Raw"], color="gray", linewidth=0.5, alpha=0.5, label="Raw"
                )
                ax.set_ylabel("EDA Raw (µS)", fontsize=10)
                ax.set_title(
                    f"{moment.capitalize()}: Raw EDA Signal",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

        # Plot 2: EDA Tonic and Phasic
        ax = axes[1, col_idx]
        if f"{moment}_processed" in eda_signals:
            df = eda_signals[f"{moment}_processed"]
            if "EDA_Tonic" in df.columns and "EDA_Phasic" in df.columns:
                ax.plot(
                    df["EDA_Tonic"], color="blue", linewidth=1, alpha=0.7, label="Tonic"
                )
                ax.plot(
                    df["EDA_Phasic"],
                    color="red",
                    linewidth=0.5,
                    alpha=0.6,
                    label="Phasic",
                )
                ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
                ax.set_ylabel("EDA (µS)", fontsize=10)
                ax.set_title(
                    f"{moment.capitalize()}: Tonic & Phasic Components",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

                # Highlight negative values
                tonic_min = df["EDA_Tonic"].min()
                phasic_min = df["EDA_Phasic"].min()
                if tonic_min < 0 or phasic_min < 0:
                    warning_text = []
                    if tonic_min < 0:
                        warning_text.append(f"Tonic min: {tonic_min:.4f}")
                    if phasic_min < 0:
                        warning_text.append(f"Phasic min: {phasic_min:.4f}")
                    ax.text(
                        0.02,
                        0.98,
                        "⚠️ NEGATIVE VALUES\n" + "\n".join(warning_text),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        color="red",
                        fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                    )

        # Plot 3: EDA metrics distribution
        ax = axes[2, col_idx]
        if "metrics" in eda_signals:
            metrics = eda_signals["metrics"]
            moment_metrics = metrics[metrics["moment"] == moment]
            if len(moment_metrics) > 0:
                row = moment_metrics.iloc[0]

                # Plot key metrics
                metric_names = []
                metric_values = []

                if "EDA_Tonic_Mean" in row:
                    metric_names.append("Tonic\nMean")
                    metric_values.append(row["EDA_Tonic_Mean"])
                if "EDA_Phasic_Mean" in row:
                    metric_names.append("Phasic\nMean")
                    metric_values.append(row["EDA_Phasic_Mean"])
                if "SCR_Peaks_N" in row:
                    metric_names.append("SCR\nPeaks")
                    metric_values.append(row["SCR_Peaks_N"])

                if metric_names:
                    bars = ax.bar(
                        metric_names,
                        metric_values,
                        color=[colors[moment]] * len(metric_names),
                        alpha=0.6,
                    )
                    ax.set_ylabel("Value", fontsize=10)
                    ax.set_title(
                        f"{moment.capitalize()}: Key EDA Metrics",
                        fontsize=11,
                        fontweight="bold",
                    )
                    ax.grid(True, alpha=0.3, axis="y")

                    # Add values on bars
                    for bar, val in zip(bars, metric_values):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{val:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )

    plt.tight_layout()
    output_file = output_dir / f"{subject}_{session}_eda_diagnostics.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved EDA diagnostic plot: {output_file}")
    plt.close()


def generate_case_report(
    case: Dict,
    bvp_signals: Dict,
    eda_signals: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    """Generate detailed markdown report for a case."""
    subject = case["subject"]
    session = case["session"]

    report = f"""# Outlier Investigation Report: {subject}/{session}

**Date**: November 11, 2025  
**Investigation Phase**: 2 - Detailed Outlier Analysis  
**Issue Type**: {case["issue_type"]}  
**Priority**: {case["priority"]}

---

## Issue Summary

"""

    if case["issue_type"] in ["HIGH_CVNN", "BOTH"]:
        report += f"""### HRV Variability Issue
- **CVNN (max)**: {case.get("cvnn_max", "N/A"):.4f} ⚠️
- **CVNN (mean)**: {case.get("cvnn_mean", "N/A"):.4f}
- **Status**: High coefficient of variation in NN intervals

"""

    if case["issue_type"] in ["NEGATIVE_EDA", "BOTH"]:
        report += f"""### EDA Decomposition Issue
- **Phasic Min**: {case.get("eda_phasic_min", "N/A"):.4f} µS ⚠️
- **Tonic Min**: {case.get("eda_tonic_min", "N/A"):.4f} µS
- **Physical constraint**: EDA must be ≥ 0 µS
- **Status**: PHYSICALLY IMPOSSIBLE (negative values detected)

"""

    report += """---

## Data Inspection

### BVP/HRV Analysis

"""

    # Add BVP metrics analysis
    if "metrics" in bvp_signals:
        metrics = bvp_signals["metrics"]
        for moment in ["restingstate", "therapy"]:
            moment_metrics = metrics[metrics["moment"] == moment]
            if len(moment_metrics) > 0:
                row = moment_metrics.iloc[0]
                rmssd = row.get("HRV_RMSSD", None)
                sdnn = row.get("HRV_SDNN", None)
                cvnn = row.get("HRV_CVNN", None)
                report += f"""#### {moment.capitalize()} Moment
- **RMSSD**: {f"{rmssd:.2f} ms" if isinstance(rmssd, (int, float)) else "N/A"}
- **SDNN**: {f"{sdnn:.2f} ms" if isinstance(sdnn, (int, float)) else "N/A"}
- **CVNN**: {f"{cvnn:.4f}" if isinstance(cvnn, (int, float)) else "N/A"}

"""

    report += """### EDA Analysis

"""

    # Add EDA metrics analysis
    if "metrics" in eda_signals:
        metrics = eda_signals["metrics"]
        for moment in ["restingstate", "therapy"]:
            moment_metrics = metrics[metrics["moment"] == moment]
            if len(moment_metrics) > 0:
                row = moment_metrics.iloc[0]
                report += f"""#### {moment.capitalize()} Moment
- **Tonic Mean**: {row.get("EDA_Tonic_Mean", "N/A"):.4f} µS
- **Tonic Min**: {row.get("EDA_Tonic_Min", "N/A"):.4f} µS
- **Phasic Mean**: {row.get("EDA_Phasic_Mean", "N/A"):.4f} µS
- **Phasic Min**: {row.get("EDA_Phasic_Min", "N/A"):.4f} µS
- **SCR Peaks**: {row.get("SCR_Peaks_N", "N/A")}

"""

    report += """---

## Diagnostic Plots

See generated PNG files:
- `{subject}_{session}_bvp_diagnostics.png`
- `{subject}_{session}_eda_diagnostics.png`

---

## Observations

### Potential Root Causes

#### HRV LF/HF Aberration
- [ ] FFT window size inappropriate for signal duration
- [ ] Respiratory frequency outside HF band (0.15-0.4 Hz)
- [ ] Signal quality degradation during therapy
- [ ] Artifacts in peak detection
- [ ] Statistical outlier (stress response?)

#### EDA Negative Values
- [ ] cvxEDA optimization issue (regularization parameters)
- [ ] Baseline correction overshoot
- [ ] High-pass filtering artifacts
- [ ] Signal calibration problem
- [ ] Sensor contact issues

### Recommended Actions

1. **Visual Inspection**: Review diagnostic plots for signal quality
2. **Parameter Review**: Check preprocessing parameters in metadata
3. **Comparison**: Compare with similar sessions from same family
4. **Validation**: Cross-check with raw data quality

---

## Conclusion

**Investigation Status**: REQUIRES MANUAL REVIEW

**Next Steps**:
- [ ] Review preprocessing parameters
- [ ] Check raw data quality
- [ ] Compare with other sessions
- [ ] Document findings in PREPROCESSING_ISSUES.md
- [ ] Consider parameter adjustments if systematic

"""

    # Save report
    report_file = output_dir / f"{subject}_{session}_investigation_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    logger.info(f"Saved investigation report: {report_file}")


def main():
    """Main investigation workflow for Phase 2."""
    parser = argparse.ArgumentParser(
        description="Detailed investigation of preprocessing outliers (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Investigate top 3 worst cases
    uv run python scripts/analysis/investigate_outliers.py --top-n 3
    
    # Investigate specific sessions
    uv run python scripts/analysis/investigate_outliers.py --sessions sub-g03p03/ses-01 sub-g04p04/ses-03
    
    # Verbose output
    uv run python scripts/analysis/investigate_outliers.py --top-n 5 --verbose
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Path to preprocessing stats CSV (default: data/derivatives/analysis/preprocessing_stats.csv)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of worst cases to investigate (default: 3)",
    )

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Specific sessions to investigate (format: sub-XXX/ses-YY)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/derivatives/analysis/outlier_reports)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.verbose)
    logger.info("=" * 80)
    logger.info("OUTLIER INVESTIGATION - Phase 2")
    logger.info("=" * 80)

    # Load config
    config = load_config(args.config)
    logger.info("Loaded configuration")

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = (
            Path(config["paths"]["derivatives"]) / "analysis" / "outlier_reports"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Identify cases to investigate
    if args.sessions:
        # Manual session selection
        cases = []
        for session_str in args.sessions:
            parts = session_str.split("/")
            if len(parts) == 2:
                cases.append(
                    {
                        "subject": parts[0],
                        "session": parts[1],
                        "issue_type": "MANUAL",
                        "priority": "HIGH",
                    }
                )
        logger.info(f"Investigating {len(cases)} manually specified sessions")
    else:
        # Auto-identify worst cases
        if args.stats_csv:
            stats_csv = args.stats_csv
        else:
            stats_csv = (
                Path(config["paths"]["derivatives"])
                / "analysis"
                / "preprocessing_stats.csv"
            )

        cases = identify_worst_cases(stats_csv, args.top_n, logger)

    # Investigate each case
    logger.info(f"\nInvestigating {len(cases)} outlier cases...")
    logger.info("=" * 80)

    for idx, case in enumerate(cases, 1):
        subject = case["subject"]
        session = case["session"]

        logger.info(f"\n[{idx}/{len(cases)}] Processing {subject}/{session}")
        logger.info(f"Issue type: {case['issue_type']}, Priority: {case['priority']}")

        # Load signals
        logger.info("Loading BVP signals...")
        bvp_signals = load_bvp_signals(subject, session, config, logger)

        logger.info("Loading EDA signals...")
        eda_signals = load_eda_signals(subject, session, config, logger)

        # Create diagnostic plots
        logger.info("Creating diagnostic plots...")
        create_bvp_diagnostic_plot(subject, session, bvp_signals, output_dir, logger)
        create_eda_diagnostic_plot(subject, session, eda_signals, output_dir, logger)

        # Generate report
        logger.info("Generating investigation report...")
        generate_case_report(case, bvp_signals, eda_signals, output_dir, logger)

        logger.info(f"✓ Completed investigation for {subject}/{session}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("INVESTIGATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Cases investigated: {len(cases)}")
    logger.info(f"Reports generated: {output_dir}")
    logger.info("=" * 80)
    logger.info("\n📋 Next Steps:")
    logger.info("1. Review diagnostic plots in output directory")
    logger.info("2. Read investigation reports for each case")
    logger.info("3. Document findings in PREPROCESSING_ISSUES.md")
    logger.info("4. Proceed to Phase 3 (validation & correction) if needed")


if __name__ == "__main__":
    main()
