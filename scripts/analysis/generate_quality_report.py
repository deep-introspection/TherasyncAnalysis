"""
Generate Data Quality Report for Preprocessing Outputs.

This script analyzes all preprocessed physiological data and flags sessions
with potentially problematic or extreme metric values for manual review.

Flags are meant to assist analysts in deciding which sessions to include/exclude
from analysis, not to automatically filter data.

Usage:
    uv run python scripts/analysis/generate_quality_report.py [--output OUTPUT]

Author: GitHub Copilot
Date: November 11, 2025
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QualityThresholds:
    """Define quality thresholds for flagging extreme/aberrant values."""

    # HRV Time-Domain (ms)
    HRV_RMSSD_MAX = 500  # > 500ms is extremely high (normal: 20-80ms)
    HRV_SDNN_MAX = 500  # > 500ms is extremely high (normal: 20-100ms)
    HRV_MEANNN_MIN = 400  # < 400ms = >150 BPM (very high HR)
    HRV_MEANNN_MAX = 1500  # > 1500ms = <40 BPM (very low HR)

    # HRV Frequency-Domain
    HRV_HF_MIN = 0.001  # < 0.001 ms² indicates poor signal quality
    HRV_LFHF_MAX = 50  # > 50 is aberrant (normal: 0.5-10)

    # BVP Signal Quality
    BVP_QUALITY_MIN = 0.5  # < 0.5 indicates poor signal quality
    BVP_PEAK_RATE_MIN = 0.5  # < 0.5 peaks/s = < 30 BPM (bradycardia)
    BVP_PEAK_RATE_MAX = 3.0  # > 3.0 peaks/s = > 180 BPM (tachycardia)

    # EDA Metrics
    EDA_TONIC_MIN = 0.01  # < 0.01 µS is very low (poor contact?)
    EDA_TONIC_MAX = 30  # > 30 µS is very high (check calibration)
    EDA_SCR_RATE_MAX = 30  # > 30 SCRs/min is very high arousal

    # HR Metrics
    HR_MEAN_MIN = 40  # < 40 BPM (bradycardia)
    HR_MEAN_MAX = 150  # > 150 BPM (tachycardia)


def load_metrics_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Load a metrics TSV file."""
    try:
        if not file_path.exists():
            return None
        df = pd.read_csv(file_path, sep="\t")
        return df
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def check_hrv_quality(
    row: pd.Series, subject: str, session: str, moment: str
) -> List[Dict]:
    """Check HRV metrics for quality issues."""
    issues = []

    # Check RMSSD
    if (
        pd.notna(row.get("HRV_RMSSD"))
        and row["HRV_RMSSD"] > QualityThresholds.HRV_RMSSD_MAX
    ):
        issues.append(
            {
                "subject": subject,
                "session": session,
                "moment": moment,
                "signal": "BVP",
                "metric": "HRV_RMSSD",
                "value": row["HRV_RMSSD"],
                "threshold": QualityThresholds.HRV_RMSSD_MAX,
                "severity": "HIGH",
                "description": f"Extremely high RMSSD ({row['HRV_RMSSD']:.1f} ms > {QualityThresholds.HRV_RMSSD_MAX} ms)",
            }
        )

    # Check SDNN
    if (
        pd.notna(row.get("HRV_SDNN"))
        and row["HRV_SDNN"] > QualityThresholds.HRV_SDNN_MAX
    ):
        issues.append(
            {
                "subject": subject,
                "session": session,
                "moment": moment,
                "signal": "BVP",
                "metric": "HRV_SDNN",
                "value": row["HRV_SDNN"],
                "threshold": QualityThresholds.HRV_SDNN_MAX,
                "severity": "HIGH",
                "description": f"Extremely high SDNN ({row['HRV_SDNN']:.1f} ms > {QualityThresholds.HRV_SDNN_MAX} ms)",
            }
        )

    # Check LF/HF ratio
    if (
        pd.notna(row.get("HRV_LFHF"))
        and row["HRV_LFHF"] > QualityThresholds.HRV_LFHF_MAX
    ):
        issues.append(
            {
                "subject": subject,
                "session": session,
                "moment": moment,
                "signal": "BVP",
                "metric": "HRV_LFHF",
                "value": row["HRV_LFHF"],
                "threshold": QualityThresholds.HRV_LFHF_MAX,
                "severity": "HIGH",
                "description": f"Aberrant LF/HF ratio ({row['HRV_LFHF']:.1f} > {QualityThresholds.HRV_LFHF_MAX})",
            }
        )

    # Check HF power
    if pd.notna(row.get("HRV_HF")) and row["HRV_HF"] < QualityThresholds.HRV_HF_MIN:
        issues.append(
            {
                "subject": subject,
                "session": session,
                "moment": moment,
                "signal": "BVP",
                "metric": "HRV_HF",
                "value": row["HRV_HF"],
                "threshold": QualityThresholds.HRV_HF_MIN,
                "severity": "MEDIUM",
                "description": f"Very low HF power ({row['HRV_HF']:.6f} ms² < {QualityThresholds.HRV_HF_MIN} ms²) - possible signal quality issue",
            }
        )

    # Check mean HR
    if pd.notna(row.get("HRV_MeanNN")):
        if row["HRV_MeanNN"] < QualityThresholds.HRV_MEANNN_MIN:
            issues.append(
                {
                    "subject": subject,
                    "session": session,
                    "moment": moment,
                    "signal": "BVP",
                    "metric": "HRV_MeanNN",
                    "value": row["HRV_MeanNN"],
                    "threshold": QualityThresholds.HRV_MEANNN_MIN,
                    "severity": "MEDIUM",
                    "description": f"Very high heart rate ({60000 / row['HRV_MeanNN']:.0f} BPM)",
                }
            )
        elif row["HRV_MeanNN"] > QualityThresholds.HRV_MEANNN_MAX:
            issues.append(
                {
                    "subject": subject,
                    "session": session,
                    "moment": moment,
                    "signal": "BVP",
                    "metric": "HRV_MeanNN",
                    "value": row["HRV_MeanNN"],
                    "threshold": QualityThresholds.HRV_MEANNN_MAX,
                    "severity": "MEDIUM",
                    "description": f"Very low heart rate ({60000 / row['HRV_MeanNN']:.0f} BPM)",
                }
            )

    # Check signal quality
    if (
        pd.notna(row.get("BVP_MeanQuality"))
        and row["BVP_MeanQuality"] < QualityThresholds.BVP_QUALITY_MIN
    ):
        issues.append(
            {
                "subject": subject,
                "session": session,
                "moment": moment,
                "signal": "BVP",
                "metric": "BVP_MeanQuality",
                "value": row["BVP_MeanQuality"],
                "threshold": QualityThresholds.BVP_QUALITY_MIN,
                "severity": "LOW",
                "description": f"Low BVP signal quality ({row['BVP_MeanQuality']:.2f} < {QualityThresholds.BVP_QUALITY_MIN})",
            }
        )

    return issues


def check_eda_quality(
    row: pd.Series, subject: str, session: str, moment: str
) -> List[Dict]:
    """Check EDA metrics for quality issues."""
    issues = []

    # Check tonic range
    if pd.notna(row.get("EDA_Tonic_Mean")):
        if row["EDA_Tonic_Mean"] < QualityThresholds.EDA_TONIC_MIN:
            issues.append(
                {
                    "subject": subject,
                    "session": session,
                    "moment": moment,
                    "signal": "EDA",
                    "metric": "EDA_Tonic_Mean",
                    "value": row["EDA_Tonic_Mean"],
                    "threshold": QualityThresholds.EDA_TONIC_MIN,
                    "severity": "MEDIUM",
                    "description": f"Very low tonic EDA ({row['EDA_Tonic_Mean']:.4f} µS) - possible poor contact",
                }
            )
        elif row["EDA_Tonic_Mean"] > QualityThresholds.EDA_TONIC_MAX:
            issues.append(
                {
                    "subject": subject,
                    "session": session,
                    "moment": moment,
                    "signal": "EDA",
                    "metric": "EDA_Tonic_Mean",
                    "value": row["EDA_Tonic_Mean"],
                    "threshold": QualityThresholds.EDA_TONIC_MAX,
                    "severity": "MEDIUM",
                    "description": f"Very high tonic EDA ({row['EDA_Tonic_Mean']:.2f} µS) - check calibration",
                }
            )

    # Check SCR rate
    if (
        pd.notna(row.get("SCR_Peaks_Rate"))
        and row["SCR_Peaks_Rate"] > QualityThresholds.EDA_SCR_RATE_MAX
    ):
        issues.append(
            {
                "subject": subject,
                "session": session,
                "moment": moment,
                "signal": "EDA",
                "metric": "SCR_Peaks_Rate",
                "value": row["SCR_Peaks_Rate"],
                "threshold": QualityThresholds.EDA_SCR_RATE_MAX,
                "severity": "LOW",
                "description": f"Very high SCR rate ({row['SCR_Peaks_Rate']:.1f}/min) - extreme arousal or artifacts",
            }
        )

    return issues


def check_hr_quality(
    row: pd.Series, subject: str, session: str, moment: str
) -> List[Dict]:
    """Check HR metrics for quality issues."""
    issues = []

    # Check mean HR
    if pd.notna(row.get("hr_mean")):
        if row["hr_mean"] < QualityThresholds.HR_MEAN_MIN:
            issues.append(
                {
                    "subject": subject,
                    "session": session,
                    "moment": moment,
                    "signal": "HR",
                    "metric": "hr_mean",
                    "value": row["hr_mean"],
                    "threshold": QualityThresholds.HR_MEAN_MIN,
                    "severity": "MEDIUM",
                    "description": f"Bradycardia ({row['hr_mean']:.0f} BPM < {QualityThresholds.HR_MEAN_MIN} BPM)",
                }
            )
        elif row["hr_mean"] > QualityThresholds.HR_MEAN_MAX:
            issues.append(
                {
                    "subject": subject,
                    "session": session,
                    "moment": moment,
                    "signal": "HR",
                    "metric": "hr_mean",
                    "value": row["hr_mean"],
                    "threshold": QualityThresholds.HR_MEAN_MAX,
                    "severity": "MEDIUM",
                    "description": f"Tachycardia ({row['hr_mean']:.0f} BPM > {QualityThresholds.HR_MEAN_MAX} BPM)",
                }
            )

    return issues


def scan_preprocessing_outputs(derivatives_path: Path) -> List[Dict]:
    """Scan all preprocessing outputs and flag quality issues."""
    all_issues = []

    # Scan all subjects/sessions
    for subject_dir in sorted(derivatives_path.glob("sub-*")):
        subject = subject_dir.name.replace("sub-", "")

        for session_dir in sorted(subject_dir.glob("ses-*")):
            session = session_dir.name.replace("ses-", "")

            # Check BVP/HRV metrics
            bvp_metrics = load_metrics_file(
                session_dir
                / "bvp"
                / f"sub-{subject}_ses-{session}_desc-bvp-metrics_physio.tsv"
            )
            if bvp_metrics is not None:
                for _, row in bvp_metrics.iterrows():
                    moment = row.get("moment", "unknown")
                    issues = check_hrv_quality(row, subject, session, moment)
                    all_issues.extend(issues)

            # Check EDA metrics
            eda_metrics = load_metrics_file(
                session_dir
                / "eda"
                / f"sub-{subject}_ses-{session}_desc-eda-metrics_physio.tsv"
            )
            if eda_metrics is not None:
                for _, row in eda_metrics.iterrows():
                    moment = row.get("moment", "unknown")
                    issues = check_eda_quality(row, subject, session, moment)
                    all_issues.extend(issues)

            # Check HR metrics
            hr_metrics = load_metrics_file(
                session_dir
                / "hr"
                / f"sub-{subject}_ses-{session}_desc-hr-metrics_physio.tsv"
            )
            if hr_metrics is not None:
                for _, row in hr_metrics.iterrows():
                    moment = row.get("moment", "unknown")
                    issues = check_hr_quality(row, subject, session, moment)
                    all_issues.extend(issues)

    return all_issues


def generate_markdown_report(issues: List[Dict], output_path: Path):
    """Generate a markdown quality report."""

    if not issues:
        report = "# Data Quality Report\n\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += (
            "✅ **No quality issues detected** - all metrics within expected ranges.\n"
        )
        output_path.write_text(report)
        logger.info(f"Quality report saved to {output_path}")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(issues)

    # Generate report
    report = "# Data Quality Report - Preprocessing Artifacts\n\n"
    report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Total flagged issues**: {len(issues)}\n\n"

    # Summary by severity
    report += "## Summary by Severity\n\n"
    severity_counts = df["severity"].value_counts()
    for severity in ["HIGH", "MEDIUM", "LOW"]:
        count = severity_counts.get(severity, 0)
        report += f"- **{severity}**: {count} issues\n"

    report += "\n"

    # Summary by signal type
    report += "## Summary by Signal Type\n\n"
    signal_counts = df["signal"].value_counts()
    for signal, count in signal_counts.items():
        report += f"- **{signal}**: {count} flagged metrics\n"

    report += "\n"

    # Summary by subject/session
    report += "## Affected Sessions\n\n"
    session_counts = (
        df.groupby(["subject", "session"]).size().reset_index(name="issues")
    )
    session_counts = session_counts.sort_values("issues", ascending=False)

    report += f"**Total affected sessions**: {len(session_counts)}\n\n"
    report += "| Subject | Session | Issues |\n"
    report += "|---------|---------|--------|\n"
    for _, row in session_counts.head(20).iterrows():
        report += f"| sub-{row['subject']} | ses-{row['session']} | {row['issues']} |\n"

    if len(session_counts) > 20:
        report += f"\n*... and {len(session_counts) - 20} more sessions*\n"

    report += "\n---\n\n"

    # Detailed issues by severity
    report += "## Detailed Issues\n\n"

    for severity in ["HIGH", "MEDIUM", "LOW"]:
        severity_issues = df[df["severity"] == severity]
        if len(severity_issues) == 0:
            continue

        report += f"### {severity} Severity ({len(severity_issues)} issues)\n\n"

        # Group by metric type
        for metric in severity_issues["metric"].unique():
            metric_issues = severity_issues[severity_issues["metric"] == metric]
            report += f"#### {metric} ({len(metric_issues)} cases)\n\n"

            report += "| Subject | Session | Moment | Value | Description |\n"
            report += "|---------|---------|--------|-------|-------------|\n"

            for _, issue in metric_issues.head(10).iterrows():
                value_str = (
                    f"{issue['value']:.2f}"
                    if isinstance(issue["value"], float)
                    else str(issue["value"])
                )
                report += f"| sub-{issue['subject']} | ses-{issue['session']} | {issue['moment']} | {value_str} | {issue['description']} |\n"

            if len(metric_issues) > 10:
                report += f"\n*... and {len(metric_issues) - 10} more cases*\n"

            report += "\n"

    # Interpretation guide
    report += "\n---\n\n"
    report += "## Interpretation Guide\n\n"
    report += "**Purpose**: This report flags potentially problematic measurements for manual review.\n\n"
    report += "**Important Notes**:\n\n"
    report += "1. **EDA Negative Values**: Negative values in EDA_Phasic/EDA_Tonic are **NORMAL** after mathematical decomposition (signal centering, filtering). They do NOT indicate errors.\n\n"
    report += (
        "2. **Extreme HRV Values**: High RMSSD/SDNN or aberrant LF/HF ratios may be:\n"
    )
    report += "   - Genuine physiological responses (stress, movement, arrhythmias)\n"
    report += "   - Signal quality issues (artifacts, poor contact)\n"
    report += "   - Requires **visual inspection** of diagnostic plots to determine\n\n"
    report += "3. **Action Required**: Review flagged sessions and decide:\n"
    report += (
        "   - ✅ **Include**: If artifacts are minor or physiologically plausible\n"
    )
    report += (
        "   - ❌ **Exclude**: If signal quality is too poor for reliable analysis\n"
    )
    report += "   - 🔍 **Investigate**: Check diagnostic plots in `data/derivatives/analysis/outlier_reports/`\n\n"
    report += "4. **Severity Levels**:\n"
    report += "   - **HIGH**: Extreme outliers, likely requires investigation\n"
    report += (
        "   - **MEDIUM**: Outside typical ranges, may be physiological or artifacts\n"
    )
    report += "   - **LOW**: Minor deviations, unlikely to affect analysis\n\n"

    # Write report
    output_path.write_text(report)
    logger.info(f"Quality report saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate data quality report for preprocessing outputs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/derivatives/analysis/DATA_QUALITY_REPORT.md",
        help="Output path for quality report (default: data/derivatives/analysis/DATA_QUALITY_REPORT.md)",
    )
    parser.add_argument(
        "--derivatives",
        type=str,
        default="data/derivatives/preprocessing",
        help="Path to preprocessing derivatives (default: data/derivatives/preprocessing)",
    )

    args = parser.parse_args()

    derivatives_path = Path(args.derivatives)
    output_path = Path(args.output)

    if not derivatives_path.exists():
        logger.error(f"Derivatives path does not exist: {derivatives_path}")
        return 1

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning preprocessing outputs for quality issues...")
    logger.info(f"Derivatives path: {derivatives_path}")

    # Scan all outputs
    issues = scan_preprocessing_outputs(derivatives_path)

    logger.info(f"Found {len(issues)} quality flags")

    # Generate report
    generate_markdown_report(issues, output_path)

    logger.info("✅ Quality report generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
