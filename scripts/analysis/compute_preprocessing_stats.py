#!/usr/bin/env python3
"""
Compute comprehensive statistics across all preprocessed sessions.

This script analyzes all preprocessed physiological data to:
1. Extract key metrics from BVP, EDA, and HR preprocessing outputs
2. Identify outliers and potential preprocessing artifacts
3. Generate distribution plots and summary statistics
4. Create a consolidated CSV report for quality control

Phase 1 of preprocessing artifacts investigation (PREPROCESSING_ISSUES.md).

Usage:
    uv run python scripts/analysis/compute_preprocessing_stats.py [--config CONFIG]
    uv run python scripts/analysis/compute_preprocessing_stats.py --verbose
    uv run python scripts/analysis/compute_preprocessing_stats.py --output results/custom_stats.csv

Author: TherasyncPipeline Team
Date: November 11, 2025
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the analysis script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def find_all_preprocessed_sessions(config: Dict, logger: logging.Logger) -> List[Tuple[str, str]]:
    """
    Find all preprocessed sessions.
    
    Returns:
        List of (subject, session) tuples
    """
    preprocessing_dir = Path(config['paths']['derivatives']) / 'preprocessing'
    
    sessions = []
    for subject_dir in sorted(preprocessing_dir.glob('sub-*')):
        if not subject_dir.is_dir():
            continue
        
        subject = subject_dir.name
        for session_dir in sorted(subject_dir.glob('ses-*')):
            if not session_dir.is_dir():
                continue
            
            session = session_dir.name
            
            # Check if BVP, EDA, HR directories exist
            bvp_dir = session_dir / 'bvp'
            eda_dir = session_dir / 'eda'
            hr_dir = session_dir / 'hr'
            
            if bvp_dir.exists() and eda_dir.exists() and hr_dir.exists():
                sessions.append((subject, session))
                logger.debug(f"Found preprocessed session: {subject}/{session}")
    
    logger.info(f"Found {len(sessions)} preprocessed sessions")
    return sessions


def extract_bvp_metrics(subject: str, session: str, config: Dict, logger: logging.Logger) -> Dict:
    """Extract BVP/HRV metrics from preprocessed data."""
    preprocessing_dir = Path(config['paths']['derivatives']) / 'preprocessing'
    bvp_dir = preprocessing_dir / subject / session / 'bvp'
    
    metrics = {
        'subject': subject,
        'session': session,
        'modality': 'BVP'
    }
    
    # Load BVP metrics TSV
    metrics_file = bvp_dir / f"{subject}_{session}_desc-bvp-metrics_physio.tsv"
    
    if not metrics_file.exists():
        logger.warning(f"BVP metrics file not found: {metrics_file}")
        return metrics
    
    try:
        df = pd.read_csv(metrics_file, sep='\t')
        
        # Aggregate across moments (restingstate + therapy) - only numeric columns
        for metric_col in df.columns:
            if metric_col == 'moment':
                continue
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[metric_col]):
                logger.debug(f"Skipping non-numeric column: {metric_col}")
                continue
            
            # Extract summary statistics
            values = df[metric_col].dropna()
            if len(values) > 0:
                metrics[f"BVP_{metric_col}_min"] = values.min()
                metrics[f"BVP_{metric_col}_max"] = values.max()
                metrics[f"BVP_{metric_col}_mean"] = values.mean()
                metrics[f"BVP_{metric_col}_std"] = values.std()
        
        # Flag specific outliers (check actual column name from metrics)
        lfhf_max_key = 'BVP_HRV_LFHF_max'
        if lfhf_max_key in metrics and pd.notna(metrics[lfhf_max_key]):
            if metrics[lfhf_max_key] > 10:
                metrics['BVP_LFHF_outlier'] = True
                logger.warning(f"{subject}/{session}: High LF/HF ratio = {metrics[lfhf_max_key]:.2f}")
        
        # Flag impossible HR values
        hr_mean_max_key = 'BVP_HR_Mean_max'
        hr_mean_min_key = 'BVP_HR_Mean_min'
        if hr_mean_max_key in metrics and pd.notna(metrics[hr_mean_max_key]):
            if metrics[hr_mean_max_key] > 200:
                metrics['BVP_HR_high_outlier'] = True
                logger.warning(f"{subject}/{session}: Very high HR = {metrics[hr_mean_max_key]:.1f} bpm")
        if hr_mean_min_key in metrics and pd.notna(metrics[hr_mean_min_key]):
            if metrics[hr_mean_min_key] < 30:
                metrics['BVP_HR_low_outlier'] = True
                logger.warning(f"{subject}/{session}: Very low HR = {metrics[hr_mean_min_key]:.1f} bpm")
    
    except Exception as e:
        logger.error(f"Error loading BVP metrics for {subject}/{session}: {e}")
    
    return metrics


def extract_eda_metrics(subject: str, session: str, config: Dict, logger: logging.Logger) -> Dict:
    """Extract EDA metrics from preprocessed data."""
    preprocessing_dir = Path(config['paths']['derivatives']) / 'preprocessing'
    eda_dir = preprocessing_dir / subject / session / 'eda'
    
    metrics = {
        'subject': subject,
        'session': session,
        'modality': 'EDA'
    }
    
    # Load EDA metrics TSV
    metrics_file = eda_dir / f"{subject}_{session}_desc-eda-metrics_physio.tsv"
    
    if not metrics_file.exists():
        logger.warning(f"EDA metrics file not found: {metrics_file}")
        return metrics
    
    try:
        df = pd.read_csv(metrics_file, sep='\t')
        
        # Aggregate across moments - only numeric columns
        for metric_col in df.columns:
            if metric_col == 'moment':
                continue
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[metric_col]):
                logger.debug(f"Skipping non-numeric column: {metric_col}")
                continue
            
            values = df[metric_col].dropna()
            if len(values) > 0:
                metrics[f"EDA_{metric_col}_min"] = values.min()
                metrics[f"EDA_{metric_col}_max"] = values.max()
                metrics[f"EDA_{metric_col}_mean"] = values.mean()
                metrics[f"EDA_{metric_col}_std"] = values.std()
        
        # Flag negative EDA values (physical impossibility)
        # Check for actual column names (could be EDA_EDA_Tonic_Mean_min, etc.)
        tonic_keys = [k for k in metrics.keys() if 'Tonic' in k and '_min' in k]
        for tonic_key in tonic_keys:
            if pd.notna(metrics[tonic_key]) and metrics[tonic_key] < 0:
                metrics['EDA_Tonic_negative'] = True
                logger.warning(f"{subject}/{session}: Negative EDA tonic ({tonic_key}) = {metrics[tonic_key]:.6f} µS")
        
        phasic_keys = [k for k in metrics.keys() if 'Phasic' in k and '_min' in k]
        for phasic_key in phasic_keys:
            if pd.notna(metrics[phasic_key]) and metrics[phasic_key] < 0:
                metrics['EDA_Phasic_negative'] = True
                logger.warning(f"{subject}/{session}: Negative EDA phasic ({phasic_key}) = {metrics[phasic_key]:.6f} µS")
    
    except Exception as e:
        logger.error(f"Error loading EDA metrics for {subject}/{session}: {e}")
    
    return metrics


def extract_hr_metrics(subject: str, session: str, config: Dict, logger: logging.Logger) -> Dict:
    """Extract HR metrics from preprocessed data."""
    preprocessing_dir = Path(config['paths']['derivatives']) / 'preprocessing'
    hr_dir = preprocessing_dir / subject / session / 'hr'
    
    metrics = {
        'subject': subject,
        'session': session,
        'modality': 'HR'
    }
    
    # Load HR metrics TSV (different naming pattern)
    metrics_file = hr_dir / f"{subject}_{session}_task-combined_hr-metrics.tsv"
    
    if not metrics_file.exists():
        logger.warning(f"HR metrics file not found: {metrics_file}")
        return metrics
    
    try:
        df = pd.read_csv(metrics_file, sep='\t')
        
        # Aggregate across moments - only numeric columns
        for metric_col in df.columns:
            if metric_col == 'moment':
                continue
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[metric_col]):
                logger.debug(f"Skipping non-numeric column: {metric_col}")
                continue
            
            values = df[metric_col].dropna()
            if len(values) > 0:
                metrics[f"HR_{metric_col}_min"] = values.min()
                metrics[f"HR_{metric_col}_max"] = values.max()
                metrics[f"HR_{metric_col}_mean"] = values.mean()
                metrics[f"HR_{metric_col}_std"] = values.std()
        
        # Flag impossible HR values
        hr_mean_max_key = 'HR_Mean_max'
        hr_mean_min_key = 'HR_Mean_min'
        if hr_mean_max_key in metrics and pd.notna(metrics[hr_mean_max_key]):
            if metrics[hr_mean_max_key] > 200:
                metrics['HR_high_outlier'] = True
                logger.warning(f"{subject}/{session}: Very high HR = {metrics[hr_mean_max_key]:.1f} bpm")
        if hr_mean_min_key in metrics and pd.notna(metrics[hr_mean_min_key]):
            if metrics[hr_mean_min_key] < 30:
                metrics['HR_low_outlier'] = True
                logger.warning(f"{subject}/{session}: Very low HR = {metrics[hr_mean_min_key]:.1f} bpm")
    
    except Exception as e:
        logger.error(f"Error loading HR metrics for {subject}/{session}: {e}")
    
    return metrics


def compute_all_stats(config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """Compute statistics for all preprocessed sessions."""
    sessions = find_all_preprocessed_sessions(config, logger)
    
    all_metrics = []
    
    for subject, session in sessions:
        logger.info(f"Processing {subject}/{session}...")
        
        # Extract metrics from each modality
        bvp_metrics = extract_bvp_metrics(subject, session, config, logger)
        eda_metrics = extract_eda_metrics(subject, session, config, logger)
        hr_metrics = extract_hr_metrics(subject, session, config, logger)
        
        # Combine all metrics into one row
        combined = {'subject': subject, 'session': session}
        combined.update({k: v for k, v in bvp_metrics.items() if k not in ['subject', 'session', 'modality']})
        combined.update({k: v for k, v in eda_metrics.items() if k not in ['subject', 'session', 'modality']})
        combined.update({k: v for k, v in hr_metrics.items() if k not in ['subject', 'session', 'modality']})
        
        all_metrics.append(combined)
    
    df = pd.DataFrame(all_metrics)
    logger.info(f"Extracted metrics for {len(df)} sessions, {len(df.columns)} columns")
    
    return df


def identify_outliers(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Identify and flag outlier sessions."""
    outliers = []
    
    for idx, row in df.iterrows():
        outlier_info = {
            'subject': row['subject'],
            'session': row['session'],
            'outlier_types': []
        }
        
        # Check for LF/HF ratio outliers (actual column is BVP_HRV_LFHF_max)
        lfhf_col = 'BVP_HRV_LFHF_max'
        if lfhf_col in df.columns and pd.notna(row[lfhf_col]):
            if row[lfhf_col] > 10:
                outlier_info['outlier_types'].append(f"High LF/HF ({row[lfhf_col]:.2f})")
        
        # Check for negative EDA Tonic (check all tonic min columns)
        tonic_min_cols = [c for c in df.columns if 'EDA_EDA_Tonic' in c and '_min' in c.lower()]
        for col in tonic_min_cols:
            if pd.notna(row[col]) and row[col] < 0:
                outlier_info['outlier_types'].append(f"Negative {col.replace('EDA_EDA_', 'EDA_')} ({row[col]:.6f} µS)")
                break  # Only report once per session
        
        # Check for negative EDA Phasic
        phasic_min_cols = [c for c in df.columns if 'EDA_EDA_Phasic' in c and '_min' in c.lower()]
        for col in phasic_min_cols:
            if pd.notna(row[col]) and row[col] < 0:
                outlier_info['outlier_types'].append(f"Negative {col.replace('EDA_EDA_', 'EDA_')} ({row[col]:.6f} µS)")
                break  # Only report once per session
        
        # Check for impossible HR values from BVP
        if 'BVP_HR_Mean_max' in df.columns and pd.notna(row['BVP_HR_Mean_max']):
            if row['BVP_HR_Mean_max'] > 200:
                outlier_info['outlier_types'].append(f"High BVP HR ({row['BVP_HR_Mean_max']:.1f} bpm)")
        
        if 'BVP_HR_Mean_min' in df.columns and pd.notna(row['BVP_HR_Mean_min']):
            if row['BVP_HR_Mean_min'] < 30:
                outlier_info['outlier_types'].append(f"Low BVP HR ({row['BVP_HR_Mean_min']:.1f} bpm)")
        
        # Check for impossible HR values from HR module
        if 'HR_Mean_max' in df.columns and pd.notna(row['HR_Mean_max']):
            if row['HR_Mean_max'] > 200:
                outlier_info['outlier_types'].append(f"High HR ({row['HR_Mean_max']:.1f} bpm)")
        
        if 'HR_Mean_min' in df.columns and pd.notna(row['HR_Mean_min']):
            if row['HR_Mean_min'] < 30:
                outlier_info['outlier_types'].append(f"Low HR ({row['HR_Mean_min']:.1f} bpm)")
        
        if len(outlier_info['outlier_types']) > 0:
            outlier_info['outlier_summary'] = '; '.join(outlier_info['outlier_types'])
            outliers.append(outlier_info)
    
    outlier_df = pd.DataFrame(outliers)
    
    if len(outlier_df) > 0:
        logger.warning(f"\n{'='*80}")
        logger.warning(f"OUTLIERS DETECTED: {len(outlier_df)} sessions flagged")
        logger.warning(f"{'='*80}")
        for _, row in outlier_df.iterrows():
            logger.warning(f"{row['subject']}/{row['session']}: {row['outlier_summary']}")
        logger.warning(f"{'='*80}\n")
    else:
        logger.info("No outliers detected ✓")
    
    return outlier_df


def create_distribution_plots(df: pd.DataFrame, output_dir: Path, logger: logging.Logger):
    """Create distribution plots for key metrics."""
    logger.info("Generating distribution plots...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Select key metrics to plot (using actual column names)
    metrics_to_plot = [
        ('BVP_HRV_MeanNN_mean', 'BVP: Mean NN Interval (ms)', (500, 1500)),
        ('BVP_HRV_RMSSD_mean', 'HRV: RMSSD (ms)', (0, 1000)),
        ('BVP_HRV_LFHF_mean', 'HRV: LF/HF Ratio (mean)', (0, 20)),
        ('BVP_HRV_LFHF_max', 'HRV: LF/HF Ratio (max) ⚠️', (0, 80)),
        ('EDA_EDA_Tonic_Mean_mean', 'EDA: Tonic Mean (µS)', (0, 5)),
        ('EDA_SCR_Peaks_N_mean', 'EDA: Number of SCR Peaks', (0, 250)),
        ('EDA_EDA_Phasic_Min_min', 'EDA: Phasic Min (µS) ⚠️', (-5, 1)),
        ('HR_hr_mean_mean', 'HR: Mean Heart Rate (bpm)', (40, 120))
    ]
    
    # Filter available metrics
    available_metrics = [(col, label, xlim) for col, label, xlim in metrics_to_plot if col in df.columns]
    
    if len(available_metrics) == 0:
        logger.warning("No metrics available for plotting")
        return
    
    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (metric_col, label, xlim) in enumerate(available_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get data
        data = df[metric_col].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        # Histogram + KDE
        ax.hist(data, bins=30, alpha=0.6, color='steelblue', edgecolor='black', density=True)
        
        # Add KDE if enough data points
        if len(data) > 5:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label} Distribution (n={len(data)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits if specified
        if xlim:
            ax.set_xlim(xlim)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'preprocessing_stats_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plot: {output_file}")
    plt.close()
    
    # Create boxplots for outlier detection
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (metric_col, label, xlim) in enumerate(available_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        data = df[metric_col].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        # Boxplot
        bp = ax.boxplot([data], vert=False, patch_artist=True, 
                        widths=0.6, showmeans=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f'{label} Boxplot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set x-axis limits if specified
        if xlim:
            ax.set_xlim(xlim)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    output_file = output_dir / 'preprocessing_stats_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved boxplot: {output_file}")
    plt.close()


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description='Compute comprehensive statistics for all preprocessed sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    uv run python scripts/analysis/compute_preprocessing_stats.py
    
    # Run with custom config and verbose output
    uv run python scripts/analysis/compute_preprocessing_stats.py --config config/custom.yaml --verbose
    
    # Specify custom output location
    uv run python scripts/analysis/compute_preprocessing_stats.py --output results/custom_stats.csv
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV file path (default: data/derivatives/analysis/preprocessing_stats.csv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    logger.info("="*80)
    logger.info("PREPROCESSING STATISTICS ANALYSIS - Phase 1")
    logger.info("="*80)
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config or 'config/config.yaml'}")
    
    # Compute statistics
    logger.info("\nStep 1: Extracting metrics from all preprocessed sessions...")
    stats_df = compute_all_stats(config, logger)
    
    # Identify outliers
    logger.info("\nStep 2: Identifying outliers...")
    outlier_df = identify_outliers(stats_df, logger)
    
    # Save results
    if args.output:
        output_csv = args.output
    else:
        output_csv = Path(config['paths']['derivatives']) / 'analysis' / 'preprocessing_stats.csv'
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, index=False)
    logger.info(f"\nSaved statistics CSV: {output_csv}")
    
    # Save outliers
    if len(outlier_df) > 0:
        outlier_csv = output_csv.parent / 'preprocessing_outliers.csv'
        outlier_df.to_csv(outlier_csv, index=False)
        logger.info(f"Saved outliers CSV: {outlier_csv}")
    
    # Create visualizations
    logger.info("\nStep 3: Creating distribution plots...")
    plot_dir = output_csv.parent / 'plots'
    create_distribution_plots(stats_df, plot_dir, logger)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Sessions analyzed: {len(stats_df)}")
    logger.info(f"Metrics extracted: {len(stats_df.columns) - 2}")  # -2 for subject, session
    logger.info(f"Outliers detected: {len(outlier_df)}")
    logger.info(f"Results saved to: {output_csv.parent}")
    logger.info("="*80)
    
    if len(outlier_df) > 0:
        logger.warning("\n⚠️  OUTLIERS REQUIRE INVESTIGATION - See PREPROCESSING_ISSUES.md Phase 2")
    else:
        logger.info("\n✓ No outliers detected - Preprocessing quality is good!")


if __name__ == '__main__':
    main()
