"""
EDA-specific visualization plots.

Implements visualizations #4, #5:
- EDA arousal profile (tonic/phasic comparison)
- SCR distribution (histogram + boxplot)

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import pandas as pd

from ..config import (
    COLORS, FIGSIZE, FONTSIZE, LINEWIDTH, ALPHA,
    apply_plot_style, get_moment_color, get_moment_label, get_moment_order,
    format_duration
)


def plot_eda_arousal_profile(
    data: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot EDA baseline arousal comparison between moments.
    
    Visualization #4: Quantitative comparison showing:
    - Tonic EDA levels: boxplot distribution (baseline arousal)
    - Phasic variability: bar chart of standard deviation (reactivity)
    
    Args:
        data: Dictionary containing EDA signals with structure:
            - 'eda': Dict with 'signals' containing moment DataFrames
                     Each should have: 'EDA_Tonic', 'EDA_Phasic'
        output_path: Where to save the figure (optional)
        show: Whether to display the figure
    
    Returns:
        Figure object
    """
    apply_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE['wide'])
    
    eda_data = data.get('eda', {})
    if not eda_data or 'signals' not in eda_data:
        ax1.text(0.5, 0.5, 'No EDA data available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return
    
    # Get all available moments dynamically
    available_moments = list(eda_data.get('signals', {}).keys())
    
    if not available_moments:
        ax1.text(0.5, 0.5, 'No EDA signals available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return
    
    # Calculate statistics for each moment
    stats = {}
    for moment in available_moments:
        df = eda_data['signals'][moment]
        if df.empty or 'EDA_Tonic' not in df.columns or 'EDA_Phasic' not in df.columns:
            continue
        
        # Clip tonic values to 0 (physiological minimum)
        # Note: Negative tonic values indicate preprocessing artifacts
        tonic_clipped = df['EDA_Tonic'].clip(lower=0)
        
        stats[moment] = {
            'tonic_mean': tonic_clipped.mean(),
            'tonic_min': tonic_clipped.min(),
            'tonic_max': tonic_clipped.max(),
            'phasic_std': df['EDA_Phasic'].std()
        }
    
    if not stats:
        ax1.text(0.5, 0.5, 'Insufficient EDA data', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return
    
    # Panel 1: Tonic EDA distribution (boxplot)
    # Prepare data for boxplot
    tonic_data = []
    moment_labels = []
    moment_colors = []
    
    for moment in available_moments:
        df = eda_data['signals'][moment]
        if df.empty or 'EDA_Tonic' not in df.columns:
            continue
        
        # Clip tonic values to 0 (physiological minimum)
        tonic_clipped = df['EDA_Tonic'].clip(lower=0)
        tonic_data.append(tonic_clipped.values)
        moment_labels.append(get_moment_label(moment))
        moment_colors.append(get_moment_color(moment))
    
    # Create boxplot
    bp = ax1.boxplot(tonic_data, labels=moment_labels, patch_artist=True,
                     widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', 
                                   markeredgecolor='black', markersize=8),
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color each box
    for patch, color in zip(bp['boxes'], moment_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Moment', fontsize=FONTSIZE['label'], fontweight='bold')
    ax1.set_ylabel('EDA Tonic Level (µS)', fontsize=FONTSIZE['label'], fontweight='bold')
    ax1.grid(True, alpha=ALPHA['fill'], axis='y', linestyle='--')
    ax1.set_title('Baseline Arousal Distribution\n(Tonic EDA)', 
                 fontsize=FONTSIZE['subtitle'], fontweight='bold')
    
    # Panel 2: Phasic variability (reactivity)
    x = np.arange(len(stats))
    for i, moment in enumerate(stats.keys()):
        moment_color = get_moment_color(moment)
        s = stats[moment]
        
        # Phasic std as bar
        ax2.bar(i, s['phasic_std'], 0.6,
               color=moment_color, alpha=0.8,
               edgecolor='white', linewidth=2)
        
        # Add value label
        ax2.text(i, s['phasic_std'] + max([stats[m]['phasic_std'] for m in stats]) * 0.02, 
                f"{s['phasic_std']:.3f}",
                ha='center', va='bottom', fontsize=FONTSIZE['annotation'],
                fontweight='bold', color=moment_color)
    
    ax2.set_xlabel('Moment', fontsize=FONTSIZE['label'], fontweight='bold')
    ax2.set_ylabel('EDA Phasic Std Dev (µS)', fontsize=FONTSIZE['label'], fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_moment_label(m) for m in stats.keys()], 
                        fontsize=FONTSIZE['tick'], fontweight='bold')
    ax2.grid(True, alpha=ALPHA['fill'], axis='y', linestyle='--')
    ax2.set_title('Emotional Reactivity\n(Phasic Variability)', 
                 fontsize=FONTSIZE['subtitle'], fontweight='bold')
    
    # Overall title
    fig.suptitle(
        f'EDA Baseline Arousal Comparison\n'
        f'Subject {data.get("subject", "Unknown")}, Session {data.get("session", "Unknown")}',
        fontsize=FONTSIZE['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_scr_distribution(
    data: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot SCR amplitude distribution with histogram and boxplot per moment.
    
    Visualization #5: Separate histogram + boxplot for each moment:
    - Left column: Histograms showing amplitude distribution per moment
    - Right column: Boxplots showing statistical summary per moment
    - One row per moment for clear comparison
    
    Args:
        data: Dictionary containing EDA events with key:
            - 'eda': Dict with 'events' sub-dict containing moment DataFrames
                     Each should have 'amplitude' column
        output_path: Where to save the figure (optional)
        show: Whether to display the figure
    
    Returns:
        Figure object
    """
    apply_plot_style()
    
    # Get all available moments dynamically
    eda_data = data.get('eda', {})
    if not eda_data or 'events' not in eda_data:
        fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
        ax.text(0.5, 0.5, 'No EDA events available', 
               ha='center', va='center', fontsize=FONTSIZE['title'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    available_moments = list(eda_data.get('events', {}).keys())
    
    if not available_moments:
        fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
        ax.text(0.5, 0.5, 'No SCR events available', 
               ha='center', va='center', fontsize=FONTSIZE['title'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Collect amplitude data per moment
    amplitude_data = {}
    for moment in available_moments:
        df = eda_data['events'][moment]
        if df.empty or 'amplitude' not in df.columns:
            continue
        # Filter out NaN values
        amplitudes = df['amplitude'].values
        amplitudes_clean = amplitudes[~np.isnan(amplitudes)]
        
        if len(amplitudes_clean) > 0:
            amplitude_data[moment] = amplitudes_clean
    
    if not amplitude_data:
        fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
        ax.text(0.5, 0.5, 'No SCR amplitude data available', 
               ha='center', va='center', fontsize=FONTSIZE['label'])
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Create figure with N rows x 2 columns (N = number of moments)
    # Column 1: Histogram, Column 2: Boxplot
    n_moments = len(amplitude_data)
    fig, axes = plt.subplots(n_moments, 2, figsize=(FIGSIZE['wide'][0], FIGSIZE['wide'][1] * n_moments / 2))
    
    # Handle single moment case (axes needs reshaping)
    if n_moments == 1:
        axes = axes.reshape(1, 2)
    
    # Calculate global min/max for amplitude normalization
    all_amplitudes = np.concatenate(list(amplitude_data.values()))
    amp_min = 0  # Start at 0 for physiological interpretation
    amp_max = all_amplitudes.max() * 1.1  # Add 10% margin
    
    # Calculate global max frequency for histogram normalization
    max_freq = 0
    hist_data = []
    for amplitudes in amplitude_data.values():
        counts, _ = np.histogram(amplitudes, bins=30, range=(amp_min, amp_max))
        max_freq = max(max_freq, counts.max())
        hist_data.append((amplitudes, counts))
    max_freq = max_freq * 1.1  # Add 10% margin
    
    # Plot each moment in its own row
    for idx, (moment, amplitudes) in enumerate(amplitude_data.items()):
        moment_color = get_moment_color(moment)
        
        # Column 1: Histogram
        ax_hist = axes[idx, 0]
        ax_hist.hist(amplitudes, bins=30, range=(amp_min, amp_max), alpha=0.7, 
                    color=moment_color, edgecolor='black', linewidth=1.2)
        ax_hist.set_ylabel('Frequency', fontsize=FONTSIZE['label'], fontweight='bold')
        ax_hist.set_xlabel('Amplitude (µS)', fontsize=FONTSIZE['label'], fontweight='bold')
        ax_hist.set_title(f'{get_moment_label(moment)} - Distribution (n={len(amplitudes)})', 
                         fontsize=FONTSIZE['subtitle'], fontweight='bold')
        ax_hist.set_xlim(amp_min, amp_max)
        ax_hist.set_ylim(0, max_freq)
        ax_hist.grid(True, alpha=ALPHA['medium'], axis='y', linestyle='--')
        ax_hist.tick_params(labelsize=FONTSIZE['tick'])
        
        # Column 2: Boxplot
        ax_box = axes[idx, 1]
        bp = ax_box.boxplot([amplitudes], vert=True, patch_artist=True,
                            widths=0.6, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='white', 
                                          markeredgecolor='black', markersize=8),
                            medianprops=dict(color='black', linewidth=2),
                            boxprops=dict(linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='red', 
                                           markersize=4, alpha=0.5))
        
        # Color the box
        bp['boxes'][0].set_facecolor(moment_color)
        bp['boxes'][0].set_alpha(0.7)
        
        ax_box.set_ylabel('Amplitude (µS)', fontsize=FONTSIZE['label'], fontweight='bold')
        ax_box.set_title(f'{get_moment_label(moment)} - Statistics', 
                        fontsize=FONTSIZE['subtitle'], fontweight='bold')
        ax_box.set_ylim(amp_min, amp_max)
        ax_box.set_xticklabels([''])  # No x-axis labels needed
        ax_box.grid(True, alpha=ALPHA['medium'], axis='y', linestyle='--')
        ax_box.tick_params(labelsize=FONTSIZE['tick'])
    
    # Overall title
    fig.suptitle(
        f'Skin Conductance Response (SCR) Amplitude Distribution\n'
        f'Subject {data.get("subject", "Unknown")}, Session {data.get("session", "Unknown")}',
        fontsize=FONTSIZE['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
