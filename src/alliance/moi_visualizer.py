"""
MOI (Moments of Interest) Visualizer.

Creates visualizations for epoched MOI alliance and emotion annotations.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from src.core.config_loader import ConfigLoader
from src.visualization.config import COLORS, apply_plot_style

logger = logging.getLogger(__name__)


class MOIVisualizer:
    """Visualizes epoched MOI annotations."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize MOI visualizer.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.derivatives_path = Path(self.config["paths"]["derivatives"])
        self.alliance_dir = self.config["output"]["alliance_dir"]

        # Apply project-wide plot style
        apply_plot_style()

        # Color schemes from project visualization config
        self.colors = {
            "positive": COLORS["positive"],  # Teal
            "negative": COLORS["negative"],  # Coral
            "mixed": COLORS["medium"],  # Amber (warning color)
            "none": COLORS["gray"],  # Neutral gray
        }

    def load_epoched_moi(self, group_id: str, session_id: str) -> pd.DataFrame:
        """
        Load epoched MOI annotations.

        Args:
            group_id: Group ID (e.g., 'g01')
            session_id: Session ID (e.g., '01')

        Returns:
            DataFrame with epoched annotations
        """
        subject_dir = f"sub-{group_id}shared"
        session_dir = f"ses-{session_id}"
        annotations_subdir = self.config["output"]["alliance_subdirs"]["annotations"]

        tsv_file = (
            self.derivatives_path
            / self.alliance_dir
            / subject_dir
            / session_dir
            / annotations_subdir
            / f"sub-{group_id}_ses-{session_id}_desc-alliance_annotations_epoched.tsv"
        )

        if not tsv_file.exists():
            raise FileNotFoundError(f"Epoched file not found: {tsv_file}")

        df = pd.read_csv(tsv_file, sep="\t")

        # Parse epoch columns from string lists to actual lists
        for col in ["epoch_fixed", "epoch_nsplit", "epoch_sliding"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

        return df

    def compute_epoch_states(
        self, df: pd.DataFrame, method: str, metadata: Optional[Dict] = None
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Compute alliance and emotion states for each epoch.

        Args:
            df: DataFrame with epoched annotations
            method: Epoching method ('fixed', 'nsplit', 'sliding')
            metadata: Optional metadata with session duration

        Returns:
            Tuple of (alliance_states, emotion_states) where each is a dict:
                epoch_id -> state (0=none, 1=positive, -1=negative, 2=mixed)
        """
        epoch_col = f"epoch_{method}"

        # Get all epochs that have annotations
        epochs_with_annotations = set()
        for epoch_list in df[epoch_col]:
            epochs_with_annotations.update(epoch_list)

        # Determine total number of epochs (includes empty ones)
        if epochs_with_annotations:
            max_epoch = max(epochs_with_annotations)
        else:
            max_epoch = 0

        # Initialize ALL epochs to state 0 (none/neutral)
        alliance_states = {i: 0 for i in range(max_epoch + 1)}
        emotion_states = {i: 0 for i in range(max_epoch + 1)}

        # Now fill in the states for epochs that have annotations
        for epoch_id in epochs_with_annotations:
            # Find all annotations that overlap this epoch
            overlapping = df[df[epoch_col].apply(lambda x: epoch_id in x)]

            # Alliance state
            alliance_values = overlapping["alliance"].dropna()
            has_positive_alliance = (alliance_values == 1).any()
            has_negative_alliance = (alliance_values == -1).any()

            if has_positive_alliance and has_negative_alliance:
                alliance_states[epoch_id] = 2  # Mixed
            elif has_positive_alliance:
                alliance_states[epoch_id] = 1  # Positive
            elif has_negative_alliance:
                alliance_states[epoch_id] = -1  # Negative
            # else: already initialized to 0 (none)

            # Emotion state
            emotion_values = overlapping["emotion"].dropna()
            has_positive_emotion = (emotion_values == 1).any()
            has_negative_emotion = (emotion_values == -1).any()

            if has_positive_emotion and has_negative_emotion:
                emotion_states[epoch_id] = 2  # Mixed
            elif has_positive_emotion:
                emotion_states[epoch_id] = 1  # Positive
            elif has_negative_emotion:
                emotion_states[epoch_id] = -1  # Negative
            # else: already initialized to 0 (none)

        return alliance_states, emotion_states

    def plot_timeseries(
        self,
        df: pd.DataFrame,
        group_id: str,
        session_id: str,
        method: str,
        output_dir: Path,
    ) -> Path:
        """
        Create timeseries visualization (3 rows × 2 columns).

        Args:
            df: DataFrame with epoched annotations
            group_id: Group ID
            session_id: Session ID
            method: Epoching method
            output_dir: Output directory for figure

        Returns:
            Path to saved figure
        """
        alliance_states, emotion_states = self.compute_epoch_states(df, method)

        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(
            f"MOI Annotations Timeline - {group_id.upper()}/Session {session_id} ({method.capitalize()} epochs)",
            fontsize=14,
            fontweight="bold",
        )

        epochs = sorted(alliance_states.keys())

        # Row 0: Positive
        self._plot_binary_line(
            axes[0, 0],
            epochs,
            alliance_states,
            target=1,
            title="Alliance: Positive",
            color=self.colors["positive"],
        )
        self._plot_binary_line(
            axes[0, 1],
            epochs,
            emotion_states,
            target=1,
            title="Emotion: Positive",
            color=self.colors["positive"],
        )

        # Row 1: Negative
        self._plot_binary_line(
            axes[1, 0],
            epochs,
            alliance_states,
            target=-1,
            title="Alliance: Negative",
            color=self.colors["negative"],
        )
        self._plot_binary_line(
            axes[1, 1],
            epochs,
            emotion_states,
            target=-1,
            title="Emotion: Negative",
            color=self.colors["negative"],
        )

        # Row 2: Mixed (Split)
        self._plot_binary_line(
            axes[2, 0],
            epochs,
            alliance_states,
            target=2,
            title="Alliance: Mixed (Split)",
            color=self.colors["mixed"],
        )
        self._plot_binary_line(
            axes[2, 1],
            epochs,
            emotion_states,
            target=2,
            title="Emotion: Mixed (Split)",
            color=self.colors["mixed"],
        )

        # Common X label
        for ax in axes[2, :]:
            ax.set_xlabel("Epoch ID", fontsize=10)

        plt.tight_layout()

        # Save figure
        output_file = (
            output_dir
            / f"sub-{group_id}_ses-{session_id}_method-{method}_desc-alliance_timeseries.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Saved timeseries: {output_file.name}")
        return output_file

    def _plot_binary_line(
        self,
        ax,
        epochs: List[int],
        states: Dict[int, int],
        target: int,
        title: str,
        color: str,
    ):
        """
        Plot binary presence/absence line.

        Args:
            ax: Matplotlib axis
            epochs: List of epoch IDs
            states: Dict mapping epoch_id -> state
            target: Target state to plot (1=positive, -1=negative, 2=mixed)
            title: Subplot title
            color: Line color
        """
        # Convert to binary: 1 if state matches target, 0 otherwise
        y = [1 if states.get(e, 0) == target else 0 for e in epochs]

        ax.plot(
            epochs, y, color=color, linewidth=2, marker="o", markersize=3, alpha=0.7
        )
        ax.fill_between(epochs, 0, y, color=color, alpha=0.2)
        ax.set_ylabel("Presence", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Absent", "Present"])
        ax.grid(True, alpha=0.3, linestyle="--")

    def plot_distribution(
        self,
        df: pd.DataFrame,
        group_id: str,
        session_id: str,
        method: str,
        output_dir: Path,
    ) -> Path:
        """
        Create distribution pie charts (alliance + emotion).

        Args:
            df: DataFrame with epoched annotations
            group_id: Group ID
            session_id: Session ID
            method: Epoching method
            output_dir: Output directory for figure

        Returns:
            Path to saved figure
        """
        alliance_states, emotion_states = self.compute_epoch_states(df, method)

        # Count states
        alliance_counts = self._count_states(alliance_states)
        emotion_counts = self._count_states(emotion_states)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"MOI Annotations Distribution - {group_id.upper()}/Session {session_id} ({method.capitalize()} epochs)",
            fontsize=14,
            fontweight="bold",
        )

        # Alliance pie chart
        self._plot_pie(ax1, alliance_counts, title="Alliance Distribution")

        # Emotion pie chart
        self._plot_pie(ax2, emotion_counts, title="Emotion Distribution")

        plt.tight_layout()

        # Save figure
        output_file = (
            output_dir
            / f"sub-{group_id}_ses-{session_id}_method-{method}_desc-alliance_distribution.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Saved distribution: {output_file.name}")
        return output_file

    def _count_states(self, states: Dict[int, int]) -> Dict[str, int]:
        """
        Count occurrences of each state.

        Args:
            states: Dict mapping epoch_id -> state

        Returns:
            Dict with counts for 'positive', 'negative', 'mixed', 'none'
        """
        counts = {
            "positive": sum(1 for s in states.values() if s == 1),
            "negative": sum(1 for s in states.values() if s == -1),
            "mixed": sum(1 for s in states.values() if s == 2),
            "none": sum(1 for s in states.values() if s == 0),
        }
        return counts

    def _plot_pie(self, ax, counts: Dict[str, int], title: str):
        """
        Plot pie chart for state distribution.

        Args:
            ax: Matplotlib axis
            counts: Dict with state counts
            title: Chart title
        """
        # Filter out zero counts
        filtered_counts = {k: v for k, v in counts.items() if v > 0}

        if not filtered_counts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(title, fontsize=12, fontweight="bold")
            return

        labels = [k.capitalize() for k in filtered_counts.keys()]
        sizes = list(filtered_counts.values())
        colors = [self.colors[k] for k in filtered_counts.keys()]

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(9)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    def visualize_session(
        self, group_id: str, session_id: str
    ) -> Dict[str, List[Path]]:
        """
        Create all visualizations for a session.

        Args:
            group_id: Group ID
            session_id: Session ID

        Returns:
            Dict mapping visualization type to list of output files
        """
        logger.info(f"Creating visualizations for {group_id}/ses-{session_id}")

        # Load data
        df = self.load_epoched_moi(group_id, session_id)

        # Create output directory (centralized in visualization/alliance/)
        subject_dir = f"sub-{group_id}shared"
        session_dir = f"ses-{session_id}"

        output_dir = (
            self.derivatives_path
            / "visualization"
            / "alliance"
            / subject_dir
            / session_dir
            / "figures"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"timeseries": [], "distribution": []}

        # Generate visualizations for each method
        for method in ["fixed", "nsplit", "sliding"]:
            # Timeseries
            ts_file = self.plot_timeseries(df, group_id, session_id, method, output_dir)
            results["timeseries"].append(ts_file)

            # Distribution
            dist_file = self.plot_distribution(
                df, group_id, session_id, method, output_dir
            )
            results["distribution"].append(dist_file)

        logger.info(
            f"✓ Created {len(results['timeseries']) + len(results['distribution'])} visualizations"
        )

        return results
