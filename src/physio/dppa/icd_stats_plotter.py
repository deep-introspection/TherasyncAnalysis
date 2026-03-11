"""
ICD Statistical Visualization Module.

Creates publication-ready visualizations comparing real vs pseudo dyad ICDs:
- Violin/boxplot distributions
- Inter-session heatmaps
- Family-level summaries
- Statistical test results

Uses the project's visualization style guide for consistency.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from src.core.config_loader import ConfigLoader

# Optional: statsmodels for mixed models
try:
    from statsmodels.formula.api import mixedlm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)


# =============================================================================
# COLOR PALETTE - Consistent with project style guide
# =============================================================================

COLORS = {
    # DPPA-specific colors
    "real_dyad": "#7EB8DA",  # Sky Blue - Real dyads (same family)
    "pseudo_dyad": "#F4A4B8",  # Rose Pink - Pseudo dyads (cross-family)
    "therapy": "#F4A4B8",  # Rose Pink
    "restingstate": "#7EB8DA",  # Sky Blue
    # Utility
    "text": "#2C3E50",
    "grid": "#CCCCCC",
    "background": "#FFFFFF",
    "dark_gray": "#2C3E50",
    "good": "#6DD47E",  # Significant difference
    "poor": "#E17055",  # No significance
}

ALPHA = {
    "fill": 0.25,
    "medium": 0.5,
    "high": 0.7,
    "line": 0.85,
}

FIGSIZE = {
    "small": (8, 6),
    "medium": (12, 8),
    "large": (16, 10),
    "wide": (14, 6),
}

FONTSIZE = {
    "title": 16,
    "subtitle": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
    "small": 8,
}

LINEWIDTH = {
    "thin": 0.5,
    "normal": 1.0,
    "medium": 1.5,
    "thick": 2.0,
}

MARKERSIZE = {
    "small": 4,
    "medium": 6,
    "large": 8,
}


def apply_plot_style() -> None:
    """Apply consistent plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "figure.dpi": 150,
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlesize": FONTSIZE["title"],
            "axes.labelsize": FONTSIZE["label"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": FONTSIZE["tick"],
            "ytick.labelsize": FONTSIZE["tick"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "legend.fontsize": FONTSIZE["legend"],
            "legend.frameon": False,
            "grid.alpha": 0.3,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "font.family": "sans-serif",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": COLORS["background"],
        }
    )


class ICDStatsPlotter:
    """
    Generate statistical visualizations for ICD analysis.

    Creates figures comparing real vs pseudo dyads, including:
    - Distribution comparisons (violin + box + swarm)
    - Epoch-wise evolution
    - Statistical test results

    Example:
        >>> plotter = ICDStatsPlotter()
        >>> plotter.plot_real_vs_pseudo(icd_df, output_path)
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ICD Stats Plotter.

        Args:
            config_path: Optional path to configuration file.
        """
        self.config = ConfigLoader(config_path)
        logger.info("ICDStatsPlotter initialized")

    def load_icd_data(
        self, icd_file: Path, task: str = "therapy"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and parse ICD CSV file.

        Args:
            icd_file: Path to ICD CSV file
            task: Task name for logging

        Returns:
            Tuple of (melted DataFrame, metadata dict)
        """
        logger.info(f"Loading ICD data from {icd_file}")

        # Read CSV (wide format: epochs x dyad pairs)
        df = pd.read_csv(icd_file)

        # Extract dyad columns (all except epoch_id)
        dyad_cols = [c for c in df.columns if c != "epoch_id"]

        # Melt to long format
        df_long = df.melt(
            id_vars=["epoch_id"],
            value_vars=dyad_cols,
            var_name="dyad_pair",
            value_name="icd_value",
        )

        # Parse dyad type from column name
        # Real dyads have format: g01p01_ses-01_vs_g01p02_ses-01 (same family g01)
        # Pseudo dyads: g01p01_ses-01_vs_g02p01_ses-01 (different families)
        def classify_dyad(dyad_pair: str) -> str:
            parts = dyad_pair.split("_vs_")
            if len(parts) != 2:
                return "unknown"

            # Extract family codes (first 3 chars: g01, g02, etc.)
            family1 = parts[0][:3]  # g01
            family2 = parts[1][:3]  # g01 or g02

            return "real" if family1 == family2 else "pseudo"

        df_long["dyad_type"] = df_long["dyad_pair"].apply(classify_dyad)

        # Remove NaN values
        df_long = df_long.dropna(subset=["icd_value"])

        # Metadata
        n_real = df_long[df_long["dyad_type"] == "real"]["dyad_pair"].nunique()
        n_pseudo = df_long[df_long["dyad_type"] == "pseudo"]["dyad_pair"].nunique()
        n_epochs = df_long["epoch_id"].nunique()

        metadata = {
            "task": task,
            "n_real_dyads": n_real,
            "n_pseudo_dyads": n_pseudo,
            "n_epochs": n_epochs,
            "total_observations": len(df_long),
        }

        logger.info(
            f"Loaded: {n_real} real dyads, {n_pseudo} pseudo dyads, {n_epochs} epochs"
        )

        return df_long, metadata

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute statistical comparison between real and pseudo dyads.

        Args:
            df: Long-format DataFrame with 'dyad_type' and 'icd_value' columns

        Returns:
            Dictionary with statistical test results
        """
        real_values = df[df["dyad_type"] == "real"]["icd_value"].values
        pseudo_values = df[df["dyad_type"] == "pseudo"]["icd_value"].values

        # Descriptive statistics
        real_stats = {
            "mean": np.mean(real_values),
            "median": np.median(real_values),
            "std": np.std(real_values),
            "n": len(real_values),
        }
        pseudo_stats = {
            "mean": np.mean(pseudo_values),
            "median": np.median(pseudo_values),
            "std": np.std(pseudo_values),
            "n": len(pseudo_values),
        }

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mw = stats.mannwhitneyu(
            real_values, pseudo_values, alternative="two-sided"
        )

        # Independent t-test (parametric)
        t_stat, p_value_t = stats.ttest_ind(real_values, pseudo_values)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (real_stats["n"] - 1) * real_stats["std"] ** 2
                + (pseudo_stats["n"] - 1) * pseudo_stats["std"] ** 2
            )
            / (real_stats["n"] + pseudo_stats["n"] - 2)
        )
        cohens_d = (real_stats["mean"] - pseudo_stats["mean"]) / pooled_std

        return {
            "real": real_stats,
            "pseudo": pseudo_stats,
            "mann_whitney": {"U": u_stat, "p": p_value_mw},
            "t_test": {"t": t_stat, "p": p_value_t},
            "cohens_d": cohens_d,
            "difference": real_stats["mean"] - pseudo_stats["mean"],
        }

    def compute_participant_level_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute statistics corrected for non-independence by aggregating per participant.

        Each participant appears in multiple dyads. This method:
        1. Computes mean ICD per participant across all their dyads
        2. Separates into real vs pseudo contributions
        3. Uses paired test (each participant has both real and pseudo mean)

        This addresses the pseudo-replication problem where the naive analysis
        treats each dyad as independent, inflating statistical significance.

        Args:
            df: Long-format DataFrame with 'dyad_type', 'icd_value', 'dyad_pair' columns

        Returns:
            Dictionary with corrected statistical test results
        """
        logger.info(
            "Computing participant-level statistics (corrected for non-independence)"
        )

        # Extract participant info from dyad_pair
        def extract_participants(dyad_pair: str) -> Tuple[str, str]:
            parts = dyad_pair.split("_vs_")
            return parts[0], parts[1]

        # Build participant-level data
        participant_real_icds = {}  # participant -> list of ICD values in real dyads
        participant_pseudo_icds = {}  # participant -> list of ICD values in pseudo dyads

        for _, row in df.iterrows():
            p1, p2 = extract_participants(row["dyad_pair"])
            icd = row["icd_value"]
            dyad_type = row["dyad_type"]

            if dyad_type == "real":
                for p in [p1, p2]:
                    if p not in participant_real_icds:
                        participant_real_icds[p] = []
                    participant_real_icds[p].append(icd)
            else:  # pseudo
                for p in [p1, p2]:
                    if p not in participant_pseudo_icds:
                        participant_pseudo_icds[p] = []
                    participant_pseudo_icds[p].append(icd)

        # Compute mean ICD per participant
        participant_real_means = {
            p: np.mean(vals) for p, vals in participant_real_icds.items()
        }
        participant_pseudo_means = {
            p: np.mean(vals) for p, vals in participant_pseudo_icds.items()
        }

        # Get participants with both real and pseudo data (for paired test)
        common_participants = set(participant_real_means.keys()) & set(
            participant_pseudo_means.keys()
        )

        real_values = np.array([participant_real_means[p] for p in common_participants])
        pseudo_values = np.array(
            [participant_pseudo_means[p] for p in common_participants]
        )

        # Descriptive statistics (participant-level)
        real_stats = {
            "mean": np.mean(real_values),
            "median": np.median(real_values),
            "std": np.std(real_values, ddof=1),
            "n": len(real_values),
            "n_original_obs": sum(len(v) for v in participant_real_icds.values()),
        }
        pseudo_stats = {
            "mean": np.mean(pseudo_values),
            "median": np.median(pseudo_values),
            "std": np.std(pseudo_values, ddof=1),
            "n": len(pseudo_values),
            "n_original_obs": sum(len(v) for v in participant_pseudo_icds.values()),
        }

        # Paired tests (since each participant contributes to both)
        # Wilcoxon signed-rank (non-parametric paired)
        try:
            w_stat, p_value_wilcoxon = stats.wilcoxon(real_values, pseudo_values)
        except ValueError:
            w_stat, p_value_wilcoxon = np.nan, 1.0

        # Paired t-test
        t_stat, p_value_paired_t = stats.ttest_rel(real_values, pseudo_values)

        # Effect size (Cohen's d for paired samples)
        diff = real_values - pseudo_values
        cohens_d_paired = (
            np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        )

        # Also compute confidence interval for the difference
        diff_mean = np.mean(diff)
        diff_sem = stats.sem(diff)
        ci_95 = stats.t.interval(0.95, len(diff) - 1, loc=diff_mean, scale=diff_sem)

        return {
            "method": "participant_aggregation",
            "real": real_stats,
            "pseudo": pseudo_stats,
            "n_participants": len(common_participants),
            "wilcoxon": {"W": w_stat, "p": p_value_wilcoxon},
            "paired_t_test": {"t": t_stat, "p": p_value_paired_t},
            "cohens_d_paired": cohens_d_paired,
            "difference": {
                "mean": diff_mean,
                "ci_95_lower": ci_95[0],
                "ci_95_upper": ci_95[1],
            },
            "participant_real_means": participant_real_means,
            "participant_pseudo_means": participant_pseudo_means,
        }

    def compute_mixed_model_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute statistics using a linear mixed model with participant as random effect.

        Model: ICD ~ dyad_type + (1 | participant)

        This properly accounts for the non-independence by modeling the correlation
        structure induced by participants appearing in multiple dyads.

        Requires statsmodels package.

        Args:
            df: Long-format DataFrame with 'dyad_type', 'icd_value', 'dyad_pair' columns

        Returns:
            Dictionary with mixed model results
        """
        if not HAS_STATSMODELS:
            logger.warning("statsmodels not installed. Cannot compute mixed model.")
            return {
                "method": "mixed_model",
                "error": "statsmodels not installed. Install with: pip install statsmodels",
            }

        logger.info("Computing mixed model statistics (participant as random effect)")

        # Create expanded dataframe with participant column
        # Each observation gets attributed to both participants in the dyad
        records = []
        for _, row in df.iterrows():
            parts = row["dyad_pair"].split("_vs_")
            if len(parts) != 2:
                continue

            # Create one record per participant-observation
            for participant in parts:
                records.append(
                    {
                        "icd_value": row["icd_value"],
                        "dyad_type": row["dyad_type"],
                        "participant": participant,
                        "epoch_id": row["epoch_id"],
                        "dyad_pair": row["dyad_pair"],
                    }
                )

        model_df = pd.DataFrame(records)

        # Convert dyad_type to numeric for easier interpretation
        model_df["is_real"] = (model_df["dyad_type"] == "real").astype(int)

        try:
            # Fit mixed model: ICD ~ is_real + (1 | participant)
            model = mixedlm(
                "icd_value ~ is_real", model_df, groups=model_df["participant"]
            )
            result = model.fit(method="powell")  # Powell is more robust

            # Extract results
            coef_real = result.params["is_real"]
            se_real = result.bse["is_real"]
            z_real = result.tvalues["is_real"]
            p_real = result.pvalues["is_real"]

            # Confidence interval
            ci_lower = coef_real - 1.96 * se_real
            ci_upper = coef_real + 1.96 * se_real

            # Random effect variance
            re_variance = (
                result.cov_re.iloc[0, 0] if hasattr(result, "cov_re") else np.nan
            )

            # Compute ICC (intraclass correlation)
            residual_var = result.scale
            icc = (
                re_variance / (re_variance + residual_var)
                if (re_variance + residual_var) > 0
                else 0
            )

            return {
                "method": "mixed_model",
                "model_formula": "icd_value ~ is_real + (1 | participant)",
                "n_observations": len(model_df),
                "n_participants": model_df["participant"].nunique(),
                "effect_of_real_dyad": {
                    "coefficient": coef_real,
                    "se": se_real,
                    "z": z_real,
                    "p": p_real,
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                },
                "random_effect_variance": re_variance,
                "residual_variance": residual_var,
                "icc": icc,
                "interpretation": (
                    f"Real dyads have {'higher' if coef_real > 0 else 'lower'} ICD "
                    f"by {abs(coef_real):.1f} ms (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]), "
                    f"p = {p_real:.2e}. "
                    f"ICC = {icc:.2f} indicates {icc * 100:.0f}% of variance is between participants."
                ),
                "model_summary": str(result.summary()),
            }

        except Exception as e:
            logger.error(f"Mixed model failed: {e}")
            return {
                "method": "mixed_model",
                "error": str(e),
            }

    def generate_corrected_statistics_report(
        self, df: pd.DataFrame, metadata: Dict, output_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive statistics report comparing naive vs corrected analyses.

        Compares:
        1. Naive analysis (treating dyads as independent)
        2. Participant-aggregated analysis (paired test)
        3. Mixed model analysis (participant as random effect)

        Args:
            df: Long-format ICD DataFrame
            metadata: Data metadata dict
            output_path: Optional path to save text report

        Returns:
            Dictionary with all statistical analyses
        """
        results = {
            "metadata": metadata,
            "naive": self.compute_statistics(df),
            "participant_level": self.compute_participant_level_statistics(df),
            "mixed_model": self.compute_mixed_model_statistics(df),
        }

        # Generate text report
        report_lines = [
            "=" * 70,
            "ICD STATISTICAL ANALYSIS: Naive vs Corrected Approaches",
            "=" * 70,
            "",
            f"Task: {metadata['task']}",
            f"N dyads (real): {metadata['n_real_dyads']}",
            f"N dyads (pseudo): {metadata['n_pseudo_dyads']}",
            f"N epochs: {metadata['n_epochs']}",
            "",
            "-" * 70,
            "1. NAIVE ANALYSIS (treating dyads as independent)",
            "-" * 70,
            "   ⚠️  WARNING: This analysis ignores non-independence!",
            f"   Real mean:   {results['naive']['real']['mean']:.1f} ± {results['naive']['real']['std']:.1f} ms (n={results['naive']['real']['n']:,})",
            f"   Pseudo mean: {results['naive']['pseudo']['mean']:.1f} ± {results['naive']['pseudo']['std']:.1f} ms (n={results['naive']['pseudo']['n']:,})",
            f"   Difference:  {results['naive']['difference']:.1f} ms",
            f"   Mann-Whitney p = {results['naive']['mann_whitney']['p']:.2e}",
            f"   Cohen's d = {results['naive']['cohens_d']:.3f}",
            "",
            "-" * 70,
            "2. PARTICIPANT-LEVEL ANALYSIS (corrected for non-independence)",
            "-" * 70,
            "   ✓ Each participant's mean ICD computed, then compared",
            f"   N participants: {results['participant_level']['n_participants']}",
            f"   Real mean:   {results['participant_level']['real']['mean']:.1f} ± {results['participant_level']['real']['std']:.1f} ms",
            f"   Pseudo mean: {results['participant_level']['pseudo']['mean']:.1f} ± {results['participant_level']['pseudo']['std']:.1f} ms",
            f"   Difference:  {results['participant_level']['difference']['mean']:.1f} ms",
            f"   95% CI: [{results['participant_level']['difference']['ci_95_lower']:.1f}, {results['participant_level']['difference']['ci_95_upper']:.1f}]",
            f"   Wilcoxon signed-rank p = {results['participant_level']['wilcoxon']['p']:.4f}",
            f"   Paired t-test p = {results['participant_level']['paired_t_test']['p']:.4f}",
            f"   Cohen's d (paired) = {results['participant_level']['cohens_d_paired']:.3f}",
            "",
        ]

        # Mixed model section
        if "error" not in results["mixed_model"]:
            mm = results["mixed_model"]
            report_lines.extend(
                [
                    "-" * 70,
                    "3. MIXED MODEL ANALYSIS (participant as random effect)",
                    "-" * 70,
                    f"   ✓ Model: {mm['model_formula']}",
                    f"   N observations: {mm['n_observations']:,}",
                    f"   N participants: {mm['n_participants']}",
                    f"   Effect of real dyad: {mm['effect_of_real_dyad']['coefficient']:.1f} ms",
                    f"   95% CI: [{mm['effect_of_real_dyad']['ci_95_lower']:.1f}, {mm['effect_of_real_dyad']['ci_95_upper']:.1f}]",
                    f"   p = {mm['effect_of_real_dyad']['p']:.4f}",
                    f"   ICC = {mm['icc']:.2f} ({mm['icc'] * 100:.0f}% variance between participants)",
                    "",
                    f"   Interpretation: {mm['interpretation']}",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    "-" * 70,
                    "3. MIXED MODEL ANALYSIS",
                    "-" * 70,
                    f"   ⚠️  Error: {results['mixed_model']['error']}",
                    "",
                ]
            )

        # Summary section
        naive_sig = results["naive"]["mann_whitney"]["p"] < 0.05
        corrected_sig = results["participant_level"]["wilcoxon"]["p"] < 0.05

        report_lines.extend(
            [
                "=" * 70,
                "SUMMARY",
                "=" * 70,
                f"   Naive p-value:     {results['naive']['mann_whitney']['p']:.2e} {'(significant)' if naive_sig else '(not significant)'}",
                f"   Corrected p-value: {results['participant_level']['wilcoxon']['p']:.4f} {'(significant)' if corrected_sig else '(not significant)'}",
                "",
            ]
        )

        if naive_sig and not corrected_sig:
            report_lines.append(
                "   ⚠️  CAUTION: Significance DISAPPEARS after correction!"
            )
            report_lines.append(
                "      The naive analysis may have inflated significance due to pseudo-replication."
            )
        elif naive_sig and corrected_sig:
            report_lines.append("   ✓ Effect remains significant after correction.")
            report_lines.append(
                "      However, note the difference in effect size estimates."
            )

        report_lines.append("")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)
        results["report_text"] = report_text

        # Print report
        print(report_text)

        # Save if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Saved statistics report to {output_path}")

        return results

    def plot_real_vs_pseudo_distribution(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        statistics: Dict,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Create violin + boxplot comparing real vs pseudo ICD distributions.

        Layout:
        ┌─────────────────────────────────────────┐
        │   Violin + Box + Points Distribution    │
        │   Real (blue) vs Pseudo (pink)          │
        ├─────────────────────────────────────────┤
        │   Statistics Panel (text)               │
        └─────────────────────────────────────────┘

        Args:
            df: Long-format ICD DataFrame
            metadata: Data metadata dict
            statistics: Statistical test results
            output_path: Where to save figure
            show: Whether to display

        Returns:
            Figure object
        """
        apply_plot_style()

        fig, (ax_main, ax_stats) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Prepare data for plotting
        real_values = df[df["dyad_type"] == "real"]["icd_value"].values
        pseudo_values = df[df["dyad_type"] == "pseudo"]["icd_value"].values

        positions = [0, 1]
        data = [real_values, pseudo_values]
        colors = [COLORS["real_dyad"], COLORS["pseudo_dyad"]]
        labels = ["Real Dyads\n(same family)", "Pseudo Dyads\n(cross-family)"]

        # ========== Main Plot: Violin + Box ==========

        # Violin plot
        vp = ax_main.violinplot(
            data, positions=positions, showextrema=False, widths=0.8
        )
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_edgecolor("white")
            body.set_alpha(ALPHA["medium"])

        # Box plot overlay
        bp = ax_main.boxplot(
            data,
            positions=positions,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 2},
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )
        for i, (box, color) in enumerate(zip(bp["boxes"], colors)):
            box.set_facecolor(color)
            box.set_alpha(ALPHA["high"])
            box.set_edgecolor("white")

        # Scatter points (jittered)
        for i, (vals, color) in enumerate(zip(data, colors)):
            # Subsample if too many points
            if len(vals) > 500:
                indices = np.random.choice(len(vals), 500, replace=False)
                vals_sample = vals[indices]
            else:
                vals_sample = vals

            jitter = np.random.normal(0, 0.05, len(vals_sample))
            ax_main.scatter(
                positions[i] + jitter,
                vals_sample,
                c=color,
                alpha=0.3,
                s=MARKERSIZE["small"],
                edgecolors="none",
                zorder=1,
            )

        # Mean markers
        for i, vals in enumerate(data):
            ax_main.scatter(
                [positions[i]],
                [np.mean(vals)],
                marker="D",
                s=100,
                c="white",
                edgecolors=colors[i],
                linewidths=2,
                zorder=10,
            )

        # Significance annotation - use corrected p-value if available
        if "corrected" in statistics and statistics["corrected"] is not None:
            p_value = statistics["corrected"]["wilcoxon"]["p"]
            pass  # corrected
        else:
            p_value = statistics["mann_whitney"]["p"]
            pass  # uncorrected

        sig_symbol = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else "ns"
        )

        # Draw significance bracket
        y_max = max(real_values.max(), pseudo_values.max())
        bracket_y = y_max * 1.05
        ax_main.plot(
            [0, 0, 1, 1],
            [bracket_y, bracket_y * 1.02, bracket_y * 1.02, bracket_y],
            color=COLORS["dark_gray"],
            linewidth=LINEWIDTH["medium"],
        )
        ax_main.text(
            0.5,
            bracket_y * 1.04,
            sig_symbol,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE["subtitle"],
            fontweight="bold",
            color=COLORS["good"] if p_value < 0.05 else COLORS["poor"],
        )

        # Formatting
        ax_main.set_xticks(positions)
        ax_main.set_xticklabels(labels, fontsize=FONTSIZE["label"], fontweight="bold")
        ax_main.set_ylabel(
            "Inter-Centroid Distance (ms)",
            fontsize=FONTSIZE["label"],
            fontweight="bold",
        )
        ax_main.set_title(
            f"ICD Distribution: Real vs Pseudo Dyads\n{metadata['task'].title()} Task",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
            pad=20,
        )
        ax_main.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax_main.set_ylim(0, y_max * 1.15)

        # Legend
        legend_elements = [
            mpatches.Patch(
                facecolor=COLORS["real_dyad"],
                alpha=ALPHA["high"],
                label=f"Real (n={metadata['n_real_dyads']} dyads)",
            ),
            mpatches.Patch(
                facecolor=COLORS["pseudo_dyad"],
                alpha=ALPHA["high"],
                label=f"Pseudo (n={metadata['n_pseudo_dyads']} dyads)",
            ),
        ]
        ax_main.legend(
            handles=legend_elements, loc="upper right", fontsize=FONTSIZE["legend"]
        )

        # ========== Statistics Panel ==========
        ax_stats.axis("off")

        # Check if we have corrected statistics
        if "corrected" in statistics and statistics["corrected"] is not None:
            corrected = statistics["corrected"]
            naive_p = statistics["mann_whitney"]["p"]
            corrected_p = corrected["wilcoxon"]["p"]

            stats_text = (
                f"Statistical Comparison (CORRECTED for non-independence)\n"
                f"{'─' * 60}\n\n"
                f"⚠️  Naive analysis (treating dyads as independent):\n"
                f"    Real: {statistics['real']['mean']:.1f} ms, Pseudo: {statistics['pseudo']['mean']:.1f} ms\n"
                f"    Difference: {statistics['difference']:.1f} ms, p = {naive_p:.2e}\n\n"
                f"✓ CORRECTED analysis (participant-level, n={corrected['n_participants']}):\n"
                f"    Real: {corrected['real']['mean']:.1f} ± {corrected['real']['std']:.1f} ms\n"
                f"    Pseudo: {corrected['pseudo']['mean']:.1f} ± {corrected['pseudo']['std']:.1f} ms\n"
                f"    Difference: {corrected['difference']['mean']:.1f} ms "
                f"(95% CI: [{corrected['difference']['ci_95_lower']:.1f}, {corrected['difference']['ci_95_upper']:.1f}])\n"
                f"    Wilcoxon p = {corrected_p:.4f}, Cohen's d = {corrected['cohens_d_paired']:.3f}\n\n"
            )

            if naive_p < 0.05 and corrected_p >= 0.05:
                stats_text += "⚠️  Significance DISAPPEARS after correction for pseudo-replication!"
            elif corrected_p < 0.05:
                stats_text += "✓ Effect remains significant after correction."
            else:
                stats_text += "No significant difference between real and pseudo dyads."
        else:
            stats_text = (
                f"Statistical Comparison\n"
                f"{'─' * 50}\n\n"
                f"Real Dyads:   Mean = {statistics['real']['mean']:.1f} ms  "
                f"(SD = {statistics['real']['std']:.1f}, n = {statistics['real']['n']:,})\n"
                f"Pseudo Dyads: Mean = {statistics['pseudo']['mean']:.1f} ms  "
                f"(SD = {statistics['pseudo']['std']:.1f}, n = {statistics['pseudo']['n']:,})\n\n"
                f"Difference: {statistics['difference']:.1f} ms "
                f"({'lower' if statistics['difference'] < 0 else 'higher'} for real dyads)\n\n"
                f"Mann-Whitney U = {statistics['mann_whitney']['U']:.0f}, "
                f"p = {statistics['mann_whitney']['p']:.2e}\n"
                f"Cohen's d = {statistics['cohens_d']:.3f} "
                f"({'small' if abs(statistics['cohens_d']) < 0.5 else 'medium' if abs(statistics['cohens_d']) < 0.8 else 'large'} effect)"
            )

        ax_stats.text(
            0.5,
            0.5,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=FONTSIZE["annotation"] + 1,
            fontfamily="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["background"],
                edgecolor=COLORS["grid"],
                linewidth=1,
            ),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved distribution plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_epoch_evolution(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot ICD evolution across epochs for real vs pseudo dyads.

        Shows mean ± SEM for each dyad type across epochs.

        Args:
            df: Long-format ICD DataFrame
            metadata: Data metadata dict
            output_path: Where to save figure
            show: Whether to display

        Returns:
            Figure object
        """
        apply_plot_style()

        fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

        # Aggregate by epoch and dyad type
        agg_df = (
            df.groupby(["epoch_id", "dyad_type"])["icd_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg_df["sem"] = agg_df["std"] / np.sqrt(agg_df["count"])

        # Plot each dyad type
        for dyad_type, color, label in [
            ("real", COLORS["real_dyad"], "Real Dyads"),
            ("pseudo", COLORS["pseudo_dyad"], "Pseudo Dyads"),
        ]:
            type_df = agg_df[agg_df["dyad_type"] == dyad_type].sort_values("epoch_id")

            epochs = type_df["epoch_id"].values
            means = type_df["mean"].values
            sems = type_df["sem"].values

            # Line plot with confidence band
            ax.plot(
                epochs,
                means,
                color=color,
                linewidth=LINEWIDTH["medium"],
                label=label,
                alpha=ALPHA["line"],
            )
            ax.fill_between(
                epochs, means - sems, means + sems, color=color, alpha=ALPHA["fill"]
            )

        # Formatting
        ax.set_xlabel("Epoch", fontsize=FONTSIZE["label"], fontweight="bold")
        ax.set_ylabel("Mean ICD (ms)", fontsize=FONTSIZE["label"], fontweight="bold")
        ax.set_title(
            f"ICD Evolution Across Epochs: Real vs Pseudo Dyads\n{metadata['task'].title()} Task",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=FONTSIZE["legend"])
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, metadata["n_epochs"] - 1)

        # Add interpretation note
        ax.text(
            0.02,
            0.98,
            "Shaded area: ± SEM",
            transform=ax.transAxes,
            fontsize=FONTSIZE["small"],
            verticalalignment="top",
            color=COLORS["dark_gray"],
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved epoch evolution plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_dyad_heatmap(
        self, icd_file: Path, output_path: Optional[Path] = None, show: bool = False
    ) -> plt.Figure:
        """
        Create heatmap of mean ICD between all dyad pairs.

        Args:
            icd_file: Path to ICD CSV file
            output_path: Where to save figure
            show: Whether to display

        Returns:
            Figure object
        """
        apply_plot_style()

        # Read wide format data
        df = pd.read_csv(icd_file)

        # Compute mean ICD per dyad (across epochs)
        dyad_cols = [c for c in df.columns if c != "epoch_id"]
        mean_icds = df[dyad_cols].mean()

        # Parse subject pairs and build matrix
        subjects = set()
        for col in dyad_cols:
            parts = col.split("_vs_")
            if len(parts) == 2:
                subjects.add(parts[0])
                subjects.add(parts[1])

        subjects = sorted(list(subjects))
        n_subjects = len(subjects)

        # Create matrix
        matrix = np.full((n_subjects, n_subjects), np.nan)
        subject_idx = {s: i for i, s in enumerate(subjects)}

        for col, icd in mean_icds.items():
            parts = col.split("_vs_")
            if len(parts) == 2:
                i = subject_idx.get(parts[0])
                j = subject_idx.get(parts[1])
                if i is not None and j is not None:
                    matrix[i, j] = icd
                    matrix[j, i] = icd  # Symmetric

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))

        # Mask diagonal
        mask = np.eye(n_subjects, dtype=bool)
        matrix_masked = np.ma.masked_where(mask, matrix)

        im = ax.imshow(matrix_masked, cmap="RdYlBu_r", aspect="auto")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean ICD (ms)", fontsize=FONTSIZE["label"])

        # Labels
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels(
            subjects, rotation=45, ha="right", fontsize=FONTSIZE["small"]
        )
        ax.set_yticklabels(subjects, fontsize=FONTSIZE["small"])

        ax.set_title(
            "Inter-Centroid Distance Matrix\n(Mean across epochs)",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved heatmap to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_intra_family_evolution(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot ICD evolution for intra-family analysis (no pseudo comparison).

        Shows mean ± SEM across epochs for real dyads only.

        Args:
            df: Long-format ICD DataFrame (real dyads only)
            metadata: Data metadata dict
            output_path: Where to save figure
            show: Whether to display

        Returns:
            Figure object
        """
        apply_plot_style()

        fig, ax = plt.subplots(figsize=FIGSIZE["wide"])

        # Aggregate by epoch
        agg_df = (
            df.groupby("epoch_id")["icd_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg_df["sem"] = agg_df["std"] / np.sqrt(agg_df["count"])

        epochs = agg_df["epoch_id"].values
        means = agg_df["mean"].values
        sems = agg_df["sem"].values

        # Line plot with confidence band
        ax.plot(
            epochs,
            means,
            color=COLORS["real_dyad"],
            linewidth=LINEWIDTH["medium"],
            label="Real Dyads",
            alpha=ALPHA["line"],
            marker="o",
            markersize=MARKERSIZE["small"],
        )
        ax.fill_between(
            epochs,
            means - sems,
            means + sems,
            color=COLORS["real_dyad"],
            alpha=ALPHA["fill"],
        )

        # Add trend line
        z = np.polyfit(epochs, means, 1)
        p = np.poly1d(z)
        ax.plot(
            epochs,
            p(epochs),
            "--",
            color=COLORS["dark_gray"],
            alpha=ALPHA["medium"],
            linewidth=LINEWIDTH["thin"],
            label=f"Trend (slope: {z[0]:.2f} ms/epoch)",
        )

        # Formatting
        ax.set_xlabel("Epoch", fontsize=FONTSIZE["label"], fontweight="bold")
        ax.set_ylabel("Mean ICD (ms)", fontsize=FONTSIZE["label"], fontweight="bold")
        ax.set_title(
            f"Intra-Family ICD Evolution\n{metadata['task'].title()} Task "
            f"({metadata['n_real_dyads']} dyads)",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=FONTSIZE["legend"])
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add stats annotation
        overall_mean = df["icd_value"].mean()
        overall_std = df["icd_value"].std()
        ax.text(
            0.02,
            0.98,
            f"Overall: {overall_mean:.1f} ± {overall_std:.1f} ms\nShaded area: ± SEM",
            transform=ax.transAxes,
            fontsize=FONTSIZE["small"],
            verticalalignment="top",
            color=COLORS["dark_gray"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved intra-family evolution plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_intra_family_distribution(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot ICD distribution for intra-family analysis (real dyads only).

        Args:
            df: Long-format ICD DataFrame
            metadata: Data metadata dict
            output_path: Where to save figure
            show: Whether to display

        Returns:
            Figure object
        """
        apply_plot_style()

        fig, ax = plt.subplots(figsize=FIGSIZE["medium"])

        icd_values = df["icd_value"].values

        # Histogram
        ax.hist(
            icd_values,
            bins=50,
            color=COLORS["real_dyad"],
            alpha=ALPHA["high"],
            edgecolor="white",
            linewidth=0.5,
        )

        # Add mean and median lines
        mean_val = np.mean(icd_values)
        median_val = np.median(icd_values)
        ax.axvline(
            mean_val,
            color=COLORS["dark_gray"],
            linewidth=LINEWIDTH["thick"],
            linestyle="--",
        )
        ax.axvline(
            median_val,
            color=COLORS["dark_gray"],
            linewidth=LINEWIDTH["medium"],
            linestyle=":",
        )

        # Formatting
        ax.set_xlabel("ICD (ms)", fontsize=FONTSIZE["label"], fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=FONTSIZE["label"], fontweight="bold")
        ax.set_title(
            f"Intra-Family ICD Distribution\n{metadata['task'].title()} Task",
            fontsize=FONTSIZE["title"],
            fontweight="bold",
        )

        # Add stats box in upper left (away from the tail of distribution)
        stats_text = (
            f"N dyads: {metadata['n_real_dyads']}\n"
            f"N epochs: {metadata['n_epochs']}\n"
            f"Mean: {mean_val:.1f} ms (--)\n"
            f"Median: {median_val:.1f} ms (..)\n"
            f"Std: {np.std(icd_values):.1f} ms"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=FONTSIZE["annotation"],
            verticalalignment="top",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=COLORS["grid"],
                alpha=0.95,
            ),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved intra-family distribution to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def generate_full_report(self, icd_file: Path, task: str, output_dir: Path) -> Dict:
        """
        Generate complete ICD statistics report with all visualizations.

        Handles both inter-session (real vs pseudo) and intra-family (real only) modes.

        Args:
            icd_file: Path to ICD CSV file
            task: Task name
            output_dir: Output directory for figures

        Returns:
            Dictionary with statistics and file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df, metadata = self.load_icd_data(icd_file, task)

        files = {}
        statistics = None

        # Check if we have pseudo dyads (inter-session mode)
        has_pseudo = metadata["n_pseudo_dyads"] > 0

        if has_pseudo:
            # Inter-session mode: real vs pseudo comparison
            # Compute both naive and corrected statistics
            naive_statistics = self.compute_statistics(df)
            corrected_statistics = self.compute_participant_level_statistics(df)

            # Combine for visualization (include corrected for proper display)
            statistics = naive_statistics.copy()
            statistics["corrected"] = corrected_statistics

            # 1. Distribution plot (with corrected stats)
            dist_path = output_dir / f"icd_distribution_{task}.png"
            self.plot_real_vs_pseudo_distribution(df, metadata, statistics, dist_path)
            files["distribution"] = dist_path

            # 2. Epoch evolution (only if multiple epochs)
            if metadata["n_epochs"] > 1:
                epoch_path = output_dir / f"icd_epoch_evolution_{task}.png"
                self.plot_epoch_evolution(df, metadata, epoch_path)
                files["epoch_evolution"] = epoch_path

            # 3. Heatmap
            heatmap_path = output_dir / f"icd_heatmap_{task}.png"
            self.plot_dyad_heatmap(icd_file, heatmap_path)
            files["heatmap"] = heatmap_path

            # 4. Save corrected statistics report
            stats_path = output_dir / f"icd_corrected_stats_{task}.txt"
            self.generate_corrected_statistics_report(df, metadata, stats_path)

        else:
            # Intra-family mode: real dyads only (no pseudo comparison)
            logger.info("Intra-family mode: generating real-dyads-only visualizations")

            # 1. Distribution histogram
            dist_path = output_dir / f"icd_distribution_{task}.png"
            self.plot_intra_family_distribution(df, metadata, dist_path)
            files["distribution"] = dist_path

            # 2. Evolution over epochs (if multiple epochs)
            if metadata["n_epochs"] > 1:
                evol_path = output_dir / f"icd_evolution_{task}.png"
                self.plot_intra_family_evolution(df, metadata, evol_path)
                files["evolution"] = evol_path

            # 3. Heatmap (still useful to see dyad relationships)
            heatmap_path = output_dir / f"icd_heatmap_{task}.png"
            self.plot_dyad_heatmap(icd_file, heatmap_path)
            files["heatmap"] = heatmap_path

            # Basic statistics for real dyads
            real_values = df[df["dyad_type"] == "real"]["icd_value"].values
            statistics = {
                "real": {
                    "mean": np.mean(real_values),
                    "median": np.median(real_values),
                    "std": np.std(real_values),
                    "n": len(real_values),
                },
                "pseudo": None,
                "mode": "intra_family",
            }

        logger.info(f"Generated {len(files)} visualizations in {output_dir}")

        return {
            "metadata": metadata,
            "statistics": statistics,
            "files": files,
        }
