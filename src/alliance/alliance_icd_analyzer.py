"""
Alliance-ICD Statistical Analyzer.

Computes statistics to analyze correlation between alliance states
and physiological synchrony (ICD).

Authors: Remy Ramadour
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from scipy import stats

from src.core.config_loader import ConfigLoader
from src.alliance.alliance_icd_loader import AllianceICDLoader

logger = logging.getLogger(__name__)


class AllianceICDAnalyzer:
    """Analyzes correlation between alliance states and ICD."""

    # Alliance labels for display
    ALLIANCE_LABELS = {0: "Neutral", 1: "Positive", -1: "Negative", 2: "Split"}

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.loader = AllianceICDLoader(config_path)
        self._data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load all merged alliance-ICD data.

        Returns:
            DataFrame with merged data
        """
        if self._data is None:
            self._data = self.loader.load_all_merged_data()
        return self._data

    def compute_descriptive_stats(
        self, data: Optional[pd.DataFrame] = None, group_by: str = "alliance_label"
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics for ICD by alliance state.

        Args:
            data: DataFrame with merged data (if None, loads from cache)
            group_by: Column to group by

        Returns:
            DataFrame with stats (mean, std, median, q25, q75, n)
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return pd.DataFrame()

        stats_df = (
            data.groupby(group_by)["icd"]
            .agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("median", "median"),
                    ("q25", lambda x: x.quantile(0.25)),
                    ("q75", lambda x: x.quantile(0.75)),
                    ("min", "min"),
                    ("max", "max"),
                    ("n", "count"),
                ]
            )
            .round(4)
        )

        return stats_df

    def compute_stats_by_dyad_type(
        self, data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute statistics separately for real and pseudo dyads.

        Args:
            data: DataFrame with merged data

        Returns:
            Dict with 'real' and 'pseudo' stats DataFrames
        """
        if data is None:
            data = self.load_data()

        results = {}

        for dyad_type in ["real", "pseudo"]:
            subset = data[data["dyad_type"] == dyad_type]
            if not subset.empty:
                results[dyad_type] = self.compute_descriptive_stats(subset)

        return results

    def test_alliance_effect(
        self, data: Optional[pd.DataFrame] = None, test_type: str = "kruskal"
    ) -> Dict:
        """
        Test if alliance state has significant effect on ICD.

        Args:
            data: DataFrame with merged data
            test_type: 'kruskal' (Kruskal-Wallis) or 'anova'

        Returns:
            Dict with test results
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        # Group ICD by alliance state
        groups = []
        group_labels = []

        for state in [0, 1, -1, 2]:
            subset = data[data["alliance_state"] == state]["icd"]
            if len(subset) > 0:
                groups.append(subset.values)
                group_labels.append(self.ALLIANCE_LABELS[state])

        if len(groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}

        if test_type == "kruskal":
            statistic, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis H-test"
        else:
            statistic, p_value = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"

        return {
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "groups": group_labels,
            "n_per_group": [len(g) for g in groups],
            "significant": p_value < 0.05,
        }

    def pairwise_comparisons(
        self, data: Optional[pd.DataFrame] = None, correction: str = "bonferroni"
    ) -> pd.DataFrame:
        """
        Perform pairwise Mann-Whitney U tests between alliance states.

        Args:
            data: DataFrame with merged data
            correction: Multiple comparison correction method

        Returns:
            DataFrame with pairwise test results
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return pd.DataFrame()

        # Get groups with data
        states_with_data = []
        for state in [0, 1, -1, 2]:
            if len(data[data["alliance_state"] == state]) > 0:
                states_with_data.append(state)

        results = []
        comparisons = []

        for i, state1 in enumerate(states_with_data):
            for state2 in states_with_data[i + 1 :]:
                g1 = data[data["alliance_state"] == state1]["icd"]
                g2 = data[data["alliance_state"] == state2]["icd"]

                statistic, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")

                # Effect size (rank-biserial correlation)
                n1, n2 = len(g1), len(g2)
                effect_size = 1 - (2 * statistic) / (n1 * n2)

                results.append(
                    {
                        "group1": self.ALLIANCE_LABELS[state1],
                        "group2": self.ALLIANCE_LABELS[state2],
                        "n1": n1,
                        "n2": n2,
                        "U_statistic": statistic,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "mean_diff": g1.mean() - g2.mean(),
                    }
                )
                comparisons.append((state1, state2))

        df = pd.DataFrame(results)

        if not df.empty and correction == "bonferroni":
            n_comparisons = len(df)
            df["p_adjusted"] = df["p_value"] * n_comparisons
            df["p_adjusted"] = df["p_adjusted"].clip(upper=1.0)
            df["significant"] = df["p_adjusted"] < 0.05

        return df

    def compare_real_vs_pseudo(
        self, data: Optional[pd.DataFrame] = None, by_alliance: bool = True
    ) -> Dict:
        """
        Compare ICD between real and pseudo dyads.

        Args:
            data: DataFrame with merged data
            by_alliance: If True, compare within each alliance state

        Returns:
            Dict with comparison results
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        results = {}

        if by_alliance:
            for state in [0, 1, -1, 2]:
                state_data = data[data["alliance_state"] == state]
                real = state_data[state_data["dyad_type"] == "real"]["icd"]
                pseudo = state_data[state_data["dyad_type"] == "pseudo"]["icd"]

                if len(real) > 0 and len(pseudo) > 0:
                    statistic, p_value = stats.mannwhitneyu(
                        real, pseudo, alternative="two-sided"
                    )

                    results[self.ALLIANCE_LABELS[state]] = {
                        "n_real": len(real),
                        "n_pseudo": len(pseudo),
                        "mean_real": float(real.mean()),
                        "mean_pseudo": float(pseudo.mean()),
                        "U_statistic": float(statistic),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    }
        else:
            real = data[data["dyad_type"] == "real"]["icd"]
            pseudo = data[data["dyad_type"] == "pseudo"]["icd"]

            if len(real) > 0 and len(pseudo) > 0:
                statistic, p_value = stats.mannwhitneyu(
                    real, pseudo, alternative="two-sided"
                )

                results["overall"] = {
                    "n_real": len(real),
                    "n_pseudo": len(pseudo),
                    "mean_real": float(real.mean()),
                    "mean_pseudo": float(pseudo.mean()),
                    "U_statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

        return results

    def compute_session_level_stats(
        self, data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute statistics at the session level (aggregated).

        This is Level B analysis: aggregate per dyad across epochs,
        then summarize across dyads.

        Args:
            data: DataFrame with merged data

        Returns:
            DataFrame with session-level stats
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return pd.DataFrame()

        # First, aggregate per dyad
        dyad_stats = (
            data.groupby(["dyad", "dyad_type", "family", "session", "alliance_label"])
            .agg({"icd": ["mean", "std", "count"]})
            .reset_index()
        )

        dyad_stats.columns = [
            "dyad",
            "dyad_type",
            "family",
            "session",
            "alliance_label",
            "mean_icd",
            "std_icd",
            "n_epochs",
        ]

        # Then aggregate across dyads
        session_stats = (
            dyad_stats.groupby(["alliance_label", "dyad_type"])
            .agg({"mean_icd": ["mean", "std"], "n_epochs": "sum", "dyad": "count"})
            .round(4)
        )

        session_stats.columns = [
            "grand_mean_icd",
            "between_dyad_std",
            "total_epochs",
            "n_dyads",
        ]

        return session_stats.reset_index()

    def generate_report(
        self, data: Optional[pd.DataFrame] = None, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a text report of all analyses.

        Args:
            data: DataFrame with merged data
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        if data is None:
            data = self.load_data()

        lines = []
        lines.append("=" * 70)
        lines.append("ALLIANCE-ICD STATISTICAL ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Data summary
        lines.append("DATA SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total observations: {len(data)}")
        lines.append(
            f"Real dyad observations: {len(data[data['dyad_type'] == 'real'])}"
        )
        lines.append(
            f"Pseudo-dyad observations: {len(data[data['dyad_type'] == 'pseudo'])}"
        )
        lines.append(f"Unique dyads: {data['dyad'].nunique()}")
        lines.append(f"Families: {sorted(data['family'].unique())}")
        lines.append("")

        # Descriptive stats
        lines.append("DESCRIPTIVE STATISTICS BY ALLIANCE STATE")
        lines.append("-" * 40)
        desc_stats = self.compute_descriptive_stats(data)
        lines.append(desc_stats.to_string())
        lines.append("")

        # Overall test
        lines.append("OVERALL ALLIANCE EFFECT TEST")
        lines.append("-" * 40)
        test_result = self.test_alliance_effect(data)
        lines.append(f"Test: {test_result.get('test', 'N/A')}")
        stat_val = test_result.get("statistic", "N/A")
        lines.append(
            f"Statistic: {stat_val:.4f}"
            if isinstance(stat_val, (int, float))
            else f"Statistic: {stat_val}"
        )
        p_val = test_result.get("p_value", "N/A")
        lines.append(
            f"p-value: {p_val:.6f}"
            if isinstance(p_val, (int, float))
            else f"p-value: {p_val}"
        )
        lines.append(f"Significant (α=0.05): {test_result.get('significant', 'N/A')}")
        lines.append("")

        # Pairwise comparisons
        lines.append("PAIRWISE COMPARISONS (Bonferroni-corrected)")
        lines.append("-" * 40)
        pairwise = self.pairwise_comparisons(data)
        if not pairwise.empty:
            lines.append(pairwise.to_string(index=False))
        lines.append("")

        # Real vs Pseudo
        lines.append("REAL vs PSEUDO-DYAD COMPARISON")
        lines.append("-" * 40)
        comparison = self.compare_real_vs_pseudo(data, by_alliance=False)
        if "overall" in comparison:
            c = comparison["overall"]
            lines.append(
                f"Real dyads - Mean ICD: {c['mean_real']:.4f} (n={c['n_real']})"
            )
            lines.append(
                f"Pseudo-dyads - Mean ICD: {c['mean_pseudo']:.4f} (n={c['n_pseudo']})"
            )
            lines.append(f"Mann-Whitney U: {c['U_statistic']:.2f}")
            lines.append(f"p-value: {c['p_value']:.6f}")
            lines.append(f"Significant: {c['significant']}")
        lines.append("")

        # By dyad type
        lines.append("STATISTICS BY DYAD TYPE")
        lines.append("-" * 40)
        by_type = self.compute_stats_by_dyad_type(data)
        for dtype, stats_df in by_type.items():
            lines.append(f"\n{dtype.upper()} DYADS:")
            lines.append(stats_df.to_string())
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")

        return report
