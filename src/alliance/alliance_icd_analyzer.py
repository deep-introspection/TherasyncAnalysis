"""
Alliance-ICD Statistical Analyzer.

Computes statistics to analyze correlation between alliance states
and physiological synchrony (ICD).

Authors: Remy Ramadour
Date: November 2025
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.core.config_loader import ConfigLoader
from src.alliance.alliance_icd_loader import AllianceICDLoader
from src.stats.corrections import compute_icc, correct_pvalues, epsilon_squared

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
        """Deprecated: use test_alliance_effect_mixed(). Naive epoch-level test."""
        warnings.warn(
            "test_alliance_effect treats epochs as independent. "
            "Use test_alliance_effect_mixed() for correct inference.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.test_alliance_effect_naive(data, test_type)

    def test_alliance_effect_naive(
        self, data: Optional[pd.DataFrame] = None, test_type: str = "kruskal"
    ) -> Dict:
        """
        Naive epoch-level test (ignores nesting structure).

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

    def test_alliance_effect_mixed(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test alliance effect using linear mixed-effects model.

        Model: icd ~ C(alliance_state) with random intercepts for family and dyad.
        Accounts for the nesting structure (epochs within dyads within families).

        Args:
            data: DataFrame with merged data

        Returns:
            Dict with coefficients, p-values, ICC, marginal R².
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        required_cols = {"icd", "alliance_state", "family", "dyad"}
        if not required_cols.issubset(data.columns):
            return {"error": f"Missing columns: {required_cols - set(data.columns)}"}

        df = data.dropna(subset=["icd", "alliance_state"]).copy()
        df["alliance_state"] = df["alliance_state"].astype(str)

        try:
            model = smf.mixedlm(
                "icd ~ C(alliance_state)",
                data=df,
                groups="family",
                re_formula="1",
                vc_formula={"dyad": "0 + C(dyad)"},
            )
            result = model.fit(reml=True)
        except Exception as e:
            logger.warning(f"Mixed model failed: {e}")
            return {"error": str(e)}

        # Extract fixed effects
        coefficients = result.fe_params.to_dict()
        p_values = result.pvalues.to_dict()

        # Marginal R²: variance explained by fixed effects / total variance
        var_fixed = np.var(result.predict(df))
        var_total = np.var(df["icd"])
        marginal_r2 = float(var_fixed / var_total) if var_total > 0 else np.nan

        # ICC at family and dyad levels
        icc_family = compute_icc(df, "family", "icd")
        icc_dyad = compute_icc(df, "dyad", "icd")

        return {
            "test": "Linear Mixed-Effects Model",
            "formula": "icd ~ C(alliance_state) | (1|family) + (1|dyad)",
            "coefficients": coefficients,
            "p_values": p_values,
            "marginal_r2": marginal_r2,
            "icc_family": icc_family,
            "icc_dyad": icc_dyad,
            "n_observations": len(df),
            "n_families": df["family"].nunique(),
            "n_dyads": df["dyad"].nunique(),
            "aic": float(result.aic),
            "bic": float(result.bic),
        }

    def compute_effect_sizes(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute effect sizes for alliance effect on ICD.

        Returns epsilon-squared from Kruskal-Wallis H statistic.

        Args:
            data: DataFrame with merged data

        Returns:
            Dict with effect size measures.
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        naive_result = self.test_alliance_effect_naive(data)
        if "error" in naive_result:
            return naive_result

        h_stat = naive_result["statistic"]
        n_total = sum(naive_result["n_per_group"])
        k_groups = len(naive_result["groups"])

        return {
            "epsilon_squared": epsilon_squared(h_stat, n_total, k_groups),
            "h_statistic": h_stat,
            "n_total": n_total,
            "k_groups": k_groups,
        }

    def compute_icc_structure(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute ICC at family and dyad levels.

        Args:
            data: DataFrame with merged data

        Returns:
            Dict with ICC values at each nesting level.
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        return {
            "icc_family": compute_icc(data, "family", "icd"),
            "icc_dyad": compute_icc(data, "dyad", "icd"),
            "n_families": data["family"].nunique(),
            "n_dyads": data["dyad"].nunique(),
            "n_observations": len(data),
        }

    def pairwise_comparisons(
        self, data: Optional[pd.DataFrame] = None, correction: str = "fdr_bh"
    ) -> pd.DataFrame:
        """
        Perform pairwise Mann-Whitney U tests between alliance states.

        Args:
            data: DataFrame with merged data
            correction: Multiple comparison correction method
                ('bonferroni', 'holm', 'fdr_bh')

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

        df = pd.DataFrame(results)

        if not df.empty:
            rejected, p_adjusted = correct_pvalues(
                df["p_value"].values, method=correction
            )
            df["p_adjusted"] = p_adjusted
            df["significant"] = rejected
            df["correction_method"] = correction

        return df

    @staticmethod
    def _extract_participants_from_dyad(dyad_name: str) -> tuple[str, str]:
        """
        Parse dyad column name into participant IDs.

        Args:
            dyad_name: e.g. "g01p02_ses-01_vs_g01p01_ses-01"

        Returns:
            Tuple of (participant1, participant2), e.g. ("g01p02", "g01p01")
        """
        parts = dyad_name.split("_vs_")
        p1 = parts[0].split("_ses-")[0]
        p2 = parts[1].split("_ses-")[0]
        return p1, p2

    def compare_real_vs_pseudo(
        self, data: Optional[pd.DataFrame] = None, by_alliance: bool = True
    ) -> Dict:
        """Deprecated: use compare_real_vs_pseudo_mixed(). Naive epoch-level test."""
        warnings.warn(
            "compare_real_vs_pseudo treats epochs as independent. "
            "Use compare_real_vs_pseudo_mixed() for correct inference.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compare_real_vs_pseudo_naive(data, by_alliance)

    def compare_real_vs_pseudo_naive(
        self, data: Optional[pd.DataFrame] = None, by_alliance: bool = True
    ) -> Dict:
        """
        Naive epoch-level comparison of real vs pseudo dyads (ignores nesting).

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

    def compare_real_vs_pseudo_mixed(
        self, data: Optional[pd.DataFrame] = None, by_alliance: bool = False
    ) -> Dict:
        """
        Compare real vs pseudo dyads using mixed-effects model.

        Model: icd ~ is_real + (1 | participant)
        Accounts for participants appearing in multiple dyads.

        Args:
            data: DataFrame with merged data
            by_alliance: If True, fit separate models per alliance state

        Returns:
            Dict with coefficient, p-value, CI, ICC, participant count.
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        if by_alliance:
            results = {}
            for state in [0, 1, -1, 2]:
                subset = data[data["alliance_state"] == state]
                if len(subset) > 0:
                    results[self.ALLIANCE_LABELS[state]] = (
                        self._fit_real_vs_pseudo_mixed(subset)
                    )
            return results

        return self._fit_real_vs_pseudo_mixed(data)

    def _fit_real_vs_pseudo_mixed(self, df: pd.DataFrame) -> Dict:
        """Fit a single mixed model for real vs pseudo comparison."""
        df = df.dropna(subset=["icd"]).copy()
        df["is_real"] = (df["dyad_type"] == "real").astype(int)

        # Extract participant IDs and explode to one row per participant
        records = []
        for _, row in df.iterrows():
            try:
                p1, p2 = self._extract_participants_from_dyad(row["dyad"])
            except (IndexError, KeyError):
                continue
            for pid in (p1, p2):
                records.append(
                    {"icd": row["icd"], "is_real": row["is_real"], "participant": pid}
                )

        if len(records) < 4:
            return {"error": "Too few observations for mixed model"}

        model_df = pd.DataFrame(records)
        n_participants = model_df["participant"].nunique()

        if n_participants < 2:
            return {"error": "Need at least 2 participants"}

        try:
            model = smf.mixedlm("icd ~ is_real", data=model_df, groups="participant")
            result = model.fit(reml=True)
        except Exception as e:
            logger.warning(f"Real-vs-pseudo mixed model failed: {e}")
            return {"error": str(e)}

        coef = float(result.params["is_real"])
        se = float(result.bse["is_real"])
        p_value = float(result.pvalues["is_real"])
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        re_var = (
            float(result.cov_re.iloc[0, 0]) if hasattr(result, "cov_re") else np.nan
        )
        resid_var = float(result.scale)
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0.0

        return {
            "test": "Linear Mixed-Effects Model",
            "formula": "icd ~ is_real + (1 | participant)",
            "coefficient": coef,
            "se": se,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "icc_participant": float(icc),
            "n_observations": len(model_df),
            "n_participants": n_participants,
            "significant": p_value < 0.05,
        }

    def compare_real_vs_pseudo_aggregated(
        self, data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Compare real vs pseudo dyads using participant-level aggregation.

        Aggregates mean ICD per participant for real and pseudo dyads,
        then runs Wilcoxon signed-rank on paired participant means.
        Fallback when mixed model can't converge.

        Args:
            data: DataFrame with merged data

        Returns:
            Dict with n_participants, Wilcoxon stat/p, paired Cohen's d.
        """
        if data is None:
            data = self.load_data()

        if data.empty:
            return {"error": "No data available"}

        # Collect ICD values per participant per dyad type
        participant_real: dict[str, list[float]] = {}
        participant_pseudo: dict[str, list[float]] = {}

        for _, row in data.iterrows():
            try:
                p1, p2 = self._extract_participants_from_dyad(row["dyad"])
            except (IndexError, KeyError):
                continue

            target = (
                participant_real if row["dyad_type"] == "real" else participant_pseudo
            )
            for pid in (p1, p2):
                target.setdefault(pid, []).append(row["icd"])

        real_means = {p: np.mean(v) for p, v in participant_real.items()}
        pseudo_means = {p: np.mean(v) for p, v in participant_pseudo.items()}

        common = sorted(set(real_means) & set(pseudo_means))
        if len(common) < 2:
            return {"error": f"Need >=2 paired participants, found {len(common)}"}

        real_arr = np.array([real_means[p] for p in common])
        pseudo_arr = np.array([pseudo_means[p] for p in common])

        try:
            w_stat, p_wilcoxon = stats.wilcoxon(real_arr, pseudo_arr)
        except ValueError:
            w_stat, p_wilcoxon = np.nan, 1.0

        diff = real_arr - pseudo_arr
        diff_std = np.std(diff, ddof=1)
        cohens_d = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0

        return {
            "test": "Wilcoxon signed-rank (participant-aggregated)",
            "n_participants": len(common),
            "mean_real": float(np.mean(real_arr)),
            "mean_pseudo": float(np.mean(pseudo_arr)),
            "W_statistic": float(w_stat),
            "p_value": float(p_wilcoxon),
            "cohens_d_paired": cohens_d,
            "significant": p_wilcoxon < 0.05,
        }

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

        session_stats = session_stats.reset_index()

        # Kruskal-Wallis on dyad-level means (correct degrees of freedom)
        groups = []
        for label in dyad_stats["alliance_label"].unique():
            group_means = dyad_stats.loc[
                dyad_stats["alliance_label"] == label, "mean_icd"
            ].values
            if len(group_means) > 0:
                groups.append(group_means)

        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            session_stats.attrs["kruskal_dyad_level"] = {
                "H": float(h_stat),
                "p_value": float(p_val),
                "n_dyads": n_total,
                "k_groups": len(groups),
                "epsilon_squared": epsilon_squared(h_stat, n_total, len(groups)),
            }

        return session_stats

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

        # ICC structure
        lines.append("INTRACLASS CORRELATION (ICC)")
        lines.append("-" * 40)
        icc = self.compute_icc_structure(data)
        if "error" not in icc:
            lines.append(f"ICC(family): {icc['icc_family']:.4f}")
            lines.append(f"ICC(dyad):   {icc['icc_dyad']:.4f}")
            lines.append(f"N families: {icc['n_families']}, N dyads: {icc['n_dyads']}")
        lines.append("")

        # Descriptive stats
        lines.append("DESCRIPTIVE STATISTICS BY ALLIANCE STATE")
        lines.append("-" * 40)
        desc_stats = self.compute_descriptive_stats(data)
        lines.append(desc_stats.to_string())
        lines.append("")

        # PRIMARY: Mixed-effects model
        lines.append("PRIMARY ANALYSIS: LINEAR MIXED-EFFECTS MODEL")
        lines.append("-" * 40)
        mixed_result = self.test_alliance_effect_mixed(data)
        if "error" not in mixed_result:
            lines.append(f"Formula: {mixed_result['formula']}")
            lines.append(
                f"N obs: {mixed_result['n_observations']}, "
                f"N families: {mixed_result['n_families']}, "
                f"N dyads: {mixed_result['n_dyads']}"
            )
            lines.append(
                f"AIC: {mixed_result['aic']:.1f}, BIC: {mixed_result['bic']:.1f}"
            )
            lines.append(f"Marginal R²: {mixed_result['marginal_r2']:.4f}")
            lines.append("Fixed effects:")
            for coef, val in mixed_result["coefficients"].items():
                p = mixed_result["p_values"].get(coef, np.nan)
                lines.append(f"  {coef}: β={val:.4f}, p={p:.6f}")
        else:
            lines.append(f"Mixed model failed: {mixed_result['error']}")
        lines.append("")

        # Effect sizes
        lines.append("EFFECT SIZES")
        lines.append("-" * 40)
        effects = self.compute_effect_sizes(data)
        if "error" not in effects:
            lines.append(f"Epsilon²: {effects['epsilon_squared']:.4f}")
            lines.append(
                f"(H={effects['h_statistic']:.2f}, "
                f"n={effects['n_total']}, k={effects['k_groups']})"
            )
        lines.append("")

        # SECONDARY: Naive Kruskal-Wallis
        lines.append("SECONDARY ANALYSIS: NAIVE KRUSKAL-WALLIS (epoch-level)")
        lines.append("-" * 40)
        lines.append(
            "NOTE: Treats epochs as independent; p-values are anti-conservative"
        )
        test_result = self.test_alliance_effect_naive(data)
        stat_val = test_result.get("statistic", "N/A")
        lines.append(
            f"H-statistic: {stat_val:.4f}"
            if isinstance(stat_val, (int, float))
            else f"Statistic: {stat_val}"
        )
        p_val = test_result.get("p_value", "N/A")
        lines.append(
            f"p-value: {p_val:.6f}"
            if isinstance(p_val, (int, float))
            else f"p-value: {p_val}"
        )
        lines.append("")

        # Pairwise comparisons
        lines.append("PAIRWISE COMPARISONS (FDR-corrected)")
        lines.append("-" * 40)
        pairwise = self.pairwise_comparisons(data)
        if not pairwise.empty:
            lines.append(pairwise.to_string(index=False))
        lines.append("")

        # Real vs Pseudo — Primary: mixed model
        lines.append("REAL vs PSEUDO-DYAD COMPARISON")
        lines.append("=" * 40)

        lines.append("")
        lines.append("PRIMARY: LINEAR MIXED-EFFECTS MODEL")
        lines.append("-" * 40)
        mixed_rp = self.compare_real_vs_pseudo_mixed(data, by_alliance=False)
        if "error" not in mixed_rp:
            lines.append(f"Formula: {mixed_rp['formula']}")
            lines.append(
                f"N obs: {mixed_rp['n_observations']}, "
                f"N participants: {mixed_rp['n_participants']}"
            )
            lines.append(
                f"Coefficient (is_real): {mixed_rp['coefficient']:.4f} "
                f"(95% CI: [{mixed_rp['ci_lower']:.4f}, {mixed_rp['ci_upper']:.4f}])"
            )
            lines.append(f"p-value: {mixed_rp['p_value']:.6f}")
            lines.append(f"ICC(participant): {mixed_rp['icc_participant']:.4f}")
        else:
            lines.append(f"Mixed model failed: {mixed_rp['error']}")
        lines.append("")

        # Real vs Pseudo — Secondary: participant-aggregated
        lines.append("SECONDARY: PARTICIPANT-AGGREGATED WILCOXON")
        lines.append("-" * 40)
        agg_rp = self.compare_real_vs_pseudo_aggregated(data)
        if "error" not in agg_rp:
            lines.append(f"N participants (paired): {agg_rp['n_participants']}")
            lines.append(
                f"Mean real: {agg_rp['mean_real']:.4f}, "
                f"Mean pseudo: {agg_rp['mean_pseudo']:.4f}"
            )
            lines.append(f"Wilcoxon W: {agg_rp['W_statistic']:.2f}")
            lines.append(f"p-value: {agg_rp['p_value']:.6f}")
            lines.append(f"Cohen's d (paired): {agg_rp['cohens_d_paired']:.4f}")
        else:
            lines.append(f"Aggregated test failed: {agg_rp['error']}")
        lines.append("")

        # Real vs Pseudo — Tertiary: naive Mann-Whitney
        lines.append("TERTIARY: NAIVE MANN-WHITNEY U (epoch-level)")
        lines.append("-" * 40)
        lines.append(
            "NOTE: Treats epochs as independent; p-values are anti-conservative"
        )
        naive_rp = self.compare_real_vs_pseudo_naive(data, by_alliance=False)
        if "overall" in naive_rp:
            c = naive_rp["overall"]
            lines.append(
                f"Real dyads - Mean ICD: {c['mean_real']:.4f} (n={c['n_real']})"
            )
            lines.append(
                f"Pseudo-dyads - Mean ICD: {c['mean_pseudo']:.4f} (n={c['n_pseudo']})"
            )
            lines.append(f"Mann-Whitney U: {c['U_statistic']:.2f}")
            lines.append(f"p-value: {c['p_value']:.6f}")
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
