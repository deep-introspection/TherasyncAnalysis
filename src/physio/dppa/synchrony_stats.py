"""
Synchrony Statistics for DPPA Analysis.

Statistical testing pipeline that computes dynamic synchrony metrics for all
dyad pairs (real + pseudo) and runs 3-tier statistical comparison:
naive, participant-aggregated, and mixed model.

Authors: Guillaume Dumas
Date: March 2026
"""

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.physio.dppa.centroid_loader import CentroidLoader
from src.physio.dppa.dyad_config_loader import DyadConfigLoader
from src.stats.corrections import correct_pvalues

try:
    from statsmodels.regression.mixed_linear_model import MixedLM

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)


def compute_synchrony_for_all_dyads(
    loader: CentroidLoader,
    dyad_config: DyadConfigLoader,
    task: str,
    method: str,
    metric_fn: Callable[[pd.DataFrame, pd.DataFrame], dict],
    value_key: str = "correlation",
) -> pd.DataFrame:
    """
    Apply a synchrony metric to all dyad pairs.

    Args:
        loader: CentroidLoader instance.
        dyad_config: DyadConfigLoader instance.
        task: Task name (e.g. 'therapy').
        method: Epoching method (e.g. 'nsplit120').
        metric_fn: Function(centroids1, centroids2) -> dict with value_key.
        value_key: Key in metric_fn result to use as the scalar value.

    Returns:
        DataFrame with columns: dyad_pair, is_real, participant1, session1,
        participant2, session2, metric_value, plus all metric_fn output keys.
    """
    pairs = dyad_config.get_all_session_pairs_with_real_flag(task=task)
    records = []

    for pair in pairs:
        s1, ses1 = pair["subject1"], pair["session1"]
        s2, ses2 = pair["subject2"], pair["session2"]

        c1 = loader.load_centroid(s1, ses1, task, method)
        c2 = loader.load_centroid(s2, ses2, task, method)

        if c1 is None or c2 is None:
            continue

        result = metric_fn(c1, c2)
        val = result.get(value_key, np.nan)
        if np.isnan(val):
            continue

        record = {
            "dyad_pair": f"{s1}_{ses1}_vs_{s2}_{ses2}",
            "is_real": pair["is_real_dyad"],
            "participant1": s1,
            "session1": ses1,
            "participant2": s2,
            "session2": ses2,
            "metric_value": val,
        }
        record.update(result)
        records.append(record)

    df = pd.DataFrame(records)
    n_real = df["is_real"].sum() if len(df) > 0 else 0
    logger.info(f"Computed synchrony: {len(df)} dyads ({n_real} real)")
    return df


def compare_real_vs_pseudo_synchrony(
    results_df: pd.DataFrame,
    value_col: str = "metric_value",
    one_sided: bool = False,
) -> dict:
    """
    Run 3-tier statistical comparison of real vs pseudo dyads.

    Args:
        results_df: DataFrame from compute_synchrony_for_all_dyads.
        value_col: Column with the metric value.
        one_sided: If True, test real > pseudo (for centroid correlation).

    Returns:
        Dict with 'naive', 'aggregated', 'mixed' sub-dicts.
    """
    real = results_df.loc[results_df["is_real"], value_col].values
    pseudo = results_df.loc[~results_df["is_real"], value_col].values

    output: dict = {}

    # --- Tier 1: Naive Mann-Whitney ---
    if len(real) >= 2 and len(pseudo) >= 2:
        u, p = sp_stats.mannwhitneyu(
            real, pseudo, alternative="greater" if one_sided else "two-sided"
        )
        pooled_std = np.std(np.concatenate([real, pseudo]), ddof=1)
        d = (np.mean(real) - np.mean(pseudo)) / pooled_std if pooled_std > 0 else 0.0
        output["naive"] = {
            "U": float(u),
            "p": float(p),
            "d": float(d),
            "n_real": len(real),
            "n_pseudo": len(pseudo),
            "mean_real": float(np.mean(real)),
            "mean_pseudo": float(np.mean(pseudo)),
        }
    else:
        output["naive"] = {"error": "Not enough data"}

    # --- Tier 2: Participant-aggregated paired Wilcoxon ---
    output["aggregated"] = _aggregated_test(results_df, value_col, one_sided)

    # --- Tier 3: Mixed model ---
    output["mixed"] = _mixed_model_test(results_df, value_col)

    return output


def _aggregated_test(df: pd.DataFrame, value_col: str, one_sided: bool) -> dict:
    """Mean per participant, then paired Wilcoxon."""
    participant_real: dict[str, list[float]] = defaultdict(list)
    participant_pseudo: dict[str, list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        target = participant_real if row["is_real"] else participant_pseudo
        for pid in (row["participant1"], row["participant2"]):
            target[pid].append(row[value_col])

    real_means = {p: np.mean(v) for p, v in participant_real.items()}
    pseudo_means = {p: np.mean(v) for p, v in participant_pseudo.items()}

    common = sorted(set(real_means) & set(pseudo_means))
    if len(common) < 5:
        return {"error": f"Only {len(common)} participants with both real and pseudo"}

    rv = np.array([real_means[p] for p in common])
    pv = np.array([pseudo_means[p] for p in common])

    alt = "greater" if one_sided else "two-sided"
    try:
        w, p = sp_stats.wilcoxon(rv, pv, alternative=alt)
    except ValueError:
        w, p = np.nan, 1.0

    diff = rv - pv
    d_paired = (
        float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    )

    return {
        "W": float(w),
        "p": float(p),
        "d_paired": d_paired,
        "n_participants": len(common),
        "mean_real": float(np.mean(rv)),
        "mean_pseudo": float(np.mean(pv)),
    }


def _mixed_model_test(df: pd.DataFrame, value_col: str) -> dict:
    """Linear mixed model: metric ~ is_real + (1 | participant)."""
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}

    records = []
    for _, row in df.iterrows():
        for pid in (row["participant1"], row["participant2"]):
            records.append(
                {
                    "value": row[value_col],
                    "is_real": int(row["is_real"]),
                    "participant": pid,
                    "dyad_pair": row["dyad_pair"],
                }
            )

    mdf = pd.DataFrame(records)
    if mdf["participant"].nunique() < 3:
        return {"error": "Too few participants"}

    try:
        model = MixedLM.from_formula("value ~ is_real", mdf, groups=mdf["participant"])
        result = model.fit(method="powell")
        coef = float(result.params["is_real"])
        se = float(result.bse["is_real"])
        p = float(result.pvalues["is_real"])
        re_var = (
            float(result.cov_re.iloc[0, 0]) if hasattr(result, "cov_re") else np.nan
        )
        resid_var = float(result.scale)
        icc_val = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else np.nan

        return {
            "coef": coef,
            "se": se,
            "p": p,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "icc": icc_val,
        }
    except Exception as e:
        logger.error(f"Mixed model failed: {e}")
        return {"error": str(e)}


def generate_synchrony_report(task: str, results_dict: dict[str, dict]) -> str:
    """
    Generate a text report for synchrony analysis results.

    Args:
        task: Task name.
        results_dict: {metric_name: output from compare_real_vs_pseudo_synchrony}.

    Returns:
        Formatted text report.
    """
    lines = [
        "=" * 70,
        f"SYNCHRONY ANALYSIS REPORT — {task.upper()}",
        "=" * 70,
        "",
    ]

    p_values = []

    for metric_name, tiers in results_dict.items():
        lines.append(f"--- {metric_name} ---")

        for tier_name in ("naive", "aggregated", "mixed"):
            tier = tiers.get(tier_name, {})
            if "error" in tier:
                lines.append(f"  {tier_name}: {tier['error']}")
                continue

            p = tier.get("p", tier.get("p_value", np.nan))
            lines.append(f"  {tier_name}: p = {p:.4f}")

            if tier_name == "naive":
                lines.append(
                    f"    U = {tier.get('U', '?')}, d = {tier.get('d', '?'):.3f}"
                )
                lines.append(
                    f"    mean_real = {tier.get('mean_real', '?'):.4f}, mean_pseudo = {tier.get('mean_pseudo', '?'):.4f}"
                )
            elif tier_name == "aggregated":
                lines.append(
                    f"    W = {tier.get('W', '?')}, d_paired = {tier.get('d_paired', '?'):.3f}"
                )
                lines.append(f"    n_participants = {tier.get('n_participants', '?')}")
            elif tier_name == "mixed":
                lines.append(
                    f"    coef = {tier.get('coef', '?'):.4f}, ICC = {tier.get('icc', '?'):.3f}"
                )

            if not np.isnan(p):
                p_values.append(p)

        lines.append("")

    # FDR correction across all collected p-values
    if len(p_values) >= 2:
        rejected, adjusted = correct_pvalues(p_values, method="fdr_bh")
        lines.append("--- FDR CORRECTION (across all tiers & metrics) ---")
        for i, (orig, adj, rej) in enumerate(zip(p_values, adjusted, rejected)):
            sig = "*" if rej else ""
            lines.append(f"  p[{i}] = {orig:.4f} -> adjusted = {adj:.4f} {sig}")
        lines.append("")

    return "\n".join(lines)
