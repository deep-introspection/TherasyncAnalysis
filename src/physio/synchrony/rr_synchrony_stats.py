"""
RR Synchrony Statistics.

Computes synchrony metrics for all dyad pairs using raw RR interval
time series, producing a DataFrame compatible with the existing
3-tier statistical testing pipeline.

Authors: Guillaume Dumas
Date: March 2026
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.physio.dppa.dyad_config_loader import DyadConfigLoader
from src.physio.synchrony.rr_loader import RRLoader, RRTimeSeries

logger = logging.getLogger(__name__)


def compute_rr_synchrony_for_all_dyads(
    loader: RRLoader,
    dyad_config: DyadConfigLoader,
    task: str,
    metric_fn: Callable[[RRTimeSeries, RRTimeSeries], dict[str, float]],
    value_key: str,
) -> pd.DataFrame:
    """Apply a synchrony metric to all dyad pairs using raw RR intervals.

    Output schema matches synchrony_stats.compute_synchrony_for_all_dyads()
    so results feed directly into compare_real_vs_pseudo_synchrony().

    Args:
        loader: RRLoader instance.
        dyad_config: DyadConfigLoader instance.
        task: Task name (e.g. 'therapy').
        metric_fn: Function(rr1, rr2) -> dict with at least value_key.
        value_key: Key in metric_fn result for the scalar metric value.

    Returns:
        DataFrame with columns: dyad_pair, is_real, participant1, session1,
        participant2, session2, metric_value, plus all metric_fn output keys.
    """
    pairs = dyad_config.get_all_session_pairs_with_real_flag(task=task)
    records: list[dict] = []

    for pair in pairs:
        s1, ses1 = pair["subject1"], pair["session1"]
        s2, ses2 = pair["subject2"], pair["session2"]

        rr1 = loader.load_rr(s1, ses1, task)
        rr2 = loader.load_rr(s2, ses2, task)

        if rr1 is None or rr2 is None:
            continue

        result = metric_fn(rr1, rr2)
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
    logger.info(f"RR synchrony: {len(df)} dyads ({n_real} real)")
    return df
