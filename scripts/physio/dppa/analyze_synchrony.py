#!/usr/bin/env python3
"""
Analyze Dynamic Synchrony Metrics for Real vs Pseudo Dyads.

Computes centroid correlation, lagged cross-correlation, and SD feature
concordance for all dyad pairs, then runs 3-tier statistical comparison
(naive, participant-aggregated, mixed model) with FDR correction.

Usage:
    uv run python scripts/physio/dppa/analyze_synchrony.py --task therapy
    uv run python scripts/physio/dppa/analyze_synchrony.py --task restingstate

Authors: Guillaume Dumas
Date: March 2026
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.core.logger_setup import setup_logging
from src.physio.dppa.centroid_loader import CentroidLoader
from src.physio.dppa.dyad_config_loader import DyadConfigLoader
from src.physio.dppa.synchrony_calculator import (
    compute_centroid_correlation,
    compute_feature_concordance,
    compute_lagged_cross_correlation,
)
from src.physio.dppa.synchrony_stats import (
    compute_synchrony_for_all_dyads,
    generate_synchrony_report,
    compare_real_vs_pseudo_synchrony,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dynamic synchrony metrics")
    parser.add_argument(
        "--task",
        choices=["therapy", "restingstate"],
        required=True,
        help="Task to analyze",
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    setup_logging(log_dir=Path("log"))
    config = ConfigLoader(args.config)
    loader = CentroidLoader(args.config)
    dyad_config = DyadConfigLoader()

    method = dyad_config.get_inter_session_method(task=args.task)

    metrics = {
        "centroid_correlation": {
            "fn": compute_centroid_correlation,
            "value_key": "correlation",
            "one_sided": True,
        },
        "lagged_cross_correlation": {
            "fn": compute_lagged_cross_correlation,
            "value_key": "peak_correlation",
            "one_sided": False,
        },
        "sd1_concordance": {
            "fn": lambda c1, c2: compute_feature_concordance(c1, c2, ["sd1"])["sd1"],
            "value_key": "correlation",
            "one_sided": False,
        },
    }

    results_dict: dict[str, dict] = {}

    for name, spec in metrics.items():
        logger.info(f"Computing {name}...")
        df = compute_synchrony_for_all_dyads(
            loader,
            dyad_config,
            args.task,
            method,
            metric_fn=spec["fn"],
            value_key=spec["value_key"],
        )
        if len(df) == 0:
            logger.warning(f"No valid dyad data for {name}")
            continue

        tier_results = compare_real_vs_pseudo_synchrony(
            df,
            value_col="metric_value",
            one_sided=spec["one_sided"],
        )
        results_dict[name] = tier_results

    # Generate and print report
    report = generate_synchrony_report(args.task, results_dict)
    print(report)

    # Save report
    derivatives_dir = Path(config.get("paths.derivatives_dir", "data/derivatives"))
    output_dir = derivatives_dir / "dppa" / "inter_session" / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"synchrony_stats_{args.task}.txt"
    output_path.write_text(report)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
