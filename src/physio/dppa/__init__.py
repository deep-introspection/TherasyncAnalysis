"""
DPPA (Dyadic Poincaré Plot Analysis) Module.

This module provides tools for analyzing physiological synchrony between dyads
using Inter-Centroid Distances (ICD) computed from Poincaré plot centroids.

Modules:
- poincare_calculator: Compute Poincaré centroids per participant/session/epoch
- centroid_loader: Load pre-computed centroid files
- icd_calculator: Calculate Inter-Centroid Distances between dyads
- dyad_config_loader: Load dyad configuration mappings
- dppa_writer: Export ICD results to BIDS-compliant CSV
- dyad_icd_loader: Load ICD data for visualization
- dyad_centroid_loader: Load centroid data for visualization
- dyad_plotter: Generate dyad visualizations
- epoch_animator: Prepare epoch-by-epoch data for animations
- poincare_plotter: Generate Poincaré plot visualizations
- icd_stats_plotter: Statistical visualizations for real vs pseudo dyads
- synchrony_calculator: Dynamic synchrony metrics between dyad members
- synchrony_stats: Statistical testing pipeline for synchrony metrics

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

from .poincare_calculator import PoincareCalculator
from .centroid_loader import CentroidLoader
from .dyad_config_loader import DyadConfigLoader
from .icd_calculator import ICDCalculator
from .dppa_writer import DPPAWriter
from .dyad_icd_loader import DyadICDLoader
from .dyad_centroid_loader import DyadCentroidLoader
from .dyad_plotter import DyadPlotter
from .epoch_animator import EpochAnimator
from .poincare_plotter import PoincarePlotter
from .icd_stats_plotter import ICDStatsPlotter
from .synchrony_calculator import (
    compute_centroid_correlation,
    compute_feature_concordance,
    compute_lagged_cross_correlation,
)
from .synchrony_stats import (
    compute_synchrony_for_all_dyads,
    generate_synchrony_report,
    compare_real_vs_pseudo_synchrony,
)

__all__ = [
    "PoincareCalculator",
    "CentroidLoader",
    "DyadConfigLoader",
    "ICDCalculator",
    "DPPAWriter",
    "DyadICDLoader",
    "DyadCentroidLoader",
    "DyadPlotter",
    "EpochAnimator",
    "PoincarePlotter",
    "ICDStatsPlotter",
    "compute_centroid_correlation",
    "compute_feature_concordance",
    "compute_lagged_cross_correlation",
    "compute_synchrony_for_all_dyads",
    "generate_synchrony_report",
    "compare_real_vs_pseudo_synchrony",
]
