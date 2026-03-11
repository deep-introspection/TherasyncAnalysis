"""
Preprocessing modules for physiological signals.

This package contains modules for loading, cleaning, extracting metrics,
and writing BIDS-compliant outputs for BVP, EDA, and HR signals.
"""

# BVP preprocessing
from .bvp_loader import BVPLoader
from .bvp_cleaner import BVPCleaner
from .bvp_metrics import BVPMetricsExtractor
from .bvp_bids_writer import BVPBIDSWriter

# EDA preprocessing
from .eda_loader import EDALoader
from .eda_cleaner import EDACleaner
from .eda_metrics import EDAMetricsExtractor
from .eda_bids_writer import EDABIDSWriter

# HR preprocessing
from .hr_loader import HRLoader
from .hr_cleaner import HRCleaner
from .hr_metrics import HRMetricsExtractor
from .hr_bids_writer import HRBIDSWriter

# TEMP preprocessing
from .temp_loader import TEMPLoader
from .temp_cleaner import TEMPCleaner
from .temp_metrics import TEMPMetricsExtractor
from .temp_bids_writer import TEMPBIDSWriter

__all__ = [
    # BVP
    "BVPLoader",
    "BVPCleaner",
    "BVPMetricsExtractor",
    "BVPBIDSWriter",
    # EDA
    "EDALoader",
    "EDACleaner",
    "EDAMetricsExtractor",
    "EDABIDSWriter",
    # HR
    "HRLoader",
    "HRCleaner",
    "HRMetricsExtractor",
    "HRBIDSWriter",
    # TEMP
    "TEMPLoader",
    "TEMPCleaner",
    "TEMPMetricsExtractor",
    "TEMPBIDSWriter",
]
