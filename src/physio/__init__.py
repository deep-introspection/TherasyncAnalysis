"""
Physiological data processing modules for Therasync Pipeline.

This package contains modules for processing BVP, EDA, HR and other
physiological signals from Empatica devices.

Subpackages:
    preprocessing: Signal preprocessing pipelines (loading, cleaning, metrics, BIDS writing)

Authors: Lena Adel, Remy Ramadour
"""

# Import from preprocessing subpackage for backward compatibility
from src.physio.preprocessing import (
    BVPLoader,
    BVPCleaner,
    BVPMetricsExtractor,
    BVPBIDSWriter,
    EDALoader,
    EDACleaner,
    EDAMetricsExtractor,
    EDABIDSWriter,
    HRLoader,
    HRCleaner,
    HRMetricsExtractor,
    HRBIDSWriter,
)

__all__ = [
    # BVP Pipeline
    "BVPLoader",
    "BVPCleaner",
    "BVPMetricsExtractor",
    "BVPBIDSWriter",
    # EDA Pipeline
    "EDALoader",
    "EDACleaner",
    "EDAMetricsExtractor",
    "EDABIDSWriter",
    # HR Pipeline
    "HRLoader",
    "HRCleaner",
    "HRMetricsExtractor",
    "HRBIDSWriter",
]
