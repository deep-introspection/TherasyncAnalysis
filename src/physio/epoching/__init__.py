"""
Epoching module for physiological signals.

This module provides functionality to segment physiological signals into time windows
(epochs) for analysis. Supports multiple epoching methods:
- Fixed windows with overlap
- N-split (equal division)
- Sliding windows

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

from .epoch_assigner import EpochAssigner

__all__ = ["EpochAssigner"]
