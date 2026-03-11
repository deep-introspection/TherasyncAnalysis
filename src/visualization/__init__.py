"""
Visualization Module for TherasyncPipeline.

This module provides comprehensive visualization tools for preprocessed
physiological data (BVP, EDA, HR).

Main components:
- data_loader: Load preprocessed data from derivatives/preprocessing/
- plotters: Individual plotting functions for different visualization types
- report_generator: Generate HTML reports with all figures
- config: Visualization configuration and styling

Authors: Lena Adel, Remy Ramadour
Date: November 2025
Version: 0.4.0
"""

__version__ = "0.4.0"
__all__ = ["data_loader", "plotters", "report_generator", "config"]
