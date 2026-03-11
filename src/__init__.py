"""
Therasync Pipeline - Physiological Data Processing

A comprehensive pipeline for analyzing physiological data from family therapy sessions.
"""

__version__ = "0.1.0"
__author__ = "ramdam17"
__email__ = "remy.ramadour.labs@gmail.com"

# Import core functionality
from .core import setup_logging, get_logger, load_config

__all__ = ["setup_logging", "get_logger", "load_config"]
