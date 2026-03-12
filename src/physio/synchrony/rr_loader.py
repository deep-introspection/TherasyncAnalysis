"""
RR Interval Time Series Loader.

Loads raw RR interval data from preprocessing derivatives and provides
uniform resampling for frequency-domain synchrony methods.

Authors: Guillaume Dumas
Date: March 2026
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class RRTimeSeries:
    """Raw RR interval time series for a single recording."""

    times: np.ndarray  # Peak times in seconds (midpoint of each interval)
    rr_ms: np.ndarray  # RR intervals in milliseconds
    subject: str
    session: str
    task: str
    duration_s: float


class RRLoader:
    """Load and cache RR interval time series from preprocessing derivatives."""

    def __init__(self, config_path: str | Path | None = None):
        config = ConfigLoader(config_path)
        paths = config.get("paths", {})
        self.preprocessing_dir = (
            Path(paths.get("derivatives", "data/derivatives")) / "preprocessing"
        )
        self._cache: dict[tuple[str, str, str], RRTimeSeries | None] = {}
        logger.info("RRLoader initialized")

    def load_rr(
        self, subject: str, session: str, task: str, use_cache: bool = True
    ) -> RRTimeSeries | None:
        """Load valid RR intervals from preprocessing derivatives.

        Args:
            subject: Subject ID (e.g., 'g01p01').
            session: Session ID (e.g., 'ses-01').
            task: Task name (e.g., 'therapy').
            use_cache: Return cached result if available.

        Returns:
            RRTimeSeries or None if file missing / no valid intervals.
        """
        if not session.startswith("ses-"):
            session = f"ses-{session}"

        key = (subject, session, task)
        if use_cache and key in self._cache:
            return self._cache[key]

        path = (
            self.preprocessing_dir
            / f"sub-{subject}"
            / session
            / "bvp"
            / f"sub-{subject}_{session}_task-{task}_desc-rrintervals_physio.tsv"
        )

        if not path.exists():
            logger.warning(f"RR file not found: {path}")
            self._cache[key] = None
            return None

        df = pd.read_csv(path, sep="\t")
        valid = df[df["is_valid"] == 1].copy()

        if len(valid) < 10:
            logger.warning(
                f"Too few valid RR intervals ({len(valid)}) for {subject}/{session}/{task}"
            )
            self._cache[key] = None
            return None

        # Use midpoint of each interval as the time coordinate
        times = ((valid["time_peak_start"] + valid["time_peak_end"]) / 2).values
        rr_ms = valid["rr_interval_ms"].values
        duration = valid["time_peak_end"].iloc[-1] - valid["time_peak_start"].iloc[0]

        ts = RRTimeSeries(
            times=times,
            rr_ms=rr_ms,
            subject=subject,
            session=session,
            task=task,
            duration_s=float(duration),
        )
        self._cache[key] = ts
        logger.debug(
            f"Loaded RR: {subject}/{session}/{task} ({len(rr_ms)} intervals, {duration:.0f}s)"
        )
        return ts

    def clear_cache(self) -> None:
        """Clear cached data."""
        self._cache.clear()


def resample_rr_to_uniform(
    rr: RRTimeSeries, fs: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic interpolation of RR intervals to a uniform time grid.

    Args:
        rr: RR time series.
        fs: Target sampling frequency in Hz.

    Returns:
        (times_uniform, rr_interpolated) arrays.
    """
    t_start, t_end = rr.times[0], rr.times[-1]
    n_samples = int((t_end - t_start) * fs) + 1
    t_uniform = np.linspace(t_start, t_end, n_samples)

    kind = "cubic" if len(rr.times) >= 4 else "linear"
    interpolator = interp1d(rr.times, rr.rr_ms, kind=kind, fill_value="extrapolate")
    rr_uniform = interpolator(t_uniform)

    return t_uniform, rr_uniform
