"""
Cross Recurrence Quantification Analysis (CRQA) on RR Intervals.

Builds the cross-recurrence matrix and extracts recurrence rate,
determinism, and mean diagonal line length — nonlinear coupling
measures that capture shared dynamics without assuming linearity.

Authors: Guillaume Dumas
Date: March 2026
"""

import numpy as np
from scipy.spatial.distance import cdist

from src.physio.synchrony.rr_loader import RRTimeSeries, resample_rr_to_uniform


def _embed(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Time-delay embedding of a 1-D signal."""
    n = len(x) - (dim - 1) * delay
    return np.column_stack([x[i * delay : i * delay + n] for i in range(dim)])


def _diagonal_lines(recurrence_matrix: np.ndarray, min_length: int = 2) -> list[int]:
    """Extract diagonal line lengths from a binary recurrence matrix."""
    n, m = recurrence_matrix.shape
    lines: list[int] = []

    for offset in range(-n + 1, m):
        diag = np.diag(recurrence_matrix, offset)
        length = 0
        for val in diag:
            if val:
                length += 1
            elif length >= min_length:
                lines.append(length)
                length = 0
            else:
                length = 0
        if length >= min_length:
            lines.append(length)

    return lines


def compute_crqa(
    rr1: RRTimeSeries,
    rr2: RRTimeSeries,
    embedding_dim: int = 2,
    time_delay: int = 1,
    radius_percentile: float = 10.0,
    max_points: int = 500,
) -> dict[str, float]:
    """Cross Recurrence Quantification Analysis on RR interval series.

    Args:
        rr1: First participant's RR time series.
        rr2: Second participant's RR time series.
        embedding_dim: Embedding dimension for phase-space reconstruction.
        time_delay: Time delay for embedding (in samples at 4 Hz).
        radius_percentile: Percentile of pairwise distances to use as radius.
        max_points: Maximum number of points (downsamples if needed).

    Returns:
        Dict with recurrence_rate, determinism, mean_diagonal_length.
    """
    fs = 4.0
    _, v1 = resample_rr_to_uniform(rr1, fs)
    _, v2 = resample_rr_to_uniform(rr2, fs)

    # Truncate to same length
    n = min(len(v1), len(v2))
    v1, v2 = v1[:n], v2[:n]

    # Downsample for computational feasibility
    if n > max_points:
        step = n // max_points
        v1 = v1[::step][:max_points]
        v2 = v2[::step][:max_points]

    # Normalize
    v1 = (v1 - np.mean(v1)) / (np.std(v1) + 1e-10)
    v2 = (v2 - np.mean(v2)) / (np.std(v2) + 1e-10)

    # Time-delay embedding
    e1 = _embed(v1, embedding_dim, time_delay)
    e2 = _embed(v2, embedding_dim, time_delay)

    if len(e1) < 10 or len(e2) < 10:
        return {
            "recurrence_rate": np.nan,
            "determinism": np.nan,
            "mean_diagonal_length": np.nan,
        }

    # Cross-distance matrix
    dist = cdist(e1, e2, metric="euclidean")

    # Threshold at given percentile
    radius = np.percentile(dist, radius_percentile)
    crm = (dist <= radius).astype(int)

    # Recurrence rate
    total = crm.size
    n_recurrent = int(crm.sum())
    rr_rate = n_recurrent / total if total > 0 else 0.0

    # Diagonal line statistics
    lines = _diagonal_lines(crm, min_length=2)
    if lines:
        det_points = sum(lines)
        determinism = det_points / n_recurrent if n_recurrent > 0 else 0.0
        mean_diag = float(np.mean(lines))
    else:
        determinism = 0.0
        mean_diag = 0.0

    return {
        "recurrence_rate": float(rr_rate),
        "determinism": float(determinism),
        "mean_diagonal_length": mean_diag,
    }
