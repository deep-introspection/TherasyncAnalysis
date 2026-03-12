"""Tests for RR interval loader and resampling."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.physio.synchrony.rr_loader import (
    RRLoader,
    RRTimeSeries,
    resample_rr_to_uniform,
)


def _make_rr_tsv(
    tmp_dir: Path,
    subject: str,
    session: str,
    task: str,
    n_intervals: int = 100,
    rr_mean: float = 800.0,
) -> Path:
    """Create a synthetic RR interval TSV file."""
    bvp_dir = tmp_dir / f"sub-{subject}" / session / "bvp"
    bvp_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rr_ms = rr_mean + rng.normal(0, 50, n_intervals)
    times_start = np.cumsum(np.concatenate([[0.0], rr_ms[:-1]])) / 1000.0
    times_end = times_start + rr_ms / 1000.0
    is_valid = np.ones(n_intervals, dtype=int)
    # Mark a few as invalid
    is_valid[0] = 0
    if n_intervals > 5:
        is_valid[5] = 0

    df = pd.DataFrame(
        {
            "time_peak_start": times_start,
            "time_peak_end": times_end,
            "rr_interval_ms": rr_ms,
            "is_valid": is_valid,
        }
    )

    path = bvp_dir / f"sub-{subject}_{session}_task-{task}_desc-rrintervals_physio.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


class TestRRTimeSeries:
    def test_dataclass_fields(self) -> None:
        ts = RRTimeSeries(
            times=np.array([0.0, 0.8, 1.6]),
            rr_ms=np.array([800.0, 800.0, 800.0]),
            subject="g01p01",
            session="ses-01",
            task="therapy",
            duration_s=2.4,
        )
        assert ts.subject == "g01p01"
        assert len(ts.times) == 3
        assert ts.duration_s == 2.4


class TestRRLoader:
    def test_load_rr_returns_valid_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_rr_tsv(tmp_path, "g01p01", "ses-01", "therapy")

            with patch.object(RRLoader, "__init__", lambda self, *a, **kw: None):
                loader = RRLoader.__new__(RRLoader)
                loader.preprocessing_dir = tmp_path
                loader._cache = {}

            ts = loader.load_rr("g01p01", "ses-01", "therapy")
            assert ts is not None
            # 100 total - 2 invalid = 98
            assert len(ts.rr_ms) == 98
            assert ts.subject == "g01p01"
            assert ts.duration_s > 0

    def test_load_rr_caches_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_rr_tsv(tmp_path, "g01p01", "ses-01", "therapy")

            with patch.object(RRLoader, "__init__", lambda self, *a, **kw: None):
                loader = RRLoader.__new__(RRLoader)
                loader.preprocessing_dir = tmp_path
                loader._cache = {}

            ts1 = loader.load_rr("g01p01", "ses-01", "therapy")
            ts2 = loader.load_rr("g01p01", "ses-01", "therapy")
            assert ts1 is ts2  # Same object from cache

    def test_load_rr_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(RRLoader, "__init__", lambda self, *a, **kw: None):
                loader = RRLoader.__new__(RRLoader)
                loader.preprocessing_dir = Path(tmp)
                loader._cache = {}

            assert loader.load_rr("g99p99", "ses-01", "therapy") is None

    def test_load_rr_too_few_valid_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_rr_tsv(tmp_path, "g01p01", "ses-01", "therapy", n_intervals=5)

            with patch.object(RRLoader, "__init__", lambda self, *a, **kw: None):
                loader = RRLoader.__new__(RRLoader)
                loader.preprocessing_dir = tmp_path
                loader._cache = {}

            # 5 intervals - 1 invalid = 4, which is < 10
            assert loader.load_rr("g01p01", "ses-01", "therapy") is None

    def test_load_rr_normalizes_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_rr_tsv(tmp_path, "g01p01", "ses-01", "therapy")

            with patch.object(RRLoader, "__init__", lambda self, *a, **kw: None):
                loader = RRLoader.__new__(RRLoader)
                loader.preprocessing_dir = tmp_path
                loader._cache = {}

            # Pass without ses- prefix
            ts = loader.load_rr("g01p01", "01", "therapy")
            assert ts is not None

    def test_clear_cache(self) -> None:
        with patch.object(RRLoader, "__init__", lambda self, *a, **kw: None):
            loader = RRLoader.__new__(RRLoader)
            loader._cache = {("a", "b", "c"): None}
            loader.clear_cache()
            assert len(loader._cache) == 0


class TestResampleRRToUniform:
    def test_output_is_uniformly_spaced(self) -> None:
        rng = np.random.default_rng(0)
        times = np.cumsum(rng.uniform(0.7, 1.0, 200))
        rr_ms = 800.0 + rng.normal(0, 30, 200)

        ts = RRTimeSeries(
            times=times,
            rr_ms=rr_ms,
            subject="test",
            session="ses-01",
            task="therapy",
            duration_s=times[-1],
        )

        t_uniform, rr_uniform = resample_rr_to_uniform(ts, fs=4.0)

        # Check uniform spacing
        dt = np.diff(t_uniform)
        np.testing.assert_allclose(dt, dt[0], atol=1e-10)

        # Check sampling rate
        assert abs(1.0 / dt[0] - 4.0) < 0.01

        # Check values are in reasonable range
        assert np.all(rr_uniform > 500)
        assert np.all(rr_uniform < 1200)

    def test_preserves_original_values_approximately(self) -> None:
        times = np.arange(0, 50, 0.8)
        rr_ms = np.full(len(times), 800.0)

        ts = RRTimeSeries(
            times=times,
            rr_ms=rr_ms,
            subject="test",
            session="ses-01",
            task="therapy",
            duration_s=times[-1],
        )

        _, rr_uniform = resample_rr_to_uniform(ts, fs=4.0)

        # Constant input should give constant output
        np.testing.assert_allclose(rr_uniform, 800.0, atol=0.1)
