#!/usr/bin/env python3
"""
Tests for generate_epoch_frames CLI script

Tests validation of frame generation script.
"""

import subprocess
from pathlib import Path


# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "physio" / "dppa" / "generate_epoch_frames.py"


def test_script_exists():
    """Test that generate_epoch_frames.py exists and is executable."""
    assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"


def test_script_help():
    """Test that script shows help message."""
    result = subprocess.run(
        ["uv", "run", "python", str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Generate epoch-by-epoch animation frames" in result.stdout
    assert "--dyad" in result.stdout
    assert "--method" in result.stdout
    assert "--task" in result.stdout


def test_frame_generation_sample():
    """
    Test frame generation with small sample (5 epochs).

    This is a real-world integration test using g01p01 vs g01p02 data.
    """
    output_dir = (
        PROJECT_ROOT / "data" / "derivatives" / "dppa" / "frames" / "test_sample"
    )

    # Clean up previous test output
    if output_dir.exists():
        for frame in output_dir.glob("frame_*.png"):
            frame.unlink()
        output_dir.rmdir()

    # Run script
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT_PATH),
            "--dyad",
            "g01p01_ses-01_vs_g01p02_ses-01",
            "--method",
            "sliding_duration30s_step5s",
            "--task",
            "therapy",
            "--max-epochs",
            "5",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    # Check exit code
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check that frames were created
    assert output_dir.exists(), "Output directory not created"
    frames = list(output_dir.glob("frame_*.png"))
    assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"

    # Check frame naming
    frame_names = sorted([f.name for f in frames])
    expected_names = [f"frame_{i:04d}.png" for i in range(5)]
    assert frame_names == expected_names

    # Check frame sizes (should be non-zero)
    for frame in frames:
        assert frame.stat().st_size > 50_000, f"Frame {frame.name} is too small"

    # Check log output (logs go to stderr because of logging handler)
    assert "GENERATE EPOCH FRAMES" in result.stderr
    assert "Generated:       5" in result.stderr
    assert "Success rate:    100.0%" in result.stderr

    # Clean up
    for frame in frames:
        frame.unlink()
    output_dir.rmdir()


def test_invalid_dyad_format():
    """Test that script rejects invalid dyad format."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT_PATH),
            "--dyad",
            "invalid_format",
            "--max-epochs",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error parsing dyad" in result.stderr
