#!/usr/bin/env python3
"""
Batch Generate DPPA Animations.

Generate epoch-by-epoch animation frames and videos for all dyads
in the intra-family configuration.

Usage:
    # Generate frames only
    uv run python scripts/physio/dppa/batch_generate_animations.py --task therapy

    # Generate frames + videos
    uv run python scripts/physio/dppa/batch_generate_animations.py --task therapy --video

    # Specific dyads
    uv run python scripts/physio/dppa/batch_generate_animations.py --dyads g01p01_ses-01_vs_g01p02_ses-01

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.core.config_loader import ConfigLoader  # noqa: E402
from src.core.logger_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def get_all_dyads(config: ConfigLoader, task: str = "therapy") -> List[str]:
    """
    Get list of all available dyads from intra-family ICD files.

    Args:
        config: ConfigLoader instance
        task: Task name

    Returns:
        List of dyad identifiers
    """
    derivatives_dir = Path(config.get("paths.derivatives_dir", "data/derivatives"))
    intra_dir = derivatives_dir / "dppa" / "intra_family"

    # Find ICD file
    pattern = f"intra_family_icd_task-{task}_*.csv"
    icd_files = list(intra_dir.glob(pattern))

    if not icd_files:
        logger.warning(f"No ICD file found for task {task}")
        return []

    # Read CSV header to get dyad columns
    import pandas as pd

    df = pd.read_csv(icd_files[0], nrows=0)

    # Filter columns that are dyad pairs (contain '_vs_')
    dyad_cols = [c for c in df.columns if "_vs_" in c]

    logger.info(f"Found {len(dyad_cols)} dyads in {icd_files[0].name}")
    return dyad_cols


def generate_frames_for_dyad(
    dyad: str,
    task: str,
    method: str,
    max_epochs: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> bool:
    """
    Generate frames for a single dyad using generate_epoch_frames.py.

    Args:
        dyad: Dyad identifier
        task: Task name
        method: Epoching method
        max_epochs: Optional max epochs limit
        config_path: Config file path

    Returns:
        True if successful
    """
    script_path = (
        project_root / "scripts" / "physio" / "dppa" / "generate_epoch_frames.py"
    )

    cmd = [
        sys.executable,
        str(script_path),
        "--dyad",
        dyad,
        "--task",
        task,
        "--method",
        method,
    ]

    if max_epochs:
        cmd.extend(["--max-epochs", str(max_epochs)])

    if config_path:
        cmd.extend(["--config", str(config_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per dyad
        )

        if result.returncode != 0:
            logger.error(f"Failed to generate frames for {dyad}: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout generating frames for {dyad}")
        return False
    except Exception as e:
        logger.error(f"Error generating frames for {dyad}: {e}")
        return False


def create_video_from_frames(
    frames_dir: Path, output_path: Path, fps: int = 10, format: str = "mp4"
) -> bool:
    """
    Create video from frames using ffmpeg.

    Args:
        frames_dir: Directory containing frame_XXXX.png files
        output_path: Output video path
        fps: Frames per second
        format: Output format (mp4 or gif)

    Returns:
        True if successful
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found. Install with: brew install ffmpeg")
        return False

    frames_pattern = str(frames_dir / "frame_%04d.png")

    if format == "mp4":
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate",
            str(fps),
            "-i",
            frames_pattern,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Pad to even dimensions
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",  # Quality (lower = better, 18-28 is good)
            str(output_path),
        ]
    elif format == "gif":
        # Two-pass for better GIF quality
        palette_path = frames_dir / "palette.png"

        # Generate palette
        cmd_palette = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frames_pattern,
            "-vf",
            f"fps={fps},scale=800:-1:flags=lanczos,palettegen",
            str(palette_path),
        ]

        try:
            subprocess.run(cmd_palette, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate palette: {e}")
            return False

        # Generate GIF using palette
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frames_pattern,
            "-i",
            str(palette_path),
            "-lavfi",
            f"fps={fps},scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(output_path),
        ]
    else:
        logger.error(f"Unknown format: {format}")
        return False

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return False

        # Clean up palette if GIF
        if format == "gif" and palette_path.exists():
            palette_path.unlink()

        logger.info(
            f"Created video: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        return True

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout")
        return False
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch generate DPPA animations for all dyads"
    )
    parser.add_argument(
        "--task",
        choices=["therapy", "restingstate"],
        default="therapy",
        help="Task to process (default: therapy)",
    )
    parser.add_argument(
        "--method", default="sliding_duration30s_step5s", help="Epoching method"
    )
    parser.add_argument(
        "--dyads", nargs="+", help="Specific dyads to process (default: all)"
    )
    parser.add_argument(
        "--max-epochs", type=int, help="Limit epochs per dyad (for testing)"
    )
    parser.add_argument(
        "--video", action="store_true", help="Also create videos from frames"
    )
    parser.add_argument(
        "--video-format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Video format (default: mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Video frames per second (default: 10)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dyads that already have frames",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_dir=project_root / "log")

    logger.info("=" * 80)
    logger.info("BATCH GENERATE DPPA ANIMATIONS")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Video: {args.video} ({args.video_format}, {args.fps} fps)")

    # Load config
    config = ConfigLoader(args.config)

    # Get dyads to process
    if args.dyads:
        dyads = args.dyads
    else:
        dyads = get_all_dyads(config, args.task)

    if not dyads:
        logger.error("No dyads found to process")
        return 1

    logger.info(f"Processing {len(dyads)} dyads")

    # Output directories
    derivatives_dir = Path(config.get("paths.derivatives_dir", "data/derivatives"))
    frames_base_dir = derivatives_dir / "dppa" / "frames"
    videos_dir = derivatives_dir / "dppa" / "videos" / args.task

    if args.video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    # Process each dyad
    success_frames = 0
    success_videos = 0
    failed = []

    for i, dyad in enumerate(dyads):
        logger.info(f"\n[{i + 1}/{len(dyads)}] Processing: {dyad}")

        frames_dir = frames_base_dir / dyad

        # Check if already exists
        if args.skip_existing and frames_dir.exists():
            frame_count = len(list(frames_dir.glob("frame_*.png")))
            if frame_count > 0:
                logger.info(f"  Skipping (already has {frame_count} frames)")
                success_frames += 1
                continue

        # Generate frames
        if generate_frames_for_dyad(
            dyad, args.task, args.method, args.max_epochs, args.config
        ):
            success_frames += 1

            # Create video if requested
            if args.video and frames_dir.exists():
                video_path = videos_dir / f"{dyad}.{args.video_format}"
                if create_video_from_frames(
                    frames_dir, video_path, args.fps, args.video_format
                ):
                    success_videos += 1
        else:
            failed.append(dyad)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total dyads:      {len(dyads)}")
    logger.info(f"Frames generated: {success_frames}")
    if args.video:
        logger.info(f"Videos created:   {success_videos}")
    logger.info(f"Failed:           {len(failed)}")

    if failed:
        logger.info("\nFailed dyads:")
        for d in failed:
            logger.info(f"  - {d}")

    logger.info("=" * 80)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
