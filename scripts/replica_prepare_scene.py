#!/usr/bin/env python3
"""Prepare one Replica scene for GPS-SLAM.

The original `replica_preprocess.py` is intentionally left untouched because it
is a simple hardcoded helper. This script is for repeatable cross-scene tests:
it reads one raw Replica scene and writes GPS-SLAM-format RGB/depth/pose files,
optionally also writing a half-resolution copy for constrained GPUs.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


FULL_INTRINSICS = np.array(
    [[600.0, 0.0, 599.5], [0.0, 600.0, 339.5], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
FULL_SHAPE = np.array([1200, 680], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Replica scene for GPS-SLAM")
    parser.add_argument("--raw-root", required=True, type=Path, help="Directory containing raw Replica scene folders")
    parser.add_argument("--scene", required=True, help="Scene name, e.g. office1 or room0")
    parser.add_argument("--output-root", required=True, type=Path, help="Destination root for full-resolution prepared scenes")
    parser.add_argument("--half-output-root", type=Path, default=None, help="Optional destination root for half-resolution scenes")
    parser.add_argument("--target-frames", type=int, default=2000, help="Uniformly sample to at most this many frames")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output directories")
    return parser.parse_args()


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    (path / "camera").mkdir(parents=True)
    (path / "depth").mkdir(parents=True)


def write_metadata(output_dir: Path, intrinsics: np.ndarray, shape: np.ndarray) -> None:
    np.savetxt(output_dir / "camera" / "intrinsics.txt", intrinsics, fmt="%.8f")
    np.savetxt(output_dir / "camera" / "img_shape.txt", shape.astype(np.int32), fmt="%d")


def sampled_indices(frame_count: int, target_frames: int) -> list[int]:
    if target_frames <= 0 or frame_count <= target_frames:
        return list(range(frame_count))
    # Match the original preprocessor's uniform stride behavior while avoiding
    # a zero stride for scenes shorter than the target.
    stride = max(1, frame_count // target_frames)
    return list(range(0, frame_count, stride))[:target_frames]


def prepare_full(raw_scene: Path, output_dir: Path, indices: list[int], overwrite: bool) -> None:
    reset_dir(output_dir, overwrite)
    poses = np.loadtxt(raw_scene / "traj.txt").reshape((-1, 4, 4))
    results_dir = raw_scene / "results"
    for out_idx, src_idx in enumerate(tqdm(indices, desc=f"{raw_scene.name} full")):
        np.savetxt(output_dir / "camera" / f"pose{out_idx:06d}.txt", poses[src_idx], fmt="%.8f")
        shutil.copy2(results_dir / f"frame{src_idx:06d}.jpg", output_dir / "camera" / f"frame{out_idx:06d}.jpg")
        shutil.copy2(results_dir / f"depth{src_idx:06d}.png", output_dir / "depth" / f"depth{out_idx:06d}.png")
    write_metadata(output_dir, FULL_INTRINSICS, FULL_SHAPE)


def prepare_half(full_dir: Path, half_dir: Path, frame_count: int, overwrite: bool) -> None:
    reset_dir(half_dir, overwrite)
    half_shape = np.array([FULL_SHAPE[0] // 2, FULL_SHAPE[1] // 2], dtype=np.int32)
    half_intrinsics = FULL_INTRINSICS.copy()
    half_intrinsics[0, :] *= 0.5
    half_intrinsics[1, :] *= 0.5

    for idx in tqdm(range(frame_count), desc=f"{half_dir.name} half"):
        shutil.copy2(full_dir / "camera" / f"pose{idx:06d}.txt", half_dir / "camera" / f"pose{idx:06d}.txt")

        color = cv2.imread(str(full_dir / "camera" / f"frame{idx:06d}.jpg"), cv2.IMREAD_COLOR)
        if color is None:
            raise RuntimeError(f"Failed to read color frame {idx}")
        color_half = cv2.resize(color, tuple(half_shape), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(half_dir / "camera" / f"frame{idx:06d}.jpg"), color_half)

        depth = cv2.imread(str(full_dir / "depth" / f"depth{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth frame {idx}")
        depth_half = cv2.resize(depth, tuple(half_shape), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(half_dir / "depth" / f"depth{idx:06d}.png"), depth_half)

    write_metadata(half_dir, half_intrinsics, half_shape)


def main() -> None:
    args = parse_args()
    raw_scene = args.raw_root / args.scene
    if not raw_scene.exists():
        raise FileNotFoundError(f"Raw Replica scene not found: {raw_scene}")
    if not (raw_scene / "traj.txt").exists() or not (raw_scene / "results").exists():
        raise FileNotFoundError(f"{raw_scene} does not look like a raw Replica scene")

    poses = np.loadtxt(raw_scene / "traj.txt").reshape((-1, 4, 4))
    indices = sampled_indices(len(poses), args.target_frames)

    full_dir = args.output_root / args.scene
    prepare_full(raw_scene, full_dir, indices, args.overwrite)

    if args.half_output_root is not None:
        half_dir = args.half_output_root / f"{args.scene}_half_clean"
        prepare_half(full_dir, half_dir, len(indices), args.overwrite)

    print(
        f"Prepared {args.scene}: raw_frames={len(poses)} prepared_frames={len(indices)} "
        f"full={full_dir}"
    )


if __name__ == "__main__":
    main()
