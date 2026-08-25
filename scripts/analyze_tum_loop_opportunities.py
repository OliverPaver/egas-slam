#!/usr/bin/env python3
"""Estimate loop-closure opportunities in processed TUM RGB-D sequences.

This is an offline benchmark-planning diagnostic only. It reads ground-truth
poses from the processed dataset and is not used by the SLAM runtime.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def load_poses(sequence_dir: Path) -> list[tuple[int, np.ndarray]]:
    pose_dir = sequence_dir / "camera"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"missing camera pose directory: {pose_dir}")

    poses: list[tuple[int, np.ndarray]] = []
    for path in sorted(pose_dir.glob("pose*.txt")):
        frame_text = path.stem.removeprefix("pose")
        if not frame_text.isdigit():
            continue
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"invalid 4x4 pose: {path}")
        poses.append((int(frame_text), pose))
    if not poses:
        raise FileNotFoundError(f"no pose*.txt files found in {pose_dir}")
    return poses


def rotation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    cos_angle = float((np.trace(rel) - 1.0) * 0.5)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def analyze_sequence(
    sequence_dir: Path,
    min_frame_separation: int,
    distance_threshold: float,
    rotation_threshold_deg: float | None,
    submap_frames: int,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    poses = load_poses(sequence_dir)
    loop_pairs: list[dict[str, object]] = []
    closest_distance = math.inf
    closest_pair = None

    for i, (frame_i, pose_i) in enumerate(poses):
        for frame_j, pose_j in poses[i + 1 :]:
            if frame_j - frame_i < min_frame_separation:
                continue
            distance = float(np.linalg.norm(pose_i[:3, 3] - pose_j[:3, 3]))
            if distance < closest_distance:
                closest_distance = distance
                closest_pair = (frame_i, frame_j)
            if distance > distance_threshold:
                continue
            rotation = rotation_degrees(pose_i, pose_j)
            if rotation_threshold_deg is not None and rotation > rotation_threshold_deg:
                continue
            submap_i = frame_i // submap_frames if submap_frames > 0 else -1
            submap_j = frame_j // submap_frames if submap_frames > 0 else -1
            if submap_i == submap_j:
                continue
            loop_pairs.append(
                {
                    "sequence": sequence_dir.name,
                    "frame_a": frame_i,
                    "frame_b": frame_j,
                    "distance_m": distance,
                    "rotation_deg": rotation,
                    "submap_a": submap_i,
                    "submap_b": submap_j,
                }
            )

    submap_pairs = {(int(p["submap_a"]), int(p["submap_b"])) for p in loop_pairs}
    loop_frames = {int(p["frame_a"]) for p in loop_pairs} | {int(p["frame_b"]) for p in loop_pairs}
    summary = {
        "sequence": sequence_dir.name,
        "frames": len(poses),
        "min_frame_separation": min_frame_separation,
        "distance_threshold_m": distance_threshold,
        "rotation_threshold_deg": "" if rotation_threshold_deg is None else rotation_threshold_deg,
        "submap_frames": submap_frames,
        "loop_frame_pairs": len(loop_pairs),
        "loop_submap_pairs": len(submap_pairs),
        "loop_frames": len(loop_frames),
        "closest_nonlocal_distance_m": "" if not math.isfinite(closest_distance) else closest_distance,
        "closest_nonlocal_pair": "" if closest_pair is None else f"{closest_pair[0]}:{closest_pair[1]}",
    }
    loop_pairs.sort(key=lambda row: (float(row["distance_m"]), float(row["rotation_deg"])))
    return summary, loop_pairs[:top_k]


def iter_sequences(root: Path, names: Iterable[str]) -> list[Path]:
    requested = list(names)
    if requested:
        return [root / name for name in requested]
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "camera").is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Processed TUM root directory")
    parser.add_argument("--sequences", nargs="*", default=[], help="Sequence directory names. Defaults to all processed sequences.")
    parser.add_argument("--min-frame-separation", type=int, default=150)
    parser.add_argument("--distance-threshold", type=float, default=0.35)
    parser.add_argument("--rotation-threshold-deg", type=float, default=60.0, help="Use a negative value to ignore rotation.")
    parser.add_argument("--submap-frames", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--pairs-csv", type=Path)
    args = parser.parse_args()

    rotation_threshold = args.rotation_threshold_deg if args.rotation_threshold_deg >= 0 else None
    summaries: list[dict[str, object]] = []
    top_pairs: list[dict[str, object]] = []
    for sequence_dir in iter_sequences(args.root, args.sequences):
        summary, pairs = analyze_sequence(
            sequence_dir,
            max(1, args.min_frame_separation),
            max(0.0, args.distance_threshold),
            rotation_threshold,
            max(1, args.submap_frames),
            max(0, args.top_k),
        )
        summaries.append(summary)
        top_pairs.extend(pairs)
        print(
            f"{summary['sequence']}: frames={summary['frames']} "
            f"loop_pairs={summary['loop_frame_pairs']} "
            f"submap_pairs={summary['loop_submap_pairs']} "
            f"closest={summary['closest_nonlocal_distance_m']} "
            f"closest_pair={summary['closest_nonlocal_pair']}"
        )

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else [])
            if summaries:
                writer.writeheader()
                writer.writerows(summaries)

    if args.pairs_csv:
        args.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sequence", "frame_a", "frame_b", "distance_m", "rotation_deg", "submap_a", "submap_b"]
        with args.pairs_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_pairs)


if __name__ == "__main__":
    main()
