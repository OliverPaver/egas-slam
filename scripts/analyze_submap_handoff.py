#!/usr/bin/env python3
"""Analyze submap handoff drift against ground-truth pose folders.

This is an offline diagnostic for development only. It does not change reported
online results; it reads a completed run's poses and submap logs to identify
which accepted handoffs correlate with trajectory error growth.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import numpy as np


def pose_id(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def read_poses(pose_dir: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    if not pose_dir.exists():
        raise FileNotFoundError(f"pose directory does not exist: {pose_dir}")
    for fname in sorted(os.listdir(pose_dir)):
        path = pose_dir / fname
        if path.suffix.lower() != ".txt":
            continue
        idx = pose_id(path)
        if idx is None:
            continue
        try:
            pose = np.loadtxt(path)
        except Exception:
            continue
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            poses[idx] = pose.astype(np.float64)
    return poses


def read_frame_assignments(path: Path) -> dict[int, int]:
    assignments: dict[int, int] = {}
    if not path.exists():
        return assignments
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                frame_id = int(row.get("frame_id", ""))
                submap_id = int(row.get("primary_submap", ""))
            except ValueError:
                continue
            assignments[frame_id] = submap_id
    return assignments


def read_transitions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def align_est_to_gt(
    est_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    common_ids: list[int],
) -> dict[int, np.ndarray]:
    est = np.stack([est_poses[idx][:3, 3] for idx in common_ids], axis=1)
    gt = np.stack([gt_poses[idx][:3, 3] for idx in common_ids], axis=1)
    est_centered = est - est.mean(axis=1, keepdims=True)
    gt_centered = gt - gt.mean(axis=1, keepdims=True)
    w = est_centered @ gt_centered.T
    u, _, vh = np.linalg.svd(w.T)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        s[2, 2] = -1.0
    rot = u @ s @ vh
    trans = gt.mean(axis=1, keepdims=True) - rot @ est.mean(axis=1, keepdims=True)

    aligned: dict[int, np.ndarray] = {}
    for idx, pose in est_poses.items():
        out = pose.copy()
        out[:3, :3] = rot @ out[:3, :3]
        out[:3, 3:4] = rot @ out[:3, 3:4] + trans
        aligned[idx] = out
    return aligned


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def rmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def percentile(values: Iterable[float], pct: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, pct))


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def window_ids(common_ids: list[int], center: int, before: bool, radius: int) -> list[int]:
    common = set(common_ids)
    if before:
        return [idx for idx in range(center - radius, center) if idx in common]
    return [idx for idx in range(center, center + radius) if idx in common]


def safe_int(value: object, default: int = -1) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def analyze(run_dir: Path, est_dir: Path, gt_dir: Path, output_dir: Path, window: int) -> dict[str, object]:
    est_poses = read_poses(est_dir)
    gt_poses = read_poses(gt_dir)
    common_ids = sorted(set(est_poses) & set(gt_poses))
    if not common_ids:
        raise RuntimeError(f"no matching poses between {est_dir} and {gt_dir}")

    aligned_est = align_est_to_gt(est_poses, gt_poses, common_ids)
    errors_cm: dict[int, float] = {
        idx: float(np.linalg.norm(aligned_est[idx][:3, 3] - gt_poses[idx][:3, 3]) * 100.0)
        for idx in common_ids
    }

    assignments = read_frame_assignments(run_dir / "frame_assignments.csv")
    submap_frames: dict[int, list[int]] = {}
    for idx in common_ids:
        if idx not in assignments:
            continue
        submap_frames.setdefault(assignments[idx], []).append(idx)

    submap_rows: list[dict[str, object]] = []
    for submap_id, frames in sorted(submap_frames.items()):
        vals = [errors_cm[idx] for idx in frames]
        submap_rows.append(
            {
                "submap_id": submap_id,
                "frames": len(frames),
                "start_frame": min(frames),
                "end_frame": max(frames),
                "ate_mean_cm": mean(vals),
                "ate_rmse_cm": rmse(vals),
                "ate_p95_cm": percentile(vals, 95.0),
            }
        )

    transitions = read_transitions(run_dir / "submap_transitions.csv")
    sequential_by_boundary: dict[tuple[int, int, int], dict[str, str]] = {}
    for row in transitions:
        if row.get("event") != "sequential_transform":
            continue
        frame = safe_int(row.get("frame"))
        src = safe_int(row.get("primary_local_map"))
        dst = safe_int(row.get("new_local_map"))
        if frame >= 0 and src >= 0 and dst >= 0:
            sequential_by_boundary[(frame, src, dst)] = row

    boundary_rows: list[dict[str, object]] = []
    for row in transitions:
        if row.get("move_to_primary") != "1":
            continue
        frame = safe_int(row.get("frame"))
        if frame < 0:
            continue
        from_submap = safe_int(row.get("primary_local_map"))
        to_submap = safe_int(row.get("new_local_map"))
        sequential_row = sequential_by_boundary.get((frame, from_submap, to_submap), {})

        before_ids = window_ids(common_ids, frame, before=True, radius=window)
        after_ids = window_ids(common_ids, frame, before=False, radius=window)
        before_vals = [errors_cm[idx] for idx in before_ids]
        after_vals = [errors_cm[idx] for idx in after_ids]

        step_trans_cm = float("nan")
        step_rot_deg = float("nan")
        prev_frame = frame - 1
        if prev_frame in aligned_est and frame in aligned_est and prev_frame in gt_poses and frame in gt_poses:
            est_rel = np.linalg.inv(aligned_est[prev_frame]) @ aligned_est[frame]
            gt_rel = np.linalg.inv(gt_poses[prev_frame]) @ gt_poses[frame]
            err = np.linalg.inv(gt_rel) @ est_rel
            step_trans_cm = float(np.linalg.norm(err[:3, 3]) * 100.0)
            step_rot_deg = rotation_angle_deg(err[:3, :3])

        boundary_rows.append(
            {
                "frame": frame,
                "from_submap": from_submap,
                "to_submap": to_submap,
                "event": row.get("event", ""),
                "action": row.get("action", ""),
                "force_handoff": row.get("force_handoff", ""),
                "sequential_action": sequential_row.get("action", ""),
                "sequential_inliers": safe_int(sequential_row.get("inliers")),
                "internal_delta_cm": safe_float(sequential_row.get("delta_m")) * 100.0,
                "internal_delta_deg": safe_float(sequential_row.get("delta_deg")),
                "before_mean_ate_cm": mean(before_vals),
                "after_mean_ate_cm": mean(after_vals),
                "after_minus_before_cm": mean(after_vals) - mean(before_vals),
                "boundary_step_trans_cm": step_trans_cm,
                "boundary_step_rot_deg": step_rot_deg,
                "before_count": len(before_vals),
                "after_count": len(after_vals),
            }
        )

    overall_values = [errors_cm[idx] for idx in common_ids]
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "est_dir": str(est_dir),
        "gt_dir": str(gt_dir),
        "matched_poses": len(common_ids),
        "assigned_poses": sum(len(frames) for frames in submap_frames.values()),
        "submaps": len(submap_rows),
        "handoff_boundaries": len(boundary_rows),
        "ate_mean_cm": mean(overall_values),
        "ate_rmse_cm": rmse(overall_values),
        "ate_p95_cm": percentile(overall_values, 95.0),
        "boundary_window": window,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "submap_handoff_audit.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (output_dir / "submap_handoff_audit_submaps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(submap_rows[0].keys()) if submap_rows else ["submap_id"])
        writer.writeheader()
        writer.writerows(submap_rows)

    with (output_dir / "submap_handoff_audit_boundaries.csv").open("w", newline="") as f:
        fieldnames = list(boundary_rows[0].keys()) if boundary_rows else ["frame"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(boundary_rows)

    return summary


def main() -> None:
    parser = ArgumentParser(description="Analyze GPS-SLAM submap handoff drift against GT poses")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--gt_path", required=True, type=Path)
    parser.add_argument("--est_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    est_dir = args.est_path if args.est_path is not None else args.run_dir / "val" / "pose"
    output_dir = args.output_dir if args.output_dir is not None else args.run_dir
    summary = analyze(args.run_dir, est_dir, args.gt_path, output_dir, max(1, args.window))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
