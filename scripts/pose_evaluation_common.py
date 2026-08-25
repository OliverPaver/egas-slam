"""Strict pose loading and pairing shared by ATE and RPE tools."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def pose_id(path: str | Path) -> int | None:
    match = re.search(r"(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def read_poses(pose_dir: str | Path, *, label: str) -> dict[int, np.ndarray]:
    pose_dir = Path(pose_dir)
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"{label} pose directory does not exist: {pose_dir}")

    poses: dict[int, np.ndarray] = {}
    for path in sorted(pose_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        idx = pose_id(path)
        if idx is None:
            # Replica camera folders include metadata such as img_shape.txt.
            # Only numbered text files are pose candidates.
            continue
        if idx in poses:
            raise ValueError(f"Duplicate {label} pose frame ID {idx}: {path}")
        try:
            pose = np.loadtxt(path, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"Unable to read {label} pose {path}: {exc}") from exc
        if pose.shape != (4, 4):
            raise ValueError(f"{label} pose must be 4x4, got {pose.shape}: {path}")
        if not np.isfinite(pose).all():
            raise ValueError(f"{label} pose contains NaN or infinity: {path}")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5, rtol=0.0):
            raise ValueError(f"{label} pose has an invalid homogeneous row: {path}")
        poses[idx] = pose
    if not poses:
        raise RuntimeError(f"No valid {label} pose files found in {pose_dir}")
    return poses


def matched_pose_ids(
    est_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    *,
    allow_partial: bool,
) -> tuple[list[int], dict[str, int | float]]:
    common_ids = sorted(set(est_poses) & set(gt_poses))
    missing_gt_ids = sorted(set(est_poses) - set(gt_poses))
    if missing_gt_ids and not allow_partial:
        preview = ", ".join(str(idx) for idx in missing_gt_ids[:10])
        suffix = " ..." if len(missing_gt_ids) > 10 else ""
        raise RuntimeError(
            f"{len(missing_gt_ids)} of {len(est_poses)} estimated poses lack ground truth "
            f"(IDs: {preview}{suffix})"
        )
    if not common_ids:
        raise RuntimeError("No matching estimated and ground-truth poses")
    return common_ids, {
        "matched_poses": len(common_ids),
        "estimated_poses": len(est_poses),
        "ground_truth_poses": len(gt_poses),
        "missing_ground_truth_poses": len(missing_gt_ids),
        "estimated_pose_coverage": len(common_ids) / len(est_poses),
    }
