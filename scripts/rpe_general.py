#!/usr/bin/env python3
"""Compute relative pose error for GPS-SLAM pose folders."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

try:
    from .evaluation_common import atomic_write_json, write_status_error
    from .pose_evaluation_common import (
        matched_pose_ids,
        pose_id,
        read_poses as _read_poses,
    )
except ImportError:
    from evaluation_common import atomic_write_json, write_status_error
    from pose_evaluation_common import (
        matched_pose_ids,
        pose_id,
        read_poses as _read_poses,
    )


def read_poses(pose_dir):
    """Compatibility wrapper for callers of the historical helper."""
    return _read_poses(pose_dir, label="pose")


def align_positions(est_poses, gt_poses, common_ids):
    est = np.stack([est_poses[idx][:3, 3] for idx in common_ids], axis=1)
    gt = np.stack([gt_poses[idx][:3, 3] for idx in common_ids], axis=1)
    est_centered = est - est.mean(axis=1, keepdims=True)
    gt_centered = gt - gt.mean(axis=1, keepdims=True)
    covariance = est_centered @ gt_centered.T
    u, _, vh = np.linalg.svd(covariance.T)
    reflection = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection[2, 2] = -1.0
    rotation = u @ reflection @ vh
    translation = gt.mean(axis=1, keepdims=True) - rotation @ est.mean(axis=1, keepdims=True)

    aligned = {}
    for idx, pose in est_poses.items():
        out = pose.copy()
        out[:3, :3] = rotation @ out[:3, :3]
        out[:3, 3:4] = rotation @ out[:3, 3:4] + translation
        aligned[idx] = out
    return aligned


def rotation_angle_deg(rotation):
    cos_angle = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def output_paths(est_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(est_dir).parent
    return output_dir / "rpe_eval.txt", output_dir / "rpe_eval.json"


def evaluate(est_dir, gt_dir, delta, align, *, allow_partial=False):
    if delta <= 0:
        raise ValueError("RPE delta must be a positive frame-ID offset")
    print("Processing poses from:", est_dir, gt_dir)
    est_poses = _read_poses(Path(est_dir), label="estimated")
    gt_poses = _read_poses(Path(gt_dir), label="ground-truth")
    common_ids, coverage = matched_pose_ids(
        est_poses,
        gt_poses,
        allow_partial=allow_partial,
    )
    print(
        f"matched poses: {len(common_ids)} "
        f"(est={len(est_poses)}, gt={len(gt_poses)}, "
        f"est_coverage={coverage['estimated_pose_coverage']:.6f})"
    )
    if len(common_ids) < 2:
        raise RuntimeError("Need at least two matching poses for RPE")

    if align:
        est_poses = align_positions(est_poses, gt_poses, common_ids)

    trans_errors = []
    rot_errors = []
    common_set = set(common_ids)
    for idx in common_ids:
        paired_idx = idx + delta
        if paired_idx not in common_set:
            continue
        est_rel = np.linalg.inv(est_poses[idx]) @ est_poses[paired_idx]
        gt_rel = np.linalg.inv(gt_poses[idx]) @ gt_poses[paired_idx]
        error = np.linalg.inv(gt_rel) @ est_rel
        if not np.isfinite(error).all():
            raise RuntimeError(f"Non-finite RPE transform for pair {idx}->{paired_idx}")
        trans_errors.append(float(np.linalg.norm(error[:3, 3])))
        rot_errors.append(rotation_angle_deg(error[:3, :3]))

    pairs = len(trans_errors)
    if pairs == 0:
        raise RuntimeError("No valid pose pairs for requested RPE delta")

    trans_array = np.asarray(trans_errors, dtype=np.float64)
    rot_array = np.asarray(rot_errors, dtype=np.float64)
    trans_rmse_cm = float(np.sqrt(np.mean(np.square(trans_array))) * 100.0)
    rot_rmse_deg = float(np.sqrt(np.mean(np.square(rot_array))))
    print(f"RPE pairs: {pairs}")
    print(f"RPE translation RMSE: {trans_rmse_cm:.8f} cm")
    print(f"RPE rotation RMSE: {rot_rmse_deg:.8f} deg")

    text_path, json_path = output_paths(est_dir)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        f"RPE pairs: {pairs}\n"
        f"RPE translation RMSE: {trans_rmse_cm:.8f} cm\n"
        f"RPE rotation RMSE: {rot_rmse_deg:.8f} deg\n",
        encoding="utf-8",
    )
    results = {
        "status": "ok",
        "delta_frame_ids": delta,
        "aligned": bool(align),
        "rpe_pairs": pairs,
        "rpe_translation_rmse_cm": trans_rmse_cm,
        "rpe_rotation_rmse_deg": rot_rmse_deg,
        "translation_units": "centimetres",
        "rotation_units": "degrees",
        **coverage,
    }
    atomic_write_json(json_path, results)
    return results


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gt_path", required=True, type=Path)
    parser.add_argument("--est_path", required=True, type=Path)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="evaluate matching poses even if an estimated frame lacks ground truth",
    )
    args = parser.parse_args()

    text_path, json_path = output_paths(args.est_path)
    text_path.unlink(missing_ok=True)
    try:
        evaluate(
            args.est_path,
            args.gt_path,
            args.delta,
            not args.no_align,
            allow_partial=args.allow_partial,
        )
    except Exception as exc:
        text_path.unlink(missing_ok=True)
        if json_path.parent.is_dir():
            write_status_error(
                json_path,
                exc,
                estimated_path=str(args.est_path),
                ground_truth_path=str(args.gt_path),
                delta_frame_ids=args.delta,
            )
        print(f"RPE evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
