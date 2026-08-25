#!/usr/bin/env python3
"""Compute rigidly aligned absolute trajectory error for pose folders."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

try:
    from .evaluation_common import atomic_write_json, write_status_error
    from .pose_evaluation_common import matched_pose_ids, read_poses
except ImportError:
    from evaluation_common import atomic_write_json, write_status_error
    from pose_evaluation_common import matched_pose_ids, read_poses


def readPoses(est_dir, gt_dir):
    """Compatibility wrapper returning pose lists in matched frame order."""
    est_by_id = read_poses(est_dir, label="estimated")
    gt_by_id = read_poses(gt_dir, label="ground-truth")
    common_ids, _ = matched_pose_ids(est_by_id, gt_by_id, allow_partial=False)
    return (
        [est_by_id[idx] for idx in common_ids],
        [gt_by_id[idx] for idx in common_ids],
        common_ids,
    )


def align(model: np.ndarray, data: np.ndarray):
    """Rigidly align 3xN ``model`` points to 3xN ``data`` points."""
    if model.shape != data.shape or model.ndim != 2 or model.shape[0] != 3:
        raise ValueError(f"Trajectories must both be 3xN, got {model.shape} and {data.shape}")
    if not np.isfinite(model).all() or not np.isfinite(data).all():
        raise ValueError("Trajectory contains NaN or infinity")

    model_center = model.mean(axis=1, keepdims=True)
    data_center = data.mean(axis=1, keepdims=True)
    model_zero = model - model_center
    data_zero = data - data_center
    covariance = model_zero @ data_zero.T
    u, _, vh = np.linalg.svd(covariance.T)
    reflection = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection[2, 2] = -1.0
    rotation = u @ reflection @ vh
    translation = data_center - rotation @ model_center
    aligned = rotation @ model + translation
    translation_error = np.linalg.norm(aligned - data, axis=0)
    return rotation, translation, translation_error


def output_paths(est_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(est_dir).parent
    return output_dir / "pose_eval.txt", output_dir / "pose_eval.json"


def evaluate(
    est_dir: str | Path,
    gt_dir: str | Path,
    *,
    allow_partial: bool = False,
    min_poses: int = 2,
) -> dict[str, float | int | str]:
    if min_poses < 2:
        raise ValueError("min_poses must be at least 2")
    print("Processing poses from:", est_dir, gt_dir)
    est_poses = read_poses(est_dir, label="estimated")
    gt_poses = read_poses(gt_dir, label="ground-truth")
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
    if len(common_ids) < min_poses:
        raise RuntimeError(
            f"Need at least {min_poses} matching poses for ATE, found {len(common_ids)}"
        )

    est_xyz = np.stack([est_poses[idx][:3, 3] for idx in common_ids], axis=1)
    gt_xyz = np.stack([gt_poses[idx][:3, 3] for idx in common_ids], axis=1)
    _, _, translation_error = align(est_xyz, gt_xyz)
    mean_cm = float(translation_error.mean() * 100.0)
    rmse_cm = float(np.sqrt(np.mean(np.square(translation_error))) * 100.0)

    # The old script called the arithmetic mean "ATE RMSE". Keep the value but
    # label it truthfully; repository runners consume the actual-RMSE line.
    print(f"ATE mean translation error: {mean_cm:.8f}")
    print(f"ATE actual RMSE: {rmse_cm:.8f}")
    text_path, json_path = output_paths(est_dir)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        f"ATE mean translation error: {mean_cm:.8f}\n"
        f"ATE actual RMSE: {rmse_cm:.8f}\n",
        encoding="utf-8",
    )
    results: dict[str, float | int | str] = {
        "status": "ok",
        "ate_mean_translation_error_cm": mean_cm,
        "ate_rmse_cm": rmse_cm,
        "units": "centimetres",
        **coverage,
    }
    atomic_write_json(json_path, results)
    return results


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gt_path", required=True, type=Path)
    parser.add_argument("--est_path", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="evaluate matching poses even if an estimated frame lacks ground truth",
    )
    parser.add_argument("--min-poses", type=int, default=2)
    args = parser.parse_args()

    text_path, json_path = output_paths(args.est_path)
    text_path.unlink(missing_ok=True)
    try:
        evaluate(
            args.est_path,
            args.gt_path,
            allow_partial=args.allow_partial,
            min_poses=args.min_poses,
        )
    except Exception as exc:
        text_path.unlink(missing_ok=True)
        if json_path.parent.is_dir():
            write_status_error(
                json_path,
                exc,
                estimated_path=str(args.est_path),
                ground_truth_path=str(args.gt_path),
            )
        print(f"ATE evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
