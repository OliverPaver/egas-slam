#!/usr/bin/env python3
"""Recompute legacy ATE mean and actual ATE RMSE for saved pose folders.

This script does not rewrite ``pose_eval.txt``. It is meant for auditing old
run folders after the project started reporting the true RMSE alongside the
legacy mean-translation value.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


def pose_id(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def read_poses(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    if not path.exists():
        return poses
    for file_path in sorted(path.glob("*.txt")):
        idx = pose_id(file_path)
        if idx is None:
            continue
        try:
            pose = np.loadtxt(file_path).astype(np.float64)
        except Exception:
            continue
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            poses[idx] = pose
    return poses


def align_positions(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est_centered = est - est.mean(axis=1, keepdims=True)
    gt_centered = gt - gt.mean(axis=1, keepdims=True)
    w = est_centered @ gt_centered.T
    u, _, vh = np.linalg.svd(w.T)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0.0:
        s[2, 2] = -1.0
    rot = u @ s @ vh
    trans = gt.mean(axis=1, keepdims=True) - rot @ est.mean(axis=1, keepdims=True)
    return rot, trans


def ate_metrics(est_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray]) -> dict[str, float | int | str]:
    common = sorted(set(est_poses) & set(gt_poses))
    out: dict[str, float | int | str] = {
        "matched_poses": len(common),
        "est_poses": len(est_poses),
        "gt_poses": len(gt_poses),
    }
    if not common:
        out["error"] = "no_matching_poses"
        return out
    est_xyz = np.stack([est_poses[idx][:3, 3] for idx in common], axis=1)
    gt_xyz = np.stack([gt_poses[idx][:3, 3] for idx in common], axis=1)
    rot, trans = align_positions(est_xyz, gt_xyz)
    aligned = rot @ est_xyz + trans
    errors = np.linalg.norm(aligned - gt_xyz, axis=0)
    out.update(
        {
            "ate_mean_cm": float(errors.mean() * 100.0),
            "ate_actual_rmse_cm": float(np.sqrt(np.mean(np.square(errors))) * 100.0),
            "ate_median_cm": float(np.median(errors) * 100.0),
            "ate_p95_cm": float(np.percentile(errors, 95) * 100.0),
            "ate_max_cm": float(errors.max() * 100.0),
        }
    )
    return out


def run_config(run_dir: Path) -> dict:
    for path in sorted(run_dir.glob("*.yaml")):
        try:
            loaded = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def infer_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        path = path.parent
    if path.name in {"pose", "corrected_pose"}:
        return path.parent.parent
    if path.name == "val":
        return path.parent
    return path


def infer_gt_path(pose_dir: Path, explicit_gt: Path | None) -> Path | None:
    if explicit_gt is not None:
        return explicit_gt
    cfg = run_config(infer_run_dir(pose_dir))
    reader = cfg.get("READER", {}) if isinstance(cfg, dict) else {}
    input_dir = reader.get("input_dir")
    pose_path = reader.get("pose_path", "camera")
    if input_dir:
        candidate = Path(input_dir) / pose_path
        if candidate.exists():
            return candidate
    return None


def pose_dirs_from_run(run_dir: Path) -> list[Path]:
    candidates = [run_dir / "val" / "pose", run_dir / "val" / "corrected_pose"]
    return [path for path in candidates if path.exists()]


def discover_pose_dirs(paths: Iterable[Path]) -> list[Path]:
    pose_dirs: list[Path] = []
    for path in paths:
        if path.is_dir() and path.name in {"pose", "corrected_pose"}:
            pose_dirs.append(path)
        elif path.is_dir() and (path / "val").exists():
            pose_dirs.extend(pose_dirs_from_run(path))
        elif path.is_dir():
            pose_dirs.extend(sorted(path.rglob("val/pose")))
            pose_dirs.extend(sorted(path.rglob("val/corrected_pose")))
    return sorted(set(pose_dirs))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Run dirs, roots, or pose dirs")
    parser.add_argument("--gt-path", type=Path, default=None, help="Ground-truth pose directory; otherwise inferred from the run YAML")
    parser.add_argument("--pose-dir", action="append", type=Path, default=[], help="Additional explicit pose directory")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    pose_dirs = discover_pose_dirs([*args.paths, *args.pose_dir])
    if not pose_dirs:
        raise SystemExit("No val/pose or val/corrected_pose folders found")

    rows: list[dict[str, object]] = []
    for pose_dir in pose_dirs:
        gt_path = infer_gt_path(pose_dir, args.gt_path)
        row: dict[str, object] = {
            "run_dir": str(infer_run_dir(pose_dir)),
            "pose_dir": str(pose_dir),
            "pose_kind": pose_dir.name,
            "gt_path": str(gt_path) if gt_path else "",
        }
        if gt_path is None:
            row["error"] = "missing_gt_path"
        else:
            row.update(ate_metrics(read_poses(pose_dir), read_poses(gt_path)))
        rows.append(row)

    if args.output_csv:
        write_csv(args.output_csv, rows)

    fields = [
        "run_dir",
        "pose_kind",
        "matched_poses",
        "ate_mean_cm",
        "ate_actual_rmse_cm",
        "ate_p95_cm",
        "ate_max_cm",
        "error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
