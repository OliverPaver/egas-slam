#!/usr/bin/env python3
"""Download and convert TUM RGB-D sequences into GPS-SLAM layout.

The output layout matches DatasetReader:
  camera/frame000000.png
  camera/pose000000.txt
  depth/depth000000.png
  metadata.json

RGB/depth/GT are associated by timestamp. Poses are interpolated at the RGB
timestamp with linear translation and quaternion slerp.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    Path(os.environ.get("GPS_SLAM_DATA_ROOT", PROJECT_ROOT / "data")) / "tum_rgbd"
)

PAPER_SEQUENCES = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg2_xyz",
    "rgbd_dataset_freiburg3_long_office_household",
]

LOOP_SEQUENCES = [
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_pioneer_slam",
    "rgbd_dataset_freiburg2_pioneer_slam2",
]

INTRINSICS = {
    "freiburg1": [517.3, 516.5, 318.6, 255.3],
    "freiburg2": [520.9, 521.0, 325.1, 249.7],
    "freiburg3": [535.4, 539.2, 320.1, 247.6],
}


def sequence_family(sequence: str) -> str:
    for family in INTRINSICS:
        if family in sequence:
            return family
    raise ValueError(f"Cannot infer Freiburg camera family from {sequence}")


def sequence_url(sequence: str) -> str:
    family = sequence_family(sequence)
    return f"https://cvg.cit.tum.de/rgbd/dataset/{family}/{sequence}.tgz"


def parse_list(path: Path) -> List[Tuple[float, List[str]]]:
    rows: List[Tuple[float, List[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((float(parts[0]), parts[1:]))
    return rows


def one_to_one_by_timestamp(
    source: Iterable[Tuple[float, List[str]]],
    target: List[Tuple[float, List[str]]],
    max_delta: float,
) -> List[Tuple[Tuple[float, List[str]], Tuple[float, List[str]], float]]:
    """Closest one-to-one association compatible with TUM's associate.py policy."""
    source_sorted = sorted(source, key=lambda item: item[0])
    target_sorted = sorted(target, key=lambda item: item[0])
    potential: List[Tuple[float, int, int]] = []
    j0 = 0
    for i, item in enumerate(source_sorted):
        ts = item[0]
        while j0 < len(target_sorted) and target_sorted[j0][0] < ts - max_delta:
            j0 += 1
        j = j0
        while j < len(target_sorted) and target_sorted[j][0] <= ts + max_delta:
            potential.append((abs(target_sorted[j][0] - ts), i, j))
            j += 1

    potential.sort(key=lambda item: item[0])
    used_source = set()
    used_target = set()
    matches: List[Tuple[Tuple[float, List[str]], Tuple[float, List[str]], float]] = []
    for delta, i, j in potential:
        if i in used_source or j in used_target:
            continue
        used_source.add(i)
        used_target.add(j)
        matches.append((source_sorted[i], target_sorted[j], delta))
    matches.sort(key=lambda item: item[0][0])
    return matches


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0 or not np.isfinite(n):
        raise ValueError("Invalid zero/non-finite quaternion")
    return q / n


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quat(q0 + alpha * (q1 - q0))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    return (math.sin(theta_0 - theta) / sin_theta_0) * q0 + (math.sin(theta) / sin_theta_0) * q1


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def interpolate_pose(gt_rows: List[Tuple[float, List[str]]], timestamp: float) -> Optional[np.ndarray]:
    if not gt_rows:
        return None
    rows = sorted(gt_rows, key=lambda item: item[0])
    if timestamp < rows[0][0] or timestamp > rows[-1][0]:
        return None
    lo = 0
    hi = len(rows) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] <= timestamp:
            lo = mid
        else:
            hi = mid
    t0, p0 = rows[lo]
    t1, p1 = rows[hi]
    vals0 = np.array([float(v) for v in p0], dtype=np.float64)
    vals1 = np.array([float(v) for v in p1], dtype=np.float64)
    if vals0.shape[0] != 7 or vals1.shape[0] != 7:
        return None
    alpha = 0.0 if t1 == t0 else (timestamp - t0) / (t1 - t0)
    trans = vals0[:3] + alpha * (vals1[:3] - vals0[:3])
    quat = slerp(vals0[3:7], vals1[3:7], alpha)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quat_xyzw_to_rot(quat)
    pose[:3, 3] = trans
    if not np.isfinite(pose).all():
        return None
    return pose


def choose_data_root(root: Optional[Path]) -> Path:
    return root if root is not None else DEFAULT_DATA_ROOT


def download(sequence: str, raw_root: Path) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    archive = raw_root / f"{sequence}.tgz"
    seq_dir = raw_root / sequence
    if seq_dir.exists():
        return seq_dir
    if not archive.exists():
        print(f"[TUM] Download {sequence}: {sequence_url(sequence)}")
        urllib.request.urlretrieve(sequence_url(sequence), archive)
    print(f"[TUM] Extract {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(raw_root)
    if not seq_dir.exists():
        raise RuntimeError(f"Extracted archive did not create {seq_dir}")
    return seq_dir


def verify_sequence(processed_dir: Path) -> Dict[str, object]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    camera_dir = processed_dir / "camera"
    depth_dir = processed_dir / "depth"
    frame_count = int(metadata["frames"])
    expected_shape = metadata.get("image_shape", [640, 480])
    expected_width, expected_height = int(expected_shape[0]), int(expected_shape[1])
    records = metadata.get("frame_records", [])
    if len(records) != frame_count:
        raise RuntimeError(f"{processed_dir.name}: metadata frame_records length does not match frames")

    rgb_sources = []
    depth_sources = []
    max_pose_abs = 0.0
    min_depth_value = None
    max_depth_value = None
    for idx in range(frame_count):
        frame_id = f"{idx:06d}"
        rgb_path = camera_dir / f"frame{frame_id}.png"
        depth_path = depth_dir / f"depth{frame_id}.png"
        pose_path = camera_dir / f"pose{frame_id}.txt"
        if not rgb_path.exists():
            raise RuntimeError(f"{processed_dir.name}: missing RGB {rgb_path}")
        if not depth_path.exists():
            raise RuntimeError(f"{processed_dir.name}: missing depth {depth_path}")
        if not pose_path.exists():
            raise RuntimeError(f"{processed_dir.name}: missing pose {pose_path}")

        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise RuntimeError(f"{processed_dir.name}: invalid pose {pose_path}")
        if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
            raise RuntimeError(f"{processed_dir.name}: invalid homogeneous pose row {pose_path}")
        max_pose_abs = max(max_pose_abs, float(np.max(np.abs(pose))))

        record = records[idx]
        if int(record["frame_id"]) != idx:
            raise RuntimeError(f"{processed_dir.name}: non-monotonic frame record at {idx}")
        rgb_sources.append(str(Path(record["rgb_source"]).resolve()))
        depth_sources.append(str(Path(record["depth_source"]).resolve()))

    if len(set(rgb_sources)) != len(rgb_sources):
        raise RuntimeError(f"{processed_dir.name}: duplicate raw RGB source in processed sequence")
    if len(set(depth_sources)) != len(depth_sources):
        raise RuntimeError(f"{processed_dir.name}: duplicate raw depth source in processed sequence")

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for TUM audit image checks") from exc

    sample_indices = sorted(set([0, frame_count // 2, max(0, frame_count - 1)]))
    for idx in sample_indices:
        frame_id = f"{idx:06d}"
        rgb = cv2.imread(str(camera_dir / f"frame{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_dir / f"depth{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise RuntimeError(f"{processed_dir.name}: OpenCV could not read frame {frame_id}")
        if rgb.shape[1] != expected_width or rgb.shape[0] != expected_height:
            raise RuntimeError(f"{processed_dir.name}: RGB shape changed for frame {frame_id}: {rgb.shape}")
        if depth.shape[1] != expected_width or depth.shape[0] != expected_height:
            raise RuntimeError(f"{processed_dir.name}: depth shape changed for frame {frame_id}: {depth.shape}")
        valid_depth = depth[depth > 0]
        if valid_depth.size:
            dmin = float(valid_depth.min()) / float(metadata["depth_scale"])
            dmax = float(valid_depth.max()) / float(metadata["depth_scale"])
            min_depth_value = dmin if min_depth_value is None else min(min_depth_value, dmin)
            max_depth_value = dmax if max_depth_value is None else max(max_depth_value, dmax)

    fairness = metadata.get("fairness_policy", {})
    required_flags = [
        "rgb_preserved_as_png",
        "depth_preserved_as_png",
        "no_denoising",
        "no_cropping",
        "no_resizing",
        "no_colour_correction",
        "no_depth_filtering",
        "association_only",
    ]
    failed_flags = [name for name in required_flags if fairness.get(name) is not True]
    if failed_flags:
        raise RuntimeError(f"{processed_dir.name}: fairness flags failed: {failed_flags}")

    source = metadata.get("source", {})
    if source.get("association_policy") != "one_to_one_closest_timestamp":
        raise RuntimeError(f"{processed_dir.name}: unexpected association policy {source.get('association_policy')}")
    association = metadata.get("association", {})
    if association.get("rgb_depth_max_delta") is None or float(association["rgb_depth_max_delta"]) > float(association["max_delta"]) + 1e-9:
        raise RuntimeError(f"{processed_dir.name}: RGB/depth association exceeds configured max delta")

    return {
        "sequence": processed_dir.name,
        "frames": frame_count,
        "image_shape": [expected_width, expected_height],
        "intrinsics": metadata.get("intrinsics"),
        "image_policy": source.get("image_policy"),
        "association_policy": source.get("association_policy"),
        "rgb_depth_mean_delta": association.get("rgb_depth_mean_delta"),
        "rgb_depth_max_delta": association.get("rgb_depth_max_delta"),
        "rgb_gt_mean_nearest_delta": association.get("rgb_gt_mean_nearest_delta"),
        "rgb_gt_max_nearest_delta": association.get("rgb_gt_max_nearest_delta"),
        "unique_rgb_sources": len(set(rgb_sources)),
        "unique_depth_sources": len(set(depth_sources)),
        "max_pose_abs": max_pose_abs,
        "sample_min_depth_m": min_depth_value,
        "sample_max_depth_m": max_depth_value,
        "fairness_policy": fairness,
    }


def convert_sequence(seq_dir: Path, out_dir: Path, max_assoc_delta: float, copy: bool) -> Dict[str, object]:
    rgb_rows = parse_list(seq_dir / "rgb.txt")
    depth_rows = parse_list(seq_dir / "depth.txt")
    gt_rows = parse_list(seq_dir / "groundtruth.txt")
    associations = one_to_one_by_timestamp(rgb_rows, depth_rows, max_assoc_delta)

    camera_dir = out_dir / "camera"
    depth_dir = out_dir / "depth"
    if camera_dir.exists():
        shutil.rmtree(camera_dir)
    if depth_dir.exists():
        shutil.rmtree(depth_dir)
    camera_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    rgb_depth_deltas: List[float] = []
    gt_deltas: List[float] = []
    frames = []
    skipped_no_gt = 0
    for rgb_entry, depth_entry, rgb_depth_delta in associations:
        rgb_ts, rgb_parts = rgb_entry
        depth_ts, depth_parts = depth_entry
        pose = interpolate_pose(gt_rows, rgb_ts)
        if pose is None:
            skipped_no_gt += 1
            continue
        nearest_gt_delta = min(abs(row[0] - rgb_ts) for row in gt_rows)
        frame_id = f"{kept:06d}"
        rgb_src = seq_dir / rgb_parts[0]
        depth_src = seq_dir / depth_parts[0]
        rgb_dst = camera_dir / f"frame{frame_id}.png"
        depth_dst = depth_dir / f"depth{frame_id}.png"
        if copy:
            shutil.copy2(rgb_src, rgb_dst)
            shutil.copy2(depth_src, depth_dst)
        else:
            if rgb_dst.exists() or rgb_dst.is_symlink():
                rgb_dst.unlink()
            if depth_dst.exists() or depth_dst.is_symlink():
                depth_dst.unlink()
            rgb_dst.symlink_to(rgb_src.resolve())
            depth_dst.symlink_to(depth_src.resolve())
        np.savetxt(camera_dir / f"pose{frame_id}.txt", pose, fmt="%.9f")
        frames.append(
            {
                "frame_id": kept,
                "rgb_timestamp": rgb_ts,
                "depth_timestamp": depth_ts,
                "rgb_depth_delta": rgb_depth_delta,
                "nearest_gt_delta": nearest_gt_delta,
                "rgb_source": str(rgb_src),
                "depth_source": str(depth_src),
            }
        )
        rgb_depth_deltas.append(rgb_depth_delta)
        gt_deltas.append(nearest_gt_delta)
        kept += 1

    family = sequence_family(seq_dir.name)
    metadata = {
        "dataset": "tum_rgbd",
        "sequence": seq_dir.name,
        "source_dir": str(seq_dir),
        "frames": kept,
        "image_extension": ".png",
        "depth_extension": ".png",
        "depth_scale": 5000.0,
        "image_shape": [640, 480],
        "intrinsics": INTRINSICS[family],
        "source": {
            "archive_url": sequence_url(seq_dir.name),
            "raw_rgb_rows": len(rgb_rows),
            "raw_depth_rows": len(depth_rows),
            "raw_groundtruth_rows": len(gt_rows),
            "image_policy": "copy_raw_png" if copy else "symlink_raw_png",
            "association_policy": "one_to_one_closest_timestamp",
            "rgb_depth_associations": len(associations),
            "skipped_without_interpolated_gt": skipped_no_gt,
            "kept_frames": kept,
        },
        "fairness_policy": {
            "rgb_preserved_as_png": True,
            "depth_preserved_as_png": True,
            "no_denoising": True,
            "no_cropping": True,
            "no_resizing": True,
            "no_colour_correction": True,
            "no_depth_filtering": True,
            "association_only": True,
        },
        "association": {
            "max_delta": max_assoc_delta,
            "rgb_depth_mean_delta": float(np.mean(rgb_depth_deltas)) if rgb_depth_deltas else None,
            "rgb_depth_max_delta": float(np.max(rgb_depth_deltas)) if rgb_depth_deltas else None,
            "rgb_gt_mean_nearest_delta": float(np.mean(gt_deltas)) if gt_deltas else None,
            "rgb_gt_max_nearest_delta": float(np.max(gt_deltas)) if gt_deltas else None,
        },
        "frame_records": frames,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (out_dir / "splits.txt").open("w", encoding="utf-8") as handle:
        for frame in range(kept):
            handle.write(f"{frame:06d}\n")
    print(f"[TUM] Converted {seq_dir.name}: {kept} frames -> {out_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TUM RGB-D for GPS-SLAM")
    parser.add_argument("--data-root", type=Path, default=None, help="Root containing raw/processed folders")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--sequences", nargs="*", default=PAPER_SEQUENCES + LOOP_SEQUENCES)
    parser.add_argument("--paper-only", action="store_true", help="Use only the three GPS-SLAM paper TUM ATE sequences")
    parser.add_argument("--loop-only", action="store_true", help="Use only the secondary loop-heavy TUM validation sequences")
    parser.add_argument("--download", action="store_true", help="Download missing archives from TUM")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--max-assoc-delta", type=float, default=0.02)
    parser.add_argument("--verify-only", action="store_true", help="Only audit already processed sequences")
    parser.add_argument("--audit-output", type=Path, default=None, help="Optional JSON path for the fairness audit summary")
    args = parser.parse_args()

    if args.paper_only and args.loop_only:
        raise RuntimeError("--paper-only and --loop-only are mutually exclusive")
    if args.paper_only:
        args.sequences = PAPER_SEQUENCES
    elif args.loop_only:
        args.sequences = LOOP_SEQUENCES

    data_root = choose_data_root(args.data_root)
    raw_root = args.raw_root or data_root / "raw"
    processed_root = args.processed_root or data_root / "processed"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    if not args.verify_only:
        for sequence in args.sequences:
            seq_dir = raw_root / sequence
            if args.download and not seq_dir.exists():
                seq_dir = download(sequence, raw_root)
            if not seq_dir.exists():
                raise FileNotFoundError(f"Missing raw TUM sequence {seq_dir}; rerun with --download")
            summary[sequence] = convert_sequence(
                seq_dir,
                processed_root / sequence,
                max_assoc_delta=args.max_assoc_delta,
                copy=args.copy,
            )

    if summary:
        summary_path = processed_root / "tum_preprocess_summary.json"
        existing_summary = {}
        if summary_path.exists():
            try:
                existing_summary = json.loads(summary_path.read_text())
            except Exception:
                existing_summary = {}
        existing_summary.update(summary)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(existing_summary, handle, indent=2)

    audit = {}
    for sequence in args.sequences:
        audit[sequence] = verify_sequence(processed_root / sequence)
        print(
            f"[TUM AUDIT] {sequence}: frames={audit[sequence]['frames']} "
            f"rgb_depth_max={audit[sequence]['rgb_depth_max_delta']:.6f}s "
            f"policy={audit[sequence]['association_policy']} "
            f"image_policy={audit[sequence]['image_policy']}"
        )
    audit_path = args.audit_output or processed_root / "tum_fairness_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(f"[TUM AUDIT] wrote {audit_path}")


if __name__ == "__main__":
    main()
