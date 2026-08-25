#!/usr/bin/env python3
"""Report ground-truth revisit statistics for processed TUM RGB-D sequences.

This script is for dataset characterization only. Do not feed its output back
into the online SLAM run or loop-candidate selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional convenience for run inference.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GPS_SLAM_DATA_ROOT", PROJECT_ROOT / "data"))
DEFAULT_ROOTS = [
    DATA_ROOT / "tum_rgbd" / "processed",
    DATA_ROOT / "tum_rgbd_local" / "processed",
]

SEQUENCES = {
    "fr1_desk": "rgbd_dataset_freiburg1_desk",
    "fr2_xyz": "rgbd_dataset_freiburg2_xyz",
    "fr3_office": "rgbd_dataset_freiburg3_long_office_household",
    "fr1_room": "rgbd_dataset_freiburg1_room",
    "fr2_pioneer_slam": "rgbd_dataset_freiburg2_pioneer_slam",
    "fr2_pioneer_slam2": "rgbd_dataset_freiburg2_pioneer_slam2",
}


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse frame id from {path}")
    return int(match.group(1))


def read_poses(sequence_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pose_dir = sequence_dir / "camera"
    pose_paths = sorted(pose_dir.glob("pose*.txt"), key=frame_number)
    if not pose_paths:
        raise FileNotFoundError(f"No pose*.txt files in {pose_dir}")
    poses = []
    frame_ids = []
    for path in pose_paths:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"{path} is not a finite 4x4 pose")
        poses.append(pose)
        frame_ids.append(frame_number(path))
    return np.asarray(frame_ids, dtype=np.int64), np.asarray(poses, dtype=np.float64)


def safe_int(value: object, default: int = -1) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    return value_f if math.isfinite(value_f) else default


def pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def counts_to_text(counts: Counter[str]) -> str:
    return ";".join(
        f"{key}:{value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def rotation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    cos_angle = (float(np.trace(rel)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def count_revisit_runs(frame_ids: set[int], max_gap_frames: int) -> int:
    if not frame_ids:
        return 0
    runs = 0
    previous = None
    for frame_id in sorted(frame_ids):
        if previous is None or frame_id - previous > max_gap_frames:
            runs += 1
        previous = frame_id
    return runs


def audit_sequence(
    root: Path,
    sequence: str,
    thresholds: list[float],
    min_separation_frames: int,
    stride: int,
    rotation_threshold_deg: float,
) -> dict:
    sequence_dir = root / sequence
    metadata_path = sequence_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    frame_ids, poses = read_poses(sequence_dir)
    positions = poses[:, :3, 3]
    step_dist = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    sampled_indices = np.arange(0, len(poses), max(1, stride), dtype=np.int64)

    threshold_rows = {}
    for threshold in thresholds:
        pair_count = 0
        max_temporal_separation = 0
        first_pair = None
        last_pair = None
        unique_source_frames: set[int] = set()
        unique_revisit_frames: set[int] = set()
        for si in sampled_indices:
            min_j = si + min_separation_frames
            if min_j >= len(poses):
                continue
            for sj in sampled_indices:
                if sj < min_j:
                    continue
                distance = float(np.linalg.norm(positions[si] - positions[sj]))
                if distance > threshold:
                    continue
                rotation = rotation_degrees(poses[si], poses[sj])
                if rotation_threshold_deg > 0.0 and rotation > rotation_threshold_deg:
                    continue
                pair_count += 1
                temporal_sep = int(frame_ids[sj] - frame_ids[si])
                max_temporal_separation = max(max_temporal_separation, temporal_sep)
                unique_source_frames.add(int(frame_ids[si]))
                unique_revisit_frames.add(int(frame_ids[sj]))
                pair = {
                    "source_frame": int(frame_ids[si]),
                    "revisit_frame": int(frame_ids[sj]),
                    "distance_m": distance,
                    "rotation_deg": rotation,
                }
                if first_pair is None:
                    first_pair = pair
                last_pair = pair
        threshold_rows[f"{threshold:.2f}m"] = {
            "pair_count": pair_count,
            "unique_source_frames": len(unique_source_frames),
            "unique_revisit_frames": len(unique_revisit_frames),
            "revisit_episodes": count_revisit_runs(unique_revisit_frames, max(1, stride * 2)),
            "max_temporal_separation_frames": max_temporal_separation,
            "first_pair": first_pair,
            "last_pair": last_pair,
        }

    return {
        "sequence": sequence,
        "short_name": next((k for k, v in SEQUENCES.items() if v == sequence), sequence),
        "frames": int(len(poses)),
        "metadata_frames": metadata.get("frames"),
        "path_length_m": float(step_dist.sum()) if len(step_dist) else 0.0,
        "min_separation_frames": min_separation_frames,
        "stride": stride,
        "rotation_threshold_deg": rotation_threshold_deg,
        "thresholds": threshold_rows,
    }


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        threshold = float(part)
        if threshold <= 0.0:
            raise ValueError("thresholds must be positive")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def default_root() -> Path:
    for root in DEFAULT_ROOTS:
        if root.exists():
            return root
    return DEFAULT_ROOTS[0]


def resolve_sequence(sequence: str) -> str:
    return SEQUENCES.get(sequence, sequence)


def infer_run_sequence(run_dir: Path, tum_root: Path, explicit_sequence: str | None) -> str:
    if explicit_sequence:
        return resolve_sequence(explicit_sequence)

    if yaml is not None:
        for config_path in sorted(run_dir.glob("*.yaml")):
            try:
                cfg = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                continue
            input_dir = cfg.get("READER", {}).get("input_dir") if isinstance(cfg, dict) else None
            if input_dir:
                candidate = Path(str(input_dir)).name
                if (tum_root / candidate).exists():
                    return candidate

    run_name = run_dir.name
    for short_name, sequence in SEQUENCES.items():
        if short_name in run_name or sequence in run_name:
            return sequence
    raise RuntimeError(
        f"Could not infer TUM sequence for {run_dir}. Pass --run-sequence with a short or full sequence name."
    )


def read_frame_assignments(run_dir: Path) -> dict[int, int]:
    path = run_dir / "frame_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frame assignments: {path}")
    assignments: dict[int, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = safe_int(row.get("frame_id"), safe_int(row.get("sequence_id"), -1))
            primary = safe_int(row.get("primary_submap"), -1)
            if frame_id >= 0 and primary >= 0:
                assignments[frame_id] = primary
    if not assignments:
        raise RuntimeError(f"No valid frame/submap assignments in {path}")
    return assignments


def evenly_limited(values: list[int], max_count: int) -> list[int]:
    if max_count <= 0 or len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[len(values) // 2]]
    indices = np.linspace(0, len(values) - 1, max_count)
    selected = [values[int(round(idx))] for idx in indices]
    return sorted(set(selected))


def gt_submap_pairs(
    frame_ids: np.ndarray,
    poses: np.ndarray,
    frame_to_submap: dict[int, int],
    distance_threshold: float,
    rotation_threshold_deg: float,
    min_submap_span: int,
    max_samples_per_submap: int,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, object]]]:
    pose_by_frame = {int(frame_id): pose for frame_id, pose in zip(frame_ids, poses)}
    frames_by_submap: dict[int, list[int]] = {}
    for frame_id, submap_id in frame_to_submap.items():
        if frame_id in pose_by_frame:
            frames_by_submap.setdefault(submap_id, []).append(frame_id)
    for frames in frames_by_submap.values():
        frames.sort()

    rows: list[dict[str, object]] = []
    best_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    submap_ids = sorted(frames_by_submap)
    sampled_by_submap = {
        submap_id: evenly_limited(frames, max_samples_per_submap)
        for submap_id, frames in frames_by_submap.items()
    }
    for idx, submap_a in enumerate(submap_ids):
        frames_a = sampled_by_submap[submap_a]
        if not frames_a:
            continue
        for submap_b in submap_ids[idx + 1 :]:
            span = abs(submap_b - submap_a)
            if span < min_submap_span:
                continue
            frames_b = sampled_by_submap[submap_b]
            if not frames_b:
                continue
            best_distance = math.inf
            best_rotation = math.inf
            best_frames = (-1, -1)
            for frame_a in frames_a:
                pose_a = pose_by_frame[frame_a]
                for frame_b in frames_b:
                    pose_b = pose_by_frame[frame_b]
                    distance = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
                    rotation = rotation_degrees(pose_a, pose_b)
                    if distance < best_distance:
                        best_distance = distance
                        best_rotation = rotation
                        best_frames = (frame_a, frame_b)
            if not math.isfinite(best_distance):
                continue
            row = {
                "submap_a": submap_a,
                "submap_b": submap_b,
                "span": span,
                "gt_min_distance_m": best_distance,
                "gt_rotation_deg": best_rotation,
                "gt_frame_a": best_frames[0],
                "gt_frame_b": best_frames[1],
                "gt_temporal_separation_frames": abs(best_frames[1] - best_frames[0]),
                "gt_revisit": int(
                    best_distance <= distance_threshold
                    and (rotation_threshold_deg <= 0.0 or best_rotation <= rotation_threshold_deg)
                ),
            }
            best_by_pair[pair_key(submap_a, submap_b)] = row
            if row["gt_revisit"]:
                rows.append(row)
    rows.sort(key=lambda row: (float(row["gt_min_distance_m"]), -int(row["span"])))
    return rows, best_by_pair


def iter_pose_graph_roots(run_dir: Path) -> list[tuple[int, int, Path]]:
    pose_graph_root = run_dir / "pose_graph_live"
    updates_path = pose_graph_root / "updates.csv"
    roots: list[tuple[int, int, Path]] = []
    if updates_path.exists():
        with updates_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                root = Path(row.get("diagnostic_root", "")).expanduser()
                if not root.is_absolute():
                    root = run_dir / root
                if (root / "pose_graph_loop_candidates.csv").exists():
                    roots.append(
                        (
                            safe_int(row.get("update_index"), len(roots)),
                            safe_int(row.get("frame_id"), -1),
                            root,
                        )
                    )
    if roots:
        return roots
    if (pose_graph_root / "pose_graph_loop_candidates.csv").exists():
        return [(0, -1, pose_graph_root)]
    return []


NON_VERIFIED_REASONS = {
    "candidate_pair_budget",
    "coarse_distance",
    "coarse_rotation",
    "non_finite_transform",
    "appearance_matches",
    "bow_score",
}


def load_online_candidate_stats(run_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    pair_stats: dict[tuple[int, int], dict[str, object]] = {}
    for update_index, update_frame, root in iter_pose_graph_roots(run_dir):
        path = root / "pose_graph_loop_candidates.csv"
        with path.open(newline="") as handle:
            for rank, row in enumerate(csv.DictReader(handle)):
                from_id = safe_int(row.get("from_hybrid_id"), safe_int(row.get("from_sdf_submap_id"), -1))
                to_id = safe_int(row.get("to_hybrid_id"), safe_int(row.get("to_sdf_submap_id"), -1))
                if from_id < 0 or to_id < 0:
                    continue
                key = pair_key(from_id, to_id)
                stats = pair_stats.setdefault(
                    key,
                    {
                        "listed_count": 0,
                        "verified_count": 0,
                        "accepted_count": 0,
                        "budget_count": 0,
                        "selection_count": 0,
                        "first_update_index": update_index,
                        "first_update_frame": update_frame,
                        "best_rank": rank,
                        "best_score": math.inf,
                        "best_bow_score": 0.0,
                        "max_icp_inliers": 0,
                        "max_consistent_observations": 0,
                        "max_support_observations": 0,
                        "reason_counts": Counter(),
                        "selection_reason_counts": Counter(),
                    },
                )
                stats["listed_count"] = int(stats["listed_count"]) + 1
                stats["first_update_index"] = min(int(stats["first_update_index"]), update_index)
                if update_frame >= 0:
                    current_first_frame = int(stats["first_update_frame"])
                    stats["first_update_frame"] = (
                        update_frame
                        if current_first_frame < 0
                        else min(current_first_frame, update_frame)
                    )
                stats["best_rank"] = min(int(stats["best_rank"]), rank)
                reason = str(row.get("reason", "") or "")
                stats["reason_counts"][reason] += 1
                if reason == "candidate_pair_budget":
                    stats["budget_count"] = int(stats["budget_count"]) + 1

                score = safe_float(row.get("score"))
                if math.isfinite(score):
                    stats["best_score"] = min(float(stats["best_score"]), score)
                bow = safe_float(row.get("bow_score"))
                if math.isfinite(bow):
                    stats["best_bow_score"] = max(float(stats["best_bow_score"]), bow)
                stats["max_icp_inliers"] = max(
                    int(stats["max_icp_inliers"]),
                    safe_int(row.get("verification_icp_inliers"), 0),
                )
                stats["max_consistent_observations"] = max(
                    int(stats["max_consistent_observations"]),
                    safe_int(row.get("verification_consistent_observations"), 0),
                )
                stats["max_support_observations"] = max(
                    int(stats["max_support_observations"]),
                    safe_int(row.get("verification_support_observations"), 0),
                )
                verified_translation = safe_float(row.get("verification_translation_distance"))
                verified_rotation = safe_float(row.get("verification_rotation_degrees"))
                verified_inliers = safe_int(row.get("verification_icp_inliers"), 0)
                has_verification_measurement = (
                    (math.isfinite(verified_translation) and verified_translation >= 0.0)
                    or (math.isfinite(verified_rotation) and verified_rotation >= 0.0)
                    or verified_inliers > 0
                )
                if (
                    str(row.get("status", "")) == "accepted"
                    or reason not in NON_VERIFIED_REASONS
                    or has_verification_measurement
                ):
                    stats["verified_count"] = int(stats["verified_count"]) + 1
                if str(row.get("status", "")) == "accepted":
                    stats["accepted_count"] = int(stats["accepted_count"]) + 1

        selection_path = root / "pose_graph_apply_loop_selection.csv"
        if selection_path.exists():
            with selection_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    from_id = safe_int(row.get("from_hybrid_id"), -1)
                    to_id = safe_int(row.get("to_hybrid_id"), -1)
                    if from_id < 0 or to_id < 0:
                        continue
                    key = pair_key(from_id, to_id)
                    stats = pair_stats.setdefault(
                        key,
                        {
                            "listed_count": 0,
                            "verified_count": 0,
                            "accepted_count": 0,
                            "budget_count": 0,
                            "selection_count": 0,
                            "first_update_index": update_index,
                            "first_update_frame": update_frame,
                            "best_rank": 0,
                            "best_score": math.inf,
                            "best_bow_score": 0.0,
                            "max_icp_inliers": 0,
                            "max_consistent_observations": 0,
                            "max_support_observations": 0,
                            "reason_counts": Counter(),
                            "selection_reason_counts": Counter(),
                        },
                    )
                    selection_reason = str(row.get("reason", "") or "")
                    stats["selection_reason_counts"][selection_reason] += 1
                    if safe_int(row.get("selected"), 0) != 0:
                        stats["selection_count"] = int(stats["selection_count"]) + 1
    return pair_stats


def most_common_reason(counts: Counter[str]) -> str:
    if not counts:
        return ""
    reason, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
    return reason


def classify_online_funnel(stats: dict[str, object]) -> tuple[str, str]:
    listed = int(stats.get("listed_count", 0))
    verified = int(stats.get("verified_count", 0))
    accepted = int(stats.get("accepted_count", 0))
    selected = int(stats.get("selection_count", 0))
    budget = int(stats.get("budget_count", 0))
    reason_counts = stats.get("reason_counts", Counter())
    selection_counts = stats.get("selection_reason_counts", Counter())
    selection_reason = most_common_reason(selection_counts)
    verify_reason = most_common_reason(reason_counts)

    if selected > 0:
        return "selected_for_apply", "selected"
    if accepted > 0:
        if selection_reason.startswith("below_candidate_") or selection_reason in {
            "below_candidate_loop_submap_span",
            "below_candidate_loop_consistent_observations",
            "below_candidate_loop_icp_inliers",
            "above_candidate_loop_score",
        }:
            return "accepted_rejected_by_apply_support", selection_reason
        if selection_reason.startswith("pcm_") or "pcm" in selection_reason:
            return "accepted_rejected_by_pcm", selection_reason
        if selection_reason in {"proposal_gate_failed", "pcm_clique_proposal_gate_failed"}:
            return "accepted_rejected_by_proposal_gate", selection_reason
        if selection_reason == "selection_budget":
            return "accepted_rejected_by_selection_budget", selection_reason
        return "accepted_not_selected", selection_reason or "no_selection_record"
    if verified > 0:
        return "verified_rejected", verify_reason
    if listed > 0:
        if budget > 0:
            return "listed_budget_only", "candidate_pair_budget"
        return "listed_not_verified", verify_reason
    return "missing_from_online_candidates", "not_surfaced"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_completed_run(
    tum_root: Path,
    run_dir: Path,
    sequence: str | None,
    distance_threshold: float,
    rotation_threshold_deg: float,
    min_submap_span: int,
    max_samples_per_submap: int,
    output_prefix: Path | None,
    top_k: int,
) -> dict[str, object]:
    sequence_name = infer_run_sequence(run_dir, tum_root, sequence)
    sequence_dir = tum_root / sequence_name
    frame_ids, poses = read_poses(sequence_dir)
    frame_to_submap = read_frame_assignments(run_dir)
    gt_rows, gt_pair_lookup = gt_submap_pairs(
        frame_ids,
        poses,
        frame_to_submap,
        distance_threshold,
        rotation_threshold_deg,
        max(1, min_submap_span),
        max(1, max_samples_per_submap),
    )
    online = load_online_candidate_stats(run_dir)

    pair_rows: list[dict[str, object]] = []
    gt_pairs = {pair_key(int(row["submap_a"]), int(row["submap_b"])) for row in gt_rows}
    for row in gt_rows:
        key = pair_key(int(row["submap_a"]), int(row["submap_b"]))
        stats = online.get(key, {})
        listed = int(stats.get("listed_count", 0))
        verified = int(stats.get("verified_count", 0))
        accepted = int(stats.get("accepted_count", 0))
        budget = int(stats.get("budget_count", 0))
        selected = int(stats.get("selection_count", 0))
        funnel_stage, funnel_blocker = classify_online_funnel(stats)
        pair_rows.append(
            {
                **row,
                "online_funnel_stage": funnel_stage,
                "online_blocker": funnel_blocker,
                "online_listed": int(listed > 0),
                "online_verified": int(verified > 0),
                "online_accepted": int(accepted > 0),
                "online_selected": int(selected > 0),
                "online_budget_only": int(listed > 0 and verified == 0 and budget > 0),
                "online_listed_count": listed,
                "online_verified_count": verified,
                "online_accepted_count": accepted,
                "online_selection_count": selected,
                "online_best_rank": stats.get("best_rank", ""),
                "online_first_update_index": stats.get("first_update_index", ""),
                "online_first_update_frame": stats.get("first_update_frame", ""),
                "online_best_score": (
                    "" if not math.isfinite(float(stats.get("best_score", math.inf))) else stats.get("best_score")
                ),
                "online_best_bow_score": stats.get("best_bow_score", ""),
                "online_max_icp_inliers": stats.get("max_icp_inliers", ""),
                "online_max_consistent_observations": stats.get("max_consistent_observations", ""),
                "online_reason_counts": counts_to_text(stats.get("reason_counts", Counter())),
                "online_selection_reason_counts": counts_to_text(
                    stats.get("selection_reason_counts", Counter())
                ),
            }
        )

    pair_rows.sort(
        key=lambda row: (
            int(row["online_verified"]) == 0,
            int(row["online_listed"]) == 0,
            float(row["gt_min_distance_m"]),
            -int(row["span"]),
        )
    )

    online_non_gt_verified = 0
    online_non_gt_accepted = 0
    online_budget_only = 0
    for key, stats in online.items():
        if key in gt_pairs:
            continue
        if int(stats.get("verified_count", 0)) > 0:
            online_non_gt_verified += 1
        if int(stats.get("accepted_count", 0)) > 0:
            online_non_gt_accepted += 1
        if int(stats.get("listed_count", 0)) > 0 and int(stats.get("verified_count", 0)) == 0 and int(stats.get("budget_count", 0)) > 0:
            online_budget_only += 1

    total_gt = len(gt_rows)
    listed_gt = sum(1 for row in pair_rows if int(row["online_listed"]))
    verified_gt = sum(1 for row in pair_rows if int(row["online_verified"]))
    accepted_gt = sum(1 for row in pair_rows if int(row["online_accepted"]))
    selected_gt = sum(1 for row in pair_rows if int(row["online_selected"]))
    budget_only_gt = sum(1 for row in pair_rows if int(row["online_budget_only"]))
    missing_gt = total_gt - listed_gt
    summary = {
        "note": "Ground-truth run recall audit only; not used by the online SLAM system.",
        "run_dir": str(run_dir),
        "tum_root": str(tum_root),
        "sequence": sequence_name,
        "distance_threshold_m": distance_threshold,
        "rotation_threshold_deg": rotation_threshold_deg,
        "min_submap_span": min_submap_span,
        "max_samples_per_submap": max_samples_per_submap,
        "assigned_frames": len(frame_to_submap),
        "assigned_submaps": len(set(frame_to_submap.values())),
        "gt_revisit_pairs": total_gt,
        "gt_revisit_listed_pairs": listed_gt,
        "gt_revisit_verified_pairs": verified_gt,
        "gt_revisit_accepted_pairs": accepted_gt,
        "gt_revisit_selected_pairs": selected_gt,
        "gt_revisit_budget_only_pairs": budget_only_gt,
        "gt_revisit_missing_pairs": missing_gt,
        "gt_revisit_listed_recall": listed_gt / total_gt if total_gt else 0.0,
        "gt_revisit_verified_recall": verified_gt / total_gt if total_gt else 0.0,
        "gt_revisit_accepted_recall": accepted_gt / total_gt if total_gt else 0.0,
        "online_pairs_seen": len(online),
        "online_non_gt_verified_pairs": online_non_gt_verified,
        "online_non_gt_accepted_pairs": online_non_gt_accepted,
        "online_non_gt_budget_only_pairs": online_budget_only,
        "gt_revisit_funnel_stage_counts": counts_to_text(
            Counter(str(row["online_funnel_stage"]) for row in pair_rows)
        ),
        "gt_revisit_blocker_counts": counts_to_text(
            Counter(str(row["online_blocker"]) for row in pair_rows)
        ),
    }

    if output_prefix is not None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        summary_path = output_prefix.with_suffix(".json")
        pairs_path = output_prefix.with_name(output_prefix.name + "_pairs.csv")
        summary_path.write_text(json.dumps({**summary, "top_pairs": pair_rows[:top_k]}, indent=2))
        fieldnames = [
            "submap_a",
            "submap_b",
            "span",
            "gt_min_distance_m",
            "gt_rotation_deg",
            "gt_frame_a",
            "gt_frame_b",
            "gt_temporal_separation_frames",
            "gt_revisit",
            "online_funnel_stage",
            "online_blocker",
            "online_listed",
            "online_verified",
            "online_accepted",
            "online_selected",
            "online_budget_only",
            "online_listed_count",
            "online_verified_count",
            "online_accepted_count",
            "online_selection_count",
            "online_best_rank",
            "online_first_update_index",
            "online_first_update_frame",
            "online_best_score",
            "online_best_bow_score",
            "online_max_icp_inliers",
            "online_max_consistent_observations",
            "online_reason_counts",
            "online_selection_reason_counts",
        ]
        write_csv(pairs_path, pair_rows, fieldnames)

    print(
        "run,sequence,gt_pairs,listed,verified,accepted,selected,budget_only,missing,"
        "listed_recall,verified_recall,accepted_recall,online_pairs,non_gt_verified"
    )
    print(
        f"{run_dir.name},{sequence_name},{total_gt},{listed_gt},{verified_gt},"
        f"{accepted_gt},{selected_gt},{budget_only_gt},{missing_gt},"
        f"{summary['gt_revisit_listed_recall']:.3f},"
        f"{summary['gt_revisit_verified_recall']:.3f},"
        f"{summary['gt_revisit_accepted_recall']:.3f},"
        f"{len(online)},{online_non_gt_verified}"
    )
    if pair_rows:
        print("top_gt_pairs:")
        for row in pair_rows[:top_k]:
            print(
                f"  {row['submap_a']}->{row['submap_b']} span={row['span']} "
                f"dist={float(row['gt_min_distance_m']):.3f}m "
                f"rot={float(row['gt_rotation_deg']):.1f}deg "
                f"listed={row['online_listed']} verified={row['online_verified']} "
                f"accepted={row['online_accepted']} stage={row['online_funnel_stage']} "
                f"blocker={row['online_blocker']} reason={row['online_reason_counts']} "
                f"selection={row['online_selection_reason_counts']}"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-root", type=Path, default=default_root())
    parser.add_argument("--sequences", nargs="*", default=list(SEQUENCES.keys()))
    parser.add_argument("--thresholds", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--min-separation-frames", type=int, default=90)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--rotation-threshold-deg", type=float, default=90.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Completed GPS-SLAM run to compare against GT revisits.")
    parser.add_argument("--run-sequence", default=None, help="Short or full TUM sequence name for --run-dir if it cannot be inferred.")
    parser.add_argument("--run-distance-threshold", type=float, default=0.50)
    parser.add_argument("--run-min-submap-span", type=int, default=6)
    parser.add_argument("--run-max-samples-per-submap", type=int, default=12)
    parser.add_argument("--run-output-prefix", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    if args.run_dir is not None:
        output_prefix = args.run_output_prefix
        if output_prefix is None:
            output_prefix = args.run_dir / "pose_graph_live" / "gt_loop_recall_audit"
        audit_completed_run(
            args.tum_root,
            args.run_dir,
            args.run_sequence,
            max(0.0, args.run_distance_threshold),
            max(0.0, args.rotation_threshold_deg),
            max(1, args.run_min_submap_span),
            max(1, args.run_max_samples_per_submap),
            output_prefix,
            max(0, args.top_k),
        )
        return

    thresholds = parse_thresholds(args.thresholds)
    rows = []
    for name in args.sequences:
        sequence = SEQUENCES.get(name, name)
        rows.append(
            audit_sequence(
                args.tum_root,
                sequence,
                thresholds,
                max(1, args.min_separation_frames),
                max(1, args.stride),
                max(0.0, args.rotation_threshold_deg),
            )
        )

    report = {
        "tum_root": str(args.tum_root),
        "note": "Ground-truth audit only; not used by the online SLAM system.",
        "sequences": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    header = (
        "sequence,frames,path_m,threshold,pairs,source_frames,"
        "revisit_frames,revisit_episodes,max_sep_frames"
    )
    print(header)
    for row in rows:
        for threshold, stats in row["thresholds"].items():
            print(
                f"{row['short_name']},{row['frames']},{row['path_length_m']:.2f},"
                f"{threshold},{stats['pair_count']},{stats['unique_source_frames']},"
                f"{stats['unique_revisit_frames']},{stats['revisit_episodes']},"
                f"{stats['max_temporal_separation_frames']}"
            )


if __name__ == "__main__":
    main()
