#!/usr/bin/env python3
"""Summarize Hybrid MultiEngine/Gaussian-submap run diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return 99.0
    return -10.0 * math.log10(mse)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def frame_color_path(directory: Path, frame_id: int, fallback_suffix: str = "") -> Path | None:
    """Return the colour image for a frame, independent of eval iteration suffix."""
    exact = directory / f"frame{frame_id:06d}{fallback_suffix}.color.jpg"
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"frame{frame_id:06d}_iter*.color.jpg"))
    if matches:
        return matches[0]
    matches = sorted(directory.glob(f"frame{frame_id:06d}.color.jpg"))
    if matches:
        return matches[0]
    return None


def summarize_submap_stats(run_dir: Path) -> dict[str, Any]:
    rows = read_csv(run_dir / "submap_stats.csv")
    if not rows:
        return {}
    final_by_submap: dict[int, dict[str, str]] = {}
    for row in rows:
        final_by_submap[to_int(row.get("submap_id"), -1)] = row

    gaussians_by_submap = {
        sid: to_int(row.get("gaussians"))
        for sid, row in final_by_submap.items()
        if sid >= 0
    }
    blocks_by_submap = {
        sid: to_int(row.get("blocks"))
        for sid, row in final_by_submap.items()
        if sid >= 0
    }
    occupancy_by_submap = {
        sid: to_float(row.get("occupancy"))
        for sid, row in final_by_submap.items()
        if sid >= 0
    }
    resident_counts = [to_int(row.get("resident_count")) for row in rows if row.get("resident_count") not in (None, "")]
    result = {
        "logged_submaps": len(final_by_submap),
        "final_gaussians_sum": sum(gaussians_by_submap.values()),
        "final_gaussians_max": max(gaussians_by_submap.values(), default=0),
        "final_gaussians_zero_submaps": sum(1 for value in gaussians_by_submap.values() if value == 0),
        "max_blocks": max(blocks_by_submap.values(), default=0),
        "max_occupancy": max(occupancy_by_submap.values(), default=0.0),
        "max_resident_count": max(resident_counts, default=0),
        "top_gaussian_submaps": [
            {"submap_id": sid, "gaussians": count, "occupancy": occupancy_by_submap.get(sid, 0.0)}
            for sid, count in sorted(gaussians_by_submap.items(), key=lambda item: item[1], reverse=True)[:10]
        ],
    }
    stage_counts = Counter(row.get("stage", "") for row in rows if row.get("stage"))
    if stage_counts:
        result["submap_stats_stage_counts"] = dict(stage_counts)

    lifecycle_fields = ["frozen", "freeze_pending", "candidate_warming", "outgoing_handoff", "retired"]
    if any(field in next(iter(final_by_submap.values()), {}) for field in lifecycle_fields):
        final_lifecycle: dict[str, int] = {}
        for field in lifecycle_fields:
            final_lifecycle[field] = sum(1 for row in final_by_submap.values() if to_int(row.get(field), -1) == 1)
        result["final_lifecycle_counts"] = final_lifecycle
        result["final_submap_readiness"] = [
            {
                "submap_id": sid,
                "gaussian_submap_id": to_int(row.get("gaussian_submap_id"), -1),
                "frames": [to_int(row.get("start_frame"), -1), to_int(row.get("end_frame"), -1)],
                "sequence": [to_int(row.get("start_sequence"), -1), to_int(row.get("end_sequence"), -1)],
                "observed_frames": to_int(row.get("observed_frames"), 0),
                "gaussians": to_int(row.get("gaussians"), 0),
                "keyframes": to_int(row.get("keyframes"), 0),
                "loop_observations": to_int(row.get("loop_observations"), 0),
                "local_window": to_int(row.get("local_window"), 0),
                "opt_cameras": to_int(row.get("opt_cameras"), 0),
                "frozen": bool(to_int(row.get("frozen"), 0)),
                "freeze_pending": bool(to_int(row.get("freeze_pending"), 0)),
                "candidate_warming": bool(to_int(row.get("candidate_warming"), 0)),
                "outgoing_handoff": bool(to_int(row.get("outgoing_handoff"), 0)),
                "retired": bool(to_int(row.get("retired"), 0)),
                "occupancy": occupancy_by_submap.get(sid, 0.0),
                "close_reason": row.get("close_reason", ""),
            }
            for sid, row in sorted(final_by_submap.items())
            if sid >= 0
        ]
        result["final_under_ready_submaps"] = [
            item for item in result["final_submap_readiness"]
            if item["observed_frames"] < 16 or item["gaussians"] == 0 or item["keyframes"] == 0
        ][:20]
    return result


def summarize_events(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "submap_events.jsonl"
    if not path.exists():
        return {}
    counts: Counter[str] = Counter()
    capped_by_submap: Counter[int] = Counter()
    insert_added_by_submap: Counter[int] = Counter()
    insert_candidate_added_by_submap: Counter[int] = Counter()
    insert_candidate_warmup_added_by_submap: Counter[int] = Counter()
    insert_candidate_mature_added_by_submap: Counter[int] = Counter()
    insert_selection_counts: Counter[str] = Counter()
    insert_selection_applied_counts: Counter[str] = Counter()
    transition_rejects_by_submap: Counter[int] = Counter()
    reject_cooldown_skips_by_sdf: Counter[int] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        event = str(payload.get("event", ""))
        counts[event] += 1
        sid = to_int(payload.get("hybrid_submap"), -1)
        if event == "gaussian_insert_capped" and sid >= 0:
            capped_by_submap[sid] += 1
        if event == "candidate_transition_reject" and sid >= 0:
            transition_rejects_by_submap[sid] += 1
        if event == "candidate_warming_skipped_reject_cooldown":
            sdf_id = to_int(payload.get("sdf_submap"), -1)
            if sdf_id >= 0:
                reject_cooldown_skips_by_sdf[sdf_id] += 1
        if event == "gaussian_insert" and sid >= 0:
            added = to_int(payload.get("added"))
            insert_added_by_submap[sid] += added
            selection = str(payload.get("insert_selection", "unknown"))
            insert_selection_counts[selection] += 1
            if to_int(payload.get("insert_selection_applied")):
                insert_selection_applied_counts[selection] += 1
            if to_int(payload.get("candidate_warming")):
                insert_candidate_added_by_submap[sid] += added
                if to_int(payload.get("candidate_warmup_insert")):
                    insert_candidate_warmup_added_by_submap[sid] += added
                else:
                    insert_candidate_mature_added_by_submap[sid] += added
    result: dict[str, Any] = {"event_counts": dict(counts)}
    if capped_by_submap:
        result["gaussian_insert_capped_by_submap"] = dict(capped_by_submap.most_common())
    if transition_rejects_by_submap:
        result["candidate_transition_rejects_by_submap"] = dict(transition_rejects_by_submap.most_common())
    if reject_cooldown_skips_by_sdf:
        result["candidate_reject_cooldown_skips_by_sdf"] = dict(reject_cooldown_skips_by_sdf.most_common())
    if insert_added_by_submap:
        result["gaussian_insert_added_total"] = sum(insert_added_by_submap.values())
        result["gaussian_insert_added_by_submap"] = dict(insert_added_by_submap.most_common())
        result["gaussian_insert_candidate_added_by_submap"] = dict(insert_candidate_added_by_submap.most_common())
        result["gaussian_insert_candidate_warmup_added_total"] = sum(insert_candidate_warmup_added_by_submap.values())
        result["gaussian_insert_candidate_warmup_added_by_submap"] = dict(insert_candidate_warmup_added_by_submap.most_common())
        result["gaussian_insert_candidate_mature_added_total"] = sum(insert_candidate_mature_added_by_submap.values())
        result["gaussian_insert_candidate_mature_added_by_submap"] = dict(insert_candidate_mature_added_by_submap.most_common())
        result["gaussian_insert_selection_counts"] = dict(insert_selection_counts.most_common())
        result["gaussian_insert_selection_applied_counts"] = dict(insert_selection_applied_counts.most_common())
    return result


def summarize_transitions(run_dir: Path) -> dict[str, Any]:
    rows = read_csv(run_dir / "submap_transitions.csv")
    if not rows:
        return {}
    event_counts = Counter(row.get("event", "") for row in rows)
    action_counts = Counter(row.get("action", "") for row in rows)
    rejects = [
        {
            "frame": to_int(row.get("frame")),
            "event": row.get("event", ""),
            "action": row.get("action", ""),
            "primary_local_map": to_int(row.get("primary_local_map"), -1),
            "new_local_map": to_int(row.get("new_local_map"), -1),
            "inliers": to_int(row.get("inliers"), -1),
            "delta_m": to_float(row.get("delta_m")),
            "delta_deg": to_float(row.get("delta_deg")),
        }
        for row in rows
        if "reject" in row.get("event", "") or "reject" in row.get("action", "")
    ]
    return {
        "transition_rows": len(rows),
        "transition_event_counts": dict(event_counts),
        "transition_action_counts": dict(action_counts),
        "transition_rejects": rejects[:20],
        "transition_reject_count": len(rejects),
    }


def summarize_frame_assignments(run_dir: Path) -> dict[str, Any]:
    rows = read_csv(run_dir / "frame_assignments.csv")
    if not rows:
        return {}
    primary_counts = Counter(to_int(row.get("primary_submap"), -1) for row in rows)
    active_widths = []
    for row in rows:
        active = [part for part in row.get("active_submaps", "").split("|") if part != ""]
        active_widths.append(len(active))
    return {
        "frame_assignment_rows": len(rows),
        "primary_frame_counts": dict(sorted(primary_counts.items())),
        "max_active_submaps": max(active_widths, default=0),
    }


def summarize_quality(run_dir: Path, limit: int = 0) -> dict[str, Any]:
    debug_rows = read_csv(run_dir / "val" / "submap_render_debug.csv")
    if not debug_rows:
        return {}
    gt_dir = run_dir / "val" / "gt"
    render_dir = run_dir / "val" / "render"
    raycast_dir = run_dir / "val" / "raycast_color"
    if not gt_dir.exists() or not render_dir.exists() or not raycast_dir.exists():
        return {}

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    owner_grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    worst: list[dict[str, Any]] = []
    processed = 0
    for row in debug_rows:
        frame_id = to_int(row.get("frame_id"), -1)
        if frame_id < 0:
            continue
        if limit > 0 and processed >= limit:
            break
        gt_path = frame_color_path(gt_dir, frame_id)
        render_path = frame_color_path(render_dir, frame_id)
        raycast_path = frame_color_path(raycast_dir, frame_id)
        if gt_path is None or render_path is None or raycast_path is None:
            continue
        gt = load_rgb(gt_path)
        hybrid = load_rgb(render_path)
        sdf = load_rgb(raycast_path)
        hybrid_psnr = psnr(gt, hybrid)
        sdf_psnr = psnr(gt, sdf)
        selected = row.get("selected_submaps", "")
        owner = row.get("owner_submap", "")
        grouped[selected].append((hybrid_psnr, sdf_psnr))
        owner_grouped[owner].append((hybrid_psnr, sdf_psnr))
        worst.append({
            "frame_id": frame_id,
            "owner_submap": owner,
            "selected_submaps": selected,
            "hybrid_psnr": hybrid_psnr,
            "sdf_psnr": sdf_psnr,
            "delta_psnr": hybrid_psnr - sdf_psnr,
        })
        processed += 1

    def pack(groups: dict[str, list[tuple[float, float]]]) -> list[dict[str, Any]]:
        rows = []
        for key, values in groups.items():
            if not values:
                continue
            hybrid_values = [item[0] for item in values]
            sdf_values = [item[1] for item in values]
            rows.append({
                "key": key,
                "frames": len(values),
                "hybrid_psnr": float(np.mean(hybrid_values)),
                "sdf_psnr": float(np.mean(sdf_values)),
                "delta_psnr": float(np.mean(hybrid_values) - np.mean(sdf_values)),
            })
        rows.sort(key=lambda item: item["delta_psnr"])
        return rows

    worst.sort(key=lambda item: item["delta_psnr"])
    return {
        "quality_frames": processed,
        "quality_by_selected_submaps": pack(grouped),
        "quality_by_owner_submap": pack(owner_grouped),
        "worst_quality_frames": worst[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--quality-limit", type=int, default=0, help="limit frames for image-based quality analysis")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    summary: dict[str, Any] = {"run_dir": str(run_dir)}
    for section in (
        summarize_submap_stats(run_dir),
        summarize_events(run_dir),
        summarize_transitions(run_dir),
        summarize_frame_assignments(run_dir),
        summarize_quality(run_dir, args.quality_limit),
    ):
        summary.update(section)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
