#!/usr/bin/env python3
"""Summarize MultiEngine submap transition causes from existing run logs.

The script is intentionally read-only. It consumes ``submap_transitions.csv``
files produced by the active-map manager and reports which policy actually
created/promoted submaps, including weak/failed tracking counters at the event.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


def safe_int(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def numeric_summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": median(values),
        f"{prefix}_max": max(values),
        f"{prefix}_p95": ordered[p95_idx],
    }


def transition_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "submap_transitions.csv":
            files.append(path)
        elif path.is_dir():
            if (path / "submap_transitions.csv").exists():
                files.append(path / "submap_transitions.csv")
            else:
                files.extend(sorted(path.rglob("submap_transitions.csv")))
    return sorted(set(files))


def summarize_transition_file(path: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    event_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    move_counts: Counter[str] = Counter()
    first_event_frame: dict[str, int] = {}
    move_frames: list[int] = []
    delta_m: list[float] = []
    delta_deg: list[float] = []
    weak_at_move: list[int] = []
    failed_at_move: list[int] = []
    visibility_move_rows = []
    primary_failure_trigger_counts: Counter[str] = Counter()
    capacity_like_moves = 0
    tracking_like_moves = 0

    for row in rows:
        event = row.get("event", "") or "unknown"
        action = row.get("action", "") or "unknown"
        frame = safe_int(row.get("frame"), -1)
        move = safe_int(row.get("move_to_primary"), 0) != 0 or action == "move"
        event_counts[event] += 1
        action_counts[action] += 1
        first_event_frame.setdefault(event, frame)
        if safe_float(row.get("delta_m"), 0.0) > 0.0:
            delta_m.append(safe_float(row.get("delta_m")))
        if safe_float(row.get("delta_deg"), 0.0) > 0.0:
            delta_deg.append(safe_float(row.get("delta_deg")))
        if not move:
            continue
        move_counts[event] += 1
        move_frames.append(frame)
        weak = safe_int(row.get("primary_raw_weak"), -1)
        failed = safe_int(row.get("primary_raw_failed"), -1)
        if weak >= 0:
            weak_at_move.append(weak)
        if failed >= 0:
            failed_at_move.append(failed)
        if event in {"primary_failure_handoff", "tracking_boundary_handoff"}:
            tracking_like_moves += 1
        if event == "primary_failure_handoff":
            detail = row.get("detail", "") or ""
            if "trigger=failed_and_weak" in detail:
                primary_failure_trigger_counts["failed_and_weak"] += 1
            elif "trigger=failed" in detail:
                primary_failure_trigger_counts["failed"] += 1
            elif "trigger=weak" in detail:
                primary_failure_trigger_counts["weak"] += 1
            else:
                primary_failure_trigger_counts["unknown"] += 1
        if event in {"capacity_handoff", "capacity_request", "occupancy_handoff"}:
            capacity_like_moves += 1
        if event in {"visibility_handoff", "primary_failure_handoff"}:
            visibility_move_rows.append(row)

    move_intervals = [
        later - earlier
        for earlier, later in zip(move_frames, move_frames[1:])
        if later >= 0 and earlier >= 0
    ]
    out: dict[str, object] = {
        "run_dir": str(path.parent),
        "transition_file": str(path),
        "rows": len(rows),
        "events": json.dumps(dict(sorted(event_counts.items())), sort_keys=True),
        "actions": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
        "moves_by_event": json.dumps(dict(sorted(move_counts.items())), sort_keys=True),
        "first_event_frame": json.dumps(dict(sorted(first_event_frame.items())), sort_keys=True),
        "move_count": sum(move_counts.values()),
        "first_move_frame": min(move_frames) if move_frames else -1,
        "last_move_frame": max(move_frames) if move_frames else -1,
        "tracking_like_move_count": tracking_like_moves,
        "capacity_like_move_count": capacity_like_moves,
        "primary_failure_move_count": move_counts.get("primary_failure_handoff", 0),
        "primary_failure_trigger_counts": json.dumps(dict(sorted(primary_failure_trigger_counts.items())), sort_keys=True),
        "primary_failure_trigger_failed": primary_failure_trigger_counts.get("failed", 0),
        "primary_failure_trigger_weak": primary_failure_trigger_counts.get("weak", 0),
        "primary_failure_trigger_failed_and_weak": primary_failure_trigger_counts.get("failed_and_weak", 0),
        "primary_failure_trigger_unknown": primary_failure_trigger_counts.get("unknown", 0),
        "visibility_move_count": move_counts.get("visibility_handoff", 0),
        "capacity_handoff_move_count": move_counts.get("capacity_handoff", 0),
        "sequential_transform_observed": event_counts.get("sequential_transform", 0),
        "sequential_transform_rejected": action_counts.get("reject", 0) + action_counts.get("rejected", 0),
    }
    out.update(numeric_summary(move_intervals, "move_interval_frames"))
    out.update(numeric_summary(delta_m, "sequential_delta_m"))
    out.update(numeric_summary(delta_deg, "sequential_delta_deg"))
    out.update(numeric_summary([float(v) for v in weak_at_move], "primary_raw_weak_at_move"))
    out.update(numeric_summary([float(v) for v in failed_at_move], "primary_raw_failed_at_move"))

    if visibility_move_rows:
        visible_deltas = [
            safe_float(row.get("new_visible"), 0.0) - safe_float(row.get("primary_visible"), 0.0)
            for row in visibility_move_rows
            if safe_float(row.get("new_visible"), -1.0) >= 0.0
            and safe_float(row.get("primary_visible"), -1.0) >= 0.0
        ]
        blocks = [
            safe_float(row.get("new_blocks"), 0.0)
            for row in visibility_move_rows
            if safe_float(row.get("new_blocks"), -1.0) >= 0.0
        ]
        out.update(numeric_summary(visible_deltas, "handoff_visible_margin"))
        out.update(numeric_summary(blocks, "handoff_new_blocks"))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Run dirs, roots, or submap_transitions.csv files")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV summary path")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    files = transition_files(args.paths)
    if not files:
        raise SystemExit("No submap_transitions.csv files found")
    rows = [summarize_transition_file(path) for path in files]
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fields = [
        "run_dir",
        "rows",
        "move_count",
        "moves_by_event",
        "first_move_frame",
        "primary_failure_move_count",
        "primary_failure_trigger_failed",
        "primary_failure_trigger_weak",
        "primary_failure_trigger_failed_and_weak",
        "primary_failure_trigger_unknown",
        "visibility_move_count",
        "capacity_handoff_move_count",
        "primary_raw_failed_at_move_max",
        "primary_raw_weak_at_move_max",
        "sequential_delta_m_max",
        "sequential_delta_deg_max",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
