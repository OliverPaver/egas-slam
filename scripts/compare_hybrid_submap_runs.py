#!/usr/bin/env python3
"""Compare two hybrid SDF/Gaussian submap runs.

This is intentionally lightweight: it summarizes the files that usually explain
quality regressions before looking at rendered images manually.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    else:
        out[prefix] = value


def read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    flat: dict[str, Any] = {}
    flatten("", data, flat)
    return flat


def default_config_for(run_dir: Path) -> Path | None:
    for candidate in [
        run_dir / "config.yaml",
        run_dir / "config.yml",
        run_dir / "run_config.yaml",
        run_dir / "generated_config.yaml",
    ]:
        if candidate.exists():
            return candidate
    return None


def event_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "submap_events.jsonl"
    result: dict[str, Any] = {"events_file": str(path), "events_present": path.exists()}
    if not path.exists():
        return result

    counts: dict[str, int] = {}
    gaussian_added = 0
    gaussian_insert_frames = 0
    novelty_suppressed = 0
    novelty_candidates = 0
    budget_cap_skips = 0
    budget_pressure_prunes = 0
    readiness_delta_after = []
    readiness_skips = 0
    malformed = 0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            event = str(payload.get("event", ""))
            if event:
                counts[event] = counts.get(event, 0) + 1
            if event == "gaussian_insert":
                gaussian_insert_frames += 1
                gaussian_added += int(payload.get("added", 0) or 0)
                novelty_suppressed += max(0, int(payload.get("cross_novelty_suppressed_pixels", 0) or 0))
                novelty_candidates += max(0, int(payload.get("cross_novelty_candidate_pixels_before", 0) or 0))
            elif event == "gaussian_insert_skipped_cap":
                budget_cap_skips += 1
            elif event == "gaussian_budget_pressure_prune":
                budget_pressure_prunes += 1
            elif event == "gaussian_submap_graduation_end":
                value = float(payload.get("readiness_delta_after", -999.0) or -999.0)
                if value > -998.0:
                    readiness_delta_after.append(value)
            elif event == "gaussian_submap_graduation_skipped":
                if payload.get("skip_reason") == "readiness_already_met":
                    readiness_skips += 1

    result.update({
        "event_counts": counts,
        "malformed_events": malformed,
        "gaussian_insert_frames": gaussian_insert_frames,
        "gaussian_added_total": gaussian_added,
        "cross_novelty_suppressed_total": novelty_suppressed,
        "cross_novelty_candidate_before_total": novelty_candidates,
        "cross_novelty_suppressed_ratio": (
            novelty_suppressed / novelty_candidates if novelty_candidates > 0 else None
        ),
        "budget_cap_skip_count": budget_cap_skips,
        "budget_pressure_prune_count": budget_pressure_prunes,
        "graduation_readiness_after_avg": (
            sum(readiness_delta_after) / len(readiness_delta_after) if readiness_delta_after else None
        ),
        "graduation_readiness_skip_count": readiness_skips,
    })
    return result


def submap_stats_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "submap_stats.csv"
    result: dict[str, Any] = {"submap_stats_file": str(path), "submap_stats_present": path.exists()}
    if not path.exists():
        return result

    last_by_gaussian: dict[int, dict[str, str]] = {}
    sdf_ids = set()
    max_sdf_count = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sdf_ids.add(int(row.get("submap_id", "") or -1))
                max_sdf_count = max(max_sdf_count, len([idx for idx in sdf_ids if idx >= 0]))
            except ValueError:
                pass
            try:
                gaussian_id = int(row.get("gaussian_submap_id", "") or -1)
            except ValueError:
                gaussian_id = -1
            if gaussian_id >= 0:
                last_by_gaussian[gaussian_id] = row

    final_gaussians = 0
    max_gaussians = 0
    cap_hit_submaps = 0
    budget_pressure_submaps = 0
    final_caps = []
    for row in last_by_gaussian.values():
        gs = int(float(row.get("gaussians", "") or 0))
        final_gaussians += gs
        max_gaussians = max(max_gaussians, gs)
        cap_hits = int(float(row.get("budget_cap_hits", "") or 0))
        prunes = int(float(row.get("budget_pressure_prunes", "") or 0))
        cap_hit_submaps += int(cap_hits > 0)
        budget_pressure_submaps += int(prunes > 0)
        cap = int(float(row.get("last_budget_cap", "") or -1))
        if cap >= 0:
            final_caps.append(cap)

    result.update({
        "accepted_sdf_submap_count": len([idx for idx in sdf_ids if idx >= 0]),
        "max_logged_sdf_submap_count": max_sdf_count,
        "gaussian_submap_count": len(last_by_gaussian),
        "final_gaussians_total": final_gaussians,
        "final_gaussians_max_per_submap": max_gaussians,
        "budget_cap_hit_submap_count": cap_hit_submaps,
        "budget_pressure_submap_count": budget_pressure_submaps,
        "final_budget_cap_avg": sum(final_caps) / len(final_caps) if final_caps else None,
        "final_budget_cap_max": max(final_caps) if final_caps else None,
    })
    return result


def metrics_summary(run_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in [
        run_dir / "val" / "render" / "results.json",
        run_dir / "results.json",
    ]:
        if path.exists():
            try:
                result.update({f"metric_{k}": v for k, v in json.loads(path.read_text()).items()})
                result["metric_file"] = str(path)
                break
            except json.JSONDecodeError:
                result["metric_file_error"] = str(path)
    time_log = run_dir / "time_log.txt"
    if time_log.exists():
        text = time_log.read_text(errors="ignore")
        match = re.search(r"\[PIPELINE AVG TIME\].*GS num:\s*([0-9]+).*FPS:\s*([0-9.]+)", text)
        if match:
            result["time_log_gs_num"] = int(match.group(1))
            result["time_log_fps"] = float(match.group(2))
    return result


def config_diff(old_config: dict[str, Any], new_config: dict[str, Any]) -> dict[str, Any]:
    changed = {}
    for key in sorted(set(old_config) | set(new_config)):
        old_value = old_config.get(key, "<missing>")
        new_value = new_config.get(key, "<missing>")
        if old_value != new_value:
            changed[key] = {"old": old_value, "new": new_value}
    return changed


def summarize_run(run_dir: Path, config_path: Path | None) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "config": str(config_path) if config_path else "",
        **submap_stats_summary(run_dir),
        **event_summary(run_dir),
        **metrics_summary(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--old-config", type=Path, default=None)
    parser.add_argument("--new-config", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    old_config_path = args.old_config or default_config_for(args.old_run)
    new_config_path = args.new_config or default_config_for(args.new_run)
    old_config = read_yaml(old_config_path)
    new_config = read_yaml(new_config_path)

    report = {
        "old": summarize_run(args.old_run, old_config_path),
        "new": summarize_run(args.new_run, new_config_path),
        "config_changed": config_diff(old_config, new_config),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
        print(args.json_out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
