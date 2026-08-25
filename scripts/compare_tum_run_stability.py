#!/usr/bin/env python3
"""Compare repeated TUM loop-closure runs for lifecycle stability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - used only when PyYAML is unavailable.
    yaml = None


CORE_FIELDS = (
    "run",
    "run_dir",
    "ate_actual_rmse_cm",
    "corrected_ate_actual_rmse_cm",
    "corrected_ate_improvement_pct",
    "fps",
    "submap_count",
    "accepted_handoff_moves",
    "capacity_handoff_moves",
    "capacity_handoff_rejects",
    "raw_tracking_non_good",
    "loop_accepted",
    "loop_exported",
    "loop_scanned_pairs",
    "pose_graph_apply_selected_correction_scale",
    "pose_graph_apply_apply_scale_trial_reasons",
    "pose_graph_apply_apply_scale_trial_observation_cache_hit",
    "pose_graph_apply_timing_scale_evaluation_ms",
    "pose_graph_apply_timing_total_ms",
    "pose_graph_max_apply_ms",
    "random_seed",
    "experiment_repeat_index",
    "experiment_repeat_count",
    "experiment_repeat_seed_step",
    "experiment_group_key",
    "experiment_profile_name",
    "experiment_profile_fingerprint",
    "experiment_config_fingerprint",
    "config_profile_fingerprint",
    "tsdf_hash_profile",
    "tsdf_max_frames",
    "tsdf_overlap_frames",
    "tsdf_trial_frames",
    "tsdf_handoff_pose_policy",
    "tsdf_new_map_handoff_policy",
    "tsdf_new_map_handoff_min_accepts",
    "gaussian_submap_render_policy",
    "new_gs_sample_ratio",
    "local_opt_iters",
    "local_opt_interval",
    "model_sh_degree",
    "pose_graph_orb_match_policy_config",
    "pose_graph_verify_orb_aggregate_pairs_config",
    "pose_graph_verify_orb_aggregate_max_pairs_config",
    "pose_graph_verify_max_observation_pairs_config",
    "pose_graph_verify_pair_selection_config",
    "pose_graph_apply_support_policy_config",
    "pose_graph_apply_min_loop_submap_span_config",
    "pose_graph_apply_adaptive_min_loop_consistent_observations_config",
    "pose_graph_apply_adaptive_min_loop_icp_inliers_config",
    "pose_graph_information_policy_config",
    "pose_graph_loop_robust_kernel_config",
    "pose_graph_loop_robust_scale_config",
)

NUMERIC_FIELDS = (
    "ate_actual_rmse_cm",
    "corrected_ate_actual_rmse_cm",
    "corrected_ate_improvement_pct",
    "fps",
    "submap_count",
    "accepted_handoff_moves",
    "capacity_handoff_moves",
    "capacity_handoff_rejects",
    "raw_tracking_non_good",
    "loop_accepted",
    "loop_exported",
    "loop_scanned_pairs",
    "pose_graph_apply_selected_correction_scale",
    "pose_graph_apply_timing_scale_evaluation_ms",
    "pose_graph_apply_timing_total_ms",
    "pose_graph_max_apply_ms",
)

STABILITY_FIELDS = (
    "ate_actual_rmse_cm",
    "corrected_ate_actual_rmse_cm",
    "corrected_ate_improvement_pct",
    "rpe_translation_rmse_cm",
    "corrected_rpe_translation_rmse_cm",
    "fps",
    "gpu_memory_mb",
    "loop_observation_est_rgbd_payload_mb",
    "loop_observation_est_pose_metadata_mb",
    "loop_observations_total_reported",
    "submap_count",
    "raw_tracking_non_good",
    "raw_tracking_failed",
    "raw_tracking_max_failed_streak",
    "loop_accepted",
    "loop_exported",
    "loop_scanned_pairs",
    "pose_graph_apply_selected_correction_scale",
    "pose_graph_apply_timing_total_ms",
    "pose_graph_max_apply_ms",
)

PROFILE_FIELDS = (
    "experiment_profile_name",
    "experiment_profile_fingerprint",
    "experiment_config_fingerprint",
    "config_profile_fingerprint",
    "tsdf_hash_profile",
    "tsdf_max_frames",
    "tsdf_overlap_frames",
    "tsdf_trial_frames",
    "tsdf_handoff_pose_policy",
    "tsdf_new_map_handoff_policy",
    "tsdf_new_map_handoff_min_accepts",
    "gaussian_submap_render_policy",
    "new_gs_sample_ratio",
    "local_opt_iters",
    "local_opt_interval",
    "model_sh_degree",
    "pose_graph_orb_match_policy_config",
    "pose_graph_verify_orb_aggregate_pairs_config",
    "pose_graph_verify_orb_aggregate_max_pairs_config",
    "pose_graph_verify_max_observation_pairs_config",
    "pose_graph_verify_pair_selection_config",
    "pose_graph_apply_support_policy_config",
    "pose_graph_apply_min_loop_submap_span_config",
    "pose_graph_apply_adaptive_min_loop_consistent_observations_config",
    "pose_graph_apply_adaptive_min_loop_icp_inliers_config",
    "pose_graph_information_policy_config",
    "pose_graph_loop_robust_kernel_config",
    "pose_graph_loop_robust_scale_config",
)


def parse_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"{path} has no data rows") from exc


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} has no data rows")
    return rows


def handoff_signature(run_dir: Path, max_items: int) -> tuple[list[int], list[int]]:
    assignments = run_dir / "frame_assignments.csv"
    if not assignments.exists():
        return [], []

    handoff_frames: list[int] = []
    primary_ids: list[int] = []
    previous_primary: int | None = None
    with assignments.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame_id = int(row.get("frame_id", ""))
                primary = int(row.get("primary_submap", ""))
            except ValueError:
                continue
            if previous_primary is None:
                previous_primary = primary
                primary_ids.append(primary)
                continue
            if primary != previous_primary:
                handoff_frames.append(frame_id)
                primary_ids.append(primary)
                previous_primary = primary

    return handoff_frames[:max_items], primary_ids[: max_items + 1]


def enrich_row(row: dict[str, str], max_signature_items: int) -> dict[str, Any]:
    out: dict[str, Any] = {field: row.get(field, "") for field in CORE_FIELDS}
    run_dir_value = row.get("run_dir", "")
    if run_dir_value:
        frames, primaries = handoff_signature(Path(run_dir_value), max_signature_items)
        out["handoff_signature_frames"] = ";".join(str(value) for value in frames)
        out["handoff_signature_primary_ids"] = ";".join(str(value) for value in primaries)
        out["handoff_signature_count"] = len(frames)
    else:
        out["handoff_signature_frames"] = ""
        out["handoff_signature_primary_ids"] = ""
        out["handoff_signature_count"] = ""
    return out


def add_reference_deltas(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    reference = rows[0]
    for row in rows:
        for field in NUMERIC_FIELDS:
            value = parse_number(row.get(field))
            ref_value = parse_number(reference.get(field))
            if value is None or ref_value is None:
                row[f"delta_{field}"] = ""
            else:
                row[f"delta_{field}"] = value - ref_value
        row["handoff_signature_matches_reference"] = (
            row.get("handoff_signature_frames") == reference.get("handoff_signature_frames")
        )


def stability_group_key(row: dict[str, Any]) -> str:
    explicit = str(row.get("experiment_group_key", "")).strip()
    if explicit:
        return explicit

    run = str(row.get("run", "")).strip()
    fingerprint = config_fingerprint_group_key(row)
    if fingerprint:
        label = infer_run_label(run)
        return f"{label}_{fingerprint}"
    if not run:
        return "unknown"
    run = re.sub(r"_rseed-?\d+(?=_rep\d+$)", "", run)
    run = re.sub(r"_rep\d+$", "", run)
    return run


def infer_run_label(run: str) -> str:
    known_variants = (
        "multi_hybrid_orb_apply",
        "multi_hybrid_orb_dry_solve",
        "multi_hybrid_dry_solve",
        "multi_hybrid_no_loop",
        "multi_hybrid_apply",
        "multi_sdf_orb_apply",
        "multi_sdf_orb_dry_solve",
        "multi_sdf_dry_solve",
        "multi_sdf_sequential_noop",
        "multi_sdf_synthetic",
        "multi_sdf",
        "basic_gaussian",
        "basic_sdf",
    )
    for variant in known_variants:
        marker = f"_{variant}_"
        index = run.find(marker)
        if index >= 0:
            return run[: index + len(marker) - 1]
    return run


def normalize_config_for_grouping(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if key == "workspace_dir":
                continue
            if key == "PIPE" and isinstance(child, dict):
                pipe = {
                    pipe_key: pipe_value
                    for pipe_key, pipe_value in child.items()
                    if pipe_key
                    not in {
                        "experiment_config_fingerprint",
                        "experiment_repeat_index",
                        "experiment_repeat_count",
                        "experiment_repeat_seed_step",
                        "experiment_group_key",
                        "experiment_profile_fingerprint",
                        "experiment_profile_name",
                        "experiment_profile_summary",
                        "random_seed",
                    }
                }
                normalized[key] = normalize_config_for_grouping(pipe)
            else:
                normalized[key] = normalize_config_for_grouping(child)
        return normalized
    if isinstance(value, list):
        return [normalize_config_for_grouping(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_fingerprint_group_key(row: dict[str, Any]) -> str:
    if yaml is None:
        return ""
    run_dir_value = str(row.get("run_dir", "")).strip()
    if not run_dir_value:
        return ""
    run_dir = Path(run_dir_value)
    try:
        config_path = next(iter(sorted(run_dir.glob("*.yaml"))))
    except StopIteration:
        return ""
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return ""
    normalized = normalize_config_for_grouping(config)
    encoded = json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = parse_number(row.get(field))
        if value is not None:
            values.append(value)
    return values


def summarize_numeric(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {"mean": "", "std": "", "min": "", "max": ""}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(stability_group_key(row), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        handoff_signatures = {
            str(row.get("handoff_signature_frames", "")) for row in group_rows
        }
        seeds = sorted(
            {
                int(value)
                for row in group_rows
                for value in [parse_number(row.get("random_seed"))]
                if value is not None
            }
        )
        aggregate: dict[str, Any] = {
            "group": key,
            "runs": len(group_rows),
            "random_seeds": ";".join(str(seed) for seed in seeds),
            "unique_handoff_signatures": len(handoff_signatures),
            "handoff_signature_stable": int(len(handoff_signatures) <= 1),
        }
        for field in PROFILE_FIELDS:
            values = sorted({str(row.get(field, "")).strip() for row in group_rows})
            values = [value for value in values if value]
            if not values:
                continue
            aggregate[field] = values[0] if len(values) == 1 else "MIXED:" + "|".join(values)
        profile_fingerprints = {
            str(row.get("config_profile_fingerprint", "")).strip()
            for row in group_rows
            if str(row.get("config_profile_fingerprint", "")).strip()
        }
        aggregate["profile_fingerprint_stable"] = int(len(profile_fingerprints) <= 1)
        for field in STABILITY_FIELDS:
            stats = summarize_numeric(numeric_values(group_rows, field))
            for stat_name, value in stats.items():
                aggregate[f"{field}_{stat_name}"] = value
        aggregate_rows.append(aggregate)
    return aggregate_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GPS-SLAM TUM summary CSVs for repeated-run stability."
    )
    parser.add_argument("summaries", nargs="+", type=Path, help="Summary CSV files to compare")
    parser.add_argument("--output-csv", type=Path, help="Optional comparison CSV path")
    parser.add_argument("--aggregate-csv", type=Path, help="Optional grouped mean/std CSV path")
    parser.add_argument("--output-json", type=Path, help="Optional comparison JSON path")
    parser.add_argument(
        "--max-signature-items",
        type=int,
        default=64,
        help="Maximum handoff frames/primary IDs to include in signatures",
    )
    args = parser.parse_args()

    rows = [
        enrich_row(row, max(0, args.max_signature_items))
        for path in args.summaries
        for row in read_rows(path)
    ]
    add_reference_deltas(rows)
    aggregate_rows = aggregate_groups(rows)

    if args.output_csv:
        write_csv(rows, args.output_csv)
    if args.aggregate_csv:
        write_csv(aggregate_rows, args.aggregate_csv)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"runs": rows, "groups": aggregate_rows}, indent=2) + "\n"
        )

    fields = (
        "run",
        "experiment_profile_name",
        "config_profile_fingerprint",
        "ate_actual_rmse_cm",
        "corrected_ate_actual_rmse_cm",
        "corrected_ate_improvement_pct",
        "submap_count",
        "loop_accepted",
        "loop_exported",
        "pose_graph_apply_selected_correction_scale",
        "pose_graph_apply_timing_total_ms",
        "delta_submap_count",
        "delta_corrected_ate_actual_rmse_cm",
        "handoff_signature_matches_reference",
    )
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})

    if aggregate_rows:
        print()
        aggregate_fields = (
            "group",
            "runs",
            "random_seeds",
            "experiment_profile_name",
            "config_profile_fingerprint",
            "profile_fingerprint_stable",
            "tsdf_hash_profile",
            "tsdf_max_frames",
            "tsdf_overlap_frames",
            "tsdf_trial_frames",
            "pose_graph_verify_orb_aggregate_pairs_config",
            "pose_graph_apply_support_policy_config",
            "pose_graph_information_policy_config",
            "pose_graph_loop_robust_kernel_config",
            "corrected_ate_actual_rmse_cm_mean",
            "corrected_ate_actual_rmse_cm_std",
            "fps_mean",
            "fps_std",
            "gpu_memory_mb_mean",
            "loop_observation_est_rgbd_payload_mb_mean",
            "submap_count_mean",
            "submap_count_std",
            "loop_accepted_mean",
            "unique_handoff_signatures",
            "handoff_signature_stable",
        )
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=aggregate_fields)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({field: row.get(field, "") for field in aggregate_fields})


if __name__ == "__main__":
    main()
