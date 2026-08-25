#!/usr/bin/env python3
"""Summarize TUM GPS-SLAM run outputs without changing the timed runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml


PROFILE_FINGERPRINT_IGNORE_PIPE_KEYS = {
    "experiment_config_fingerprint",
    "experiment_group_key",
    "experiment_profile_fingerprint",
    "experiment_profile_name",
    "experiment_profile_summary",
    "experiment_repeat_count",
    "experiment_repeat_index",
    "experiment_repeat_seed_step",
    "random_seed",
}


def normalize_for_profile_fingerprint(value):
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if key == "workspace_dir":
                continue
            if key == "PIPE" and isinstance(child, dict):
                child = {
                    pipe_key: pipe_value
                    for pipe_key, pipe_value in child.items()
                    if pipe_key not in PROFILE_FINGERPRINT_IGNORE_PIPE_KEYS
                }
            normalized[key] = normalize_for_profile_fingerprint(child)
        return normalized
    if isinstance(value, list):
        return [normalize_for_profile_fingerprint(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_profile_fingerprint(cfg: dict) -> str:
    if not cfg:
        return ""
    encoded = json.dumps(
        normalize_for_profile_fingerprint(cfg),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def nested_get(cfg: dict, path: tuple[str, ...], default=""):
    value = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def slug_key(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return cleaned or "empty"


def format_counts(counts: dict[str, int]) -> str:
    return ";".join(
        f"{key}:{value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


CONFIG_PROFILE_FIELDS = (
    ("benchmark_protocol", ("PIPE", "benchmark_protocol")),
    ("benchmark_role", ("PIPE", "benchmark_role")),
    ("sequence_input_dir", ("READER", "input_dir")),
    ("reader_preload_images", ("READER", "preload_images")),
    ("reader_start_frame", ("READER", "start_frame")),
    ("reader_end_frame", ("READER", "end_frame")),
    ("reader_downscale_factor", ("READER", "downscale_factor")),
    ("work_mode", ("work_mode",)),
    ("config_save_after_train", ("PIPE", "save_after_train")),
    ("config_eval_after_train", ("PIPE", "eval_after_train")),
    ("config_eval_split", ("PIPE", "eval_split")),
    ("config_eval_stride", ("PIPE", "eval_stride")),
    ("config_eval_max_frames", ("PIPE", "eval_max_frames")),
    ("tsdf_engine", ("PIPE", "TSDF", "engine")),
    ("tsdf_tracking_failure_mode", ("PIPE", "TSDF", "tracking_failure_mode")),
    ("tsdf_tracker_config", ("PIPE", "TSDF", "tracker_config")),
    ("tsdf_tracking_profile", ("PIPE", "TSDF", "tracking_profile")),
    ("tsdf_use_bilateral_filter", ("PIPE", "TSDF", "use_bilateral_filter")),
    ("tsdf_use_approximate_raycast", ("PIPE", "TSDF", "use_approximate_raycast")),
    ("tsdf_input_mode", ("PIPE", "TSDF", "input_mode")),
    ("tsdf_use_gt_pose", ("PIPE", "TSDF", "use_gt_pose")),
    ("tsdf_reject_nonfinite_tracking_pose", ("PIPE", "TSDF", "reject_nonfinite_tracking_pose")),
    ("tsdf_require_raw_tracking_success", ("PIPE", "TSDF", "require_raw_tracking_success")),
    ("tsdf_basic_debug_tracking_start_frame", ("PIPE", "TSDF", "basic_debug_tracking_start_frame")),
    ("tsdf_basic_debug_tracking_end_frame", ("PIPE", "TSDF", "basic_debug_tracking_end_frame")),
    ("tsdf_force_single_submap", ("PIPE", "TSDF", "multi", "force_single_submap")),
    ("tsdf_console_diagnostics", ("PIPE", "TSDF", "multi", "console_diagnostics")),
    ("tsdf_hash_profile", ("PIPE", "TSDF", "multi", "hash_profile")),
    ("tsdf_paging", ("PIPE", "TSDF", "multi", "paging")),
    ("tsdf_paging_storage", ("PIPE", "TSDF", "multi", "paging_storage")),
    ("tsdf_resident_gpu_maps", ("PIPE", "TSDF", "multi", "resident_gpu_maps")),
    ("tsdf_cpu_page_cache_limit_mb", ("PIPE", "TSDF", "multi", "cpu_page_cache_limit_mb")),
    ("tsdf_boundary_policy", ("PIPE", "TSDF", "multi", "boundary_policy")),
    ("tsdf_allow_visibility_handoff", ("PIPE", "TSDF", "multi", "allow_visibility_handoff")),
    ("tsdf_occupancy_threshold_policy", ("PIPE", "TSDF", "multi", "occupancy_threshold_policy")),
    ("tsdf_adaptive_new_map_occupancy", ("PIPE", "TSDF", "multi", "adaptive_new_map_block_occupancy")),
    ("tsdf_adaptive_max_occupancy", ("PIPE", "TSDF", "multi", "adaptive_max_block_occupancy")),
    ("tsdf_adaptive_min_primary_frames", ("PIPE", "TSDF", "multi", "adaptive_occupancy_min_primary_frames")),
    ("tsdf_adaptive_min_processed_frames", ("PIPE", "TSDF", "multi", "adaptive_occupancy_min_processed_frames")),
    ("tsdf_adaptive_max_raw_weak_streak", ("PIPE", "TSDF", "multi", "adaptive_occupancy_max_raw_weak_streak")),
    ("tsdf_adaptive_max_raw_failed_streak", ("PIPE", "TSDF", "multi", "adaptive_occupancy_max_raw_failed_streak")),
    ("tsdf_max_frames", ("PIPE", "TSDF", "multi", "max_frames")),
    ("tsdf_overlap_frames", ("PIPE", "TSDF", "multi", "overlap_frames")),
    ("tsdf_emergency_overlap_frames", ("PIPE", "TSDF", "multi", "emergency_overlap_frames")),
    ("tsdf_trial_frames", ("PIPE", "TSDF", "multi", "trial_frames")),
    ("tsdf_new_map_block_occupancy", ("PIPE", "TSDF", "multi", "new_map_block_occupancy")),
    ("tsdf_max_block_occupancy", ("PIPE", "TSDF", "multi", "max_block_occupancy")),
    ("tsdf_handoff_pose_policy", ("PIPE", "TSDF", "multi", "handoff_pose_policy")),
    ("tsdf_new_map_handoff_policy", ("PIPE", "TSDF", "multi", "new_map_handoff_policy")),
    ("tsdf_new_map_handoff_min_accepts", ("PIPE", "TSDF", "multi", "new_map_handoff_min_accepts")),
    ("tsdf_new_map_handoff_emergency_min_accepts", ("PIPE", "TSDF", "multi", "new_map_handoff_emergency_min_accepts")),
    ("tsdf_sequential_transform_policy", ("PIPE", "TSDF", "multi", "sequential_transform_policy")),
    ("tsdf_sequential_transform_max_translation", ("PIPE", "TSDF", "multi", "sequential_transform_max_translation")),
    ("tsdf_occupancy_growth_lookahead_frames", ("PIPE", "TSDF", "multi", "occupancy_growth_lookahead_frames")),
    ("tsdf_tracking_boundary_policy", ("PIPE", "TSDF", "multi", "tracking_boundary_policy")),
    ("tsdf_tracking_boundary_handoff_policy", ("PIPE", "TSDF", "multi", "tracking_boundary_handoff_policy")),
    ("tsdf_raw_weak_boundary_frames", ("PIPE", "TSDF", "multi", "raw_weak_boundary_frames")),
    ("tsdf_raw_failed_boundary_frames", ("PIPE", "TSDF", "multi", "raw_failed_boundary_frames")),
    ("tsdf_tracking_boundary_min_primary_frames", ("PIPE", "TSDF", "multi", "tracking_boundary_min_primary_frames")),
    ("tsdf_tracking_boundary_cooldown_frames", ("PIPE", "TSDF", "multi", "tracking_boundary_cooldown_frames")),
    ("tsdf_primary_failure_handoff_min_raw_failed_frames", ("PIPE", "TSDF", "multi", "primary_failure_handoff_min_raw_failed_frames")),
    ("tsdf_primary_failure_handoff_raw_gate", ("PIPE", "TSDF", "multi", "primary_failure_handoff_raw_gate")),
    ("tsdf_primary_failure_handoff_min_raw_weak_frames", ("PIPE", "TSDF", "multi", "primary_failure_handoff_min_raw_weak_frames")),
    ("tsdf_primary_failure_handoff_require_pose_continuity", ("PIPE", "TSDF", "multi", "primary_failure_handoff_require_pose_continuity")),
    ("gaussian_submaps_enabled", ("PIPE", "gaussian_submaps", "enable")),
    ("gaussian_submap_ownership", ("PIPE", "gaussian_submaps", "ownership")),
    ("gaussian_submap_coordinate_frame", ("PIPE", "gaussian_submaps", "coordinate_frame")),
    ("gaussian_submap_render_policy", ("PIPE", "gaussian_submaps", "render_policy")),
    ("gaussian_submap_max_frames", ("PIPE", "gaussian_submaps", "max_frames")),
    ("gaussian_submap_max_gaussians", ("PIPE", "gaussian_submaps", "max_gaussians")),
    ("gaussian_submap_max_new_per_frame", ("PIPE", "gaussian_submaps", "max_new_gaussians_per_frame")),
    ("gaussian_submap_page_frozen", ("PIPE", "gaussian_submaps", "page_frozen_gaussians")),
    ("gaussian_submap_resident_count", ("PIPE", "gaussian_submaps", "resident_gaussian_submaps")),
    ("gaussian_render_min_ready_frames", ("PIPE", "gaussian_submaps", "render_min_ready_frames")),
    ("gaussian_render_owner_min_ready_frames", ("PIPE", "gaussian_submaps", "render_owner_min_ready_frames")),
    ("gaussian_render_neighbor_min_ready_frames", ("PIPE", "gaussian_submaps", "render_neighbor_min_ready_frames")),
    ("pose_graph_overlap_roundtrip_check", ("PIPE", "gaussian_submaps", "pose_graph_overlap_roundtrip_check")),
    ("pose_graph_overlap_roundtrip_tolerance", ("PIPE", "gaussian_submaps", "pose_graph_overlap_roundtrip_tolerance")),
    ("new_gs_sample_ratio", ("PIPE", "new_gs_sample_ratio")),
    ("local_opt_iters", ("PIPE", "local_opt_iters")),
    ("local_opt_interval", ("PIPE", "local_opt_interval")),
    ("keyframe_select_max", ("PIPE", "keyframe_select_max")),
    ("model_sh_degree", ("MODEL", "sh_degree")),
    ("pose_graph_apply_mode_config", ("PIPE", "pose_graph", "apply_mode")),
    ("pose_graph_candidate_detection_config", ("PIPE", "pose_graph", "candidate_detection")),
    ("pose_graph_online_updates_config", ("PIPE", "pose_graph", "online_updates")),
    ("pose_graph_run_on_submap_events_config", ("PIPE", "pose_graph", "run_on_submap_events")),
    ("pose_graph_candidate_interval_frames_config", ("PIPE", "pose_graph", "candidate_interval_frames")),
    ("pose_graph_online_min_submaps_config", ("PIPE", "pose_graph", "online_min_submaps")),
    ("pose_graph_online_min_submap_growth_config", ("PIPE", "pose_graph", "online_min_submap_growth")),
    ("pose_graph_max_updates_per_run_config", ("PIPE", "pose_graph", "max_updates_per_run")),
    ("pose_graph_terminal_online_scan_config", ("PIPE", "pose_graph", "terminal_online_scan")),
    ("pose_graph_candidate_order_config", ("PIPE", "pose_graph", "candidate_order")),
    ("pose_graph_candidate_verification_config", ("PIPE", "pose_graph", "candidate_verification")),
    ("pose_graph_loop_max_candidates_config", ("PIPE", "pose_graph", "loop_max_candidates")),
    ("pose_graph_loop_candidate_export_policy_config", ("PIPE", "pose_graph", "loop_candidate_export_policy")),
    ("pose_graph_loop_candidate_export_span_reserve_fraction_config", ("PIPE", "pose_graph", "loop_candidate_export_span_reserve_fraction")),
    ("pose_graph_loop_candidate_export_min_span_config", ("PIPE", "pose_graph", "loop_candidate_export_min_span")),
    ("pose_graph_orb_match_policy_config", ("PIPE", "pose_graph", "orb_match_policy")),
    ("pose_graph_verify_orb_aggregate_pairs_config", ("PIPE", "pose_graph", "verify_orb_aggregate_pairs")),
    ("pose_graph_verify_orb_aggregate_max_pairs_config", ("PIPE", "pose_graph", "verify_orb_aggregate_max_pairs")),
    ("pose_graph_verify_max_observation_pairs_config", ("PIPE", "pose_graph", "verify_max_observation_pairs")),
    ("pose_graph_verify_pair_selection_config", ("PIPE", "pose_graph", "verify_pair_selection")),
    ("pose_graph_verify_icp_max_translation_delta_config", ("PIPE", "pose_graph", "verify_icp_max_translation_delta")),
    ("pose_graph_verify_icp_max_rotation_delta_config", ("PIPE", "pose_graph", "verify_icp_max_rotation_delta_deg")),
    ("pose_graph_verify_icp_max_score_config", ("PIPE", "pose_graph", "verify_icp_max_score")),
    ("pose_graph_verify_icp_drift_aware_delta_config", ("PIPE", "pose_graph", "verify_icp_drift_aware_delta")),
    ("pose_graph_verify_icp_drift_reference_span_config", ("PIPE", "pose_graph", "verify_icp_drift_reference_span")),
    ("pose_graph_verify_icp_drift_max_scale_config", ("PIPE", "pose_graph", "verify_icp_drift_max_scale")),
    ("pose_graph_verify_icp_drift_score_margin_config", ("PIPE", "pose_graph", "verify_icp_drift_score_margin")),
    ("pose_graph_verify_icp_weighted_kabsch_config", ("PIPE", "pose_graph", "verify_icp_weighted_kabsch")),
    ("pose_graph_verify_icp_depth_adaptive_ransac_config", ("PIPE", "pose_graph", "verify_icp_depth_adaptive_ransac")),
    ("pose_graph_verify_icp_depth_noise_scale_config", ("PIPE", "pose_graph", "verify_icp_depth_noise_scale")),
    ("pose_graph_verify_icp_depth_adaptive_threshold_scale_config", ("PIPE", "pose_graph", "verify_icp_depth_adaptive_threshold_scale")),
    ("pose_graph_solve_only_max_translation_delta_config", ("PIPE", "pose_graph", "solve_only_max_translation_delta")),
    ("pose_graph_solve_only_max_rotation_delta_config", ("PIPE", "pose_graph", "solve_only_max_rotation_delta_deg")),
    ("pose_graph_apply_max_translation_delta_config", ("PIPE", "pose_graph", "apply_max_translation_delta")),
    ("pose_graph_apply_max_rotation_delta_config", ("PIPE", "pose_graph", "apply_max_rotation_delta_deg")),
    ("pose_graph_apply_support_policy_config", ("PIPE", "pose_graph", "apply_support_policy")),
    ("pose_graph_apply_min_loop_constraints_config", ("PIPE", "pose_graph", "apply_min_loop_constraints")),
    ("pose_graph_apply_min_loop_submap_span_config", ("PIPE", "pose_graph", "apply_min_loop_submap_span")),
    ("pose_graph_apply_min_loop_consistent_observations_config", ("PIPE", "pose_graph", "apply_min_loop_consistent_observations")),
    ("pose_graph_apply_min_loop_icp_inliers_config", ("PIPE", "pose_graph", "apply_min_loop_icp_inliers")),
    ("pose_graph_apply_min_translation_delta_config", ("PIPE", "pose_graph", "apply_min_translation_delta")),
    ("pose_graph_apply_adaptive_min_loop_constraints_config", ("PIPE", "pose_graph", "apply_adaptive_min_loop_constraints")),
    ("pose_graph_apply_adaptive_min_loop_submap_span_config", ("PIPE", "pose_graph", "apply_adaptive_min_loop_submap_span")),
    ("pose_graph_apply_adaptive_min_loop_consistent_observations_config", ("PIPE", "pose_graph", "apply_adaptive_min_loop_consistent_observations")),
    ("pose_graph_apply_adaptive_min_loop_icp_inliers_config", ("PIPE", "pose_graph", "apply_adaptive_min_loop_icp_inliers")),
    ("pose_graph_apply_high_confidence_min_loop_constraints_config", ("PIPE", "pose_graph", "apply_high_confidence_min_loop_constraints")),
    ("pose_graph_apply_high_confidence_min_loop_submap_span_config", ("PIPE", "pose_graph", "apply_high_confidence_min_loop_submap_span")),
    ("pose_graph_apply_high_confidence_min_loop_consistent_observations_config", ("PIPE", "pose_graph", "apply_high_confidence_min_loop_consistent_observations")),
    ("pose_graph_apply_high_confidence_min_loop_icp_inliers_config", ("PIPE", "pose_graph", "apply_high_confidence_min_loop_icp_inliers")),
    ("pose_graph_apply_high_confidence_max_loop_score_config", ("PIPE", "pose_graph", "apply_high_confidence_max_loop_score")),
    ("pose_graph_apply_high_confidence_max_icp_rmse_config", ("PIPE", "pose_graph", "apply_high_confidence_max_icp_rmse")),
    ("pose_graph_apply_observation_gate_config", ("PIPE", "pose_graph", "apply_observation_gate")),
    ("pose_graph_apply_observation_gate_min_samples_config", ("PIPE", "pose_graph", "apply_observation_gate_min_samples")),
    ("pose_graph_apply_observation_gate_max_samples_config", ("PIPE", "pose_graph", "apply_observation_gate_max_samples")),
    ("pose_graph_apply_observation_gate_sample_policy_config", ("PIPE", "pose_graph", "apply_observation_gate_sample_policy")),
    ("pose_graph_apply_observation_prefetch_loop_endpoints_config", ("PIPE", "pose_graph", "apply_observation_prefetch_loop_endpoints")),
    ("pose_graph_apply_observation_prefetch_max_maps_config", ("PIPE", "pose_graph", "apply_observation_prefetch_max_maps")),
    ("pose_graph_apply_observation_prefetch_pin_ttl_frames_config", ("PIPE", "pose_graph", "apply_observation_prefetch_pin_ttl_frames")),
    ("pose_graph_apply_observation_defer_nonresident_config", ("PIPE", "pose_graph", "apply_observation_defer_nonresident")),
    ("pose_graph_apply_observation_defer_retry_frames_config", ("PIPE", "pose_graph", "apply_observation_defer_retry_frames")),
    ("pose_graph_apply_observation_pin_deferred_maps_config", ("PIPE", "pose_graph", "apply_observation_pin_deferred_maps")),
    ("pose_graph_apply_observation_defer_stops_scale_backoff_config", ("PIPE", "pose_graph", "apply_observation_defer_stops_scale_backoff")),
    ("pose_graph_apply_observation_terminal_sync_page_in_config", ("PIPE", "pose_graph", "apply_observation_terminal_sync_page_in")),
    ("pose_graph_apply_observation_skip_resident_trim_config", ("PIPE", "pose_graph", "apply_observation_skip_resident_trim")),
    ("pose_graph_apply_observation_gate_tolerance_policy_config", ("PIPE", "pose_graph", "apply_observation_gate_tolerance_policy")),
    ("pose_graph_apply_observation_depth_rmse_ratio_floor_config", ("PIPE", "pose_graph", "apply_observation_depth_rmse_ratio_floor")),
    ("pose_graph_apply_observation_max_depth_rmse_absolute_increase_config", ("PIPE", "pose_graph", "apply_observation_max_depth_rmse_absolute_increase")),
    ("pose_graph_apply_observation_max_valid_fraction_drop_config", ("PIPE", "pose_graph", "apply_observation_max_valid_fraction_drop")),
    ("pose_graph_apply_loop_selection_policy_config", ("PIPE", "pose_graph", "apply_loop_selection_policy")),
    ("pose_graph_apply_pcm_min_clique_size_config", ("PIPE", "pose_graph", "apply_pcm_min_clique_size")),
    ("pose_graph_apply_pcm_max_trial_cliques_config", ("PIPE", "pose_graph", "apply_pcm_max_trial_cliques")),
    ("pose_graph_apply_pcm_event_cluster_radius_config", ("PIPE", "pose_graph", "apply_pcm_event_cluster_radius")),
    ("pose_graph_apply_pcm_min_distinct_events_config", ("PIPE", "pose_graph", "apply_pcm_min_distinct_events")),
    ("pose_graph_apply_pcm_min_long_span_config", ("PIPE", "pose_graph", "apply_pcm_min_long_span")),
    ("pose_graph_apply_pcm_prefer_long_span_config", ("PIPE", "pose_graph", "apply_pcm_prefer_long_span")),
    ("pose_graph_apply_pcm_scale_threshold_by_cycle_length_config", ("PIPE", "pose_graph", "apply_pcm_scale_threshold_by_cycle_length")),
    ("pose_graph_apply_pcm_translation_threshold_config", ("PIPE", "pose_graph", "apply_pcm_translation_threshold")),
    ("pose_graph_apply_pcm_rotation_threshold_deg_config", ("PIPE", "pose_graph", "apply_pcm_rotation_threshold_deg")),
    ("pose_graph_apply_repeat_min_new_nodes_config", ("PIPE", "pose_graph", "apply_repeat_min_new_nodes")),
    ("pose_graph_information_policy_config", ("PIPE", "pose_graph", "information_policy")),
    ("pose_graph_loop_robust_kernel_config", ("PIPE", "pose_graph", "loop_robust_kernel")),
    ("pose_graph_loop_robust_scale_config", ("PIPE", "pose_graph", "loop_robust_scale")),
    ("pose_graph_apply_health_gate_config", ("PIPE", "pose_graph", "apply_health_gate")),
    ("pose_graph_apply_health_max_raw_failed_ratio_config", ("PIPE", "pose_graph", "apply_health_max_raw_failed_ratio")),
    ("pose_graph_apply_health_max_raw_non_good_ratio_config", ("PIPE", "pose_graph", "apply_health_max_raw_non_good_ratio")),
    ("pose_graph_apply_health_max_raw_failed_streak_config", ("PIPE", "pose_graph", "apply_health_max_raw_failed_streak")),
    ("pose_graph_apply_health_min_frames_config", ("PIPE", "pose_graph", "apply_health_min_frames")),
    ("pose_graph_sequential_constraint_weight_config", ("PIPE", "pose_graph", "sequential_constraint_weight")),
    ("pose_graph_loop_constraint_weight_config", ("PIPE", "pose_graph", "loop_constraint_weight")),
    ("pose_graph_anchor_policy_config", ("PIPE", "pose_graph", "anchor_policy")),
)


def collect_config_profile_fields(cfg: dict) -> dict[str, float | int | str]:
    if not cfg:
        return {}
    pipe = cfg.get("PIPE", {}) if isinstance(cfg.get("PIPE"), dict) else {}
    row: dict[str, float | int | str] = {
        "config_profile_fingerprint": config_profile_fingerprint(cfg),
    }
    for key in (
        "experiment_config_fingerprint",
        "experiment_profile_fingerprint",
        "experiment_profile_name",
    ):
        if key in pipe:
            row[key] = pipe[key]
    for out_key, path in CONFIG_PROFILE_FIELDS:
        value = nested_get(cfg, path)
        if isinstance(value, bool):
            row[out_key] = int(value)
        elif value != "":
            row[out_key] = value
    return row


def pose_id(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def read_poses(pose_dir: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    if not pose_dir.exists():
        return poses
    for path in sorted(pose_dir.glob("*.txt")):
        idx = pose_id(path)
        if idx is None:
            continue
        pose = np.loadtxt(path)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            poses[idx] = pose.astype(np.float64)
    return poses


def align_positions(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est_centered = est - est.mean(axis=1, keepdims=True)
    gt_centered = gt - gt.mean(axis=1, keepdims=True)
    w = est_centered @ gt_centered.T
    u, _, vh = np.linalg.svd(w.T)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        s[2, 2] = -1.0
    rot = u @ s @ vh
    trans = gt.mean(axis=1, keepdims=True) - rot @ est.mean(axis=1, keepdims=True)
    return rot, trans


def ate_error_series(est_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
    common = sorted(set(est_poses) & set(gt_poses))
    if not common:
        return [], np.asarray([], dtype=np.float64)
    est = np.stack([est_poses[idx][:3, 3] for idx in common], axis=1)
    gt = np.stack([gt_poses[idx][:3, 3] for idx in common], axis=1)
    rot, trans = align_positions(est, gt)
    err = rot @ est + trans - gt
    return common, np.sqrt(np.sum(err * err, axis=0))


def ate_metrics(est_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray]) -> dict[str, float | int | str]:
    common, trans_error = ate_error_series(est_poses, gt_poses)
    if not common:
        return {"matched_poses": 0, "ate_error": "no_matching_poses"}
    return {
        "matched_poses": len(common),
        "ate_mean_cm": float(trans_error.mean() * 100.0),
        "ate_actual_rmse_cm": float(np.sqrt(np.mean(np.square(trans_error))) * 100.0),
    }


def ate_segment_metrics(
    est_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    prefix: str,
) -> dict[str, float | int | str]:
    common, trans_error = ate_error_series(est_poses, gt_poses)
    if not common:
        return {f"{prefix}_segment_error": "no_matching_poses"}
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    segments = {
        "before": [i for i, frame_id in enumerate(common) if frame_id <= start_frame],
        "between": [i for i, frame_id in enumerate(common) if start_frame <= frame_id <= end_frame],
        "after": [i for i, frame_id in enumerate(common) if frame_id >= end_frame],
    }
    out: dict[str, float | int | str] = {
        f"{prefix}_segment_start_frame": start_frame,
        f"{prefix}_segment_end_frame": end_frame,
    }
    for name, indices in segments.items():
        key = f"{prefix}_{name}"
        out[f"{key}_matched_poses"] = len(indices)
        if not indices:
            out[f"{key}_ate_error"] = "empty_segment"
            continue
        errors = trans_error[np.asarray(indices, dtype=np.int64)]
        out[f"{key}_ate_mean_cm"] = float(errors.mean() * 100.0)
        out[f"{key}_ate_actual_rmse_cm"] = float(np.sqrt(np.mean(np.square(errors))) * 100.0)
    return out


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def aligned_est_poses(
    est_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray]
) -> tuple[list[int], dict[int, np.ndarray]]:
    common = sorted(set(est_poses) & set(gt_poses))
    if not common:
        return [], {}
    est_xyz = np.stack([est_poses[idx][:3, 3] for idx in common], axis=1)
    gt_xyz = np.stack([gt_poses[idx][:3, 3] for idx in common], axis=1)
    rot, trans = align_positions(est_xyz, gt_xyz)
    aligned_est = {}
    for idx, pose in est_poses.items():
        aligned = pose.copy()
        aligned[:3, :3] = rot @ aligned[:3, :3]
        aligned[:3, 3:4] = rot @ aligned[:3, 3:4] + trans
        aligned_est[idx] = aligned
    return common, aligned_est


def relative_pose_error_metrics_for_pairs(
    est_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    pairs: list[tuple[int, int]],
    prefix: str,
) -> dict[str, float | int | str]:
    common, aligned_est = aligned_est_poses(est_poses, gt_poses)
    if prefix == "rpe":
        pair_key = "rpe_pairs"
        error_key = "rpe_error"
        translation_key = "rpe_translation_rmse_cm"
        rotation_key = "rpe_rotation_rmse_deg"
    else:
        pair_key = f"{prefix}_rpe_pairs"
        error_key = f"{prefix}_rpe_error"
        translation_key = f"{prefix}_rpe_translation_rmse_cm"
        rotation_key = f"{prefix}_rpe_rotation_rmse_deg"
    if not common:
        return {pair_key: 0, error_key: "no_matching_poses"}
    if not pairs:
        return {pair_key: 0, error_key: "no_pairs"}

    trans_errors = []
    rot_errors = []
    for i, j in pairs:
        if i not in aligned_est or j not in aligned_est or i not in gt_poses or j not in gt_poses:
            continue
        est_rel = np.linalg.inv(aligned_est[i]) @ aligned_est[j]
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
        err = np.linalg.inv(gt_rel) @ est_rel
        trans_errors.append(float(np.linalg.norm(err[:3, 3])))
        rot_errors.append(rotation_angle_deg(err[:3, :3]))
    if not trans_errors:
        return {pair_key: 0, error_key: "no_valid_pairs"}
    trans = np.asarray(trans_errors)
    rot_err = np.asarray(rot_errors)
    return {
        pair_key: int(len(trans_errors)),
        translation_key: float(np.sqrt(np.mean(np.square(trans))) * 100.0),
        rotation_key: float(np.sqrt(np.mean(np.square(rot_err)))),
    }


def rpe_metrics(est_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray], delta: int) -> dict[str, float | int | str]:
    common, aligned_est = aligned_est_poses(est_poses, gt_poses)
    if len(common) <= delta:
        return {"rpe_pairs": 0, "rpe_error": "not_enough_poses"}
    common_set = set(common)
    pairs = [(idx, idx + delta) for idx in common if idx + delta in common_set]
    metrics = relative_pose_error_metrics_for_pairs(est_poses, gt_poses, pairs, "rpe")
    if metrics.get("rpe_pairs", 0) == 0:
        return {"rpe_pairs": 0, "rpe_error": "no_delta_pairs"}
    return metrics


def frame_assignment_boundary_pairs(run_dir: Path) -> list[tuple[int, int]]:
    path = run_dir / "frame_assignments.csv"
    if not path.exists():
        return []
    pairs: list[tuple[int, int]] = []
    previous_frame = -1
    previous_primary = -1
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = safe_int(row.get("frame_id"), safe_int(row.get("sequence_id"), -1))
            primary = safe_int(row.get("primary_submap"), safe_int(row.get("owner_submap"), -1))
            if frame < 0 or primary < 0:
                continue
            if previous_frame >= 0 and previous_primary >= 0 and primary != previous_primary:
                pairs.append((previous_frame, frame))
            previous_frame = frame
            previous_primary = primary
    return pairs


def parse_time_log(run_dir: Path) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    path = run_dir / "time_log.txt"
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    fps = re.search(r"FPS:\s*([0-9.]+)", text)
    gpu = re.search(r"GPU memory usage:\s*([0-9.]+)\s*MB", text)
    if fps:
        out["fps"] = float(fps.group(1))
    if gpu:
        out["gpu_memory_mb"] = int(float(gpu.group(1)))
    return out


def parse_runtime_tsdf_log(run_dir: Path) -> dict[str, str]:
    """Read the launcher-captured stdout log for runtime TSDF settings.

    Some generated configs keep MultiEngine-only defaults in
    ``PIPE.TSDF.multi`` even when the BasicEngine path is selected. The C++
    startup log is the authoritative source for the compiled hash profile used
    by both BasicEngine and MultiEngine binaries.
    """
    log_path = run_dir.parent / "run_logs" / f"{run_dir.name}.log"
    if not log_path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        return {"tsdf_runtime_log_error": str(exc)}

    engine_match = re.search(r"\[TSDF\]\s+engine=(basic|multi)\b", text)
    hash_match = re.search(r"\[TSDF\]\s+engine=(?:basic|multi)\b[^\n]*\bhash_profile=([^\s]+)", text)
    input_match = re.search(r"\[TSDF\]\s+engine=(?:basic|multi)\s+input_mode=([^\s]+)", text)
    if engine_match:
        out["tsdf_runtime_engine"] = engine_match.group(1)
    if hash_match:
        out["tsdf_runtime_hash_profile"] = hash_match.group(1)
    if input_match:
        out["tsdf_runtime_input_mode"] = input_match.group(1)
    return out


def safe_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_submap_stats(run_dir: Path) -> dict[str, float | int]:
    path = run_dir / "submap_stats.csv"
    if not path.exists():
        return {}
    max_submap_id = -1
    max_resident = 0
    max_occupancy = 0.0
    final_by_submap: dict[int, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            submap_id = safe_int(row.get("submap_id"), -1)
            max_submap_id = max(max_submap_id, submap_id)
            if submap_id >= 0:
                final_by_submap[submap_id] = row
            max_resident = max(max_resident, safe_int(row.get("resident_count"), 0))
            occupancy = safe_float(row.get("occupancy"))
            if occupancy is not None:
                max_occupancy = max(max_occupancy, occupancy)
    loop_observation_counts = [
        safe_int(row.get("loop_observations"), 0) for row in final_by_submap.values()
    ]
    keyframe_counts = [
        safe_int(row.get("keyframes"), 0) for row in final_by_submap.values()
    ]
    return {
        "submap_count": max_submap_id + 1 if max_submap_id >= 0 else 0,
        "max_resident_sdf_maps": max_resident,
        "max_sdf_occupancy": max_occupancy,
        "loop_observations_total_reported": sum(loop_observation_counts),
        "loop_observations_max_per_submap_reported": max(loop_observation_counts, default=0),
        "keyframes_total_reported": sum(keyframe_counts),
        "keyframes_max_per_submap_reported": max(keyframe_counts, default=0),
    }


def add_loop_memory_estimate(row: dict[str, float | int | str], cfg: dict) -> None:
    """Estimate CPU payload if all stored loop observations loaded RGB-D at once.

    Runtime loop-observation cameras keep poses and file paths, not resident RGB-D
    tensors. This estimate is intentionally conservative and exists to separate
    bounded loop-observation metadata from the measured peak GPU memory in
    time_log.txt.
    """
    observations = safe_int(row.get("loop_observations_total_reported"), 0)
    image_shape = nested_get(cfg, ("READER", "image_shape"), [])
    if not observations or not isinstance(image_shape, list) or len(image_shape) < 2:
        return
    width = safe_int(image_shape[0], 0)
    height = safe_int(image_shape[1], 0)
    if width <= 0 or height <= 0:
        return
    bytes_per_observation = width * height * 4 * 4  # RGB float32 + depth float32.
    row["loop_observation_est_rgbd_payload_mb"] = observations * bytes_per_observation / 1024.0**2
    row["loop_observation_est_pose_metadata_mb"] = observations * 3 * 16 * 4 / 1024.0**2


def parse_frame_assignments(run_dir: Path) -> dict[str, float | int]:
    path = run_dir / "frame_assignments.csv"
    if not path.exists():
        return {}
    out = {
        "frame_assignment_rows": 0,
        "assignment_keyframes": 0,
        "loop_observed_frames": 0,
        "loop_observation_assignments": 0,
        "max_loop_observed_submaps_per_frame": 0,
    }
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out["frame_assignment_rows"] += 1
            if safe_int(row.get("is_keyframe"), 0) != 0:
                out["assignment_keyframes"] += 1
            observed = str(row.get("loop_observed_submaps", "") or "").strip()
            if not observed:
                continue
            submaps = [part for part in observed.split("|") if part.strip()]
            if not submaps:
                continue
            out["loop_observed_frames"] += 1
            out["loop_observation_assignments"] += len(submaps)
            out["max_loop_observed_submaps_per_frame"] = max(
                int(out["max_loop_observed_submaps_per_frame"]),
                len(submaps),
            )
    return out


def parse_transitions(run_dir: Path) -> dict[str, float | int]:
    path = run_dir / "submap_transitions.csv"
    if not path.exists():
        return {}
    out = {
        "transition_rows": 0,
        "sequential_observed": 0,
        "sequential_initial": 0,
        "sequential_emergency": 0,
        "sequential_graph_rejects": 0,
        "sequential_restarts": 0,
        "link_rejects": 0,
        "capacity_handoff_rejects": 0,
        "handoff_moves": 0,
        "normal_handoff_moves": 0,
        "capacity_handoff_moves": 0,
        "accepted_handoff_moves": 0,
        "visibility_handoff_moves": 0,
        "primary_failure_handoffs": 0,
        "primary_failure_rejected_transform": 0,
        "primary_failure_insufficient_overlap": 0,
        "primary_failure_trial_window": 0,
        "primary_failure_trigger_failed": 0,
        "primary_failure_trigger_weak": 0,
        "primary_failure_trigger_failed_and_weak": 0,
        "primary_failure_trigger_unknown": 0,
        "candidate_recovery_retries": 0,
        "handoff_waits": 0,
        "handoff_wait_visibility": 0,
        "handoff_wait_transform": 0,
        "handoff_wait_maturity": 0,
        "max_rejected_transform_delta_cm": 0.0,
        "max_rejected_transform_delta_deg": 0.0,
        "max_primary_raw_weak_at_handoff": 0,
        "max_primary_raw_failed_at_handoff": 0,
        "max_candidate_good_at_handoff": 0,
        "max_candidate_failed_at_handoff": 0,
    }
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            event = row.get("event", "")
            action = row.get("action", "")
            detail = row.get("detail", "")
            out["transition_rows"] += 1
            if event == "sequential_transform" and action == "observed":
                out["sequential_observed"] += 1
            if event == "sequential_transform" and action == "initial":
                out["sequential_initial"] += 1
            if event == "sequential_transform" and str(action).startswith("emergency_"):
                out["sequential_emergency"] += 1
            if event == "sequential_transform_reject":
                out["sequential_graph_rejects"] += 1
                delta_m = safe_float(row.get("delta_m"), 0.0)
                delta_deg = safe_float(row.get("delta_deg"), 0.0)
                if delta_m is not None:
                    out["max_rejected_transform_delta_cm"] = max(
                        float(out["max_rejected_transform_delta_cm"]), delta_m * 100.0
                    )
                if delta_deg is not None:
                    out["max_rejected_transform_delta_deg"] = max(
                        float(out["max_rejected_transform_delta_deg"]), delta_deg
                    )
            if event == "sequential_transform_restart":
                out["sequential_restarts"] += 1
            if event == "link_reject":
                out["link_rejects"] += 1
            if event == "capacity_handoff_reject":
                out["capacity_handoff_rejects"] += 1
            if event == "primary_failure_handoff":
                out["primary_failure_handoffs"] += 1
                if detail.startswith("rejected_transform"):
                    out["primary_failure_rejected_transform"] += 1
                elif detail.startswith("insufficient_overlap"):
                    out["primary_failure_insufficient_overlap"] += 1
                elif detail.startswith("trial_window"):
                    out["primary_failure_trial_window"] += 1
                if "trigger=failed_and_weak" in detail:
                    out["primary_failure_trigger_failed_and_weak"] += 1
                elif "trigger=failed" in detail:
                    out["primary_failure_trigger_failed"] += 1
                elif "trigger=weak" in detail:
                    out["primary_failure_trigger_weak"] += 1
                else:
                    out["primary_failure_trigger_unknown"] += 1
            if event == "link_recovery_keep":
                out["candidate_recovery_retries"] += 1
            if row.get("move_to_primary") == "1":
                out["handoff_moves"] += 1
                if event != "primary_failure_handoff":
                    out["normal_handoff_moves"] += 1
                if event == "capacity_handoff":
                    out["capacity_handoff_moves"] += 1
                elif event == "accepted_handoff":
                    out["accepted_handoff_moves"] += 1
                elif event == "visibility_handoff":
                    out["visibility_handoff_moves"] += 1
                out["max_primary_raw_weak_at_handoff"] = max(
                    out["max_primary_raw_weak_at_handoff"],
                    safe_int(row.get("primary_raw_weak"), 0),
                )
                out["max_primary_raw_failed_at_handoff"] = max(
                    out["max_primary_raw_failed_at_handoff"],
                    safe_int(row.get("primary_raw_failed"), 0),
                )
                out["max_candidate_good_at_handoff"] = max(
                    out["max_candidate_good_at_handoff"],
                    safe_int(row.get("candidate_good"), 0),
                )
                out["max_candidate_failed_at_handoff"] = max(
                    out["max_candidate_failed_at_handoff"],
                    safe_int(row.get("candidate_failed"), 0),
                )
            if action == "wait_transform":
                out["handoff_waits"] += 1
                out["handoff_wait_transform"] += 1
            if action == "wait_visibility_gate":
                out["handoff_waits"] += 1
                out["handoff_wait_visibility"] += 1
            if action == "wait_maturity":
                out["handoff_waits"] += 1
                out["handoff_wait_maturity"] += 1
    return out


def parse_paging(run_dir: Path) -> dict[str, float | int | str]:
    path = run_dir / "paging.csv"
    if not path.exists():
        return {}
    out: dict[str, float | int | str] = {
        "paging_rows": 0,
        "paging_total_spill_bytes": 0,
        "paging_max_duration_ms": 0.0,
        "paging_max_resident_estimated_mb": 0.0,
        "paging_max_pending_estimated_mb": 0.0,
        "paging_max_pending_pagein_estimated_mb": 0.0,
        "paging_max_passive_cpu_cache_estimated_mb": 0.0,
        "paging_cpu_page_cache_limit_mb": 0.0,
    }
    action_counts: dict[str, int] = {}
    action_max_ms: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out["paging_rows"] = int(out["paging_rows"]) + 1
            action = slug_key(str(row.get("action", "") or "unknown"))
            action_counts[action] = action_counts.get(action, 0) + 1
            duration = safe_float(row.get("duration_ms"), 0.0) or 0.0
            action_max_ms[action] = max(action_max_ms.get(action, 0.0), duration)
            out["paging_max_duration_ms"] = max(float(out["paging_max_duration_ms"]), duration)
            page_bytes = safe_int(row.get("page_bytes"), -1)
            if page_bytes > 0 and action == "cpu_cache_spill":
                out["paging_total_spill_bytes"] = int(out["paging_total_spill_bytes"]) + page_bytes
            for src, dst in (
                ("resident_estimated_bytes", "paging_max_resident_estimated_mb"),
                ("pending_estimated_bytes", "paging_max_pending_estimated_mb"),
                ("pending_pagein_estimated_bytes", "paging_max_pending_pagein_estimated_mb"),
                ("passive_cpu_cache_estimated_bytes", "paging_max_passive_cpu_cache_estimated_mb"),
                ("cpu_page_cache_limit_bytes", "paging_cpu_page_cache_limit_mb"),
            ):
                value = safe_float(row.get(src))
                if value is None:
                    continue
                out[dst] = max(float(out[dst]), value / 1024.0**2)
    if action_counts:
        out["paging_action_counts"] = format_counts(action_counts)
        for action, count in sorted(action_counts.items()):
            out[f"paging_{action}_count"] = count
            out[f"paging_{action}_max_ms"] = action_max_ms.get(action, 0.0)
    out["paging_total_spill_mb"] = int(out["paging_total_spill_bytes"]) / 1024.0**2
    return out


def parse_tracking(run_dir: Path) -> dict[str, int]:
    path = run_dir / "tracking.csv"
    if not path.exists():
        return {}
    out = {
        "tracking_rows": 0,
        "raw_tracking_good": 0,
        "raw_tracking_poor": 0,
        "raw_tracking_failed": 0,
        "raw_tracking_non_good": 0,
        "policy_tracking_good": 0,
        "policy_tracking_poor": 0,
        "policy_tracking_failed": 0,
        "policy_tracking_non_good": 0,
        "raw_tracking_max_failed_streak": 0,
        "raw_tracking_first_weak_frame": -1,
        "raw_tracking_first_failed_frame": -1,
        "tracking_containment_pose_rollbacks": 0,
        "tracking_containment_fusion_suppressed": 0,
    }
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out["tracking_rows"] += 1
            frame = safe_int(row.get("frame_id"), safe_int(row.get("sequence_id"), -1))
            raw = safe_int(row.get("raw_primary_tracking_result"), -1)
            policy = safe_int(row.get("policy_primary_tracking_result"), -1)
            if raw == 2:
                out["raw_tracking_good"] += 1
            elif raw == 1:
                out["raw_tracking_poor"] += 1
                out["raw_tracking_non_good"] += 1
                if out["raw_tracking_first_weak_frame"] < 0:
                    out["raw_tracking_first_weak_frame"] = frame
            elif raw == 0:
                out["raw_tracking_failed"] += 1
                out["raw_tracking_non_good"] += 1
                if out["raw_tracking_first_weak_frame"] < 0:
                    out["raw_tracking_first_weak_frame"] = frame
                if out["raw_tracking_first_failed_frame"] < 0:
                    out["raw_tracking_first_failed_frame"] = frame

            if policy == 2:
                out["policy_tracking_good"] += 1
            elif policy == 1:
                out["policy_tracking_poor"] += 1
                out["policy_tracking_non_good"] += 1
            elif policy == 0:
                out["policy_tracking_failed"] += 1
                out["policy_tracking_non_good"] += 1

            out["raw_tracking_max_failed_streak"] = max(
                out["raw_tracking_max_failed_streak"],
                safe_int(row.get("raw_failed_streak"), 0),
            )
            out["tracking_containment_pose_rollbacks"] += int(
                truthy(row.get("containment_pose_rollback"))
            )
            out["tracking_containment_fusion_suppressed"] += int(
                truthy(row.get("containment_fusion_suppressed"))
            )
    return out


def numeric(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def truthy(value: str | int | float | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def ratio_or_empty(numerator: object, denominator: object) -> float | str:
    num = safe_float(numerator)
    den = safe_float(denominator)
    if num is None or den is None or den <= 0.0:
        return ""
    return float(num / den)


def latest_applied_loop_endpoint_frames(root: Path) -> tuple[int, int] | None:
    updates_path = root / "updates.csv"
    if not updates_path.exists():
        return None

    latest: tuple[int, int] | None = None
    with updates_path.open(newline="") as f:
        for update_row in csv.DictReader(f):
            if not truthy(update_row.get("applied")):
                continue
            diagnostic_root_raw = str(update_row.get("diagnostic_root", "")).strip()
            if not diagnostic_root_raw:
                continue
            diagnostic_root = Path(diagnostic_root_raw)
            selection_path = diagnostic_root / "pose_graph_apply_loop_selection.csv"
            candidates_path = diagnostic_root / "pose_graph_loop_candidates.csv"
            if not selection_path.exists() or not candidates_path.exists():
                continue

            selected_pairs: set[tuple[int, int]] = set()
            with selection_path.open(newline="") as selection_file:
                for row in csv.DictReader(selection_file):
                    if truthy(row.get("selected")):
                        selected_pairs.add(
                            (
                                safe_int(row.get("from_hybrid_id"), -1),
                                safe_int(row.get("to_hybrid_id"), -1),
                            )
                        )
            if not selected_pairs:
                continue

            endpoint_frames: list[int] = []
            with candidates_path.open(newline="") as candidates_file:
                for row in csv.DictReader(candidates_file):
                    pair = (
                        safe_int(row.get("from_hybrid_id"), -1),
                        safe_int(row.get("to_hybrid_id"), -1),
                    )
                    if pair not in selected_pairs:
                        continue
                    for key in ("from_frame_id", "to_frame_id"):
                        frame_id = safe_int(row.get(key), -1)
                        if frame_id >= 0:
                            endpoint_frames.append(frame_id)
            if endpoint_frames:
                latest = (min(endpoint_frames), max(endpoint_frames))
    return latest


def parse_timing(run_dir: Path) -> dict[str, float | int]:
    path = run_dir / "timing.csv"
    if not path.exists():
        return {}
    totals = []
    pose_graph = []
    pose_graph_warm = []
    fusion = []
    gaussian_ops = []
    local_opt = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key, dest in (
                ("total_measured_ms", totals),
                ("poseGraph_ms", pose_graph),
                ("poseGraphWarm_ms", pose_graph_warm),
                ("fusion_ms", fusion),
                ("gaussianOps_ms", gaussian_ops),
                ("localOptimize_ms", local_opt),
            ):
                value = numeric(row, key)
                if value is not None and math.isfinite(value):
                    dest.append(value)
    out: dict[str, float | int] = {}
    if totals:
        total_ms = float(np.asarray(totals, dtype=np.float64).sum())
        out["timing_frame_count"] = len(totals)
        out["total_frame_ms_sum"] = total_ms
        out["steady_state_fps"] = 1000.0 * len(totals) / total_ms if total_ms > 0.0 else 0.0
    for label, values in (
        ("total_frame_ms", totals),
        ("pose_graph_frame_ms", pose_graph),
        ("pose_graph_warm_frame_ms", pose_graph_warm),
        ("fusion_frame_ms", fusion),
        ("gaussian_ops_frame_ms", gaussian_ops),
        ("local_opt_frame_ms", local_opt),
    ):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[f"{label}_avg"] = float(arr.mean())
        out[f"{label}_p95"] = float(np.percentile(arr, 95))
        out[f"{label}_p99"] = float(np.percentile(arr, 99))
        out[f"{label}_max"] = float(arr.max())
    return out


def add_pose_graph_apply_detail(
    out: dict[str, float | int | str],
    section: dict,
    prefix: str,
) -> None:
    timing = section.get("timing_ms")
    if isinstance(timing, dict):
        for key in (
            "setup_ms",
            "repeat_guard_ms",
            "loop_selection_ms",
            "support_gate_ms",
            "graph_build_ms",
            "graph_solve_ms",
            "scale_evaluation_ms",
            "transform_apply_ms",
            "total_ms",
        ):
            if key in timing:
                out[f"{prefix}_timing_{key}"] = timing[key]
    for src, dst in (
        ("observation_sample_build_ms", "observation_sample_build_ms"),
        ("observation_initial_trim_ms", "observation_initial_trim_ms"),
        ("observation_before_score_ms", "observation_before_score_ms"),
        ("observation_after_score_ms", "observation_after_score_ms"),
        ("observation_transform_update_ms", "observation_transform_update_ms"),
        ("observation_final_trim_ms", "observation_final_trim_ms"),
        ("observation_restore_trim_ms", "observation_restore_trim_ms"),
        ("selection_failure_reason", "selection_failure_reason"),
        ("skip_reason", "skip_reason"),
    ):
        if src in section:
            out[f"{prefix}_{dst}"] = section[src]
    if isinstance(section.get("observation_pinned_sdf_submap_ids"), list):
        out[f"{prefix}_observation_pinned_sdf_submap_ids"] = ";".join(
            str(value) for value in section["observation_pinned_sdf_submap_ids"]
        )
    trials = section.get("apply_scale_trials")
    if not isinstance(trials, list):
        return
    out[f"{prefix}_apply_scale_trials"] = len(trials)
    selected_idx = safe_int(section.get("apply_scale_selected_trial"), -1)
    if selected_idx < 0 or selected_idx >= len(trials) or not isinstance(trials[selected_idx], dict):
        return
    selected = trials[selected_idx]
    for src, dst in (
        ("evaluation_ms", "selected_trial_evaluation_ms"),
        ("node_proposal_ms", "selected_trial_node_proposal_ms"),
        ("sequential_residual_ms", "selected_trial_sequential_residual_ms"),
        ("frame_relative_ms", "selected_trial_frame_relative_ms"),
        ("observation_gate_ms", "selected_trial_observation_gate_ms"),
        ("observation_sample_build_ms", "selected_trial_observation_sample_build_ms"),
        ("observation_initial_trim_ms", "selected_trial_observation_initial_trim_ms"),
        ("observation_before_score_ms", "selected_trial_observation_before_score_ms"),
        ("observation_after_score_ms", "selected_trial_observation_after_score_ms"),
        ("observation_transform_update_ms", "selected_trial_observation_transform_update_ms"),
        ("observation_final_trim_ms", "selected_trial_observation_final_trim_ms"),
        ("observation_restore_trim_ms", "selected_trial_observation_restore_trim_ms"),
    ):
        if src in selected:
            out[f"{prefix}_{dst}"] = selected[src]
    if isinstance(selected.get("observation_pinned_sdf_submap_ids"), list):
        out[f"{prefix}_selected_trial_observation_pinned_sdf_submap_ids"] = ";".join(
            str(value) for value in selected["observation_pinned_sdf_submap_ids"]
        )


def parse_pose_graph(run_dir: Path) -> dict[str, float | int | str]:
    root = run_dir / "pose_graph_live"
    out: dict[str, float | int | str] = {}
    updates_path = root / "updates.csv"
    if updates_path.exists():
        update_rows = 0
        applied_updates = 0
        max_update_ms = 0.0
        max_candidate_scan_ms = 0.0
        max_dry_run_ms = 0.0
        max_solve_only_ms = 0.0
        max_apply_ms = 0.0
        max_candidate_pair_slot_build_ms = 0.0
        max_candidate_coarse_score_ms = 0.0
        max_candidate_sort_ms = 0.0
        max_candidate_verify_ms = 0.0
        max_candidate_loop_pool_export_ms = 0.0
        max_candidate_depth_tensor_load_ms = 0.0
        max_candidate_image_tensor_load_ms = 0.0
        max_candidate_orb_feature_total_ms = 0.0
        max_candidate_orb_feature_detect_ms = 0.0
        max_candidate_orb_match_ms = 0.0
        max_candidate_bow_histogram_build_ms = 0.0
        max_candidate_bow_score_ms = 0.0
        max_loop_observation_prefetch_ms = 0.0
        max_loop_observation_prefetch_requests = 0
        max_loop_observation_prefetch_unique_maps = 0
        max_loop_observation_prefetch_resident_final = 0
        max_loop_observation_prefetch_repeat_guard_skipped = 0
        max_loop_observation_prefetch_support_skipped = 0
        max_loop_scanned_pairs = 0
        max_loop_verified_pairs = 0
        max_loop_accepted = 0
        max_loop_exported = 0
        max_loop_constraints = 0
        max_nodes = 0
        dry_failures = 0
        solve_failures = 0
        apply_failures = 0
        max_apply_t = 0.0
        max_apply_r = 0.0
        update_total_ms = 0.0
        terminal_update_rows = 0
        terminal_update_ms = 0.0
        terminal_candidate_scan_ms = 0.0
        terminal_apply_ms = 0.0
        terminal_scanned_pairs = 0
        terminal_verified_pairs = 0
        terminal_accepted = 0
        support_policy_counts: dict[str, int] = {}
        applied_support_policy_counts: dict[str, int] = {}
        apply_skip_reason_counts: dict[str, int] = {}
        latest_applied_apply_json: Path | None = None
        with updates_path.open(newline="") as f:
            for row in csv.DictReader(f):
                update_rows += 1
                update_ms = numeric(row, "update_duration_ms")
                if update_ms is not None and math.isfinite(update_ms):
                    update_total_ms += update_ms
                source = str(row.get("source", "")).strip()
                is_terminal = source in {"live_end_of_sequence", "terminal", "end_of_sequence"}
                if is_terminal:
                    terminal_update_rows += 1
                    if update_ms is not None and math.isfinite(update_ms):
                        terminal_update_ms += update_ms
                    scan_ms = numeric(row, "candidate_scan_ms")
                    apply_ms = numeric(row, "apply_ms")
                    if scan_ms is not None and math.isfinite(scan_ms):
                        terminal_candidate_scan_ms += scan_ms
                    if apply_ms is not None and math.isfinite(apply_ms):
                        terminal_apply_ms += apply_ms
                    terminal_scanned_pairs += safe_int(row.get("loop_scanned_pairs"), 0)
                    terminal_verified_pairs += safe_int(row.get("loop_verified_pairs"), 0)
                    terminal_accepted += safe_int(row.get("loop_accepted"), 0)
                row_applied = truthy(row.get("applied"))
                if row_applied:
                    applied_updates += 1
                diagnostic_root = str(row.get("diagnostic_root", "")).strip()
                if diagnostic_root:
                    apply_diag_path = Path(diagnostic_root) / "pose_graph_apply.json"
                    if row_applied:
                        latest_applied_apply_json = apply_diag_path
                    if apply_diag_path.exists():
                        try:
                            apply_diag = json.loads(apply_diag_path.read_text(errors="replace"))
                            support_policy = str(apply_diag.get("support_policy_used", "")).strip()
                            if support_policy:
                                support_policy_counts[support_policy] = (
                                    support_policy_counts.get(support_policy, 0) + 1
                                )
                                if row_applied:
                                    applied_support_policy_counts[support_policy] = (
                                        applied_support_policy_counts.get(support_policy, 0) + 1
                                    )
                            skip_reason = str(apply_diag.get("skip_reason", "")).strip()
                            if skip_reason:
                                apply_skip_reason_counts[skip_reason] = (
                                    apply_skip_reason_counts.get(skip_reason, 0) + 1
                                )
                        except json.JSONDecodeError:
                            out["pose_graph_update_apply_json_error"] = "json_decode"
                max_nodes = max(max_nodes, safe_int(row.get("nodes"), 0))
                max_loop_constraints = max(max_loop_constraints, safe_int(row.get("loop_constraints"), 0))
                max_loop_scanned_pairs = max(max_loop_scanned_pairs, safe_int(row.get("loop_scanned_pairs"), 0))
                max_loop_verified_pairs = max(max_loop_verified_pairs, safe_int(row.get("loop_verified_pairs"), 0))
                max_loop_accepted = max(max_loop_accepted, safe_int(row.get("loop_accepted"), 0))
                max_loop_exported = max(max_loop_exported, safe_int(row.get("loop_exported"), 0))
                max_loop_observation_prefetch_requests = max(
                    max_loop_observation_prefetch_requests,
                    safe_int(row.get("loop_observation_prefetch_requests"), 0),
                )
                max_loop_observation_prefetch_unique_maps = max(
                    max_loop_observation_prefetch_unique_maps,
                    safe_int(row.get("loop_observation_prefetch_unique_maps"), 0),
                )
                max_loop_observation_prefetch_resident_final = max(
                    max_loop_observation_prefetch_resident_final,
                    safe_int(row.get("loop_observation_prefetch_resident_final"), 0),
                )
                max_loop_observation_prefetch_repeat_guard_skipped = max(
                    max_loop_observation_prefetch_repeat_guard_skipped,
                    safe_int(row.get("loop_observation_prefetch_repeat_guard_skipped"), 0),
                )
                max_loop_observation_prefetch_support_skipped = max(
                    max_loop_observation_prefetch_support_skipped,
                    safe_int(row.get("loop_observation_prefetch_support_skipped"), 0),
                )
                dry_failures += safe_int(row.get("dry_run_failed"), 0)
                solve_failures += safe_int(row.get("solve_only_failed"), 0)
                apply_failures += safe_int(row.get("apply_failed"), 0)
                for key, current in (
                    ("update_duration_ms", max_update_ms),
                    ("candidate_scan_ms", max_candidate_scan_ms),
                    ("dry_run_ms", max_dry_run_ms),
                    ("solve_only_ms", max_solve_only_ms),
                    ("apply_ms", max_apply_ms),
                    ("candidate_pair_slot_build_ms", max_candidate_pair_slot_build_ms),
                    ("candidate_coarse_score_ms", max_candidate_coarse_score_ms),
                    ("candidate_sort_ms", max_candidate_sort_ms),
                    ("candidate_verify_ms", max_candidate_verify_ms),
                    ("candidate_loop_pool_export_ms", max_candidate_loop_pool_export_ms),
                    ("candidate_depth_tensor_load_ms", max_candidate_depth_tensor_load_ms),
                    ("candidate_image_tensor_load_ms", max_candidate_image_tensor_load_ms),
                    ("candidate_orb_feature_total_ms", max_candidate_orb_feature_total_ms),
                    ("candidate_orb_feature_detect_ms", max_candidate_orb_feature_detect_ms),
                    ("candidate_orb_match_ms", max_candidate_orb_match_ms),
                    ("candidate_bow_histogram_build_ms", max_candidate_bow_histogram_build_ms),
                    ("candidate_bow_score_ms", max_candidate_bow_score_ms),
                    ("loop_observation_prefetch_ms", max_loop_observation_prefetch_ms),
                    ("apply_max_translation_delta", max_apply_t),
                    ("apply_max_rotation_delta_degrees", max_apply_r),
                ):
                    value = numeric(row, key)
                    if value is None or not math.isfinite(value):
                        continue
                    if key == "update_duration_ms":
                        max_update_ms = max(current, value)
                    elif key == "candidate_scan_ms":
                        max_candidate_scan_ms = max(current, value)
                    elif key == "dry_run_ms":
                        max_dry_run_ms = max(current, value)
                    elif key == "solve_only_ms":
                        max_solve_only_ms = max(current, value)
                    elif key == "apply_ms":
                        max_apply_ms = max(current, value)
                    elif key == "candidate_pair_slot_build_ms":
                        max_candidate_pair_slot_build_ms = max(current, value)
                    elif key == "candidate_coarse_score_ms":
                        max_candidate_coarse_score_ms = max(current, value)
                    elif key == "candidate_sort_ms":
                        max_candidate_sort_ms = max(current, value)
                    elif key == "candidate_verify_ms":
                        max_candidate_verify_ms = max(current, value)
                    elif key == "candidate_loop_pool_export_ms":
                        max_candidate_loop_pool_export_ms = max(current, value)
                    elif key == "candidate_depth_tensor_load_ms":
                        max_candidate_depth_tensor_load_ms = max(current, value)
                    elif key == "candidate_image_tensor_load_ms":
                        max_candidate_image_tensor_load_ms = max(current, value)
                    elif key == "candidate_orb_feature_total_ms":
                        max_candidate_orb_feature_total_ms = max(current, value)
                    elif key == "candidate_orb_feature_detect_ms":
                        max_candidate_orb_feature_detect_ms = max(current, value)
                    elif key == "candidate_orb_match_ms":
                        max_candidate_orb_match_ms = max(current, value)
                    elif key == "candidate_bow_histogram_build_ms":
                        max_candidate_bow_histogram_build_ms = max(current, value)
                    elif key == "candidate_bow_score_ms":
                        max_candidate_bow_score_ms = max(current, value)
                    elif key == "loop_observation_prefetch_ms":
                        max_loop_observation_prefetch_ms = max(current, value)
                    elif key == "apply_max_translation_delta":
                        max_apply_t = max(current, value)
                    else:
                        max_apply_r = max(current, value)
        out.update(
            {
                "pose_graph_updates": update_rows,
                "pose_graph_applied_updates": applied_updates,
                "pose_graph_max_update_ms": max_update_ms,
                "pose_graph_max_candidate_scan_ms": max_candidate_scan_ms,
                "pose_graph_max_dry_run_ms": max_dry_run_ms,
                "pose_graph_max_solve_only_ms": max_solve_only_ms,
                "pose_graph_max_apply_ms": max_apply_ms,
                "pose_graph_max_candidate_pair_slot_build_ms": max_candidate_pair_slot_build_ms,
                "pose_graph_max_candidate_coarse_score_ms": max_candidate_coarse_score_ms,
                "pose_graph_max_candidate_sort_ms": max_candidate_sort_ms,
                "pose_graph_max_candidate_verify_ms": max_candidate_verify_ms,
                "pose_graph_max_candidate_loop_pool_export_ms": max_candidate_loop_pool_export_ms,
                "pose_graph_max_candidate_depth_tensor_load_ms": max_candidate_depth_tensor_load_ms,
                "pose_graph_max_candidate_image_tensor_load_ms": max_candidate_image_tensor_load_ms,
                "pose_graph_max_candidate_orb_feature_total_ms": max_candidate_orb_feature_total_ms,
                "pose_graph_max_candidate_orb_feature_detect_ms": max_candidate_orb_feature_detect_ms,
                "pose_graph_max_candidate_orb_match_ms": max_candidate_orb_match_ms,
                "pose_graph_max_candidate_bow_histogram_build_ms": max_candidate_bow_histogram_build_ms,
                "pose_graph_max_candidate_bow_score_ms": max_candidate_bow_score_ms,
                "pose_graph_max_loop_observation_prefetch_ms": max_loop_observation_prefetch_ms,
                "pose_graph_max_loop_observation_prefetch_requests": max_loop_observation_prefetch_requests,
                "pose_graph_max_loop_observation_prefetch_unique_maps": max_loop_observation_prefetch_unique_maps,
                "pose_graph_max_loop_observation_prefetch_resident_final": max_loop_observation_prefetch_resident_final,
                "pose_graph_max_loop_observation_prefetch_repeat_guard_skipped": max_loop_observation_prefetch_repeat_guard_skipped,
                "pose_graph_max_loop_observation_prefetch_support_skipped": max_loop_observation_prefetch_support_skipped,
                "pose_graph_max_loop_scanned_pairs": max_loop_scanned_pairs,
                "pose_graph_max_loop_verified_pairs": max_loop_verified_pairs,
                "pose_graph_max_loop_accepted": max_loop_accepted,
                "pose_graph_max_loop_exported": max_loop_exported,
                "pose_graph_max_nodes": max_nodes,
                "pose_graph_max_loop_constraints": max_loop_constraints,
                "pose_graph_dry_run_failures": dry_failures,
                "pose_graph_solve_only_failures": solve_failures,
                "pose_graph_apply_failures": apply_failures,
                "pose_graph_max_apply_translation_delta_m": max_apply_t,
                "pose_graph_max_apply_rotation_delta_deg": max_apply_r,
                "pose_graph_update_total_ms": update_total_ms,
                "terminal_pose_graph_updates": terminal_update_rows,
                "terminal_pose_graph_ms": terminal_update_ms,
                "terminal_candidate_scan_ms": terminal_candidate_scan_ms,
                "terminal_pose_graph_apply_ms": terminal_apply_ms,
                "terminal_loop_scanned_pairs": terminal_scanned_pairs,
                "terminal_loop_verified_pairs": terminal_verified_pairs,
                "terminal_loop_accepted": terminal_accepted,
            }
        )
        if support_policy_counts:
            out["pose_graph_apply_support_policy_used_counts"] = format_counts(support_policy_counts)
            for policy, count in sorted(support_policy_counts.items()):
                out[f"pose_graph_apply_support_policy_used_{slug_key(policy)}_updates"] = count
        if applied_support_policy_counts:
            out["pose_graph_applied_support_policy_used_counts"] = format_counts(applied_support_policy_counts)
            for policy, count in sorted(applied_support_policy_counts.items()):
                out[f"pose_graph_applied_support_policy_used_{slug_key(policy)}_updates"] = count
        if apply_skip_reason_counts:
            out["pose_graph_apply_skip_reason_counts"] = format_counts(apply_skip_reason_counts)
            for reason, count in sorted(apply_skip_reason_counts.items()):
                out[f"pose_graph_apply_skip_reason_{slug_key(reason)}_updates"] = count
        if latest_applied_apply_json is not None and latest_applied_apply_json.exists():
            try:
                latest_apply_diag = json.loads(latest_applied_apply_json.read_text(errors="replace"))
                out["pose_graph_latest_applied_apply_json"] = str(latest_applied_apply_json)
                add_pose_graph_apply_detail(out, latest_apply_diag, "pose_graph_latest_applied")
            except json.JSONDecodeError:
                out["pose_graph_latest_applied_apply_json_error"] = "json_decode"

    manifest_path = root / "pose_graph_live.json"
    if not manifest_path.exists():
        manifest_path = root / "pose_graph_live_latest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(errors="replace"))
            pose_graph = manifest.get("pose_graph", {})
            stats = pose_graph.get("candidate_stats", {})
            if stats:
                for src_key, dst_key in (
                    ("scanned_pairs", "loop_scanned_pairs"),
                    ("verified_pairs", "loop_verified_pairs"),
                    ("span_reserved_pairs", "pose_graph_span_reserved_pairs"),
                    ("span_injected_pairs", "pose_graph_span_injected_pairs"),
                    ("span_reserve_candidate_count", "pose_graph_span_reserve_candidate_count"),
                    ("span_reserve_fraction", "pose_graph_span_reserve_fraction"),
                    ("span_reserve_pairs_config", "pose_graph_span_reserve_pairs_config"),
                    ("accepted", "loop_accepted"),
                    ("exported", "loop_exported"),
                    ("orb_feature_cache_entries", "orb_cache_entries"),
                    ("orb_feature_cache_hits", "orb_cache_hits"),
                    ("orb_feature_cache_misses", "orb_cache_misses"),
                    ("orb_feature_cache_evictions", "orb_cache_evictions"),
                    ("orb_feature_cache_limit", "orb_cache_limit"),
                    ("warm_max_per_frame", "warm_max_per_frame"),
                    ("warm_skip_update_frames", "warm_skip_update_frames"),
                    ("warm_queue_limit", "warm_queue_limit"),
                    ("warm_orb_requests", "warm_orb_requests"),
                    ("warm_orb_cache_hits", "warm_orb_cache_hits"),
                    ("warm_orb_cache_misses", "warm_orb_cache_misses"),
                    ("warm_orb_cache_evictions", "warm_orb_cache_evictions"),
                    ("warm_orb_extract_calls", "warm_orb_extract_calls"),
                    ("warm_orb_queued", "warm_orb_queued"),
                    ("warm_orb_dropped", "warm_orb_dropped"),
                    ("warm_orb_deduplicated", "warm_orb_deduplicated"),
                    ("warm_orb_completed", "warm_orb_completed"),
                    ("warm_orb_queue_peak", "warm_orb_queue_peak"),
                    ("timing_warm_orb_total_ms", "pose_graph_timing_warm_orb_total_ms"),
                    ("timing_warm_orb_detect_ms", "pose_graph_timing_warm_orb_detect_ms"),
                    ("depth_tensor_cache_hits", "depth_tensor_cache_hits"),
                    ("depth_tensor_cache_misses", "depth_tensor_cache_misses"),
                    ("image_tensor_cache_hits", "image_tensor_cache_hits"),
                    ("image_tensor_cache_misses", "image_tensor_cache_misses"),
                    ("orb_feature_extract_calls", "orb_feature_extract_calls"),
                    ("orb_match_calls", "orb_match_calls"),
                    ("orb_match_cache_entries", "orb_match_cache_entries"),
                    ("orb_match_cache_hits", "orb_match_cache_hits"),
                    ("orb_match_cache_misses", "orb_match_cache_misses"),
                    ("bow_histogram_cache_submaps", "bow_histogram_cache_submaps"),
                    ("bow_histogram_cache_hits", "bow_histogram_cache_hits"),
                    ("bow_histogram_cache_misses", "bow_histogram_cache_misses"),
                    ("bow_histogram_cache_rebuilds", "bow_histogram_cache_rebuilds"),
                    ("bow_histogram_build_calls", "bow_histogram_build_calls"),
                    ("bow_score_calls", "bow_score_calls"),
                    ("orb_max_features", "orb_max_features"),
                    ("orb_min_quadrants", "orb_min_quadrants"),
                    ("orb_depth_search_radius", "orb_depth_search_radius"),
                    ("orb_pnp_iterations", "orb_pnp_iterations"),
                    ("verify_orb_aggregate_pairs", "pose_graph_verify_orb_aggregate_pairs"),
                    ("verify_orb_aggregate_max_pairs", "pose_graph_verify_orb_aggregate_max_pairs"),
                    ("verify_icp_weighted_kabsch", "pose_graph_verify_icp_weighted_kabsch"),
                    ("verify_icp_depth_adaptive_ransac", "pose_graph_verify_icp_depth_adaptive_ransac"),
                    ("verify_icp_depth_noise_scale", "pose_graph_verify_icp_depth_noise_scale"),
                    ("verify_icp_depth_adaptive_threshold_scale", "pose_graph_verify_icp_depth_adaptive_threshold_scale"),
                    ("verify_icp_drift_aware_delta", "pose_graph_verify_icp_drift_aware_delta"),
                    ("verify_icp_drift_reference_span", "pose_graph_verify_icp_drift_reference_span"),
                    ("verify_icp_drift_max_scale", "pose_graph_verify_icp_drift_max_scale"),
                    ("verify_icp_drift_score_margin", "pose_graph_verify_icp_drift_score_margin"),
                    ("appearance_max_observation_pairs", "appearance_max_observation_pairs"),
                    ("appearance_min_orb_matches", "appearance_min_orb_matches"),
                    ("keyframe_observation_max_per_submap", "keyframe_observation_max_per_submap"),
                    ("candidate_interval_frames", "pose_graph_candidate_interval_frames"),
                    ("candidate_accept_target_per_check", "pose_graph_candidate_accept_target_per_check"),
                    ("loop_candidate_export_policy", "pose_graph_loop_candidate_export_policy"),
                    ("loop_candidate_export_span_reserve_fraction", "pose_graph_loop_candidate_export_span_reserve_fraction"),
                    ("loop_candidate_export_min_span", "pose_graph_loop_candidate_export_min_span"),
                    ("loop_max_candidates", "pose_graph_loop_max_candidates"),
                    ("online_min_submaps", "pose_graph_online_min_submaps"),
                    ("online_min_submap_growth", "pose_graph_online_min_submap_growth"),
                    ("max_updates_per_run", "pose_graph_max_updates_per_run"),
                    ("terminal_online_scan", "pose_graph_terminal_online_scan"),
                    ("final_max_scanned_pairs", "pose_graph_final_max_scanned_pairs"),
                    ("final_rank_all_candidates", "pose_graph_final_rank_all_candidates"),
                    ("final_rank_all_candidate_pairs", "pose_graph_final_rank_all_candidate_pairs"),
                    ("verification_pair_budget", "pose_graph_verification_pair_budget"),
                    ("verified_loop_pool_max", "pose_graph_verified_loop_pool_max"),
                    ("verified_loop_pool_size_before", "pose_graph_verified_loop_pool_size_before"),
                    ("verified_loop_pool_size_after", "pose_graph_verified_loop_pool_size_after"),
                    ("verified_loop_pool_new", "pose_graph_verified_loop_pool_new"),
                    ("verified_loop_pool_replaced", "pose_graph_verified_loop_pool_replaced"),
                    ("skip_verified_loop_pairs", "pose_graph_skip_verified_loop_pairs"),
                    ("skipped_verified_loop_pairs", "pose_graph_skipped_verified_loop_pairs"),
                    ("stale_verified_loop_pairs", "pose_graph_stale_verified_loop_pairs"),
                    ("pooled_loop_constraints_exported", "pose_graph_pooled_loop_constraints_exported"),
                    ("timing_pair_slot_build_ms", "pose_graph_timing_pair_slot_build_ms"),
                    ("timing_coarse_score_ms", "pose_graph_timing_coarse_score_ms"),
                    ("timing_candidate_sort_ms", "pose_graph_timing_candidate_sort_ms"),
                    ("timing_candidate_verify_ms", "pose_graph_timing_candidate_verify_ms"),
                    ("timing_loop_pool_export_ms", "pose_graph_timing_loop_pool_export_ms"),
                    ("timing_depth_tensor_load_ms", "pose_graph_timing_depth_tensor_load_ms"),
                    ("timing_image_tensor_load_ms", "pose_graph_timing_image_tensor_load_ms"),
                    ("timing_orb_feature_total_ms", "pose_graph_timing_orb_feature_total_ms"),
                    ("timing_orb_feature_image_ms", "pose_graph_timing_orb_feature_image_ms"),
                    ("timing_orb_feature_detect_ms", "pose_graph_timing_orb_feature_detect_ms"),
                    ("timing_orb_match_ms", "pose_graph_timing_orb_match_ms"),
                    ("timing_bow_histogram_build_ms", "pose_graph_timing_bow_histogram_build_ms"),
                    ("timing_bow_score_ms", "pose_graph_timing_bow_score_ms"),
                    ("apply_correction_scale", "pose_graph_apply_correction_scale"),
                    ("apply_scale_backoff", "pose_graph_apply_scale_backoff"),
                    ("apply_scale_backoff_min_scale", "pose_graph_apply_scale_backoff_min_scale"),
                    ("apply_scale_backoff_factor", "pose_graph_apply_scale_backoff_factor"),
                    ("apply_scale_backoff_max_trials", "pose_graph_apply_scale_backoff_max_trials"),
                    ("apply_observation_gate_in_selection", "pose_graph_apply_observation_gate_in_selection"),
                    ("apply_observation_gate_max_samples", "pose_graph_apply_observation_gate_max_samples"),
                    ("apply_observation_gate_min_samples", "pose_graph_apply_observation_gate_min_samples"),
                    ("apply_observation_gate_sample_policy", "pose_graph_apply_observation_gate_sample_policy"),
                    ("apply_observation_max_render_maps_per_sample", "pose_graph_apply_observation_max_render_maps_per_sample"),
                    ("apply_observation_max_pinned_maps", "pose_graph_apply_observation_max_pinned_maps"),
                    ("apply_observation_prefetch_loop_endpoints", "pose_graph_apply_observation_prefetch_loop_endpoints"),
                    ("apply_observation_prefetch_max_maps", "pose_graph_apply_observation_prefetch_max_maps"),
                    ("loop_observation_prefetch_effective_max_maps", "pose_graph_loop_observation_prefetch_effective_max_maps"),
                    ("loop_observation_prefetch_requests", "pose_graph_loop_observation_prefetch_requests"),
                    ("loop_observation_prefetch_unique_maps", "pose_graph_loop_observation_prefetch_unique_maps"),
                    ("loop_observation_prefetch_resident_final", "pose_graph_loop_observation_prefetch_resident_final"),
                    ("loop_observation_prefetch_nonresident_requests", "pose_graph_loop_observation_prefetch_nonresident_requests"),
                    ("loop_observation_prefetch_limit_skipped", "pose_graph_loop_observation_prefetch_limit_skipped"),
                    ("loop_observation_prefetch_repeat_guard_skipped", "pose_graph_loop_observation_prefetch_repeat_guard_skipped"),
                    ("loop_observation_prefetch_support_skipped", "pose_graph_loop_observation_prefetch_support_skipped"),
                    ("timing_loop_observation_prefetch_ms", "pose_graph_timing_loop_observation_prefetch_ms"),
                    ("apply_observation_defer_nonresident", "pose_graph_apply_observation_defer_nonresident"),
                    ("apply_observation_defer_retry_frames", "pose_graph_apply_observation_defer_retry_frames"),
                    ("apply_observation_pin_deferred_maps", "pose_graph_apply_observation_pin_deferred_maps"),
                    ("apply_observation_defer_stops_scale_backoff", "pose_graph_apply_observation_defer_stops_scale_backoff"),
                    ("apply_observation_terminal_sync_page_in", "pose_graph_apply_observation_terminal_sync_page_in"),
                    ("apply_observation_skip_resident_trim", "pose_graph_apply_observation_skip_resident_trim"),
                    ("apply_observation_gate_tolerance_policy", "pose_graph_apply_observation_gate_tolerance_policy"),
                    ("apply_observation_max_color_mse_ratio", "pose_graph_apply_observation_max_color_mse_ratio"),
                    ("apply_observation_max_color_mse_increase", "pose_graph_apply_observation_max_color_mse_increase"),
                    ("apply_observation_max_depth_rmse_ratio", "pose_graph_apply_observation_max_depth_rmse_ratio"),
                    ("apply_observation_max_depth_rmse_increase", "pose_graph_apply_observation_max_depth_rmse_increase"),
                    ("apply_observation_depth_rmse_ratio_floor", "pose_graph_apply_observation_depth_rmse_ratio_floor"),
                    ("apply_observation_max_depth_rmse_absolute_increase", "pose_graph_apply_observation_max_depth_rmse_absolute_increase"),
                    ("apply_observation_max_valid_fraction_drop", "pose_graph_apply_observation_max_valid_fraction_drop"),
                    ("apply_pcm_min_clique_size", "pose_graph_apply_pcm_min_clique_size"),
                    ("apply_pcm_max_trial_cliques", "pose_graph_apply_pcm_max_trial_cliques"),
                    ("apply_pcm_event_cluster_radius", "pose_graph_apply_pcm_event_cluster_radius"),
                    ("apply_pcm_min_distinct_events", "pose_graph_apply_pcm_min_distinct_events"),
                    ("apply_pcm_min_long_span", "pose_graph_apply_pcm_min_long_span"),
                    ("apply_pcm_prefer_long_span", "pose_graph_apply_pcm_prefer_long_span"),
                    ("apply_pcm_scale_threshold_by_cycle_length", "pose_graph_apply_pcm_scale_threshold_by_cycle_length"),
                    ("apply_pcm_translation_threshold", "pose_graph_apply_pcm_translation_threshold"),
                    ("apply_pcm_rotation_threshold_deg", "pose_graph_apply_pcm_rotation_threshold_deg"),
                ):
                    if src_key in stats:
                        out[dst_key] = stats[src_key]
                if "online_updates" in stats:
                    out["pose_graph_online_updates_enabled"] = int(bool(stats["online_updates"]))
                if "run_on_submap_events" in stats:
                    out["pose_graph_run_on_submap_events"] = int(bool(stats["run_on_submap_events"]))
                if "final_scan" in stats:
                    out["pose_graph_final_scan"] = int(bool(stats["final_scan"]))
                if "final_candidate_scan" in stats:
                    out["pose_graph_final_candidate_scan"] = int(bool(stats["final_candidate_scan"]))
                if "final_pair_scan_skipped" in stats:
                    out["pose_graph_final_pair_scan_skipped"] = int(bool(stats["final_pair_scan_skipped"]))
                if "final_rank_all_candidates" in stats:
                    out["pose_graph_final_rank_all_candidates_enabled"] = int(
                        bool(stats["final_rank_all_candidates"])
                    )
                if "final_rank_all_candidate_pairs" in stats:
                    out["pose_graph_final_rank_all_candidate_pairs_used"] = int(
                        bool(stats["final_rank_all_candidate_pairs"])
                    )
                if "candidate_accept_target_reached" in stats:
                    out["pose_graph_candidate_accept_target_reached"] = int(
                        bool(stats["candidate_accept_target_reached"])
                    )
                if "persistent_loop_pool" in stats:
                    out["pose_graph_persistent_loop_pool"] = int(bool(stats["persistent_loop_pool"]))
                if "include_keyframes_with_loop_observations" in stats:
                    out["include_keyframes_with_loop_observations"] = int(
                        bool(stats["include_keyframes_with_loop_observations"])
                    )
                if "warm_loop_observations" in stats:
                    out["warm_loop_observations"] = int(bool(stats["warm_loop_observations"]))
                if "warm_skip_update_frames" in stats:
                    out["warm_skip_update_frames"] = int(bool(stats["warm_skip_update_frames"]))
                if "warm_async" in stats:
                    out["warm_async"] = int(bool(stats["warm_async"]))
                if "orb_pnp_fallback" in stats:
                    out["orb_pnp_fallback"] = int(bool(stats["orb_pnp_fallback"]))
                if "orb_pnp_reprojection_error" in stats:
                    out["orb_pnp_reprojection_error"] = stats["orb_pnp_reprojection_error"]
                if "orb_pnp_confidence" in stats:
                    out["orb_pnp_confidence"] = stats["orb_pnp_confidence"]
                if "orb_ratio" in stats:
                    out["orb_ratio"] = stats["orb_ratio"]
                if "orb_match_policy" in stats:
                    out["orb_match_policy"] = str(stats["orb_match_policy"])
                if "orb_feature_cache_enabled" in stats:
                    out["orb_cache_enabled"] = int(bool(stats["orb_feature_cache_enabled"]))
                depth_total = safe_int(out.get("depth_tensor_cache_hits"), 0) + safe_int(out.get("depth_tensor_cache_misses"), 0)
                image_total = safe_int(out.get("image_tensor_cache_hits"), 0) + safe_int(out.get("image_tensor_cache_misses"), 0)
                orb_total = safe_int(out.get("orb_cache_hits"), 0) + safe_int(out.get("orb_cache_misses"), 0)
                warm_total = safe_int(out.get("warm_orb_cache_hits"), 0) + safe_int(out.get("warm_orb_cache_misses"), 0)
                match_total = safe_int(out.get("orb_match_cache_hits"), 0) + safe_int(out.get("orb_match_cache_misses"), 0)
                bow_total = safe_int(out.get("bow_histogram_cache_hits"), 0) + safe_int(out.get("bow_histogram_cache_misses"), 0)
                out["depth_tensor_cache_hit_rate"] = ratio_or_empty(out.get("depth_tensor_cache_hits"), depth_total)
                out["image_tensor_cache_hit_rate"] = ratio_or_empty(out.get("image_tensor_cache_hits"), image_total)
                out["orb_cache_hit_rate"] = ratio_or_empty(out.get("orb_cache_hits"), orb_total)
                out["warm_orb_cache_hit_rate"] = ratio_or_empty(out.get("warm_orb_cache_hits"), warm_total)
                out["orb_match_cache_hit_rate"] = ratio_or_empty(out.get("orb_match_cache_hits"), match_total)
                out["bow_histogram_cache_hit_rate"] = ratio_or_empty(out.get("bow_histogram_cache_hits"), bow_total)
                if "apply_observation_gate" in stats:
                    out["pose_graph_apply_observation_gate_enabled"] = int(bool(stats["apply_observation_gate"]))
                if "apply_observation_gate_in_selection" in stats:
                    out["pose_graph_apply_observation_gate_in_selection_enabled"] = int(
                        bool(stats["apply_observation_gate_in_selection"])
                    )
                if "loop_observation_prefetch_enabled" in stats:
                    out["pose_graph_loop_observation_prefetch_enabled"] = int(
                        bool(stats["loop_observation_prefetch_enabled"])
                    )
                if isinstance(stats.get("loop_observation_prefetch_sdf_submap_ids"), list):
                    out["pose_graph_loop_observation_prefetch_sdf_submap_ids"] = ";".join(
                        str(value) for value in stats["loop_observation_prefetch_sdf_submap_ids"]
                    )
                if "apply_skip_repeated_loop_pairs" in stats:
                    out["pose_graph_apply_skip_repeated_loop_pairs_enabled"] = int(
                        bool(stats["apply_skip_repeated_loop_pairs"])
                    )
                if "apply_repeat_min_new_nodes" in stats:
                    out["pose_graph_apply_repeat_min_new_nodes"] = stats["apply_repeat_min_new_nodes"]
                if "bow_word_bits" in stats:
                    out["bow_word_bits"] = stats["bow_word_bits"]
                if "bow_word_count" in stats:
                    out["bow_word_count"] = stats["bow_word_count"]
                if "bow_min_score" in stats:
                    out["bow_min_score"] = stats["bow_min_score"]
                if "bow_defer_appearance" in stats:
                    out["bow_defer_appearance"] = int(bool(stats["bow_defer_appearance"]))
                if "bow_appearance_deferred" in stats:
                    out["bow_appearance_deferred"] = int(bool(stats["bow_appearance_deferred"]))
                if "bow_histogram_cache_enabled" in stats:
                    out["bow_histogram_cache_enabled"] = int(bool(stats["bow_histogram_cache_enabled"]))
                if "appearance_candidates_scored" in stats:
                    out["appearance_candidates_scored"] = stats["appearance_candidates_scored"]
                if "bow_candidates_scored" in stats:
                    out["bow_candidates_scored"] = stats["bow_candidates_scored"]
                if "bow_histogram_submaps" in stats:
                    out["bow_histogram_submaps"] = stats["bow_histogram_submaps"]
                if "verification" in stats:
                    out["loop_verification"] = str(stats["verification"])
                if "verify_pair_selection" in stats:
                    out["loop_verify_pair_selection"] = str(stats["verify_pair_selection"])
                if "candidate_order" in stats:
                    out["loop_candidate_order"] = str(stats["candidate_order"])
                if "total_pair_slots" in stats:
                    out["loop_total_pair_slots"] = stats["total_pair_slots"]
                if "scan_budget" in stats:
                    out["loop_scan_budget"] = stats["scan_budget"]
                if "incremental_scan" in stats:
                    out["loop_incremental_scan"] = int(bool(stats["incremental_scan"]))
                if "scan_cursor_start" in stats:
                    out["loop_scan_cursor_start"] = stats["scan_cursor_start"]
                if "scan_cursor_next" in stats:
                    out["loop_scan_cursor_next"] = stats["scan_cursor_next"]
                if "scan_wrapped" in stats:
                    out["loop_scan_wrapped"] = int(bool(stats["scan_wrapped"]))
                if "max_scanned_pairs_per_check" in stats:
                    out["loop_max_scanned_pairs_per_check"] = stats["max_scanned_pairs_per_check"]
            for mode in ("dry_run", "solve_only", "apply"):
                section = pose_graph.get(mode, {})
                if not isinstance(section, dict):
                    continue
                if "nodes" in section:
                    out[f"pose_graph_{mode}_nodes"] = section["nodes"]
                if "loop_edges" in section:
                    out[f"pose_graph_{mode}_loop_edges"] = section["loop_edges"]
                elif "loop_constraints" in section:
                    out[f"pose_graph_{mode}_loop_edges"] = section["loop_constraints"]
                if "max_sequential_translation_residual" in section:
                    out[f"pose_graph_{mode}_max_seq_residual_m"] = section["max_sequential_translation_residual"]
                if "max_translation_delta" in section:
                    out[f"pose_graph_{mode}_max_translation_delta_m"] = section["max_translation_delta"]
                if "failed_frame_relative_constraints" in section:
                    out[f"pose_graph_{mode}_failed_frame_relative_constraints"] = section["failed_frame_relative_constraints"]
                if "frame_relative_pairs" in section:
                    out[f"pose_graph_{mode}_frame_relative_pairs"] = section["frame_relative_pairs"]
                if "max_frame_relative_translation_change" in section:
                    out[f"pose_graph_{mode}_max_frame_relative_translation_change_m"] = section["max_frame_relative_translation_change"]
                if "max_frame_relative_rotation_change_degrees" in section:
                    out[f"pose_graph_{mode}_max_frame_relative_rotation_change_deg"] = section["max_frame_relative_rotation_change_degrees"]
                if "mean_frame_relative_translation_change" in section:
                    out[f"pose_graph_{mode}_mean_frame_relative_translation_change_m"] = section["mean_frame_relative_translation_change"]
                if "mean_frame_relative_rotation_change_degrees" in section:
                    out[f"pose_graph_{mode}_mean_frame_relative_rotation_change_deg"] = section["mean_frame_relative_rotation_change_degrees"]
                if "max_allowed_frame_relative_translation_change" in section:
                    out[f"pose_graph_{mode}_max_allowed_frame_relative_translation_change_m"] = section["max_allowed_frame_relative_translation_change"]
                if "max_allowed_frame_relative_rotation_change_degrees" in section:
                    out[f"pose_graph_{mode}_max_allowed_frame_relative_rotation_change_deg"] = section["max_allowed_frame_relative_rotation_change_degrees"]
                for src, dst in (
                    ("failed_observation_constraints", "failed_observation_constraints"),
                    ("observation_samples", "observation_samples"),
                    ("configured_correction_scale", "configured_correction_scale"),
                    ("correction_scale", "correction_scale"),
                    ("selected_correction_scale", "selected_correction_scale"),
                    ("apply_scale_selected_trial", "apply_scale_selected_trial"),
                    ("selection_failure_reason", "selection_failure_reason"),
                    ("max_full_translation_delta", "max_full_translation_delta_m"),
                    ("max_full_rotation_delta_degrees", "max_full_rotation_delta_deg"),
                    ("observation_min_required_samples", "observation_min_required_samples"),
                    ("observation_color_mse_before", "observation_color_mse_before"),
                    ("observation_color_mse_after", "observation_color_mse_after"),
                    ("observation_color_mse_ratio", "observation_color_mse_ratio"),
                    ("observation_tolerance_policy", "observation_tolerance_policy"),
                    ("observation_color_ratio_pass", "observation_color_ratio_pass"),
                    ("observation_color_increase_pass", "observation_color_increase_pass"),
                    ("observation_max_allowed_color_mse_ratio", "observation_max_allowed_color_mse_ratio"),
                    ("observation_max_allowed_color_mse_increase", "observation_max_allowed_color_mse_increase"),
                    ("observation_depth_rmse_before", "observation_depth_rmse_before"),
                    ("observation_depth_rmse_after", "observation_depth_rmse_after"),
                    ("observation_depth_rmse_ratio", "observation_depth_rmse_ratio"),
                    ("observation_depth_ratio_pass", "observation_depth_ratio_pass"),
                    ("observation_depth_increase_pass", "observation_depth_increase_pass"),
                    ("observation_max_allowed_depth_rmse_ratio", "observation_max_allowed_depth_rmse_ratio"),
                    ("observation_max_allowed_depth_rmse_increase", "observation_max_allowed_depth_rmse_increase"),
                    ("observation_defer_nonresident_enabled", "observation_defer_nonresident_enabled"),
                    ("observation_defer_retry_frames", "observation_defer_retry_frames"),
                    ("observation_pin_deferred_maps", "observation_pin_deferred_maps"),
                    ("observation_failure_reason", "observation_failure_reason"),
                    ("observation_baseline_cache_hit", "observation_baseline_cache_hit"),
                    ("observation_sample_build_ms", "observation_sample_build_ms"),
                    ("observation_initial_trim_ms", "observation_initial_trim_ms"),
                    ("observation_before_score_ms", "observation_before_score_ms"),
                    ("observation_after_score_ms", "observation_after_score_ms"),
                    ("observation_transform_update_ms", "observation_transform_update_ms"),
                    ("observation_final_trim_ms", "observation_final_trim_ms"),
                    ("observation_restore_trim_ms", "observation_restore_trim_ms"),
	                    ("support_policy", "support_policy"),
	                    ("support_policy_used", "support_policy_used"),
	                    ("adaptive_support_pass", "adaptive_support_pass"),
	                    ("high_confidence_support_pass", "high_confidence_support_pass"),
	                    ("adaptive_min_required_loop_constraints", "adaptive_min_required_loop_constraints"),
	                    ("adaptive_min_required_loop_submap_span", "adaptive_min_required_loop_submap_span"),
	                    ("adaptive_min_required_loop_consistent_observations", "adaptive_min_required_loop_consistent_observations"),
	                    ("adaptive_min_required_loop_icp_inliers", "adaptive_min_required_loop_icp_inliers"),
	                    ("adaptive_max_allowed_loop_score", "adaptive_max_allowed_loop_score"),
	                    ("high_confidence_min_required_loop_constraints", "high_confidence_min_required_loop_constraints"),
	                    ("high_confidence_min_required_loop_submap_span", "high_confidence_min_required_loop_submap_span"),
	                    ("high_confidence_min_required_loop_consistent_observations", "high_confidence_min_required_loop_consistent_observations"),
	                    ("high_confidence_min_required_loop_icp_inliers", "high_confidence_min_required_loop_icp_inliers"),
	                    ("high_confidence_max_allowed_loop_score", "high_confidence_max_allowed_loop_score"),
	                    ("high_confidence_max_allowed_loop_icp_rmse", "high_confidence_max_allowed_loop_icp_rmse"),
	                ):
                    if src in section:
                        out[f"pose_graph_{mode}_{dst}"] = section[src]
                if "observation_gate_enabled" in section:
                    out[f"pose_graph_{mode}_observation_gate_enabled"] = int(bool(section["observation_gate_enabled"]))
                if "apply_scale_backoff" in section:
                    out[f"pose_graph_{mode}_apply_scale_backoff"] = int(bool(section["apply_scale_backoff"]))
                if "apply_scale_backoff_min_scale" in section:
                    out[f"pose_graph_{mode}_apply_scale_backoff_min_scale"] = section[
                        "apply_scale_backoff_min_scale"
                    ]
                if "apply_scale_backoff_factor" in section:
                    out[f"pose_graph_{mode}_apply_scale_backoff_factor"] = section[
                        "apply_scale_backoff_factor"
                    ]
                if "apply_scale_backoff_max_trials" in section:
                    out[f"pose_graph_{mode}_apply_scale_backoff_max_trials"] = section[
                        "apply_scale_backoff_max_trials"
                    ]
                if "apply_scale_backoff_stopped_by_deferred_observation" in section:
                    stopped = int(bool(section["apply_scale_backoff_stopped_by_deferred_observation"]))
                    out[f"pose_graph_{mode}_apply_scale_backoff_stopped_by_deferred_observation"] = stopped
                    if mode == "apply":
                        out["pose_graph_apply_scale_backoff_stopped_by_deferred_observation"] = stopped
                if "rejector_count" in section:
                    out[f"pose_graph_{mode}_rejector_count"] = section["rejector_count"]
                if "unique_rejector" in section:
                    out[f"pose_graph_{mode}_unique_rejector"] = str(section["unique_rejector"])
                if "terminal_update" in section:
                    out[f"pose_graph_{mode}_terminal_update"] = int(bool(section["terminal_update"]))
                if "observation_defer_suppressed_for_terminal_sync" in section:
                    out[f"pose_graph_{mode}_observation_defer_suppressed_for_terminal_sync"] = int(
                        bool(section["observation_defer_suppressed_for_terminal_sync"])
                    )
                if isinstance(section.get("apply_scale_candidates"), list):
                    candidate_scales = ";".join(
                        str(value) for value in section["apply_scale_candidates"]
                    )
                    out[f"pose_graph_{mode}_apply_scale_candidates"] = candidate_scales
                    if mode == "apply":
                        out["pose_graph_apply_scale_candidates"] = candidate_scales
                if isinstance(section.get("apply_scale_trials"), list):
                    trials = section["apply_scale_trials"]
                    out[f"pose_graph_{mode}_apply_scale_trials"] = len(trials)
                    if mode == "apply":
                        out["pose_graph_apply_scale_trials"] = len(trials)
                    out[f"pose_graph_{mode}_apply_scale_trial_scales"] = ";".join(
                        str(trial.get("correction_scale", "")) for trial in trials if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_passed"] = ";".join(
                        str(int(bool(trial.get("passed", False)))) for trial in trials if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_reasons"] = ";".join(
                        str(trial.get("failure_reason", "")) for trial in trials if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_eval_ms"] = ";".join(
                        str(trial.get("evaluation_ms", "")) for trial in trials if isinstance(trial, dict)
                    )
                    for src, suffix in (
                        ("node_proposal_ms", "node_proposal_ms"),
                        ("sequential_residual_ms", "sequential_residual_ms"),
                        ("frame_relative_ms", "frame_relative_ms"),
                        ("observation_initial_trim_ms", "observation_initial_trim_ms"),
                        ("observation_final_trim_ms", "observation_final_trim_ms"),
                        ("observation_restore_trim_ms", "observation_restore_trim_ms"),
                        ("observation_valid_fraction_before", "observation_valid_fraction_before"),
                        ("observation_valid_fraction_after", "observation_valid_fraction_after"),
                        ("observation_valid_fraction_ratio", "observation_valid_fraction_ratio"),
                        ("rejector_count", "rejector_count"),
                        ("unique_rejector", "unique_rejector"),
                    ):
                        out[f"pose_graph_{mode}_apply_scale_trial_{suffix}"] = ";".join(
                            str(trial.get(src, "")) for trial in trials if isinstance(trial, dict)
                        )
                    out[f"pose_graph_{mode}_apply_scale_trial_observation_ms"] = ";".join(
                        str(trial.get("observation_gate_ms", "")) for trial in trials if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_observation_cache_hit"] = ";".join(
                        str(int(bool(trial.get("observation_baseline_cache_hit", False))))
                        for trial in trials
                        if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_observation_before_ms"] = ";".join(
                        str(trial.get("observation_before_score_ms", "")) for trial in trials if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_observation_after_ms"] = ";".join(
                        str(trial.get("observation_after_score_ms", "")) for trial in trials if isinstance(trial, dict)
                    )
                    out[f"pose_graph_{mode}_apply_scale_trial_observation_nonresident_ids"] = "|".join(
                        ";".join(str(value) for value in trial.get("observation_nonresident_sdf_submap_ids", []))
                        for trial in trials
                        if isinstance(trial, dict)
                    )
                timing = section.get("timing_ms")
                if isinstance(timing, dict):
                    for key in (
                        "setup_ms",
                        "repeat_guard_ms",
                        "loop_selection_ms",
                        "support_gate_ms",
                        "graph_build_ms",
                        "graph_solve_ms",
                        "scale_evaluation_ms",
                        "transform_apply_ms",
                        "total_ms",
                    ):
                        if key in timing:
                            out[f"pose_graph_{mode}_timing_{key}"] = timing[key]
                if "observation_gate_skipped_by_min_translation" in section:
                    out[f"pose_graph_{mode}_observation_gate_skipped_by_min_translation"] = int(
                        bool(section["observation_gate_skipped_by_min_translation"])
                    )
                if "observation_sample_policy" in section:
                    out[f"pose_graph_{mode}_observation_sample_policy"] = str(section["observation_sample_policy"])
                for src in (
                    "observation_valid_fraction_before",
                    "observation_valid_fraction_after",
                    "observation_valid_fraction_ratio",
                    "observation_valid_fraction_pass",
                    "observation_depth_rmse_ratio_floor",
                    "observation_max_allowed_depth_rmse_absolute_increase",
                    "observation_max_allowed_valid_fraction_drop",
                    "observation_defer_suppressed_for_terminal_sync",
                ):
                    if src in section:
                        out[f"pose_graph_{mode}_{src}"] = section[src]
                if isinstance(section.get("observation_sample_frame_ids"), list):
                    out[f"pose_graph_{mode}_observation_sample_frame_ids"] = ";".join(
                        str(value) for value in section["observation_sample_frame_ids"]
                    )
                if isinstance(section.get("observation_sample_sdf_submap_ids"), list):
                    out[f"pose_graph_{mode}_observation_sample_sdf_submap_ids"] = ";".join(
                        str(value) for value in section["observation_sample_sdf_submap_ids"]
                    )
                if isinstance(section.get("observation_sample_resident_flags"), list):
                    flags = section["observation_sample_resident_flags"]
                    out[f"pose_graph_{mode}_observation_sample_resident_count"] = sum(1 for value in flags if int(value) != 0)
                if isinstance(section.get("observation_pinned_sdf_submap_ids"), list):
                    out[f"pose_graph_{mode}_observation_pinned_sdf_submap_ids"] = ";".join(
                        str(value) for value in section["observation_pinned_sdf_submap_ids"]
                    )
                if isinstance(section.get("observation_nonresident_sdf_submap_ids"), list):
                    out[f"pose_graph_{mode}_observation_nonresident_sdf_submap_ids"] = ";".join(
                        str(value) for value in section["observation_nonresident_sdf_submap_ids"]
                    )
                if isinstance(section.get("observation_deferred_retry_pending_sdf_submap_ids"), list):
                    out[f"pose_graph_{mode}_observation_deferred_retry_pending_sdf_submap_ids"] = ";".join(
                        str(value) for value in section["observation_deferred_retry_pending_sdf_submap_ids"]
                    )
                if "min_required_translation_delta" in section:
                    out[f"pose_graph_{mode}_min_required_translation_delta_m"] = section["min_required_translation_delta"]
                if "skip_reason" in section:
                    out[f"pose_graph_{mode}_skip_reason"] = str(section["skip_reason"])
                loop_selection = section.get("loop_selection")
                if isinstance(loop_selection, dict):
                    if "policy" in loop_selection:
                        out[f"pose_graph_{mode}_loop_selection_policy"] = str(loop_selection["policy"])
                    if "input_loop_constraints" in loop_selection:
                        out[f"pose_graph_{mode}_loop_selection_input"] = loop_selection["input_loop_constraints"]
                    if "selected_loop_constraints" in loop_selection:
                        out[f"pose_graph_{mode}_loop_selection_selected"] = loop_selection["selected_loop_constraints"]
                    if "rejected_loop_constraints" in loop_selection:
                        out[f"pose_graph_{mode}_loop_selection_rejected"] = loop_selection["rejected_loop_constraints"]
                    if "repeat_guard_skipped_loop_constraints" in loop_selection:
                        out[f"pose_graph_{mode}_repeat_guard_skipped_loop_constraints"] = loop_selection[
                            "repeat_guard_skipped_loop_constraints"
                        ]
                    if "repeat_guard_removed_all_loop_constraints" in loop_selection:
                        out[f"pose_graph_{mode}_repeat_guard_removed_all_loop_constraints"] = int(
                            bool(loop_selection["repeat_guard_removed_all_loop_constraints"])
                        )
                    if "repeat_guard_input_before_filter" in loop_selection:
                        out[f"pose_graph_{mode}_repeat_guard_input_before_filter"] = loop_selection[
                            "repeat_guard_input_before_filter"
                        ]
                    if "repeat_guard_input_after_filter" in loop_selection:
                        out[f"pose_graph_{mode}_repeat_guard_input_after_filter"] = loop_selection[
                            "repeat_guard_input_after_filter"
                        ]
                    pcm = loop_selection.get("pcm")
                    if isinstance(pcm, dict):
                        for src, dst in (
                            ("support_candidates", "pcm_support_candidates"),
                            ("clustered_support_candidates", "pcm_clustered_support_candidates"),
                            ("event_clusters", "pcm_event_clusters"),
                            ("event_cluster_radius", "pcm_event_cluster_radius"),
                            ("min_distinct_events", "pcm_min_distinct_events"),
                            ("min_long_span", "pcm_min_long_span"),
                            ("prefer_long_span", "pcm_prefer_long_span"),
                            ("rejected_cliques_min_long_span", "pcm_rejected_cliques_min_long_span"),
                            ("compatible_pairs", "pcm_compatible_pairs"),
                            ("enumerated_cliques", "pcm_enumerated_cliques"),
                            ("evaluated_cliques", "pcm_evaluated_cliques"),
                            ("selected_clique_size", "pcm_selected_clique_size"),
                        ):
                            if src in pcm:
                                out[f"pose_graph_{mode}_{dst}"] = pcm[src]
                        if "scale_threshold_by_cycle_length" in pcm:
                            out[f"pose_graph_{mode}_pcm_scale_threshold_by_cycle_length"] = int(
                                bool(pcm["scale_threshold_by_cycle_length"])
                            )
                        if "selection_rejection_reason" in pcm:
                            out[f"pose_graph_{mode}_pcm_selection_rejection_reason"] = str(
                                pcm["selection_rejection_reason"]
                            )
                for src, dst in (
                    ("loop_support_constraints", "loop_support_constraints"),
                    ("loop_support_max_submap_span", "loop_support_max_submap_span"),
	                    ("loop_support_min_consistent_observations", "loop_support_min_consistent_observations"),
	                    ("loop_support_min_icp_inliers", "loop_support_min_icp_inliers"),
	                    ("loop_support_max_score", "loop_support_max_score"),
	                    ("loop_support_max_icp_rmse", "loop_support_max_icp_rmse"),
	                    ("min_required_loop_constraints", "min_required_loop_constraints"),
                    ("min_required_loop_submap_span", "min_required_loop_submap_span"),
                    ("min_required_loop_consistent_observations", "min_required_loop_consistent_observations"),
                    ("min_required_loop_icp_inliers", "min_required_loop_icp_inliers"),
                    ("max_allowed_loop_score", "max_allowed_loop_score"),
                ):
                    if src in section:
                        out[f"pose_graph_{mode}_{dst}"] = section[src]
        except json.JSONDecodeError:
            out["pose_graph_manifest_error"] = "json_decode"

    candidates_path = root / "pose_graph_loop_candidates.csv"
    if candidates_path.exists():
        reason_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        total_rows = 0
        max_icp_inliers = 0
        best_icp_rmse: float | None = None
        max_allowed_translation_delta: float | None = None
        max_allowed_rotation_delta: float | None = None
        max_allowed_score: float | None = None

        with candidates_path.open(newline="") as f:
            for row in csv.DictReader(f):
                total_rows += 1
                status = str(row.get("status", "")).strip() or "unknown"
                reason = str(row.get("reason", "")).strip() or "unknown"
                status_counts[status] = status_counts.get(status, 0) + 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                max_icp_inliers = max(
                    max_icp_inliers,
                    safe_int(row.get("verification_icp_inliers"), 0),
                )
                rmse = safe_float(row.get("verification_icp_rmse"))
                if rmse is not None and rmse >= 0.0:
                    best_icp_rmse = rmse if best_icp_rmse is None else min(best_icp_rmse, rmse)
                allowed_translation = safe_float(row.get("verification_icp_allowed_translation_delta"))
                if allowed_translation is not None and allowed_translation >= 0.0:
                    max_allowed_translation_delta = (
                        allowed_translation if max_allowed_translation_delta is None
                        else max(max_allowed_translation_delta, allowed_translation)
                    )
                allowed_rotation = safe_float(row.get("verification_icp_allowed_rotation_delta_degrees"))
                if allowed_rotation is not None and allowed_rotation >= 0.0:
                    max_allowed_rotation_delta = (
                        allowed_rotation if max_allowed_rotation_delta is None
                        else max(max_allowed_rotation_delta, allowed_rotation)
                    )
                allowed_score = safe_float(row.get("verification_icp_allowed_score"))
                if allowed_score is not None and allowed_score >= 0.0:
                    max_allowed_score = (
                        allowed_score if max_allowed_score is None
                        else max(max_allowed_score, allowed_score)
                    )

        out["loop_candidate_rows"] = total_rows
        for status, count in sorted(status_counts.items()):
            out[f"loop_status_{slug_key(status)}"] = count
        for reason, count in sorted(reason_counts.items()):
            out[f"loop_reject_{slug_key(reason)}"] = count
        if reason_counts:
            ordered_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
            out["loop_reject_top_reason"] = ordered_reasons[0][0]
            out["loop_reject_top_count"] = ordered_reasons[0][1]
            out["loop_reject_reasons"] = ";".join(
                f"{reason}:{count}" for reason, count in ordered_reasons
            )
        out["loop_candidate_max_icp_inliers"] = max_icp_inliers
        if best_icp_rmse is not None:
            out["loop_candidate_best_icp_rmse"] = best_icp_rmse
        if max_allowed_translation_delta is not None:
            out["loop_candidate_max_allowed_translation_delta"] = max_allowed_translation_delta
        if max_allowed_rotation_delta is not None:
            out["loop_candidate_max_allowed_rotation_delta_degrees"] = max_allowed_rotation_delta
        if max_allowed_score is not None:
            out["loop_candidate_max_allowed_score"] = max_allowed_score
    return out


def add_terminal_inclusive_timing(row: dict[str, float | int | str]) -> None:
    """Add end-to-end throughput without double-counting in-frame updates.

    Online pose-graph work is already present in timing.csv. Only the explicit
    end-of-sequence update is outside that per-frame timer.
    """
    frame_count = safe_int(row.get("timing_frame_count"), 0)
    frame_ms = safe_float(row.get("total_frame_ms_sum"))
    terminal_ms = safe_float(row.get("terminal_pose_graph_ms")) or 0.0
    if frame_count <= 0 or frame_ms is None or frame_ms < 0.0:
        return
    total_ms = frame_ms + terminal_ms
    row["end_to_end_sequence_ms"] = total_ms
    row["end_to_end_fps"] = 1000.0 * frame_count / total_ms if total_ms > 0.0 else 0.0
    row["terminal_latency_ms"] = terminal_ms
    row["terminal_overhead_pct"] = 100.0 * terminal_ms / total_ms if total_ms > 0.0 else 0.0
    pipeline_fps = safe_float(row.get("fps"))
    if pipeline_fps is not None and pipeline_fps > 0.0:
        # C++ starts this wall clock immediately before the frame loop and
        # stops it after the terminal pose-graph update and EOS checks.
        row["terminal_inclusive_pipeline_fps"] = pipeline_fps
        row["terminal_inclusive_pipeline_ms"] = 1000.0 * frame_count / pipeline_fps
    else:
        row["terminal_inclusive_pipeline_fps"] = row["end_to_end_fps"]
        row["terminal_inclusive_pipeline_ms"] = total_ms


def annotate_acceptance_status(row: dict[str, float | int | str]) -> None:
    blockers: list[str] = []
    warnings: list[str] = []
    raw_failure_ratio_block_threshold = 0.05
    raw_failed_streak_block_threshold = 10

    matched = safe_int(row.get("matched_poses"), 0)
    est_poses = safe_int(row.get("est_poses"), 0)
    tracking_rows = safe_int(row.get("tracking_rows"), 0)
    raw_failed = safe_int(row.get("raw_tracking_failed"), 0)
    policy_failed = safe_int(row.get("policy_tracking_failed"), 0)
    raw_non_good = safe_int(row.get("raw_tracking_non_good"), 0)
    policy_non_good = safe_int(row.get("policy_tracking_non_good"), 0)
    raw_max_failed_streak = safe_int(row.get("raw_tracking_max_failed_streak"), 0)
    capacity_rejects = safe_int(row.get("capacity_handoff_rejects"), 0)
    link_rejects = safe_int(row.get("link_rejects"), 0)
    raw_failure_ratio = float(raw_failed / tracking_rows) if tracking_rows > 0 else 0.0
    policy_failure_ratio = float(policy_failed / tracking_rows) if tracking_rows > 0 else 0.0

    pose_metrics_valid = matched > 0 and "ate_actual_rmse_cm" in row
    if not pose_metrics_valid:
        blockers.append(str(row.get("ate_error", "no_valid_ate")))
    if est_poses == 0 and tracking_rows > 0:
        blockers.append("pose_export_missing")
    if policy_failed > 0:
        blockers.append(f"policy_tracking_failed:{policy_failed}")
    if raw_failed > 0:
        warnings.append(f"raw_tracking_failed:{raw_failed}")
    if raw_failure_ratio > raw_failure_ratio_block_threshold:
        blockers.append(f"raw_tracking_failure_ratio:{raw_failure_ratio:.3f}")
    if raw_max_failed_streak >= raw_failed_streak_block_threshold:
        blockers.append(f"raw_tracking_failed_streak:{raw_max_failed_streak}")
    if raw_max_failed_streak > 0:
        warnings.append(f"raw_failed_streak:{raw_max_failed_streak}")
    if raw_non_good > 0 and raw_failed == 0:
        warnings.append(f"raw_tracking_non_good:{raw_non_good}")
    if policy_non_good > 0 and policy_failed == 0:
        warnings.append(f"policy_tracking_non_good:{policy_non_good}")
    if capacity_rejects > 0:
        warnings.append(f"capacity_handoff_rejects:{capacity_rejects}")
    if link_rejects > 0:
        warnings.append(f"link_rejects:{link_rejects}")

    row["pose_metrics_valid"] = int(pose_metrics_valid)
    row["tracking_valid_for_acceptance"] = int(
        policy_failed == 0
        and raw_failure_ratio <= raw_failure_ratio_block_threshold
        and raw_max_failed_streak < raw_failed_streak_block_threshold
    )
    row["run_valid_for_acceptance"] = int(pose_metrics_valid and not blockers)
    row["acceptance_blockers"] = ";".join(blockers)
    row["acceptance_warnings"] = ";".join(warnings)
    row["raw_tracking_failure_ratio"] = raw_failure_ratio
    row["policy_tracking_failure_ratio"] = policy_failure_ratio


def run_config(run_dir: Path) -> dict:
    configs = sorted(run_dir.glob("*.yaml"))
    if not configs:
        return {}
    return yaml.safe_load(configs[0].read_text())


def parse_image_quality_results(run_dir: Path) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    metric_dirs = (
        ("render", run_dir / "val" / "render" / "results.json"),
        ("raycast", run_dir / "val" / "raycast_color" / "results.json"),
    )
    for prefix, path in metric_dirs:
        if not path.exists():
            continue
        try:
            metrics = json.loads(path.read_text(errors="replace"))
        except Exception as exc:
            out[f"{prefix}_metric_error"] = str(exc)
            continue
        for src_key, dst_suffix in (("PSNR", "psnr"), ("SSIM", "ssim"), ("LPIPS", "lpips")):
            if src_key in metrics:
                out[f"{prefix}_{dst_suffix}"] = metrics[src_key]
    if "render_psnr" in out and "raycast_psnr" in out:
        out["render_minus_raycast_psnr"] = float(out["render_psnr"]) - float(out["raycast_psnr"])
    if "render_ssim" in out and "raycast_ssim" in out:
        out["render_minus_raycast_ssim"] = float(out["render_ssim"]) - float(out["raycast_ssim"])
    if "render_lpips" in out and "raycast_lpips" in out:
        out["render_minus_raycast_lpips"] = float(out["render_lpips"]) - float(out["raycast_lpips"])
    return out


def parse_realtime_summary(run_dir: Path) -> dict[str, float | int | str]:
    path = run_dir / "realtime_summary.json"
    if not path.is_file():
        return {}
    out: dict[str, float | int | str] = {
        "realtime_summary_path": str(path),
    }
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        out["realtime_summary_error"] = str(exc)
        return out
    out["realtime_all_frames_processed"] = int(
        bool(payload.get("all_frames_processed", False))
    )
    models = payload.get("arrival_models", {})
    if not isinstance(models, dict):
        out["realtime_summary_error"] = "arrival_models is not an object"
        return out
    for source, prefix in (
        ("fixed_rate", "realtime_fixed"),
        ("dataset_timestamps", "realtime_dataset"),
    ):
        values = models.get(source)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, bool):
                out[f"{prefix}_{key}"] = int(value)
            elif isinstance(value, (int, float, str)):
                out[f"{prefix}_{key}"] = value
    return out


def parse_runtime_resources(run_dir: Path) -> dict[str, float | int | str]:
    path = run_dir / "runtime_resources.json"
    if not path.is_file():
        return {}
    out: dict[str, float | int | str] = {
        "runtime_resources_path": str(path),
    }
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        out["runtime_resources_error"] = str(exc)
        return out
    for key, value in payload.items():
        if isinstance(value, bool):
            out[f"runtime_{key}"] = int(value)
        elif isinstance(value, (int, float, str)):
            out[f"runtime_{key}"] = value
    return out


def resolve_pose_dir(run_dir: Path, pose_subdir: str) -> Path:
    candidate = run_dir / pose_subdir
    if candidate.exists():
        return candidate
    if "/" not in pose_subdir and "\\" not in pose_subdir:
        val_candidate = run_dir / "val" / pose_subdir
        if val_candidate.exists():
            return val_candidate
    return candidate


def summarize_run(run_dir: Path, pose_subdir: str, rpe_delta: int) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"run_dir": str(run_dir), "run": run_dir.name}
    cfg = run_config(run_dir)
    row.update(collect_config_profile_fields(cfg))
    pipe = cfg.get("PIPE", {}) if cfg else {}
    for key in (
        "random_seed",
        "experiment_repeat_index",
        "experiment_repeat_count",
        "experiment_repeat_seed_step",
        "experiment_group_key",
    ):
        if key in pipe:
            row[key] = pipe[key]
    reader = cfg.get("READER", {}) if cfg else {}
    gt_dir = Path(reader.get("input_dir", "")) / reader.get("pose_path", "camera")
    est_dir = resolve_pose_dir(run_dir, pose_subdir)
    est_poses = read_poses(est_dir)
    gt_poses = read_poses(gt_dir)
    row.update({"gt_pose_dir": str(gt_dir), "est_pose_dir": str(est_dir), "est_poses": len(est_poses), "gt_poses": len(gt_poses)})
    row.update(ate_metrics(est_poses, gt_poses))
    row.update(rpe_metrics(est_poses, gt_poses, rpe_delta))
    boundary_pairs = frame_assignment_boundary_pairs(run_dir)
    row["boundary_transition_pairs_candidate"] = len(boundary_pairs)
    row.update(relative_pose_error_metrics_for_pairs(est_poses, gt_poses, boundary_pairs, "boundary"))

    corrected_dir = resolve_pose_dir(run_dir, "val/corrected_pose")
    corrected_poses = read_poses(corrected_dir)
    if corrected_poses:
        corrected_ate = ate_metrics(corrected_poses, gt_poses)
        corrected_rpe = rpe_metrics(corrected_poses, gt_poses, rpe_delta)
        corrected_boundary_rpe = relative_pose_error_metrics_for_pairs(
            corrected_poses, gt_poses, boundary_pairs, "corrected_boundary"
        )
        row["corrected_pose_dir"] = str(corrected_dir)
        row["corrected_est_poses"] = len(corrected_poses)
        for key, value in corrected_ate.items():
            row[f"corrected_{key}"] = value
        for key, value in corrected_rpe.items():
            row[f"corrected_{key}"] = value
        row.update(corrected_boundary_rpe)
        if "ate_actual_rmse_cm" in row and "corrected_ate_actual_rmse_cm" in row:
            raw_rmse = float(row["ate_actual_rmse_cm"])
            corrected_rmse = float(row["corrected_ate_actual_rmse_cm"])
            row["corrected_ate_delta_cm"] = corrected_rmse - raw_rmse
            row["corrected_ate_improvement_pct"] = (
                (raw_rmse - corrected_rmse) / raw_rmse * 100.0 if raw_rmse > 0.0 else 0.0
            )
        if "rpe_translation_rmse_cm" in row and "corrected_rpe_translation_rmse_cm" in row:
            row["corrected_rpe_translation_delta_cm"] = (
                float(row["corrected_rpe_translation_rmse_cm"]) - float(row["rpe_translation_rmse_cm"])
            )
        if "boundary_rpe_translation_rmse_cm" in row and "corrected_boundary_rpe_translation_rmse_cm" in row:
            row["corrected_boundary_rpe_translation_delta_cm"] = (
                float(row["corrected_boundary_rpe_translation_rmse_cm"])
                - float(row["boundary_rpe_translation_rmse_cm"])
            )

    online_end_dir = resolve_pose_dir(run_dir, "pose_graph_live/online_end_poses")
    online_end_poses = read_poses(online_end_dir)
    if online_end_poses:
        row["online_end_pose_dir"] = str(online_end_dir)
        row["online_end_est_poses"] = len(online_end_poses)
        row["online_end_complete"] = int(len(online_end_poses) == len(est_poses))
        for key, value in ate_metrics(online_end_poses, gt_poses).items():
            row[f"online_end_{key}"] = value
        for key, value in rpe_metrics(online_end_poses, gt_poses, rpe_delta).items():
            row[f"online_end_{key}"] = value
        row.update(
            relative_pose_error_metrics_for_pairs(
                online_end_poses,
                gt_poses,
                boundary_pairs,
                "online_end_boundary",
            )
        )
    loop_segment = latest_applied_loop_endpoint_frames(run_dir / "pose_graph_live")
    if loop_segment is not None:
        start_frame, end_frame = loop_segment
        row.update(ate_segment_metrics(est_poses, gt_poses, start_frame, end_frame, "raw_loop"))
        if corrected_poses:
            row.update(ate_segment_metrics(corrected_poses, gt_poses, start_frame, end_frame, "corrected_loop"))
            for segment in ("before", "between", "after"):
                raw_key = f"raw_loop_{segment}_ate_actual_rmse_cm"
                corrected_key = f"corrected_loop_{segment}_ate_actual_rmse_cm"
                if raw_key in row and corrected_key in row:
                    row[f"corrected_loop_{segment}_ate_delta_cm"] = (
                        float(row[corrected_key]) - float(row[raw_key])
                    )
    row.update(parse_time_log(run_dir))
    row.update(parse_runtime_tsdf_log(run_dir))
    row.update(parse_timing(run_dir))
    row.update(parse_submap_stats(run_dir))
    row.update(parse_frame_assignments(run_dir))
    row.update(parse_tracking(run_dir))
    row.update(parse_transitions(run_dir))
    row.update(parse_paging(run_dir))
    row.update(parse_pose_graph(run_dir))
    add_terminal_inclusive_timing(row)
    row.update(parse_image_quality_results(run_dir))
    row.update(parse_realtime_summary(run_dir))
    row.update(parse_runtime_resources(run_dir))
    add_loop_memory_estimate(row, cfg)
    annotate_acceptance_status(row)
    return row


def acceptance_blocked_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Return rows that do not satisfy the summarizer's acceptance contract."""
    return [row for row in rows if safe_int(row.get("run_valid_for_acceptance"), 0) != 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Run directories to summarize. Defaults to immediate children of --root.")
    parser.add_argument("--root", type=Path, default=None, help="Root whose immediate child directories are summarized when runs are omitted.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--pose-subdir", default="val/pose")
    parser.add_argument("--rpe-delta", type=int, default=1)
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit nonzero after writing the summary if any run has acceptance blockers.",
    )
    args = parser.parse_args()

    runs = args.runs
    if not runs:
        if args.root is None:
            parser.error("provide run directories or --root")
        runs = sorted(path for path in args.root.iterdir() if path.is_dir())
    elif args.root is not None:
        runs = [path if path.is_absolute() or path.exists() else args.root / path for path in runs]
    rows = [summarize_run(path, args.pose_subdir, max(1, args.rpe_delta)) for path in runs]
    fieldnames = sorted({key for row in rows for key in row})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.fail_on_blockers:
        if not rows:
            print("[ERROR] No runs were available to summarize.", file=sys.stderr)
            return 1
        blocked = acceptance_blocked_rows(rows)
        if blocked:
            print(
                f"[ERROR] {len(blocked)} of {len(rows)} summarized run(s) have acceptance blockers:",
                file=sys.stderr,
            )
            for row in blocked:
                reason = str(row.get("acceptance_blockers", "")).strip() or "run_valid_for_acceptance=0"
                print(f"  {row.get('run', '<unnamed>')}: {reason}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
