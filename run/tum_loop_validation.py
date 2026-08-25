#!/usr/bin/env python3
"""Generate and optionally run canonical TUM RGB-D loop-closure configs."""

from __future__ import annotations

import argparse
import copy
from collections import deque
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from benchmark_protocol import (  # noqa: E402
    PROTOCOL_NONE,
    TUM_DEPLOYED_V1,
    TUM_INTEGRATED_V1,
    TUM_TRACKING_V1,
    apply_protocol,
    infer_binary_hash_profile,
    protocol_release_status,
    require_max_performance_state,
    sha256_file,
    tum_role_for_variant,
    write_manifest,
)
DATA_ROOT = Path(os.environ.get("GPS_SLAM_DATA_ROOT", PROJECT_ROOT / "data"))
RUN_ROOT = Path(os.environ.get("GPS_SLAM_RUN_ROOT", PROJECT_ROOT / "output"))
DEFAULT_TUM_ROOT = DATA_ROOT / "tum_rgbd" / "processed"

HASH_PROFILE_ESTIMATED_SCENE_BYTES = {
    # Mirrors ITMLocalMap::estimateSceneBytes() for ITMVoxel_s_rgb +
    # ITMVoxelBlockHash. CPU paging stores this full scene copy per passive map.
    "compact_64k": 273_547_264,
    "medium_128k": 547_094_528,
    "legacy_large": 1_094_189_056,
}

SEQUENCES = {
    "fr1_desk": "rgbd_dataset_freiburg1_desk",
    "fr2_desk": "rgbd_dataset_freiburg2_desk",
    "fr2_xyz": "rgbd_dataset_freiburg2_xyz",
    "fr3_office": "rgbd_dataset_freiburg3_long_office_household",
    "fr1_room": "rgbd_dataset_freiburg1_room",
    "fr2_pioneer_slam": "rgbd_dataset_freiburg2_pioneer_slam",
    "fr2_pioneer_slam2": "rgbd_dataset_freiburg2_pioneer_slam2",
}

VARIANTS = {
    "basic_gaussian",
    "basic_sdf",
    "multi_hybrid_apply",
    "multi_hybrid_dry_solve",
    "multi_hybrid_no_loop",
    "multi_hybrid_orb_apply",
    "multi_hybrid_orb_dry_solve",
    "multi_sdf",
    "multi_sdf_dry_solve",
    "multi_sdf_orb_apply",
    "multi_sdf_orb_dry_solve",
    "multi_sdf_sequential_noop",
    "multi_sdf_synthetic",
}

SAFE_PATH_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_path_token(value: str, option_name: str, allow_empty: bool = False) -> str:
    """Validate a user-provided token before it becomes part of a path."""
    if not value:
        if allow_empty:
            return value
        raise ValueError(f"{option_name} must not be empty")
    if value != value.strip() or not SAFE_PATH_TOKEN_RE.fullmatch(value):
        raise ValueError(
            f"{option_name}={value!r} is unsafe; use only letters, digits, '.', '_', and '-' "
            "and start with a letter or digit"
        )
    return value


def validate_known_labels(values: list[str], known: set[str], option_name: str) -> None:
    """Reject empty, duplicate, unsafe, or unknown sequence/variant labels."""
    if not values:
        raise ValueError(f"{option_name} requires at least one value")
    for value in values:
        validate_path_token(value, option_name)
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{option_name} contains duplicate value(s): {', '.join(duplicates)}")
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"unknown {option_name} value(s): {', '.join(unknown)}; choose from: "
            + ", ".join(sorted(known))
        )


PROFILE_FINGERPRINT_IGNORE_PIPE_KEYS = {
    "bypass_libtorch_atexit",
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


def profile_fingerprint(cfg: dict) -> str:
    encoded = json.dumps(
        normalize_for_profile_fingerprint(cfg),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def nested_get(cfg: dict, path: tuple[str, ...], default=None):
    value = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def deep_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def tsdf_residency_profile_defaults(profile: str) -> dict:
    """Named MultiEngine residency policies for reproducible TUM loop runs."""
    if profile == "manual":
        return {}
    if profile == "thor_cpu_bounded":
        return {
            "paging": True,
            "paging_storage": "cpu",
            "resident_gpu_maps": 8,
            "async_page_out": True,
            "max_pending_page_snapshots": 1,
            "max_pending_page_snapshot_mb": 0,
            "async_page_in": True,
            "max_pending_page_ins": 1,
            "max_pending_page_in_mb": 512,
            "cpu_page_cache_limit_mb": 8192,
            "render_prefetch": True,
            "render_prefetch_lookahead_frames": 1,
            "render_prefetch_lookahead_max_maps": 0,
            "render_pin_primary_submap": False,
        }
    if profile == "thor_streaming_bounded":
        return {
            "paging": True,
            "paging_storage": "cpu",
            "resident_gpu_maps": 8,
            "async_page_out": True,
            "max_pending_page_snapshots": 3,
            "max_pending_page_snapshot_mb": 2048,
            "async_page_in": True,
            "max_pending_page_ins": 2,
            "max_pending_page_in_mb": 1024,
            "cpu_page_cache_limit_mb": 12288,
            "render_prefetch": True,
            "render_prefetch_lookahead_frames": 1,
            "render_prefetch_lookahead_max_maps": 0,
            "render_pin_primary_submap": False,
        }
    if profile == "eval_resident64":
        return {
            "paging": True,
            "resident_gpu_maps": 64,
            "render_prefetch": False,
            "render_prefetch_lookahead_frames": 1,
            "render_prefetch_lookahead_max_maps": 0,
            "render_pin_primary_submap": False,
        }
    if profile == "disk_stress":
        return {
            "paging": True,
            "paging_storage": "disk",
            "resident_gpu_maps": 8,
            "async_page_out": True,
            "max_pending_page_snapshots": 1,
            "max_pending_page_snapshot_mb": 0,
            "async_page_in": True,
            "max_pending_page_ins": 1,
            "max_pending_page_in_mb": 512,
            "render_prefetch": True,
            "render_prefetch_lookahead_frames": 1,
            "render_prefetch_lookahead_max_maps": 0,
            "render_pin_primary_submap": False,
        }
    raise ValueError(f"Unknown TSDF residency profile: {profile}")


def apply_tsdf_lifecycle_profile(cfg: dict, profile: str) -> dict:
    """Apply named MultiEngine lifecycle policies without changing defaults."""
    cfg = copy.deepcopy(cfg)
    if profile == "default":
        return cfg
    multi = cfg["PIPE"]["TSDF"]["multi"]
    if profile == "paper_parity":
        # Keep submaps available, but avoid unnecessary fragmentation on the
        # short/full-resolution TUM paper scenes. Weak tracking now warms a
        # candidate; only a real failed streak may force promotion.
        cfg["PIPE"]["TSDF"]["tracking_failure_mode"] = "contain_failed"
        cfg["PIPE"]["TSDF"]["reject_nonfinite_tracking_pose"] = True
        multi["boundary_policy"] = "capacity"
        multi["allow_visibility_handoff"] = True
        multi["max_frames"] = 5000
        multi["new_map_block_occupancy"] = 0.82
        multi["max_block_occupancy"] = 0.92
        multi["fusion_reserve_blocks"] = 4096
        multi["occupancy_growth_lookahead_frames"] = 0
        multi["occupancy_growth_min_primary_frames"] = 120
        multi["occupancy_growth_sample_frames"] = 10
        multi["tracking_boundary_policy"] = "raw_weak_streak"
        multi["tracking_boundary_handoff_policy"] = "failed_only"
        multi["raw_weak_boundary_frames"] = 4
        multi["raw_failed_boundary_frames"] = 5
        multi["tracking_boundary_min_primary_frames"] = 80
        multi["tracking_boundary_cooldown_frames"] = 80
        multi["tracking_boundary_force_handoff"] = True
        multi["primary_failure_handoff_policy"] = "promote_visible_candidate"
        multi["primary_failure_handoff_min_raw_failed_frames"] = 5
        multi["primary_failure_handoff_raw_gate"] = "failed_only"
        multi["primary_failure_handoff_min_raw_weak_frames"] = 0
        multi["primary_failure_handoff_min_candidate_visible_ratio"] = 0.20
        multi["primary_failure_handoff_min_candidate_blocks"] = 1024
        multi["primary_failure_handoff_min_candidate_good_frames"] = 8
        multi["primary_failure_handoff_min_overlap_frames"] = 0
        # Retain the fresh candidate tracker estimate only after validated
        # normal overlap; emergency failure promotion still preserves the
        # last finite primary global pose.
        multi["handoff_pose_policy"] = "validated_candidate"
        multi["primary_failure_handoff_require_pose_continuity"] = True
        multi["candidate_recovery_policy"] = "keep_if_visible"
        multi["candidate_recovery_min_visible_ratio"] = 0.20
        multi["candidate_recovery_min_blocks"] = 1024
        multi["candidate_recovery_max_attempts"] = 2
        multi["new_map_handoff_policy"] = "validated_visibility"
        multi["new_map_handoff_min_visible_ratio"] = 0.05
        multi["new_map_handoff_min_visible_margin"] = 0.0
        multi["new_map_handoff_min_accepts"] = 3
        multi["new_map_handoff_emergency_min_accepts"] = 2
        return cfg
    if profile == "paper_parity_adaptive_longspan":
        cfg = apply_tsdf_lifecycle_profile(cfg, "paper_parity")
        multi = cfg["PIPE"]["TSDF"]["multi"]
        multi["occupancy_threshold_policy"] = "adaptive_longspan"
        multi["adaptive_new_map_block_occupancy"] = 0.86
        multi["adaptive_max_block_occupancy"] = 0.95
        multi["adaptive_occupancy_min_primary_frames"] = 180
        multi["adaptive_occupancy_min_processed_frames"] = 600
        multi["adaptive_occupancy_max_raw_weak_streak"] = 1
        multi["adaptive_occupancy_max_raw_failed_streak"] = 0
        return cfg
    raise ValueError(f"Unknown TSDF lifecycle profile: {profile}")


def profile_summary(cfg: dict) -> dict:
    """Small stable summary used to spot invalid cross-profile comparisons."""
    return {
        "work_mode": cfg.get("work_mode"),
        "bypass_libtorch_atexit": nested_get(cfg, ("PIPE", "bypass_libtorch_atexit")),
        "reader_frames": [
            nested_get(cfg, ("READER", "start_frame")),
            nested_get(cfg, ("READER", "end_frame")),
        ],
        "tsdf_engine": nested_get(cfg, ("PIPE", "TSDF", "engine")),
        "tracking_failure_mode": nested_get(cfg, ("PIPE", "TSDF", "tracking_failure_mode")),
        "reject_nonfinite_tracking_pose": nested_get(cfg, ("PIPE", "TSDF", "reject_nonfinite_tracking_pose")),
        "hash_profile": nested_get(cfg, ("PIPE", "TSDF", "multi", "hash_profile")),
        "tsdf_paging": nested_get(cfg, ("PIPE", "TSDF", "multi", "paging")),
        "tsdf_paging_storage": nested_get(cfg, ("PIPE", "TSDF", "multi", "paging_storage")),
        "resident_gpu_maps": nested_get(cfg, ("PIPE", "TSDF", "multi", "resident_gpu_maps")),
        "tsdf_async_page_out": nested_get(cfg, ("PIPE", "TSDF", "multi", "async_page_out")),
        "tsdf_max_pending_page_snapshots": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "max_pending_page_snapshots")
        ),
        "tsdf_max_pending_page_snapshot_mb": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "max_pending_page_snapshot_mb")
        ),
        "tsdf_async_page_in": nested_get(cfg, ("PIPE", "TSDF", "multi", "async_page_in")),
        "tsdf_max_pending_page_ins": nested_get(cfg, ("PIPE", "TSDF", "multi", "max_pending_page_ins")),
        "tsdf_max_pending_page_in_mb": nested_get(cfg, ("PIPE", "TSDF", "multi", "max_pending_page_in_mb")),
        "tsdf_cpu_page_cache_limit_mb": nested_get(cfg, ("PIPE", "TSDF", "multi", "cpu_page_cache_limit_mb")),
        "tsdf_render_prefetch": nested_get(cfg, ("PIPE", "TSDF", "multi", "render_prefetch")),
        "tsdf_render_prefetch_lookahead_frames": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "render_prefetch_lookahead_frames")
        ),
        "tsdf_render_prefetch_lookahead_max_maps": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "render_prefetch_lookahead_max_maps")
        ),
        "tsdf_render_pin_primary_submap": nested_get(cfg, ("PIPE", "TSDF", "multi", "render_pin_primary_submap")),
        "force_single_submap": nested_get(cfg, ("PIPE", "TSDF", "multi", "force_single_submap")),
        "tsdf_max_frames": nested_get(cfg, ("PIPE", "TSDF", "multi", "max_frames")),
        "tsdf_new_map_occupancy": nested_get(cfg, ("PIPE", "TSDF", "multi", "new_map_block_occupancy")),
        "tsdf_max_occupancy": nested_get(cfg, ("PIPE", "TSDF", "multi", "max_block_occupancy")),
        "tsdf_occupancy_growth_lookahead_frames": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "occupancy_growth_lookahead_frames")
        ),
        "tsdf_overlap_frames": nested_get(cfg, ("PIPE", "TSDF", "multi", "overlap_frames")),
        "tsdf_allow_visibility_handoff": nested_get(cfg, ("PIPE", "TSDF", "multi", "allow_visibility_handoff")),
        "tsdf_occupancy_threshold_policy": nested_get(cfg, ("PIPE", "TSDF", "multi", "occupancy_threshold_policy")),
        "tsdf_adaptive_new_map_occupancy": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "adaptive_new_map_block_occupancy")
        ),
        "tsdf_adaptive_max_occupancy": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "adaptive_max_block_occupancy")
        ),
        "tsdf_adaptive_min_primary_frames": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "adaptive_occupancy_min_primary_frames")
        ),
        "tsdf_adaptive_min_processed_frames": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "adaptive_occupancy_min_processed_frames")
        ),
        "tsdf_adaptive_max_raw_weak_streak": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "adaptive_occupancy_max_raw_weak_streak")
        ),
        "tsdf_adaptive_max_raw_failed_streak": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "adaptive_occupancy_max_raw_failed_streak")
        ),
        "tsdf_emergency_overlap_frames": nested_get(cfg, ("PIPE", "TSDF", "multi", "emergency_overlap_frames")),
        "tsdf_trial_frames": nested_get(cfg, ("PIPE", "TSDF", "multi", "trial_frames")),
        "sequential_transform_max_translation": nested_get(cfg, ("PIPE", "TSDF", "multi", "sequential_transform_max_translation")),
        "handoff_pose_policy": nested_get(cfg, ("PIPE", "TSDF", "multi", "handoff_pose_policy")),
        "new_map_handoff_policy": nested_get(cfg, ("PIPE", "TSDF", "multi", "new_map_handoff_policy")),
        "new_map_handoff_min_accepts": nested_get(cfg, ("PIPE", "TSDF", "multi", "new_map_handoff_min_accepts")),
        "new_map_handoff_emergency_min_accepts": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "new_map_handoff_emergency_min_accepts")
        ),
        "primary_failure_handoff_min_raw_failed_frames": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "primary_failure_handoff_min_raw_failed_frames")
        ),
        "primary_failure_handoff_raw_gate": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "primary_failure_handoff_raw_gate")
        ),
        "primary_failure_handoff_min_raw_weak_frames": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "primary_failure_handoff_min_raw_weak_frames")
        ),
        "primary_failure_handoff_require_pose_continuity": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "primary_failure_handoff_require_pose_continuity")
        ),
        "tracking_boundary_policy": nested_get(cfg, ("PIPE", "TSDF", "multi", "tracking_boundary_policy")),
        "tracking_boundary_handoff_policy": nested_get(
            cfg, ("PIPE", "TSDF", "multi", "tracking_boundary_handoff_policy")
        ),
        "pose_graph_overlap_roundtrip_check": nested_get(
            cfg, ("PIPE", "gaussian_submaps", "pose_graph_overlap_roundtrip_check")
        ),
        "pose_graph_overlap_roundtrip_tolerance": nested_get(
            cfg, ("PIPE", "gaussian_submaps", "pose_graph_overlap_roundtrip_tolerance")
        ),
        "gaussian_submaps": nested_get(cfg, ("PIPE", "gaussian_submaps", "enable")),
        "gaussian_render_policy": nested_get(cfg, ("PIPE", "gaussian_submaps", "render_policy")),
        "gaussian_page_frozen": nested_get(cfg, ("PIPE", "gaussian_submaps", "page_frozen_gaussians")),
        "gaussian_resident_submaps": nested_get(cfg, ("PIPE", "gaussian_submaps", "resident_gaussian_submaps")),
        "new_gs_sample_ratio": nested_get(cfg, ("PIPE", "new_gs_sample_ratio")),
        "local_opt_iters": nested_get(cfg, ("PIPE", "local_opt_iters")),
        "local_opt_interval": nested_get(cfg, ("PIPE", "local_opt_interval")),
        "model_sh_degree": nested_get(cfg, ("MODEL", "sh_degree")),
        "pose_graph_apply_mode": nested_get(cfg, ("PIPE", "pose_graph", "apply_mode")),
        "pose_graph_online_updates": nested_get(cfg, ("PIPE", "pose_graph", "online_updates")),
        "pose_graph_run_on_submap_events": nested_get(cfg, ("PIPE", "pose_graph", "run_on_submap_events")),
        "pose_graph_candidate_interval_frames": nested_get(cfg, ("PIPE", "pose_graph", "candidate_interval_frames")),
        "pose_graph_max_updates_per_run": nested_get(cfg, ("PIPE", "pose_graph", "max_updates_per_run")),
        "pose_graph_terminal_online_scan": nested_get(cfg, ("PIPE", "pose_graph", "terminal_online_scan")),
        "loop_candidate_order": nested_get(cfg, ("PIPE", "pose_graph", "candidate_order")),
        "loop_verification": nested_get(cfg, ("PIPE", "pose_graph", "candidate_verification")),
        "orb_match_policy": nested_get(cfg, ("PIPE", "pose_graph", "orb_match_policy")),
        "verify_orb_aggregate_pairs": nested_get(cfg, ("PIPE", "pose_graph", "verify_orb_aggregate_pairs")),
        "verify_orb_aggregate_max_pairs": nested_get(cfg, ("PIPE", "pose_graph", "verify_orb_aggregate_max_pairs")),
        "verify_max_observation_pairs": nested_get(cfg, ("PIPE", "pose_graph", "verify_max_observation_pairs")),
        "verify_pair_selection": nested_get(cfg, ("PIPE", "pose_graph", "verify_pair_selection")),
        "graduation_optimize": nested_get(cfg, ("PIPE", "gaussian_submaps", "graduation_optimize")),
        "graduation_opt_iters": nested_get(cfg, ("PIPE", "gaussian_submaps", "graduation_opt_iters")),
        "graduation_keyframes": nested_get(cfg, ("PIPE", "gaussian_submaps", "graduation_keyframes")),
        "graduation_max_opt_passes": nested_get(cfg, ("PIPE", "gaussian_submaps", "graduation_max_opt_passes")),
        "graduation_keyframe_policy": nested_get(cfg, ("PIPE", "gaussian_submaps", "graduation_keyframe_policy")),
        "apply_support_policy": nested_get(cfg, ("PIPE", "pose_graph", "apply_support_policy")),
        "apply_min_loop_submap_span": nested_get(cfg, ("PIPE", "pose_graph", "apply_min_loop_submap_span")),
        "apply_min_loop_consistent_observations": nested_get(cfg, ("PIPE", "pose_graph", "apply_min_loop_consistent_observations")),
        "apply_min_loop_icp_inliers": nested_get(cfg, ("PIPE", "pose_graph", "apply_min_loop_icp_inliers")),
        "apply_max_loop_score": nested_get(cfg, ("PIPE", "pose_graph", "apply_max_loop_score")),
        "apply_min_translation_delta": nested_get(cfg, ("PIPE", "pose_graph", "apply_min_translation_delta")),
        "apply_max_sequential_translation_residual": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_max_sequential_translation_residual")
        ),
        "apply_max_sequential_rotation_residual_deg": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_max_sequential_rotation_residual_deg")
        ),
        "apply_adaptive_min_loop_submap_span": nested_get(cfg, ("PIPE", "pose_graph", "apply_adaptive_min_loop_submap_span")),
        "apply_adaptive_min_loop_consistent_observations": nested_get(cfg, ("PIPE", "pose_graph", "apply_adaptive_min_loop_consistent_observations")),
        "apply_adaptive_min_loop_icp_inliers": nested_get(cfg, ("PIPE", "pose_graph", "apply_adaptive_min_loop_icp_inliers")),
        "apply_adaptive_max_loop_score": nested_get(cfg, ("PIPE", "pose_graph", "apply_adaptive_max_loop_score")),
        "apply_high_confidence_min_loop_submap_span": nested_get(cfg, ("PIPE", "pose_graph", "apply_high_confidence_min_loop_submap_span")),
        "apply_high_confidence_min_loop_consistent_observations": nested_get(cfg, ("PIPE", "pose_graph", "apply_high_confidence_min_loop_consistent_observations")),
        "apply_high_confidence_min_loop_icp_inliers": nested_get(cfg, ("PIPE", "pose_graph", "apply_high_confidence_min_loop_icp_inliers")),
        "apply_high_confidence_max_loop_score": nested_get(cfg, ("PIPE", "pose_graph", "apply_high_confidence_max_loop_score")),
        "apply_high_confidence_max_icp_rmse": nested_get(cfg, ("PIPE", "pose_graph", "apply_high_confidence_max_icp_rmse")),
        "apply_observation_gate": nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_gate")),
        "apply_observation_gate_samples": [
            nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_gate_min_samples")),
            nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_gate_max_samples")),
        ],
        "apply_observation_defer_nonresident": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_defer_nonresident")
        ),
        "apply_observation_defer_retry_frames": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_defer_retry_frames")
        ),
        "apply_observation_pin_deferred_maps": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_pin_deferred_maps")
        ),
        "apply_observation_defer_stops_scale_backoff": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_defer_stops_scale_backoff")
        ),
        "apply_observation_terminal_sync_page_in": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_terminal_sync_page_in")
        ),
        "apply_observation_skip_resident_trim": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_skip_resident_trim")
        ),
        "apply_observation_prefetch_loop_endpoints": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_prefetch_loop_endpoints")
        ),
        "apply_observation_prefetch_max_maps": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_prefetch_max_maps")
        ),
        "apply_observation_prefetch_pin_ttl_frames": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_observation_prefetch_pin_ttl_frames")
        ),
        "apply_max_transform_updates_per_frame": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_max_transform_updates_per_frame")
        ),
        "apply_loop_selection_policy": nested_get(cfg, ("PIPE", "pose_graph", "apply_loop_selection_policy")),
        "apply_pcm_min_clique_size": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_min_clique_size")),
        "apply_pcm_event_cluster_radius": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_event_cluster_radius")),
        "apply_pcm_min_distinct_events": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_min_distinct_events")),
        "apply_pcm_min_long_span": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_min_long_span")),
        "apply_pcm_prefer_long_span": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_prefer_long_span")),
        "apply_pcm_scale_threshold_by_cycle_length": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_scale_threshold_by_cycle_length")),
        "apply_pcm_translation_threshold": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_translation_threshold")),
        "apply_pcm_rotation_threshold_deg": nested_get(cfg, ("PIPE", "pose_graph", "apply_pcm_rotation_threshold_deg")),
        "apply_observation_gate_sample_policy": nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_gate_sample_policy")),
        "apply_observation_gate_tolerance_policy": nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_gate_tolerance_policy")),
        "apply_observation_depth_rmse_ratio_floor": nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_depth_rmse_ratio_floor")),
        "apply_observation_max_depth_rmse_absolute_increase": nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_max_depth_rmse_absolute_increase")),
        "apply_observation_max_valid_fraction_drop": nested_get(cfg, ("PIPE", "pose_graph", "apply_observation_max_valid_fraction_drop")),
        "verify_icp_drift_aware_delta": nested_get(cfg, ("PIPE", "pose_graph", "verify_icp_drift_aware_delta")),
        "verify_icp_drift_reference_span": nested_get(cfg, ("PIPE", "pose_graph", "verify_icp_drift_reference_span")),
        "verify_icp_drift_max_scale": nested_get(cfg, ("PIPE", "pose_graph", "verify_icp_drift_max_scale")),
        "verify_icp_drift_score_margin": nested_get(cfg, ("PIPE", "pose_graph", "verify_icp_drift_score_margin")),
        "loop_candidate_export_policy": nested_get(cfg, ("PIPE", "pose_graph", "loop_candidate_export_policy")),
        "loop_candidate_export_span_reserve_fraction": nested_get(cfg, ("PIPE", "pose_graph", "loop_candidate_export_span_reserve_fraction")),
        "loop_candidate_export_min_span": nested_get(cfg, ("PIPE", "pose_graph", "loop_candidate_export_min_span")),
        "loop_max_candidates": nested_get(cfg, ("PIPE", "pose_graph", "loop_max_candidates")),
        "information_policy": nested_get(cfg, ("PIPE", "pose_graph", "information_policy")),
        "loop_robust_kernel": nested_get(cfg, ("PIPE", "pose_graph", "loop_robust_kernel")),
        "loop_robust_scale": nested_get(cfg, ("PIPE", "pose_graph", "loop_robust_scale")),
        "apply_health_gate": nested_get(cfg, ("PIPE", "pose_graph", "apply_health_gate")),
        "apply_health_max_raw_failed_ratio": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_health_max_raw_failed_ratio")
        ),
        "apply_health_max_raw_non_good_ratio": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_health_max_raw_non_good_ratio")
        ),
        "apply_health_max_raw_failed_streak": nested_get(
            cfg, ("PIPE", "pose_graph", "apply_health_max_raw_failed_streak")
        ),
        "anchor_policy": nested_get(cfg, ("PIPE", "pose_graph", "anchor_policy")),
    }


def apply_loop_profile(cfg: dict, profile: str) -> dict:
    """Apply named loop profiles without changing default behavior."""
    cfg = copy.deepcopy(cfg)
    pose = cfg["PIPE"]["pose_graph"]
    if profile == "default":
        return cfg
    if profile == "conservative":
        pose["orb_match_policy"] = "mutual_ratio"
        pose["verify_orb_aggregate_pairs"] = False
        pose["verify_orb_aggregate_max_pairs"] = 8
        pose["verify_max_observation_pairs"] = 32
        pose["verify_pair_selection"] = "pose"
        pose["apply_support_policy"] = "strict"
        pose["apply_min_loop_constraints"] = 1
        pose["apply_min_loop_submap_span"] = 10
        pose["apply_min_loop_consistent_observations"] = 4
        pose["apply_min_loop_icp_inliers"] = 64
        pose["apply_max_loop_score"] = 0.55
        pose["apply_observation_gate"] = True
        pose["apply_loop_selection_policy"] = "greedy_sequential_gate"
        return cfg
    if profile == "aggregate":
        pose["orb_match_policy"] = "ratio"
        pose["verify_orb_aggregate_pairs"] = True
        pose["verify_orb_aggregate_max_pairs"] = 16
        pose["verify_max_observation_pairs"] = 32
        pose["verify_pair_selection"] = "appearance"
        pose["apply_support_policy"] = "adaptive_small_loop"
        pose["apply_min_loop_constraints"] = 1
        pose["apply_min_loop_submap_span"] = 10
        pose["apply_min_loop_consistent_observations"] = 4
        pose["apply_min_loop_icp_inliers"] = 64
        pose["apply_max_loop_score"] = 0.55
        pose["apply_adaptive_min_loop_constraints"] = 1
        pose["apply_adaptive_min_loop_submap_span"] = 2
        pose["apply_adaptive_min_loop_consistent_observations"] = 16
        pose["apply_adaptive_min_loop_icp_inliers"] = 64
        pose["apply_adaptive_max_loop_score"] = 0.45
        pose["apply_observation_gate"] = True
        pose["apply_loop_selection_policy"] = "greedy_sequential_gate"
        return cfg
    if profile == "aggregate_strict":
        pose["candidate_order"] = "coarse_score"
        pose["orb_match_policy"] = "ratio"
        pose["verify_orb_aggregate_pairs"] = True
        pose["verify_orb_aggregate_max_pairs"] = 16
        pose["verify_max_observation_pairs"] = 32
        pose["verify_pair_selection"] = "appearance"
        pose["apply_support_policy"] = "strict"
        pose["apply_min_loop_constraints"] = 1
        pose["apply_min_loop_submap_span"] = 10
        pose["apply_min_loop_consistent_observations"] = 4
        pose["apply_min_loop_icp_inliers"] = 64
        pose["apply_max_loop_score"] = 0.55
        pose["apply_repeat_min_new_nodes"] = 4
        pose["apply_observation_gate"] = True
        pose["apply_loop_selection_policy"] = "greedy_sequential_gate"
        return cfg
    if profile == "aggregate_pcm":
        cfg = apply_loop_profile(cfg, "aggregate_strict")
        pose = cfg["PIPE"]["pose_graph"]
        pose["apply_min_loop_constraints"] = 2
        pose["apply_loop_selection_policy"] = "pcm_consistency"
        pose["apply_pcm_min_clique_size"] = 2
        pose["apply_pcm_max_trial_cliques"] = 32
        pose["apply_pcm_event_cluster_radius"] = 1
        pose["apply_pcm_min_distinct_events"] = 2
        pose["apply_pcm_min_long_span"] = 10
        pose["apply_pcm_prefer_long_span"] = False
        pose["apply_pcm_scale_threshold_by_cycle_length"] = True
        pose["apply_pcm_translation_threshold"] = 0.02
        pose["apply_pcm_rotation_threshold_deg"] = 2.0
        return cfg
    if profile == "aggregate_pcm_drift":
        cfg = apply_loop_profile(cfg, "aggregate_pcm")
        pose = cfg["PIPE"]["pose_graph"]
        pose["verify_icp_drift_aware_delta"] = True
        pose["verify_icp_drift_reference_span"] = 10.0
        pose["verify_icp_drift_max_scale"] = 3.0
        pose["verify_icp_drift_score_margin"] = 0.10
        pose["verify_icp_max_score"] = 0.55
        pose["loop_candidate_export_policy"] = "span_reserve_score"
        pose["loop_candidate_export_span_reserve_fraction"] = 0.5
        pose["loop_candidate_export_min_span"] = 0
        pose["loop_max_candidates"] = 24
        pose["apply_loop_selection_max_candidates"] = 24
        pose["apply_pcm_translation_threshold"] = 0.03
        pose["solve_only_max_translation_delta"] = 0.70
        pose["solve_only_max_rotation_delta_deg"] = 60.0
        pose["apply_max_translation_delta"] = 0.70
        pose["apply_max_rotation_delta_deg"] = 60.0
        pose["apply_pcm_prefer_long_span"] = True
        return cfg
    if profile == "aggregate_pcm_drift_online":
        cfg = apply_loop_profile(cfg, "aggregate_pcm_drift")
        pose = cfg["PIPE"]["pose_graph"]
        pose["online_updates"] = True
        pose["run_on_submap_events"] = True
        pose["candidate_interval_frames"] = 480
        pose["online_min_submaps"] = 8
        pose["online_min_submap_growth"] = 8
        pose["max_updates_per_run"] = 0
        pose["max_scanned_pairs_per_check"] = 64
        pose["max_candidate_pairs_per_check"] = 8
        pose["span_reserve_pairs"] = 8
        pose["final_candidate_scan"] = True
        # Mutually exclusive with final_candidate_scan. Keep the exhaustive
        # final pass for the validated accuracy profile and describe it
        # truthfully instead of emitting an inert terminal-online flag.
        pose["terminal_online_scan"] = False
        pose["final_max_scanned_pairs"] = 128
        pose["apply_min_translation_delta"] = 0.05
        pose["apply_pcm_translation_threshold"] = 0.035
        pose["apply_max_transform_updates_per_frame"] = 4
        return cfg
    if profile == "aggregate_pcm_drift_online_single_high":
        cfg = apply_loop_profile(cfg, "aggregate_pcm_drift_online")
        pose = cfg["PIPE"]["pose_graph"]
        pose["apply_support_policy"] = "high_confidence_single_loop"
        pose["apply_pcm_min_clique_size"] = 1
        pose["apply_pcm_min_distinct_events"] = 1
        pose["apply_min_loop_constraints"] = 1
        pose["apply_min_loop_consistent_observations"] = 12
        pose["apply_min_loop_icp_inliers"] = 80
        pose["apply_adaptive_min_loop_constraints"] = 1
        pose["apply_adaptive_min_loop_consistent_observations"] = 12
        pose["apply_adaptive_min_loop_icp_inliers"] = 80
        pose["apply_high_confidence_min_loop_constraints"] = 1
        pose["apply_high_confidence_min_loop_submap_span"] = 10
        pose["apply_high_confidence_min_loop_consistent_observations"] = 3
        pose["apply_high_confidence_min_loop_icp_inliers"] = 64
        pose["apply_high_confidence_max_loop_score"] = 0.50
        pose["apply_high_confidence_max_icp_rmse"] = 0.025
        return cfg
    if profile == "aggregate_pcm_drift_online_longspan":
        cfg = apply_loop_profile(cfg, "aggregate_pcm_drift_online_single_high")
        pose = cfg["PIPE"]["pose_graph"]
        pose["verify_icp_drift_reference_span"] = 1.0
        pose["verify_icp_drift_max_scale"] = 5.0
        pose["apply_max_sequential_translation_residual"] = 0.01
        pose["apply_max_sequential_rotation_residual_deg"] = 2.5
        return cfg
    if profile == "recommended_longspan":
        return apply_loop_profile(cfg, "aggregate_pcm_drift_online_longspan")
    if profile == "recommended_longspan_robust":
        cfg = apply_loop_profile(cfg, "recommended_longspan")
        pose = cfg["PIPE"]["pose_graph"]
        # Repeated online loop updates can over-correct rigid submap
        # boundaries at full scale. A bounded three-quarter update preserved
        # the global ATE gain while materially reducing local RPE regression.
        pose["apply_correction_scale"] = 0.75
        pose["information_policy"] = "confidence_diagonal"
        pose["loop_robust_kernel"] = "cauchy"
        pose["loop_robust_scale"] = 1.0
        pose["apply_health_gate"] = True
        pose["apply_health_max_raw_failed_ratio"] = 0.02
        pose["apply_health_max_raw_non_good_ratio"] = 0.10
        pose["apply_health_max_raw_failed_streak"] = 5
        pose["apply_health_min_frames"] = 30
        return cfg
    if profile == "recommended_longspan_robust_streaming":
        cfg = apply_loop_profile(cfg, "recommended_longspan_robust")
        pose = cfg["PIPE"]["pose_graph"]
        pose["run_on_submap_events"] = False
        pose["candidate_interval_frames"] = 60
        pose["max_scanned_pairs_per_check"] = 8
        pose["max_candidate_pairs_per_check"] = 1
        pose["span_reserve_pairs"] = 1
        pose["orb_feature_cache"] = True
        pose["orb_feature_cache_limit"] = 512
        pose["final_candidate_scan"] = False
        pose["terminal_online_scan"] = False
        pose["final_rank_all_candidates"] = False
        return cfg
    if profile == "recommended_longspan_lowlatency":
        cfg = apply_loop_profile(cfg, "recommended_longspan")
        pose = cfg["PIPE"]["pose_graph"]
        pose["max_candidate_pairs_per_check"] = 2
        pose["final_max_scanned_pairs"] = 128
        pose["apply_high_confidence_max_loop_score"] = 0.65
        pose["apply_high_confidence_max_icp_rmse"] = 0.03
        return cfg
    if profile == "aggregate_pcm_drift_online_guarded_recovery":
        cfg = apply_loop_profile(cfg, "aggregate_pcm_drift_online_longspan")
        pipe = cfg["PIPE"]
        pose = pipe["pose_graph"]
        multi = pipe["TSDF"]["multi"]
        pose["apply_high_confidence_max_loop_score"] = 0.55
        multi["sequential_transform_max_translation"] = 0.05
        multi["emergency_overlap_frames"] = 10
        multi["new_map_handoff_min_accepts"] = 2
        multi["primary_failure_handoff_min_raw_failed_frames"] = 4
        gaussian = pipe.get("gaussian_submaps", {})
        gaussian["page_frozen_gaussians"] = True
        gaussian["resident_gaussian_submaps"] = 2
        return cfg
    raise ValueError(f"Unknown loop profile {profile}")


def load_metadata(tum_root: Path, sequence: str) -> dict:
    metadata_path = tum_root / sequence / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}; run scripts/tum_preprocess.py first")
    metadata = json.loads(metadata_path.read_text())
    validate_fair_tum_metadata(metadata_path, metadata)
    return metadata


def validate_fair_tum_metadata(metadata_path: Path, metadata: dict) -> None:
    sequence = metadata.get("sequence", metadata_path.parent.name)
    if metadata.get("dataset") != "tum_rgbd":
        raise ValueError(f"{sequence}: expected TUM RGB-D metadata, got {metadata.get('dataset')}")
    if metadata.get("image_extension") != ".png" or metadata.get("depth_extension") != ".png":
        raise ValueError(f"{sequence}: TUM paper benchmarking must use original PNG RGB/depth files")
    if float(metadata.get("depth_scale", 0.0)) != 5000.0:
        raise ValueError(f"{sequence}: TUM depth scale must be 5000")
    if metadata.get("image_shape") != [640, 480]:
        raise ValueError(f"{sequence}: TUM paper benchmarking must stay at full 640x480 resolution")
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
    failed = [name for name in required_flags if fairness.get(name) is not True]
    if failed:
        raise ValueError(f"{sequence}: unfair preprocessing flags in {metadata_path}: {failed}")
    source = metadata.get("source", {})
    if source.get("association_policy") != "one_to_one_closest_timestamp":
        raise ValueError(
            f"{sequence}: expected one_to_one_closest_timestamp association, "
            f"got {source.get('association_policy')}; rerun scripts/tum_preprocess.py"
        )
    association = metadata.get("association", {})
    max_delta = float(association.get("max_delta", 0.02))
    rgb_depth_max = association.get("rgb_depth_max_delta")
    if rgb_depth_max is None or float(rgb_depth_max) > max_delta + 1e-9:
        raise ValueError(f"{sequence}: RGB/depth association exceeds {max_delta}s")


def base_config(template_path: Path) -> dict:
    cfg = yaml.safe_load(template_path.read_text())
    cfg["work_mode"] = "train"
    reader = cfg["READER"]
    reader["image_path"] = "camera"
    reader["pose_path"] = "camera"
    reader["depth_path"] = "depth"
    reader["pcd_name"] = "no_init_points.ply"
    reader["image_prefix"] = "frame"
    reader["image_extension"] = ".png"
    reader["depth_prefix"] = "depth"
    reader["depth_extension"] = ".png"
    reader["pose_prefix"] = "pose"
    reader["pose_extension"] = ".txt"
    reader["depth_scale"] = 5000.0
    reader["downscale_factor"] = 1
    reader["test_split_interval"] = -1
    pipe = cfg["PIPE"]
    pipe["random_seed"] = 42
    pipe["eval_split"] = "train"
    pipe["eval_after_train"] = False
    pipe["eval_sdf_only"] = False
    pipe["save_after_train"] = True
    pipe["bypass_libtorch_atexit"] = True
    pipe["TSDF"]["saved_mesh"] = ""
    pipe["eval_stride"] = 1
    pipe["eval_max_frames"] = 0
    pipe["eval_start_offset"] = 0
    pipe["TSDF"]["use_gt_pose"] = False
    pipe["TSDF"]["engine"] = "multi"
    pipe["TSDF"]["tracking_failure_mode"] = "legacy_ignore"
    pipe["TSDF"]["reject_nonfinite_tracking_pose"] = False
    pipe["TSDF"]["require_raw_tracking_success"] = False
    pipe["TSDF"]["raw_tracking_failure_grace_frames"] = 0
    pipe["TSDF"]["basic_debug_tracking_start_frame"] = -1
    pipe["TSDF"]["basic_debug_tracking_end_frame"] = -1
    reader["preload_images"] = False
    pipe["TSDF"]["input_mode"] = "on_demand"
    pipe["TSDF"]["multi"]["force_single_submap"] = False
    pipe["TSDF"]["multi"]["console_diagnostics"] = False
    pipe["TSDF"]["multi"]["loop_closure"] = False
    pipe["TSDF"]["multi"]["global_adjustment"] = False
    pipe["TSDF"]["multi"]["sequential_transform_policy"] = "validated_overlap"
    pipe["TSDF"]["multi"]["sequential_transform_max_translation"] = 0.03
    pipe["TSDF"]["multi"]["sequential_transform_max_rotation_deg"] = 3.0
    pipe["TSDF"]["multi"]["invalid_transform_restart_attempts"] = 2
    pipe["TSDF"]["multi"]["emergency_overlap_frames"] = 0
    pipe["TSDF"]["multi"]["handoff_pose_policy"] = "candidate"
    pipe["TSDF"]["multi"]["new_map_handoff_policy"] = "validated_visibility"
    pipe["TSDF"]["multi"]["new_map_handoff_min_visible_ratio"] = 0.0
    pipe["TSDF"]["multi"]["new_map_handoff_min_visible_margin"] = 0.0
    pipe["TSDF"]["multi"]["new_map_handoff_min_accepts"] = 2
    pipe["TSDF"]["multi"]["new_map_handoff_emergency_min_accepts"] = -1
    pipe["TSDF"]["multi"]["candidate_recovery_policy"] = "keep_if_visible"
    pipe["TSDF"]["multi"]["candidate_recovery_min_visible_ratio"] = 0.15
    pipe["TSDF"]["multi"]["candidate_recovery_min_blocks"] = 512
    pipe["TSDF"]["multi"]["candidate_recovery_max_attempts"] = 2
    pipe["TSDF"]["multi"]["primary_failure_handoff_policy"] = "promote_visible_candidate"
    pipe["TSDF"]["multi"]["primary_failure_handoff_min_raw_failed_frames"] = 8
    pipe["TSDF"]["multi"]["primary_failure_handoff_raw_gate"] = "failed_only"
    pipe["TSDF"]["multi"]["primary_failure_handoff_min_raw_weak_frames"] = 0
    pipe["TSDF"]["multi"]["primary_failure_handoff_min_candidate_visible_ratio"] = 0.15
    pipe["TSDF"]["multi"]["primary_failure_handoff_min_candidate_blocks"] = 512
    pipe["TSDF"]["multi"]["primary_failure_handoff_min_candidate_good_frames"] = 4
    pipe["TSDF"]["multi"]["primary_failure_handoff_min_overlap_frames"] = 0
    pipe["TSDF"]["multi"]["primary_failure_handoff_require_pose_continuity"] = False
    pipe["TSDF"]["multi"]["tracking_boundary_policy"] = "raw_weak_streak"
    pipe["TSDF"]["multi"]["raw_weak_boundary_frames"] = 3
    pipe["TSDF"]["multi"]["raw_failed_boundary_frames"] = 1
    pipe["TSDF"]["multi"]["tracking_boundary_min_primary_frames"] = 20
    pipe["TSDF"]["multi"]["tracking_boundary_cooldown_frames"] = 12
    pipe["TSDF"]["multi"]["tracking_boundary_force_handoff"] = True
    pipe["TSDF"]["multi"]["tracking_boundary_handoff_policy"] = "weak_or_failed"
    pipe["TSDF"]["multi"]["boundary_policy"] = "visibility"
    pipe["TSDF"]["multi"]["overlap_frames"] = 12
    pipe["TSDF"]["multi"]["trial_frames"] = 24
    pipe["TSDF"]["multi"]["async_page_out"] = True
    pipe["TSDF"]["multi"]["paging_storage"] = "cpu"
    pipe["TSDF"]["multi"]["cpu_page_cache_limit_mb"] = 8192
    pipe["TSDF"]["multi"]["max_pending_page_snapshots"] = 1
    pipe["TSDF"]["multi"]["max_pending_page_snapshot_mb"] = 0
    pipe["TSDF"]["multi"]["async_page_in"] = False
    pipe["TSDF"]["multi"]["max_pending_page_ins"] = 1
    pipe["TSDF"]["multi"]["max_pending_page_in_mb"] = 0
    pipe["gaussian_submaps"]["pose_graph_overlap_roundtrip_check"] = False
    pipe["gaussian_submaps"]["pose_graph_overlap_roundtrip_tolerance"] = 1e-4
    pipe["TSDF"]["multi"]["render_prefetch"] = False
    pipe["TSDF"]["multi"]["render_compositor"] = "handoff_aware"
    pipe["TSDF"]["multi"]["max_frames"] = 750
    pipe["TSDF"]["multi"]["new_map_block_occupancy"] = 0.55
    pipe["TSDF"]["multi"]["max_block_occupancy"] = 0.75
    pipe["TSDF"]["multi"]["fusion_reserve_blocks"] = 8192
    pipe["TSDF"]["multi"]["occupancy_growth_lookahead_frames"] = 30
    pipe["TSDF"]["multi"]["occupancy_growth_min_primary_frames"] = 20
    pipe["TSDF"]["multi"]["occupancy_growth_sample_frames"] = 5
    pipe["TSDF"]["multi"]["skip_primary_fusion_on_capacity_handoff"] = True
    pipe["gaussian_submaps"]["enable"] = True
    pipe["gaussian_submaps"]["ownership"] = "tsdf"
    pipe["gaussian_submaps"]["coordinate_frame"] = "local"
    pipe["gaussian_submaps"]["max_new_gaussians_per_frame"] = 1000
    pipe["gaussian_submaps"]["gaussian_insert_min_age_frames"] = 0
    pipe["gaussian_submaps"]["gaussian_insert_min_sdf_blocks"] = 0
    pipe["gaussian_submaps"]["render_policy"] = "handoff_aware_fill"
    pipe["gaussian_submaps"]["render_max_visible_gaussians"] = 0
    pipe["gaussian_submaps"]["render_past_frame_margin"] = 20
    pipe["gaussian_submaps"]["render_future_frame_margin"] = 0
    pipe["gaussian_submaps"]["render_min_ready_frames"] = 16
    pipe["gaussian_submaps"]["render_owner_min_ready_frames"] = 0
    pipe["gaussian_submaps"]["render_neighbor_min_ready_frames"] = 5
    pipe["gaussian_submaps"]["render_owner_alpha_fill_thres"] = 0.25
    pipe["gaussian_submaps"]["render_neighbor_alpha_fill_thres"] = 0.05
    pipe["gaussian_submaps"]["render_blend_alpha_thres"] = 0.10
    pipe["gaussian_submaps"]["render_blend_depth_thres"] = 0.05
    pipe["gaussian_submaps"]["render_blend_max_weight"] = 0.20
    pipe["gaussian_submaps"]["render_blend_weight_scale"] = 0.20
    pipe["gaussian_submaps"]["render_min_neighbor_gaussians"] = 0
    pipe["gaussian_submaps"]["render_single_submap_passthrough"] = True
    pipe["gaussian_submaps"]["render_single_submap_passthrough_requires_single_logical_map"] = False
    pipe["gaussian_submaps"]["render_use_local_sdf_reference"] = True
    pipe["gaussian_submaps"]["render_include_handoff_submaps"] = False
    pipe["gaussian_submaps"]["retire_unaccepted_candidates"] = True
    pipe["gaussian_submaps"]["gaussian_insert_selection"] = "error_topk"
    pipe["gaussian_submaps"]["gaussian_insert_tile_size"] = 32
    pipe["gaussian_submaps"]["gaussian_insert_tile_max_pixels"] = 0
    pipe["gaussian_submaps"]["keyframe_selection_policy"] = "random"
    pipe["gaussian_submaps"]["candidate_reject_retire"] = True
    pipe["gaussian_submaps"]["candidate_reject_retire_min_rejects"] = 4
    pipe["gaussian_submaps"]["candidate_reject_retire_min_observed_frames"] = 16
    pipe["gaussian_submaps"]["candidate_reject_retire_min_gaussians"] = 1
    pipe["gaussian_submaps"]["candidate_reject_retire_max_inliers"] = 1
    pipe["gaussian_submaps"]["candidate_reject_retire_cooldown_frames"] = 30
    pipe["gaussian_submaps"]["candidate_gaussian_insert_min_frames"] = 0
    pipe["gaussian_submaps"]["candidate_immature_new_gs_sample_ratio"] = -1.0
    pipe["gaussian_submaps"]["candidate_immature_max_new_gaussians_per_frame"] = 0
    pipe["gaussian_submaps"]["candidate_extend_while_overlap"] = True
    pipe["gaussian_submaps"]["candidate_max_warm_frames"] = 80
    pipe["gaussian_submaps"]["candidate_handoff_max_opt_passes"] = 6
    pipe["gaussian_submaps"]["graduation_optimize"] = False
    pipe["gaussian_submaps"]["graduation_opt_iters"] = 0
    pipe["gaussian_submaps"]["graduation_keyframes"] = 8
    pipe["gaussian_submaps"]["graduation_max_opt_passes"] = 1
    pipe["gaussian_submaps"]["graduation_keyframe_policy"] = "loss_temporal_mix"
    pipe["gaussian_submaps"]["candidate_max_new_gaussians_per_frame"] = 1000
    pipe["pose_graph"]["candidate_detection"] = False
    pipe["pose_graph"]["optimize"] = False
    pipe["pose_graph"]["dry_run"] = False
    pipe["pose_graph"]["solve_only"] = False
    pipe["pose_graph"]["apply_mode"] = "disabled"
    pipe["pose_graph"]["apply_correction_scale"] = 1.0
    pipe["pose_graph"]["apply_scale_backoff"] = False
    pipe["pose_graph"]["apply_scale_backoff_min_scale"] = 0.25
    pipe["pose_graph"]["apply_scale_backoff_factor"] = 0.5
    pipe["pose_graph"]["apply_scale_backoff_max_trials"] = 3
    pipe["pose_graph"]["solve_only_max_translation_delta"] = 0.30
    pipe["pose_graph"]["solve_only_max_rotation_delta_deg"] = 30.0
    pipe["pose_graph"]["apply_max_translation_delta"] = 0.30
    pipe["pose_graph"]["apply_max_rotation_delta_deg"] = 30.0
    pipe["pose_graph"]["online_updates"] = False
    pipe["pose_graph"]["candidate_interval_frames"] = 300
    pipe["pose_graph"]["run_on_submap_events"] = False
    pipe["pose_graph"]["online_min_submaps"] = 4
    pipe["pose_graph"]["online_min_submap_growth"] = 2
    pipe["pose_graph"]["max_updates_per_run"] = 0
    pipe["pose_graph"]["persistent_loop_pool"] = True
    pipe["pose_graph"]["verified_loop_pool_max"] = 64
    pipe["pose_graph"]["skip_verified_loop_pairs"] = True
    pipe["pose_graph"]["max_scanned_pairs_per_check"] = 0
    pipe["pose_graph"]["final_candidate_scan"] = True
    pipe["pose_graph"]["terminal_online_scan"] = False
    pipe["pose_graph"]["final_max_scanned_pairs"] = 16
    pipe["pose_graph"]["final_rank_all_candidates"] = True
    pipe["pose_graph"]["incremental_candidate_scan"] = True
    pipe["pose_graph"]["candidate_verification"] = "orb_icp"
    pipe["pose_graph"]["candidate_order"] = "orb_bow"
    pipe["pose_graph"]["span_reserve_pairs"] = 4
    pipe["pose_graph"]["appearance_max_observation_pairs"] = 4
    pipe["pose_graph"]["appearance_min_orb_matches"] = 0
    pipe["pose_graph"]["include_keyframes_with_loop_observations"] = True
    pipe["pose_graph"]["keyframe_observation_max_per_submap"] = 12
    pipe["pose_graph"]["orb_feature_cache"] = True
    pipe["pose_graph"]["orb_feature_cache_limit"] = 256
    pipe["pose_graph"]["warm_loop_observations"] = True
    pipe["pose_graph"]["warm_max_per_frame"] = 2
    pipe["pose_graph"]["warm_skip_update_frames"] = True
    pipe["pose_graph"]["warm_async"] = True
    pipe["pose_graph"]["warm_queue_limit"] = 32
    pipe["pose_graph"]["orb_max_features"] = 1000
    pipe["pose_graph"]["orb_ratio"] = 0.75
    pipe["pose_graph"]["orb_match_policy"] = "mutual_ratio"
    pipe["pose_graph"]["orb_min_quadrants"] = 3
    pipe["pose_graph"]["orb_depth_search_radius"] = 0
    pipe["pose_graph"]["orb_pnp_fallback"] = False
    pipe["pose_graph"]["orb_pnp_iterations"] = 100
    pipe["pose_graph"]["orb_pnp_reprojection_error"] = 4.0
    pipe["pose_graph"]["orb_pnp_confidence"] = 0.99
    pipe["pose_graph"]["bow_word_bits"] = 12
    pipe["pose_graph"]["bow_max_observations_per_submap"] = 8
    pipe["pose_graph"]["bow_min_score"] = 0.0
    pipe["pose_graph"]["bow_defer_appearance"] = True
    pipe["pose_graph"]["bow_histogram_cache"] = True
    pipe["pose_graph"]["verify_orb_aggregate_pairs"] = False
    pipe["pose_graph"]["verify_orb_aggregate_max_pairs"] = 8
    pipe["pose_graph"]["apply_min_loop_constraints"] = 1
    pipe["pose_graph"]["apply_min_loop_submap_span"] = 10
    pipe["pose_graph"]["apply_min_loop_consistent_observations"] = 4
    pipe["pose_graph"]["apply_min_loop_icp_inliers"] = 64
    pipe["pose_graph"]["apply_max_loop_score"] = 0.55
    pipe["pose_graph"]["apply_support_policy"] = "strict"
    pipe["pose_graph"]["apply_adaptive_min_loop_constraints"] = 1
    pipe["pose_graph"]["apply_adaptive_min_loop_submap_span"] = 2
    pipe["pose_graph"]["apply_adaptive_min_loop_consistent_observations"] = 3
    pipe["pose_graph"]["apply_adaptive_min_loop_icp_inliers"] = 48
    pipe["pose_graph"]["apply_adaptive_max_loop_score"] = 0.4
    pipe["pose_graph"]["apply_high_confidence_min_loop_constraints"] = 1
    pipe["pose_graph"]["apply_high_confidence_min_loop_submap_span"] = 10
    pipe["pose_graph"]["apply_high_confidence_min_loop_consistent_observations"] = 3
    pipe["pose_graph"]["apply_high_confidence_min_loop_icp_inliers"] = 64
    pipe["pose_graph"]["apply_high_confidence_max_loop_score"] = 0.50
    pipe["pose_graph"]["apply_high_confidence_max_icp_rmse"] = 0.025
    pipe["pose_graph"]["apply_max_sequential_translation_residual"] = 0.008
    pipe["pose_graph"]["apply_max_sequential_rotation_residual_deg"] = 1.2
    pipe["pose_graph"]["apply_max_frame_relative_translation_change"] = 0.05
    pipe["pose_graph"]["apply_max_frame_relative_rotation_change_deg"] = 5.0
    pipe["pose_graph"]["apply_loop_selection_policy"] = "greedy_sequential_gate"
    pipe["pose_graph"]["apply_loop_selection_max_candidates"] = 16
    pipe["pose_graph"]["apply_pcm_min_clique_size"] = 2
    pipe["pose_graph"]["apply_pcm_max_trial_cliques"] = 32
    pipe["pose_graph"]["apply_pcm_event_cluster_radius"] = 1
    pipe["pose_graph"]["apply_pcm_min_distinct_events"] = 2
    pipe["pose_graph"]["apply_pcm_min_long_span"] = 0
    pipe["pose_graph"]["apply_pcm_prefer_long_span"] = False
    pipe["pose_graph"]["apply_pcm_scale_threshold_by_cycle_length"] = True
    pipe["pose_graph"]["apply_pcm_translation_threshold"] = 0.02
    pipe["pose_graph"]["apply_pcm_rotation_threshold_deg"] = 2.0
    pipe["pose_graph"]["apply_skip_repeated_loop_pairs"] = True
    pipe["pose_graph"]["apply_repeat_min_new_nodes"] = 1
    pipe["pose_graph"]["apply_observation_gate"] = True
    pipe["pose_graph"]["apply_observation_gate_in_selection"] = False
    pipe["pose_graph"]["apply_observation_gate_max_samples"] = 3
    pipe["pose_graph"]["apply_observation_gate_min_samples"] = 2
    pipe["pose_graph"]["apply_observation_gate_sample_policy"] = "loop_aware"
    pipe["pose_graph"]["apply_observation_max_render_maps_per_sample"] = 2
    pipe["pose_graph"]["apply_observation_max_pinned_maps"] = 2
    pipe["pose_graph"]["apply_observation_defer_nonresident"] = True
    pipe["pose_graph"]["apply_observation_defer_retry_frames"] = 120
    pipe["pose_graph"]["apply_observation_pin_deferred_maps"] = True
    pipe["pose_graph"]["apply_observation_defer_stops_scale_backoff"] = True
    pipe["pose_graph"]["apply_observation_terminal_sync_page_in"] = True
    pipe["pose_graph"]["apply_observation_skip_resident_trim"] = True
    pipe["pose_graph"]["apply_observation_prefetch_loop_endpoints"] = True
    pipe["pose_graph"]["apply_observation_prefetch_max_maps"] = 4
    pipe["pose_graph"]["apply_observation_prefetch_pin_ttl_frames"] = 120
    pipe["pose_graph"]["apply_observation_gate_tolerance_policy"] = "catastrophe_depth"
    pipe["pose_graph"]["apply_observation_max_color_mse_ratio"] = 1.02
    pipe["pose_graph"]["apply_observation_max_color_mse_increase"] = 0.0005
    pipe["pose_graph"]["apply_observation_max_depth_rmse_ratio"] = 1.25
    pipe["pose_graph"]["apply_observation_max_depth_rmse_increase"] = 0.01
    pipe["pose_graph"]["apply_observation_depth_rmse_ratio_floor"] = 0.01
    pipe["pose_graph"]["apply_observation_max_depth_rmse_absolute_increase"] = 0.05
    pipe["pose_graph"]["apply_observation_max_valid_fraction_drop"] = 0.30
    pipe["pose_graph"]["apply_max_transform_updates_per_frame"] = 0
    pipe["pose_graph"]["apply_health_gate"] = False
    pipe["pose_graph"]["apply_health_max_raw_failed_ratio"] = 0.02
    pipe["pose_graph"]["apply_health_max_raw_non_good_ratio"] = 0.10
    pipe["pose_graph"]["apply_health_max_raw_failed_streak"] = 5
    pipe["pose_graph"]["apply_health_min_frames"] = 30
    pipe["pose_graph"]["sequential_constraint_weight"] = 4.0
    pipe["pose_graph"]["loop_constraint_weight"] = 1.0
    pipe["pose_graph"]["information_policy"] = "scalar"
    pipe["pose_graph"]["loop_robust_kernel"] = "none"
    pipe["pose_graph"]["loop_robust_scale"] = 1.0
    pipe["pose_graph"]["loop_information_min_confidence"] = 0.25
    pipe["pose_graph"]["loop_information_max_confidence"] = 8.0
    pipe["pose_graph"]["anchor_policy"] = "active"
    pipe["pose_graph"]["verify_icp_max_score"] = 0.55
    pipe["pose_graph"]["verify_icp_drift_aware_delta"] = False
    pipe["pose_graph"]["verify_icp_drift_reference_span"] = 10.0
    pipe["pose_graph"]["verify_icp_drift_max_scale"] = 3.0
    pipe["pose_graph"]["verify_icp_drift_score_margin"] = 0.0
    pipe["pose_graph"]["verify_icp_weighted_kabsch"] = False
    pipe["pose_graph"]["verify_icp_depth_adaptive_ransac"] = False
    pipe["pose_graph"]["verify_icp_depth_noise_scale"] = 0.002
    pipe["pose_graph"]["verify_icp_depth_adaptive_threshold_scale"] = 3.0
    pipe["pose_graph"]["verify_max_observation_pairs"] = 32
    pipe["pose_graph"]["max_candidate_pairs_per_check"] = 4
    pipe["pose_graph"]["candidate_accept_target_per_check"] = 1
    pipe["pose_graph"]["loop_max_candidates"] = 16
    pipe["pose_graph"]["loop_candidate_export_policy"] = "score"
    pipe["pose_graph"]["loop_candidate_export_span_reserve_fraction"] = 0.0
    pipe["pose_graph"]["loop_candidate_export_min_span"] = 0
    return cfg


def configure_sequence(cfg: dict, tum_root: Path, output_root: Path, short_name: str, max_frames: int | None) -> dict:
    sequence = SEQUENCES[short_name]
    metadata = load_metadata(tum_root, sequence)
    cfg = copy.deepcopy(cfg)
    cfg["workspace_dir"] = str(output_root / short_name)
    cfg["READER"]["input_dir"] = str(tum_root / sequence)
    cfg["READER"]["intrinsics"] = metadata["intrinsics"]
    cfg["READER"]["image_shape"] = metadata["image_shape"]
    cfg["READER"]["start_frame"] = 0
    end_frame = int(metadata["frames"]) - 1
    if max_frames is not None:
        end_frame = min(end_frame, max_frames - 1)
    cfg["READER"]["end_frame"] = end_frame
    return cfg


def configure_variant(cfg: dict, variant: str) -> dict:
    cfg = copy.deepcopy(cfg)
    pipe = cfg["PIPE"]

    def enable_online_pose_graph_updates(max_updates: int = 4) -> None:
        pipe["pose_graph"]["online_updates"] = True
        pipe["pose_graph"]["max_updates_per_run"] = max_updates

    def enable_orb_loop_candidates() -> None:
        pipe["pose_graph"]["candidate_detection"] = True
        pipe["pose_graph"]["dry_run"] = True
        pipe["pose_graph"]["dry_run_include_loop_candidates"] = True
        pipe["pose_graph"]["solve_only"] = True
        pipe["pose_graph"]["solve_only_include_loop_candidates"] = True
        pipe["pose_graph"]["candidate_verification"] = "orb_icp"

    if variant == "basic_sdf":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "basic"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
    elif variant == "basic_gaussian":
        pipe["TSDF"]["engine"] = "basic"
        pipe["TSDF"]["multi"]["force_single_submap"] = False
        pipe["gaussian_submaps"]["enable"] = False
    elif variant == "multi_sdf":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "multi"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
    elif variant == "multi_sdf_dry_solve":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "multi"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
        pipe["pose_graph"]["dry_run"] = True
        pipe["pose_graph"]["solve_only"] = True
    elif variant == "multi_sdf_orb_dry_solve":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "multi"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
        enable_orb_loop_candidates()
    elif variant == "multi_sdf_orb_apply":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "multi"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
        enable_orb_loop_candidates()
        pipe["pose_graph"]["apply_mode"] = "loop_candidates"
    elif variant == "multi_sdf_sequential_noop":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "multi"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
        pipe["pose_graph"]["dry_run"] = True
        pipe["pose_graph"]["solve_only"] = True
        pipe["pose_graph"]["apply_mode"] = "sequential_noop"
    elif variant == "multi_sdf_synthetic":
        cfg["work_mode"] = "recon"
        pipe["TSDF"]["engine"] = "multi"
        pipe["gaussian_submaps"]["enable"] = False
        pipe["save_after_train"] = False
        pipe["pose_graph"]["apply_mode"] = "synthetic"
    elif variant == "multi_hybrid_no_loop":
        pass
    elif variant == "multi_hybrid_dry_solve":
        pipe["pose_graph"]["candidate_detection"] = True
        pipe["pose_graph"]["dry_run"] = True
        pipe["pose_graph"]["dry_run_include_loop_candidates"] = True
        pipe["pose_graph"]["solve_only"] = True
        pipe["pose_graph"]["solve_only_include_loop_candidates"] = True
    elif variant == "multi_hybrid_orb_dry_solve":
        enable_orb_loop_candidates()
    elif variant == "multi_hybrid_apply":
        pipe["pose_graph"]["candidate_detection"] = True
        pipe["pose_graph"]["dry_run"] = True
        pipe["pose_graph"]["dry_run_include_loop_candidates"] = True
        pipe["pose_graph"]["solve_only"] = True
        pipe["pose_graph"]["solve_only_include_loop_candidates"] = True
        pipe["pose_graph"]["apply_mode"] = "loop_candidates"
        pipe["pose_graph"]["candidate_verification"] = "orb_icp"
    elif variant == "multi_hybrid_orb_apply":
        enable_orb_loop_candidates()
        pipe["pose_graph"]["apply_mode"] = "loop_candidates"
    else:
        raise ValueError(f"Unknown variant {variant}")
    return cfg


def disable_inert_online_pose_graph(cfg: dict) -> None:
    pose = cfg.get("PIPE", {}).get("pose_graph", {})
    if not isinstance(pose, dict) or not bool(pose.get("online_updates", False)):
        return
    apply_mode = str(pose.get("apply_mode", "disabled")).strip().lower()
    has_live_operation = (
        bool(pose.get("candidate_detection", False))
        or bool(pose.get("dry_run", False))
        or bool(pose.get("solve_only", False))
        or apply_mode not in {"", "disabled", "none", "false", "0"}
    )
    if has_live_operation:
        return
    pose["online_updates"] = False
    pose["run_on_submap_events"] = False
    pose["terminal_online_scan"] = False
    pose["final_candidate_scan"] = False


def variant_uses_gaussians(variant: str) -> bool:
    return variant == "basic_gaussian" or variant.startswith("multi_hybrid")


def suffix_value(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def resident_gpu_map_estimate(value) -> int:
    if value is None:
        return 2
    text = str(value).strip().lower()
    if text == "auto":
        # The C++ engine resolves this from CUDA free memory. Use the lower end
        # of its clamp so CPU-page estimates stay conservative.
        return 2
    try:
        return max(1, int(text))
    except ValueError:
        return 2


def estimate_multi_submap_usage(cfg: dict, margin: int) -> dict:
    reader_cfg = cfg["READER"]
    multi_cfg = cfg["PIPE"]["TSDF"]["multi"]
    start_frame = int(reader_cfg.get("start_frame", 0))
    end_frame = int(reader_cfg.get("end_frame", start_frame - 1))
    frame_count = max(0, end_frame - start_frame + 1)
    max_frames_per_map = max(1, int(multi_cfg.get("max_frames", frame_count or 1)))
    occupancy_lookahead = max(0, int(multi_cfg.get("occupancy_growth_lookahead_frames", 0)))
    effective_frames_per_map = max_frames_per_map
    if occupancy_lookahead > 0:
        effective_frames_per_map = max(1, min(effective_frames_per_map, occupancy_lookahead))
    if config_bool(multi_cfg.get("force_single_submap"), False):
        estimated_submaps = 1
    else:
        estimated_submaps = (frame_count + effective_frames_per_map - 1) // effective_frames_per_map
        estimated_submaps += max(0, int(margin))
    resident_maps = resident_gpu_map_estimate(multi_cfg.get("resident_gpu_maps", "auto"))
    passive_maps = max(0, estimated_submaps - resident_maps)
    hash_profile = str(multi_cfg.get("hash_profile", "compact_64k"))
    scene_bytes = HASH_PROFILE_ESTIMATED_SCENE_BYTES.get(hash_profile, HASH_PROFILE_ESTIMATED_SCENE_BYTES["legacy_large"])
    paging_storage = str(multi_cfg.get("paging_storage", "disk")).lower()
    cpu_cache_limit_mb = max(0, int(multi_cfg.get("cpu_page_cache_limit_mb", 0)))
    cpu_cache_limit_bytes = cpu_cache_limit_mb * 1024**2
    unbounded_cpu_cache_bytes = passive_maps * scene_bytes
    if paging_storage == "cpu" and cpu_cache_limit_bytes > 0:
        cpu_cache_allowance_maps = max(0, cpu_cache_limit_bytes // scene_bytes)
        cpu_cached_maps = min(passive_maps, cpu_cache_allowance_maps)
        spilled_maps = max(0, passive_maps - cpu_cached_maps)
        # One just-completed page-out may temporarily exceed the limit before
        # the spill pass runs, so keep one scene of headroom in the estimate.
        estimated_cpu_cache_bytes = min(unbounded_cpu_cache_bytes, cpu_cache_limit_bytes + scene_bytes)
        estimated_disk_page_bytes = spilled_maps * scene_bytes
    elif paging_storage == "cpu":
        cpu_cached_maps = passive_maps
        spilled_maps = 0
        estimated_cpu_cache_bytes = unbounded_cpu_cache_bytes
        estimated_disk_page_bytes = 0
    else:
        cpu_cached_maps = 0
        spilled_maps = passive_maps
        estimated_cpu_cache_bytes = 0
        estimated_disk_page_bytes = passive_maps * scene_bytes
    return {
        "frame_count": frame_count,
        "max_frames_per_map": max_frames_per_map,
        "occupancy_lookahead": occupancy_lookahead,
        "effective_frames_per_map": effective_frames_per_map,
        "estimated_submaps": estimated_submaps,
        "resident_maps": resident_maps,
        "passive_maps": passive_maps,
        "hash_profile": hash_profile,
        "scene_bytes": scene_bytes,
        "paging_storage": paging_storage,
        "cpu_page_cache_limit_mb": cpu_cache_limit_mb,
        "cpu_cached_maps": cpu_cached_maps,
        "spilled_maps": spilled_maps,
        "estimated_cpu_cache_gb": estimated_cpu_cache_bytes / 1024**3,
        # Disk sparse pages are usually smaller, but full scene bytes are the
        # safe upper bound while long-run paging paths are still being hardened.
        "estimated_disk_page_gb": estimated_disk_page_bytes / 1024**3,
    }


def enforce_multi_run_safety(cfg: dict, seq: str, variant: str, args) -> None:
    tsdf_cfg = cfg.get("PIPE", {}).get("TSDF", {})
    if str(tsdf_cfg.get("engine", "")).lower() != "multi":
        return
    multi_cfg = tsdf_cfg.get("multi", {})
    usage = estimate_multi_submap_usage(cfg, args.estimated_submap_margin)
    print(
        "[SAFETY] "
        f"{seq}/{variant}: estimated_submaps={usage['estimated_submaps']} "
        f"frames={usage['frame_count']} effective_frames_per_map={usage['effective_frames_per_map']} "
        f"resident_gpu_maps~{usage['resident_maps']} hash={usage['hash_profile']} "
        f"paging_storage={usage['paging_storage']} passive_maps~{usage['passive_maps']} "
        f"cpu_cached_maps~{usage['cpu_cached_maps']} spilled_maps~{usage['spilled_maps']} "
        f"cpu_cache_limit_mb={usage['cpu_page_cache_limit_mb']}"
    )

    if args.max_estimated_submaps > 0 and usage["estimated_submaps"] > args.max_estimated_submaps:
        raise RuntimeError(
            f"Abort: estimated logical submaps={usage['estimated_submaps']} for {seq}/{variant} "
            f"(frames={usage['frame_count']}, TSDF.multi.max_frames={usage['max_frames_per_map']}, "
            f"effective_frames_per_map={usage['effective_frames_per_map']}, "
            f"margin={max(0, int(args.estimated_submap_margin))}) exceeds "
            f"--max-estimated-submaps={args.max_estimated_submaps}. Increase TSDF.max_frames, "
            "raise the guard intentionally, or run a shorter staged test."
        )

    paging_enabled = config_bool(multi_cfg.get("paging"), False)
    paging_storage = str(multi_cfg.get("paging_storage", "disk")).lower()
    if not paging_enabled:
        return

    if paging_storage == "cpu" and args.max_estimated_cpu_page_cache_gb > 0:
        estimated_gb = usage["estimated_cpu_cache_gb"]
        if estimated_gb > args.max_estimated_cpu_page_cache_gb:
            raise RuntimeError(
                f"Abort: estimated passive CPU TSDF page cache is {estimated_gb:.1f} GB for {seq}/{variant} "
                f"({usage['passive_maps']} passive maps x {usage['scene_bytes'] / 1024**2:.1f} MB, "
                f"hash={usage['hash_profile']}). This PC has limited RAM, and previous long fr3_office "
                "runs reset the desktop in this configuration. Use --tsdf-paging-storage disk, "
                "a smaller hash profile, fewer estimated submaps, or raise "
                "--max-estimated-cpu-page-cache-gb intentionally."
            )

    if args.run and not args.disable_disk_page_fit_guard and usage["estimated_disk_page_gb"] > 0:
        free_gb = shutil.disk_usage(args.output_root).free / 1024**3
        required_remaining_gb = max(0.0, args.min_free_gb) + max(0.0, args.estimated_disk_page_margin_gb)
        remaining_after_pages_gb = free_gb - usage["estimated_disk_page_gb"]
        if remaining_after_pages_gb < required_remaining_gb:
            raise RuntimeError(
                f"Abort: TSDF paging spill/pages for {seq}/{variant} may need up to "
                f"{usage['estimated_disk_page_gb']:.1f} GB under {args.output_root}, leaving "
                f"{remaining_after_pages_gb:.1f} GB free. Required remaining space is "
                f"{required_remaining_gb:.1f} GB (--min-free-gb plus --estimated-disk-page-margin-gb). "
                "Use a roomier output root, run a shorter staged test, compact the hash/submap cadence, "
                "or pass --disable-disk-page-fit-guard for an intentional risk."
            )


def tail_log(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "\n".join(deque(handle, maxlen=max_lines)).rstrip()


def run_cmd(
    cmd: list[str],
    cwd: Path,
    log_path: Path | None = None,
    timeout_sec: float | None = None,
) -> None:
    print("[RUN]", " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, check=True, timeout=timeout_sec)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] logging stdout/stderr to {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("[RUN] " + " ".join(cmd) + "\n")
        log_file.flush()
        try:
            subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.CalledProcessError:
            tail = tail_log(log_path)
            if tail:
                print(f"[RUN] tail of failed log {log_path}:")
                print(tail)
            raise
        except subprocess.TimeoutExpired:
            log_file.write(f"\n[TIMEOUT] command exceeded {timeout_sec} seconds\n")
            log_file.flush()
            tail = tail_log(log_path)
            if tail:
                print(f"[RUN] tail of timed-out log {log_path}:")
                print(tail)
            raise


def run_sampled_quality_metrics(run_dir: Path, run_log_dir: Path | None = None) -> None:
    """Compute metrics for sampled eval outputs without changing the timed run."""
    gt_dir = run_dir / "val" / "gt"
    if not gt_dir.exists():
        print(f"[METRIC] skip sampled metrics for {run_dir.name}: missing {gt_dir}")
        return

    for subdir in ("render", "raycast_color"):
        render_dir = run_dir / "val" / subdir
        if not render_dir.exists():
            print(f"[METRIC] skip {subdir} metrics for {run_dir.name}: missing {render_dir}")
            continue
        log_path = None
        if run_log_dir is not None:
            log_path = run_log_dir / f"{run_dir.name}_{subdir}_sampled_metric.log"
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/metric_sampled_eval.py"),
                "--gt_path",
                str(gt_dir),
                "--render_path",
                str(render_dir),
            ],
            PROJECT_ROOT,
            log_path,
        )


def default_output_root_name() -> str:
    return f"tum_loop_outputs_{date.today():%Y%m%d}"


def choose_output_root(requested: Path | None, min_free_gb: float) -> Path:
    if requested is not None:
        return requested
    del min_free_gb
    return RUN_ROOT / default_output_root_name()


def path_storage_summary(label: str, path: Path) -> None:
    storage_path = path if path.exists() else path.parent
    usage = shutil.disk_usage(storage_path)
    print(
        f"[STORAGE] {label}={path} "
        f"free={usage.free / 1024**3:.1f}GB "
        f"used={usage.used / 1024**3:.1f}GB "
        f"total={usage.total / 1024**3:.1f}GB"
    )


def output_storage_summary(output_root: Path, largest_count: int) -> None:
    path_storage_summary("output_root", output_root)
    print(
        "[STORAGE] heavy run artifacts should stay under output_root; "
        "repo output/ is kept for lightweight summaries and legacy logs only"
    )
    if largest_count <= 0 or not output_root.exists():
        return
    entries: list[tuple[int, Path]] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        total = 0
        for path in child.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        entries.append((total, child))
    if not entries:
        return
    print(f"[STORAGE] largest existing output dirs under {output_root}:")
    for total, path in sorted(entries, reverse=True)[:largest_count]:
        print(f"  {total / 1024**3:7.2f}GB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-root", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "configs/release/replica/office0.yaml")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs/experiments/tum_loop")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for heavy run workspaces. Defaults to New Volume when present, "
            "then B6AC, then repo output/."
        ),
    )
    parser.add_argument("--trainer", type=Path, default=None, help="Explicit slam_trainer binary to run")
    parser.add_argument(
        "--benchmark-protocol",
        choices=[PROTOCOL_NONE, TUM_TRACKING_V1, TUM_INTEGRATED_V1, TUM_DEPLOYED_V1],
        default=PROTOCOL_NONE,
        help="lock a TUM comparison envelope and emit a benchmark manifest",
    )
    parser.add_argument(
        "--benchmark-role",
        choices=["auto", "baseline", "modified", "integrated"],
        default="auto",
        help="comparison role; auto derives it from the single selected variant",
    )
    parser.add_argument(
        "--allow-unpromoted-protocol",
        action="store_true",
        help="reproduce a rejected deployment screen; resulting runs are not paper-eligible",
    )
    parser.add_argument(
        "--benchmark-dataset-hash",
        choices=["none", "inventory", "full"],
        default="inventory",
        help="dataset fingerprint strength stored in benchmark_manifest.json",
    )
    parser.add_argument(
        "--thor-preflight",
        action="store_true",
        help="run scripts/thor_preflight.py on every generated config before execution",
    )
    parser.add_argument(
        "--analyze-realtime",
        action="store_true",
        help="write realtime_trace.csv and realtime_summary.json after every successful run",
    )
    parser.add_argument(
        "--tsdf-input-mode",
        choices=["on_demand", "preload"],
        default=None,
        help=(
            "Override PIPE.TSDF.input_mode and matching READER.preload_images. "
            "Use preload for paper-like TUM runs that fit in memory; on_demand for embedded-style streaming."
        ),
    )
    parser.add_argument(
        "--tsdf-input-prefetch",
        choices=["default", "on", "off"],
        default="default",
        help="override PIPE.TSDF.input_prefetch for generated experiments",
    )
    parser.add_argument(
        "--tsdf-input-tensor-bridge",
        choices=["direct", "opencv_legacy"],
        default="direct",
        help="select the streaming tensor-to-InfiniTAM bridge; direct is the Thor-ready path",
    )
    parser.add_argument(
        "--tsdf-input-bridge-validate",
        action="store_true",
        help="compare direct and legacy input bytes on every frame; diagnostic runs only",
    )
    parser.add_argument(
        "--submap-stats-interval-frames",
        type=int,
        default=None,
        help="sample submap_stats.csv every N frames; 0 writes only end-of-sequence state",
    )
    parser.add_argument(
        "--basic-nonfinite-recovery",
        action="store_true",
        help=(
            "For BasicEngine runs only, reject non-finite tracker poses and restore the previous finite pose. "
            "Default off preserves the unguarded legacy path."
        ),
    )
    parser.add_argument("--basic-debug-tracking-start-frame", type=int, default=None, help="Enable BasicEngine tracking/raycast diagnostics starting at this processed frame")
    parser.add_argument("--basic-debug-tracking-end-frame", type=int, default=None, help="Stop BasicEngine tracking/raycast diagnostics at this processed frame")
    parser.add_argument(
        "--tracking-failure-mode",
        choices=["legacy_ignore", "contain_failed", "stop_integration", "relocalise"],
        default=None,
        help="Override PIPE.TSDF.tracking_failure_mode; legacy_ignore reproduces the old baseline",
    )
    parser.add_argument(
        "--tracker-config",
        default=None,
        help="Override the InfiniTAM tracker string (intended for diagnostics and deterministic failure tests)",
    )
    parser.add_argument(
        "--tracking-profile",
        choices=["auto", "legacy", "efficient", "robust_rpe"],
        default="auto",
        help="Select a named depth-tracking profile; auto preserves the template default",
    )
    parser.add_argument(
        "--use-bilateral-filter",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override PIPE.TSDF.use_bilateral_filter; auto preserves the template default",
    )
    parser.add_argument(
        "--use-approximate-raycast",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override PIPE.TSDF.use_approximate_raycast; auto preserves the template default",
    )
    parser.add_argument("--hash-profile", choices=["compact_64k", "medium_128k", "legacy_large"], default=None, help="Override PIPE.TSDF.multi.hash_profile")
    parser.add_argument("--force-single-submap", action="store_true", help="Run MultiEngine through one logical submap for BasicEngine parity checks")
    parser.add_argument("--tsdf-paging", choices=["auto", "true", "false"], default="auto", help="Override PIPE.TSDF.multi.paging; auto disables paging for legacy_large")
    parser.add_argument("--tsdf-paging-storage", choices=["disk", "cpu"], default=None, help="Store paged passive SDF maps on disk or in CPU memory")
    parser.add_argument("--resident-gpu-maps", default=None, help="Override PIPE.TSDF.multi.resident_gpu_maps, e.g. 2, 3, or auto")
    parser.add_argument(
        "--tsdf-residency-profile",
        choices=["manual", "thor_cpu_bounded", "thor_streaming_bounded", "eval_resident64", "disk_stress"],
        default="manual",
        help="Apply a named MultiEngine residency policy before low-level TSDF residency overrides",
    )
    parser.add_argument(
        "--tsdf-lifecycle-profile",
        choices=["default", "paper_parity", "paper_parity_adaptive_longspan"],
        default="default",
        help="Apply a named MultiEngine lifecycle policy before low-level TSDF boundary overrides",
    )
    parser.add_argument("--tsdf-async-page-out", choices=["auto", "true", "false"], default="auto", help="Detach paged-out TSDF maps and snapshot them on a background thread")
    parser.add_argument("--tsdf-max-pending-page-snapshots", type=int, default=None, help="Maximum pending async TSDF page snapshots before blocking")
    parser.add_argument("--tsdf-max-pending-page-snapshot-mb", type=int, default=None, help="Approximate pending async TSDF snapshot memory budget before blocking; 0 disables the MB cap")
    parser.add_argument("--tsdf-async-page-in", choices=["auto", "true", "false"], default="auto", help="Enable or disable CPU-staged async TSDF page-in prefetch for MultiEngine eval/render runs")
    parser.add_argument("--tsdf-max-pending-page-ins", type=int, default=None, help="Maximum pending async TSDF page-ins before blocking")
    parser.add_argument("--tsdf-max-pending-page-in-mb", type=int, default=None, help="Approximate pending async TSDF page-in memory budget before blocking; 0 disables the MB cap")
    parser.add_argument("--tsdf-cpu-page-cache-limit-mb", type=int, default=None, help="Bound passive CPU TSDF page caches before spilling old snapshots to sparse disk pages; 0 disables")
    parser.add_argument("--tsdf-render-prefetch", choices=["auto", "true", "false"], default="auto", help="Prefetch selected paged TSDF render maps before eval raycasting and report the cost separately")
    parser.add_argument("--tsdf-render-prefetch-lookahead-frames", type=int, default=None, help="Override PIPE.TSDF.multi.render_prefetch_lookahead_frames")
    parser.add_argument("--tsdf-render-prefetch-lookahead-max-maps", type=int, default=None, help="Override PIPE.TSDF.multi.render_prefetch_lookahead_max_maps; 0 means unlimited")
    parser.add_argument("--tsdf-render-pin-primary", choices=["auto", "true", "false"], default="auto", help="Override PIPE.TSDF.multi.render_pin_primary_submap")
    parser.add_argument("--tsdf-render-compositor", choices=["nearest_depth", "owner_priority", "handoff_aware"], default=None, help="Override PIPE.TSDF.multi.render_compositor for paged combined TSDF raycasts")
    parser.add_argument("--tsdf-boundary-policy", choices=["visibility", "capacity"], default=None, help="Override PIPE.TSDF.multi.boundary_policy")
    parser.add_argument("--allow-visibility-handoff", choices=["auto", "true", "false"], default="auto", help="Allow unforced visibility-based primary switches between local maps")
    parser.add_argument("--tsdf-occupancy-threshold-policy", choices=["fixed", "adaptive_longspan"], default=None, help="Override PIPE.TSDF.multi.occupancy_threshold_policy")
    parser.add_argument("--tsdf-adaptive-new-map-occupancy", type=float, default=None, help="Relaxed new-map occupancy threshold used by adaptive_longspan")
    parser.add_argument("--tsdf-adaptive-max-occupancy", type=float, default=None, help="Relaxed fusion stop occupancy threshold used by adaptive_longspan")
    parser.add_argument("--tsdf-adaptive-occupancy-min-primary-frames", type=int, default=None, help="Minimum current-primary age before adaptive occupancy thresholds can activate")
    parser.add_argument("--tsdf-adaptive-occupancy-min-processed-frames", type=int, default=None, help="Minimum run age before adaptive occupancy thresholds can activate")
    parser.add_argument("--tsdf-adaptive-occupancy-max-raw-weak-streak", type=int, default=None, help="Maximum raw weak tracking streak allowed while activating adaptive occupancy thresholds")
    parser.add_argument("--tsdf-adaptive-occupancy-max-raw-failed-streak", type=int, default=None, help="Maximum raw failed tracking streak allowed while activating adaptive occupancy thresholds")
    parser.add_argument("--sequential-transform-policy", choices=["validated_overlap", "overlap", "initial"], default=None, help="Override PIPE.TSDF.multi.sequential_transform_policy")
    parser.add_argument("--sequential-transform-max-translation", type=float, default=None, help="Override PIPE.TSDF.multi.sequential_transform_max_translation in meters")
    parser.add_argument("--sequential-transform-max-rotation", type=float, default=None, help="Override PIPE.TSDF.multi.sequential_transform_max_rotation_deg")
    parser.add_argument("--invalid-transform-restart-attempts", type=int, default=None, help="Restart a NEW_LOCAL_MAP candidate after this many rejected sequential transforms; 0 disables")
    parser.add_argument("--handoff-pose-policy", choices=["preserve_global", "candidate", "validated_candidate"], default=None, help="Override PIPE.TSDF.multi.handoff_pose_policy")
    parser.add_argument("--new-map-handoff-policy", choices=["visibility", "accepted", "validated_visibility"], default=None, help="Override PIPE.TSDF.multi.new_map_handoff_policy")
    parser.add_argument("--new-map-handoff-min-visible-ratio", type=float, default=None, help="Minimum candidate visible-original-block ratio before promoting a NEW_LOCAL_MAP")
    parser.add_argument("--new-map-handoff-min-visible-margin", type=float, default=None, help="Minimum candidate-visible minus primary-visible margin before promoting a NEW_LOCAL_MAP")
    parser.add_argument("--new-map-handoff-min-accepts", type=int, default=None, help="Accepted overlap windows required before promoting a NEW_LOCAL_MAP")
    parser.add_argument("--new-map-handoff-emergency-min-accepts", type=int, default=None, help="Emergency/capacity accepted overlap windows required before promoting a NEW_LOCAL_MAP; -1 inherits the normal minimum")
    parser.add_argument("--candidate-recovery-policy", choices=["disabled", "keep_if_visible"], default=None, help="Keep a promising failed NEW_LOCAL_MAP candidate for another bounded overlap window")
    parser.add_argument("--candidate-recovery-min-visible-ratio", type=float, default=None, help="Minimum candidate visible-original-block ratio required for candidate recovery")
    parser.add_argument("--candidate-recovery-min-blocks", type=int, default=None, help="Minimum candidate TSDF blocks required for candidate recovery")
    parser.add_argument("--candidate-recovery-max-attempts", type=int, default=None, help="Maximum candidate recovery windows before hard rejection")
    parser.add_argument("--primary-failure-handoff-policy", choices=["disabled", "promote_visible_candidate"], default=None, help="Promote a stable candidate when the primary raw tracker has failed for a bounded streak")
    parser.add_argument("--primary-failure-handoff-min-raw-failed-frames", type=int, default=None, help="Consecutive raw failed primary frames required before primary-failure candidate promotion")
    parser.add_argument("--primary-failure-handoff-raw-gate", choices=["failed_only", "failed_or_weak"], default=None, help="Allow primary-failure handoff only from hard raw failures, or also from raw non-good weak streaks")
    parser.add_argument("--primary-failure-handoff-min-raw-weak-frames", type=int, default=None, help="Consecutive raw non-good primary frames required when primary-failure handoff raw gate is failed_or_weak")
    parser.add_argument("--primary-failure-handoff-min-candidate-visible-ratio", type=float, default=None, help="Minimum candidate visible-original-block ratio before primary-failure promotion")
    parser.add_argument("--primary-failure-handoff-min-candidate-blocks", type=int, default=None, help="Minimum candidate TSDF blocks before primary-failure promotion")
    parser.add_argument("--primary-failure-handoff-min-candidate-good-frames", type=int, default=None, help="Minimum policy-good candidate tracking frames before primary-failure promotion")
    parser.add_argument("--primary-failure-handoff-min-overlap-frames", type=int, default=None, help="Minimum accepted overlap observations before primary-failure promotion; 0 allows pure recovery handoff")
    parser.add_argument(
        "--primary-failure-handoff-require-pose-continuity",
        choices=["auto", "true", "false"],
        default="auto",
        help="Block primary-failure promotion unless the candidate can preserve the current global camera pose",
    )
    parser.add_argument("--tracking-boundary-policy", choices=["disabled", "raw_weak_streak"], default=None, help="Create a new TSDF submap when primary raw tracking quality stays weak")
    parser.add_argument("--raw-weak-boundary-frames", type=int, default=None, help="Weak raw-tracking streak required before requesting a new TSDF submap")
    parser.add_argument("--raw-failed-boundary-frames", type=int, default=None, help="Failed raw-tracking streak required before requesting a new TSDF submap")
    parser.add_argument("--tracking-boundary-min-primary-frames", type=int, default=None, help="Minimum primary-submap age before tracking-quality boundary requests")
    parser.add_argument("--tracking-boundary-cooldown-frames", type=int, default=None, help="Cooldown between tracking-quality boundary requests")
    parser.add_argument("--tracking-boundary-force-handoff", choices=["auto", "true", "false"], default="auto", help="Request a primary handoff with tracking-quality boundary requests")
    parser.add_argument(
        "--tracking-boundary-handoff-policy",
        choices=["weak_or_failed", "failed_only", "never"],
        default=None,
        help="Control whether tracking-boundary requests may force primary promotion; failed_only still warms on weak frames",
    )
    parser.add_argument(
        "--pose-graph-overlap-roundtrip-check",
        choices=["auto", "true", "false"],
        default="auto",
        help="Enable a read-only SE3 round-trip consistency check over sequential submap constraints",
    )
    parser.add_argument(
        "--pose-graph-overlap-roundtrip-tolerance",
        type=float,
        default=None,
        help="Max absolute transform error allowed by the overlap round-trip consistency check",
    )
    parser.add_argument("--require-raw-tracking-success", choices=["auto", "true", "false"], default="auto", help="Fail validation runs after sustained raw primary tracking failure, instead of accepting InfiniTAM's legacy failure policy")
    parser.add_argument("--raw-tracking-failure-grace-frames", type=int, default=None, help="Consecutive raw failed primary frames allowed before strict raw tracking mode aborts")
    parser.add_argument("--sequences", nargs="*", default=["fr1_desk", "fr2_xyz", "fr3_office", "fr1_room"])
    parser.add_argument("--variants", nargs="*", default=["basic_sdf", "multi_sdf", "multi_hybrid_no_loop", "multi_hybrid_dry_solve", "multi_hybrid_apply"])
    parser.add_argument(
        "--loop-profile",
        choices=[
            "default",
            "conservative",
            "aggregate",
            "aggregate_strict",
            "aggregate_pcm",
            "aggregate_pcm_drift",
            "aggregate_pcm_drift_online",
            "aggregate_pcm_drift_online_single_high",
            "aggregate_pcm_drift_online_longspan",
            "recommended_longspan",
            "recommended_longspan_robust",
            "recommended_longspan_robust_streaming",
            "recommended_longspan_lowlatency",
            "aggregate_pcm_drift_online_guarded_recovery",
        ],
        default="default",
        help=(
            "Named loop-closure evidence profile. default preserves current runner defaults; "
            "conservative keeps strict long-span gates; aggregate uses multi-observation ORB support "
            "as the only allowed small-loop bypass; aggregate_strict uses aggregate ORB verification "
            "with geometry-first scheduling and strict long-span apply gates; aggregate_pcm adds "
            "pairwise loop-consistency clique selection before apply; aggregate_pcm_drift keeps "
            "that selection and adds span-aware ORB ICP drift-disagreement gates; "
            "aggregate_pcm_drift_online enables bounded live updates with final scan fallback; "
            "aggregate_pcm_drift_online_single_high also permits one high-confidence long-span loop; "
            "aggregate_pcm_drift_online_longspan keeps that support policy but widens drift-aware "
            "verification and sequential residual gates for long real RGB-D revisit corrections; "
            "recommended_longspan is a stable alias for the current best-tested long-run TUM profile; "
            "recommended_longspan_robust additionally enables confidence-diagonal loop information "
            "and Cauchy loop-edge robustification; "
            "recommended_longspan_lowlatency uses the same profile with a smaller online verification "
            "budget and a larger terminal verification budget to reduce live-frame spikes; "
            "aggregate_pcm_drift_online_guarded_recovery adds the validated guarded emergency "
            "handoff and high-confidence support cap for loop-heavy long runs."
        ),
    )
    parser.add_argument("--eval-after-train", action="store_true", help="Render the configured eval split after training for image-quality metrics")
    parser.add_argument("--eval-split", choices=["train", "val", "all"], default=None, help="Override PIPE.eval_split for eval-after-train runs")
    parser.add_argument("--eval-stride", type=int, default=None, help="Render every Nth eval camera; default 1 preserves full eval output")
    parser.add_argument("--eval-max-frames", type=int, default=None, help="Maximum eval cameras to render; 0 preserves all selected cameras")
    parser.add_argument("--eval-start-offset", type=int, default=None, help="Zero-based offset inside the selected eval split before stride sampling")
    parser.add_argument(
        "--sampled-quality-metrics",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Run PSNR/SSIM/LPIPS over sampled eval outputs after each successful trainer run. "
            "auto enables this only with --eval-after-train."
        ),
    )
    parser.add_argument(
        "--bypass-libtorch-atexit",
        choices=["true", "false"],
        default="true",
        help=(
            "Set PIPE.bypass_libtorch_atexit in generated configs. This skips LibTorch static "
            "finalizers after all run outputs are flushed, avoiding post-eval exit crashes without "
            "changing SLAM metrics."
        ),
    )
    parser.add_argument("--no-save-after-train", action="store_true", help="Export poses and diagnostics only; skip model, Gaussian submap, TSDF, and mesh checkpoints")
    parser.add_argument("--keep-recon-save", action="store_true", help="Keep save_after_train enabled for recon variants")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=None, help="Override PIPE.random_seed for reproducible comparison runs")
    parser.add_argument("--tsdf-max-frames", type=int, default=None, help="Override PIPE.TSDF.multi.max_frames for forced-boundary stress runs")
    parser.add_argument("--tsdf-overlap-frames", type=int, default=None, help="Override PIPE.TSDF.multi.overlap_frames")
    parser.add_argument("--tsdf-trial-frames", type=int, default=None, help="Override PIPE.TSDF.multi.trial_frames")
    parser.add_argument("--tsdf-emergency-overlap-frames", type=int, default=None, help="Override PIPE.TSDF.multi.emergency_overlap_frames; 0 disables emergency link acceptance")
    parser.add_argument("--tsdf-new-map-occupancy", type=float, default=None, help="Override PIPE.TSDF.multi.new_map_block_occupancy")
    parser.add_argument("--tsdf-max-occupancy", type=float, default=None, help="Override PIPE.TSDF.multi.max_block_occupancy")
    parser.add_argument("--tsdf-fusion-reserve-blocks", type=int, default=None, help="Reserve this many free TSDF blocks before allowing another fusion; 0 uses the MultiEngine auto reserve")
    parser.add_argument("--tsdf-occupancy-growth-lookahead-frames", type=int, default=None, help="Request a new TSDF map early if block growth projects stop-threshold saturation within this many frames")
    parser.add_argument("--tsdf-occupancy-growth-min-primary-frames", type=int, default=None, help="Minimum primary-submap age before projected-growth boundaries are allowed")
    parser.add_argument("--tsdf-occupancy-growth-sample-frames", type=int, default=None, help="Minimum frame interval used to estimate TSDF block-growth rate for projected boundaries")
    parser.add_argument("--skip-primary-fusion-on-capacity-handoff", choices=["auto", "true", "false"], default="auto", help="Stop fusing the outgoing primary only during capacity-driven handoff overlap")
    parser.add_argument("--new-gs-sample-ratio", type=float, default=None, help="Override PIPE.new_gs_sample_ratio for Gaussian-producing variants")
    parser.add_argument("--local-opt-iters", type=int, default=None, help="Override PIPE.local_opt_iters for Gaussian-producing variants")
    parser.add_argument("--local-opt-interval", type=int, default=None, help="Override PIPE.local_opt_interval for Gaussian-producing variants")
    parser.add_argument(
        "--gaussian-schedule",
        choices=["auto", "immediate", "sliced"],
        default="auto",
        help="Override PIPE.gaussian_schedule.mode for Gaussian-producing variants",
    )
    parser.add_argument(
        "--gaussian-opt-iters-per-frame",
        type=int,
        default=None,
        help="Iterations executed per frame when PIPE.gaussian_schedule.mode=sliced",
    )
    parser.add_argument(
        "--export-online-end-trajectory",
        choices=["auto", "true", "false"],
        default="auto",
        help="Export the causal trajectory immediately before terminal pose-graph work",
    )
    parser.add_argument("--keyframe-select-max", type=int, default=None, help="Override PIPE.keyframe_select_max for Gaussian-producing variants")
    parser.add_argument("--submap-keyframe-selection-policy", choices=["auto", "random", "temporal_spread", "loss_temporal_mix"], default="auto", help="Override PIPE.gaussian_submaps.keyframe_selection_policy")
    parser.add_argument("--model-sh-degree", type=int, default=None, help="Override MODEL.sh_degree for Gaussian-producing variants")
    parser.add_argument("--gaussian-max-frames", type=int, default=None, help="Override PIPE.gaussian_submaps.max_frames for forced-boundary stress runs")
    parser.add_argument("--gaussian-max-gaussians", type=int, default=None, help="Override PIPE.gaussian_submaps.max_gaussians")
    parser.add_argument("--gaussian-max-new-per-frame", type=int, default=None, help="Override PIPE.gaussian_submaps.max_new_gaussians_per_frame")
    parser.add_argument("--page-frozen-gaussians", choices=["auto", "true", "false"], default="auto", help="Override PIPE.gaussian_submaps.page_frozen_gaussians")
    parser.add_argument("--resident-gaussian-submaps", type=int, default=None, help="Override PIPE.gaussian_submaps.resident_gaussian_submaps")
    parser.add_argument("--gaussian-insert-selection", choices=["auto", "random", "error_topk", "tile_error_topk", "adaptive_tile_error_topk"], default="auto", help="Override PIPE.gaussian_submaps.gaussian_insert_selection")
    parser.add_argument("--gaussian-insert-tile-size", type=int, default=None, help="Tile size in pixels for tile_error_topk Gaussian insertion")
    parser.add_argument("--gaussian-insert-tile-max-pixels", type=int, default=None, help="Max selected pixels per tile for tile_error_topk; 0 derives a balanced cap")
    parser.add_argument("--gaussian-insert-min-age-frames", type=int, default=None, help="Delay Gaussian insertion until the active hybrid submap has this many processed frames")
    parser.add_argument("--gaussian-insert-min-sdf-blocks", type=int, default=None, help="Delay Gaussian insertion until the owner TSDF submap has this many allocated voxel blocks")
    parser.add_argument("--ges-residual-scale", type=float, default=None, help="Override MODEL.ges_residual_scale; 1.0 preserves the original GES blend")
    parser.add_argument("--ges-base-color-weight", type=float, default=None, help="Override MODEL.ges_base_color_weight; 1.0 preserves the original GES blend")
    parser.add_argument("--candidate-reject-retire", choices=["auto", "true", "false"], default="auto", help="Override PIPE.gaussian_submaps.candidate_reject_retire")
    parser.add_argument("--candidate-reject-retire-min-rejects", type=int, default=None, help="Minimum repeated TSDF link rejects before a warming Gaussian candidate can be retired")
    parser.add_argument("--candidate-reject-retire-min-observed-frames", type=int, default=None, help="Minimum observed frames before TSDF-reject retirement can remove a warming Gaussian candidate")
    parser.add_argument("--candidate-reject-retire-min-gaussians", type=int, default=None, help="Minimum Gaussian count before TSDF-reject retirement is allowed")
    parser.add_argument("--candidate-reject-retire-max-inliers", type=int, default=None, help="Maximum link-reject inliers still considered weak enough for TSDF-reject retirement")
    parser.add_argument("--candidate-reject-retire-cooldown-frames", type=int, default=None, help="Frames to suppress recreating a Gaussian candidate after TSDF-reject retirement")
    parser.add_argument("--candidate-gaussian-insert-min-frames", type=int, default=None, help="Delay warming-candidate Gaussian insertion until this many overlap frames have been observed")
    parser.add_argument("--candidate-immature-new-gs-sample-ratio", type=float, default=None, help="Sample ratio for warming candidates before candidate_gaussian_insert_min_frames; -1 preserves the old full-skip behavior unless an immature cap is set")
    parser.add_argument("--candidate-immature-max-new-per-frame", type=int, default=None, help="Per-frame insertion cap for warming candidates before candidate_gaussian_insert_min_frames; 0 leaves the ramp disabled")
    parser.add_argument("--candidate-extend-while-overlap", choices=["auto", "true", "false"], default="auto", help="Override PIPE.gaussian_submaps.candidate_extend_while_overlap")
    parser.add_argument("--candidate-max-warm-frames", type=int, default=None, help="Override PIPE.gaussian_submaps.candidate_max_warm_frames; 0 disables the cap")
    parser.add_argument("--candidate-handoff-max-opt-passes", type=int, default=None, help="Override PIPE.gaussian_submaps.candidate_handoff_max_opt_passes; -1 inherits handoff_max_opt_passes and 0 disables the candidate cap")
    parser.add_argument("--handoff-frames", type=int, default=None, help="Override PIPE.gaussian_submaps.handoff_frames")
    parser.add_argument("--handoff-opt-iters", type=int, default=None, help="Override PIPE.gaussian_submaps.handoff_opt_iters")
    parser.add_argument("--handoff-max-opt-passes", type=int, default=None, help="Override PIPE.gaussian_submaps.handoff_max_opt_passes")
    parser.add_argument("--handoff-add-outgoing-gaussians", choices=["auto", "true", "false"], default="auto", help="Allow outgoing handoff submaps to add/prune new Gaussians instead of only optimizing existing ones")
    parser.add_argument("--graduation-optimize", choices=["auto", "true", "false"], default="auto", help="Run a bounded final Gaussian optimization pass before freezing outgoing submaps")
    parser.add_argument("--graduation-opt-iters", type=int, default=None, help="Override PIPE.gaussian_submaps.graduation_opt_iters")
    parser.add_argument("--graduation-keyframes", type=int, default=None, help="Override PIPE.gaussian_submaps.graduation_keyframes")
    parser.add_argument("--graduation-max-opt-passes", type=int, default=None, help="Override PIPE.gaussian_submaps.graduation_max_opt_passes")
    parser.add_argument("--graduation-keyframe-policy", choices=["auto", "random", "temporal_spread", "loss_temporal_mix"], default="auto", help="Override PIPE.gaussian_submaps.graduation_keyframe_policy")
    parser.add_argument("--candidate-warm-interval", type=int, default=None, help="Override PIPE.gaussian_submaps.candidate_warm_interval")
    parser.add_argument("--candidate-warming", choices=["auto", "true", "false"], default="auto", help="Override PIPE.gaussian_submaps.candidate_warming")
    parser.add_argument("--boundary-observation-handoff", choices=["auto", "true", "false"], default="auto", help="Override PIPE.gaussian_submaps.boundary_observation_handoff")
    parser.add_argument("--render-policy", choices=[
        "owner_only",
        "owner_neighbors",
        "owner_neighbors_all",
        "owner_neighbor_fill",
        "sdf_guided_fill",
        "sdf_guided_adaptive_fill",
        "sdf_depth_consistent_blend",
        "owner_confidence_fill",
        "handoff_aware_fill",
        "weight_normalized",
        "nearest_depth",
    ], default=None, help="Override PIPE.gaussian_submaps.render_policy")
    parser.add_argument("--render-max-visible-gaussians", type=int, default=None, help="Override PIPE.gaussian_submaps.render_max_visible_gaussians; 0 disables the cap")
    parser.add_argument("--render-include-handoff-submaps", choices=["auto", "true", "false"], default="auto", help="Keep outgoing handoff submaps in Gaussian render selection")
    parser.add_argument("--render-single-submap-passthrough", choices=["auto", "true", "false"], default="auto", help="Return direct GES output when exactly one Gaussian submap is selected")
    parser.add_argument("--allow-multi-logical-single-passthrough", action="store_true", help="Allow passthrough for one selected Gaussian submap even when the saved graph has multiple logical submaps; kept for compatibility because this is now the default")
    parser.add_argument("--require-single-logical-single-passthrough", action="store_true", help="Only allow single-submap passthrough when the whole saved graph has one logical Gaussian submap")
    parser.add_argument("--disable-local-sdf-reference-render", action="store_true", help="Render local Gaussian submaps against the combined global SDF reference for ablation")
    parser.add_argument("--render-fill-require-sdf-hole", choices=["auto", "true", "false"], default="auto", help="Override PIPE.gaussian_submaps.render_fill_require_sdf_hole")
    parser.add_argument("--render-fill-blend-weight", type=float, default=None, help="Override PIPE.gaussian_submaps.render_fill_blend_weight")
    parser.add_argument("--render-owner-alpha-fill-thres", type=float, default=None, help="Override PIPE.gaussian_submaps.render_owner_alpha_fill_thres")
    parser.add_argument("--render-min-owner-age-frames", type=int, default=None, help="Keep a mature overlap submap first in render selection while the TSDF owner is younger than this many frames")
    parser.add_argument("--render-min-ready-frames", type=int, default=None, help="Skip Gaussian submaps in rendering until they have accumulated this many frames")
    parser.add_argument("--render-owner-min-ready-frames", type=int, default=None, help="Owner submap readiness override; defaults to render_min_ready_frames")
    parser.add_argument("--render-neighbor-min-ready-frames", type=int, default=None, help="Neighbor submap readiness override; defaults to render_min_ready_frames")
    parser.add_argument("--render-past-frame-margin", type=int, default=None, help="Override PIPE.gaussian_submaps.render_past_frame_margin")
    parser.add_argument("--render-future-frame-margin", type=int, default=None, help="Override PIPE.gaussian_submaps.render_future_frame_margin")
    parser.add_argument("--render-min-neighbor-gaussians", type=int, default=None, help="Override PIPE.gaussian_submaps.render_min_neighbor_gaussians")
    parser.add_argument("--render-blend-alpha-thres", type=float, default=None, help="Override PIPE.gaussian_submaps.render_blend_alpha_thres")
    parser.add_argument("--render-blend-depth-thres", type=float, default=None, help="Override PIPE.gaussian_submaps.render_blend_depth_thres")
    parser.add_argument("--render-blend-max-weight", type=float, default=None, help="Override PIPE.gaussian_submaps.render_blend_max_weight")
    parser.add_argument("--render-blend-weight-scale", type=float, default=None, help="Override PIPE.gaussian_submaps.render_blend_weight_scale")
    parser.add_argument("--render-neighbor-alpha-fill-thres", type=float, default=None, help="Override PIPE.gaussian_submaps.render_neighbor_alpha_fill_thres")
    parser.add_argument("--loop-min-separation", type=int, default=None, help="Override PIPE.pose_graph.loop_min_submap_separation")
    parser.add_argument("--loop-max-distance", type=float, default=None, help="Override PIPE.pose_graph.loop_max_distance")
    parser.add_argument("--loop-max-rotation", type=float, default=None, help="Override PIPE.pose_graph.loop_max_rotation_deg")
    parser.add_argument("--loop-max-candidates", type=int, default=None, help="Override PIPE.pose_graph.loop_max_candidates")
    parser.add_argument("--loop-candidate-export-policy", choices=["score", "span_reserve_score"], default=None, help="Policy for exporting accepted loop constraints into apply selection")
    parser.add_argument("--loop-candidate-export-span-reserve-fraction", type=float, default=None, help="Fraction of exported loop slots reserved for long-span candidates")
    parser.add_argument("--loop-candidate-export-min-span", type=int, default=None, help="Minimum span eligible for loop export reserve; 0 uses apply_min_loop_submap_span")
    parser.add_argument("--max-scanned-pairs", type=int, default=None, help="Maximum submap-pair slots scanned per online candidate check; 0 scans all")
    parser.add_argument("--final-candidate-scan", choices=["auto", "true", "false"], default="auto", help="Run a final end-of-sequence pair scan; false reuses only the online verified loop pool")
    parser.add_argument("--terminal-online-scan", choices=["auto", "true", "false"], default="auto", help="When final candidate scan is disabled, allow one bounded online-style scan at end of sequence")
    parser.add_argument("--final-max-scanned-pairs", type=int, default=None, help="Maximum submap-pair slots scanned by the final end-of-sequence candidate check; 0 keeps the full final scan")
    parser.add_argument("--final-rank-all-candidates", choices=["auto", "true", "false"], default="auto", help="For ordered final scans, rank all cheap candidate pairs before applying the final expensive verification budget")
    parser.add_argument("--incremental-candidate-scan", choices=["auto", "true", "false"], default="auto", help="Resume bounded online loop-candidate scans from the previous cursor instead of rescanning from the start")
    parser.add_argument("--max-candidate-pairs", type=int, default=None, help="Override PIPE.pose_graph.max_candidate_pairs_per_check")
    parser.add_argument("--candidate-accept-target", type=int, default=None, help="Stop expensive loop verification after this many accepted candidates in one update; 0 disables")
    parser.add_argument("--candidate-order", choices=["temporal", "coarse_score", "coarse_span", "orb_appearance", "orb_bow"], default=None, help="Order eligible loop candidates before expensive verification")
    parser.add_argument("--span-reserve-pairs", type=int, default=None, help="For coarse_span ordering, reserve this many final verification slots for longest-span submap pairs; 0 uses the fraction")
    parser.add_argument("--span-reserve-fraction", type=float, default=None, help="For coarse_span ordering, reserve this fraction of final verification slots for longest-span submap pairs")
    parser.add_argument("--candidate-verification", choices=["none", "keyframe_pose", "keyframe_depth", "keyframe_icp", "orb_icp"], default=None, help="Override PIPE.pose_graph.candidate_verification")
    parser.add_argument("--appearance-max-observation-pairs", type=int, default=None, help="Maximum observation pairs per submap pair for cheap ORB appearance ranking; 0 checks all")
    parser.add_argument("--appearance-min-orb-matches", type=int, default=None, help="Optional hard prefilter on cheap ORB mutual matches; 0 keeps ranking-only behavior")
    parser.add_argument("--loop-observation-interval", type=int, default=None, help="Override PIPE.pose_graph.loop_observation_interval_frames")
    parser.add_argument("--loop-observation-max", type=int, default=None, help="Override PIPE.pose_graph.loop_observation_max_per_submap")
    parser.add_argument("--include-keyframes-with-loop-observations", choices=["auto", "true", "false"], default="auto", help="Mix bounded Gaussian keyframes with loop-observation packets during loop verification")
    parser.add_argument("--keyframe-observation-max", type=int, default=None, help="Maximum time-spaced Gaussian keyframes per submap to include in loop verification; 0 disables keyframe additions")
    parser.add_argument("--orb-feature-cache", choices=["auto", "true", "false"], default="auto", help="Cache ORB keyframe features during online loop-candidate checks")
    parser.add_argument("--orb-feature-cache-limit", type=int, default=None, help="Maximum cached ORB keyframe packets per run; 0 disables caching")
    parser.add_argument("--warm-loop-observations", choices=["auto", "true", "false"], default="auto", help="Precompute ORB packets when loop observations are stored")
    parser.add_argument("--warm-max-per-frame", type=int, default=None, help="Maximum loop-observation ORB packets to warm in one frame; 0 disables the per-frame cap")
    parser.add_argument("--warm-skip-update-frames", choices=["auto", "true", "false"], default="auto", help="Skip synchronous ORB warming on frames that are already due for an online pose-graph scan")
    parser.add_argument("--warm-async", choices=["auto", "true", "false"], default="auto", help="Run ORB loop-observation warming on a bounded CPU worker thread")
    parser.add_argument("--warm-queue-limit", type=int, default=None, help="Maximum queued async ORB warm requests; 0 disables the queue limit")
    parser.add_argument("--orb-max-features", type=int, default=None, help="Override PIPE.pose_graph.orb_max_features")
    parser.add_argument("--orb-ratio", type=float, default=None, help="Override PIPE.pose_graph.orb_ratio")
    parser.add_argument("--orb-match-policy", choices=["mutual_ratio", "ratio"], default=None, help="ORB descriptor match policy for appearance ranking and RGB-D loop verification")
    parser.add_argument("--orb-min-quadrants", type=int, default=None, help="Override PIPE.pose_graph.orb_min_quadrants")
    parser.add_argument("--orb-depth-search-radius", type=int, default=None, help="Search this pixel radius around ORB keypoints for valid TUM depth during RGB-D loop verification")
    parser.add_argument("--orb-pnp-fallback", choices=["auto", "true", "false"], default="auto", help="Use source-depth plus target-2D PnP if ORB 3D-3D loop verification has too few inliers")
    parser.add_argument("--orb-pnp-iterations", type=int, default=None, help="Override PIPE.pose_graph.orb_pnp_iterations")
    parser.add_argument("--orb-pnp-reprojection-error", type=float, default=None, help="Override PIPE.pose_graph.orb_pnp_reprojection_error in pixels")
    parser.add_argument("--orb-pnp-confidence", type=float, default=None, help="Override PIPE.pose_graph.orb_pnp_confidence")
    parser.add_argument("--bow-word-bits", type=int, default=None, help="Number of ORB descriptor hash bits for dependency-free BoW retrieval")
    parser.add_argument("--bow-max-observations", type=int, default=None, help="Maximum observations per submap used to build the BoW histogram; 0 uses all candidate observations")
    parser.add_argument("--bow-min-score", type=float, default=None, help="Optional BoW prefilter threshold; 0 keeps ranking-only behavior")
    parser.add_argument("--bow-defer-appearance", choices=["auto", "true", "false"], default="auto", help="For orb_bow ordering, rank with cheap BoW first and only run ORB appearance scoring inside the verifier budget")
    parser.add_argument("--bow-histogram-cache", choices=["auto", "true", "false"], default="auto", help="Persist BoW histograms across online loop-candidate scans")
    parser.add_argument("--verify-icp-min-inliers", type=int, default=None, help="Override PIPE.pose_graph.verify_icp_min_inliers")
    parser.add_argument("--verify-icp-max-rmse", type=float, default=None, help="Override PIPE.pose_graph.verify_icp_max_rmse")
    parser.add_argument("--verify-icp-max-translation-delta", type=float, default=None, help="Override PIPE.pose_graph.verify_icp_max_translation_delta")
    parser.add_argument("--verify-icp-max-rotation-delta", type=float, default=None, help="Override PIPE.pose_graph.verify_icp_max_rotation_delta_deg")
    parser.add_argument("--verify-icp-ransac", choices=["auto", "true", "false"], default="auto", help="Override PIPE.pose_graph.verify_icp_ransac")
    parser.add_argument("--verify-icp-ransac-iterations", type=int, default=None, help="Override PIPE.pose_graph.verify_icp_ransac_iterations")
    parser.add_argument("--verify-icp-ransac-threshold", type=float, default=None, help="Override PIPE.pose_graph.verify_icp_ransac_threshold")
    parser.add_argument("--verify-icp-ransac-min-inlier-ratio", type=float, default=None, help="Override PIPE.pose_graph.verify_icp_ransac_min_inlier_ratio")
    parser.add_argument("--verify-icp-weighted-kabsch", choices=["auto", "true", "false"], default="auto", help="Use depth-variance weights in the ORB RGB-D rigid fit")
    parser.add_argument("--verify-icp-depth-adaptive-ransac", choices=["auto", "true", "false"], default="auto", help="Use depth-dependent RANSAC inlier thresholds for ORB RGB-D verification")
    parser.add_argument("--verify-icp-depth-noise-scale", type=float, default=None, help="Depth-noise scale k for sigma_z ~= k*z^2 in weighted/adaptive ORB RGB-D verification")
    parser.add_argument("--verify-icp-depth-adaptive-threshold-scale", type=float, default=None, help="Multiplier applied to depth-derived sigma for adaptive RANSAC thresholds")
    parser.add_argument("--verify-icp-max-score", type=float, default=None, help="Override PIPE.pose_graph.verify_icp_max_score")
    parser.add_argument("--verify-icp-drift-aware-delta", choices=["auto", "true", "false"], default="auto", help="Scale ORB ICP graph-disagreement gates by loop submap span")
    parser.add_argument("--verify-icp-drift-reference-span", type=float, default=None, help="Submap span where drift-aware ICP gates start expanding")
    parser.add_argument("--verify-icp-drift-max-scale", type=float, default=None, help="Maximum multiplier for drift-aware ICP translation/rotation gates")
    parser.add_argument("--verify-icp-drift-score-margin", type=float, default=None, help="Extra allowed verification score per unit of drift gate scale beyond 1")
    parser.add_argument("--verify-orb-aggregate-pairs", choices=["auto", "true", "false"], default="auto", help="Aggregate multiple weak ORB RGB-D observation pairs before geometric verification")
    parser.add_argument("--verify-orb-aggregate-max-pairs", type=int, default=None, help="Maximum observation pairs used by aggregate ORB RGB-D verification")
    parser.add_argument("--verify-max-observation-pairs", type=int, default=None, help="Override PIPE.pose_graph.verify_max_observation_pairs")
    parser.add_argument("--verify-pair-selection", choices=["pose", "appearance"], default=None, help="Order observation pairs for loop verification by pose proximity or ORB appearance support")
    parser.add_argument("--verify-min-observations", type=int, default=None, help="Override PIPE.pose_graph.verify_min_observations")
    parser.add_argument("--verify-observation-max-translation-delta", type=float, default=None, help="Override PIPE.pose_graph.verify_observation_max_translation_delta")
    parser.add_argument("--verify-observation-max-rotation-delta", type=float, default=None, help="Override PIPE.pose_graph.verify_observation_max_rotation_delta_deg")
    parser.add_argument("--solve-only-max-translation-delta", type=float, default=None, help="Override PIPE.pose_graph.solve_only_max_translation_delta")
    parser.add_argument("--solve-only-max-rotation-delta", type=float, default=None, help="Override PIPE.pose_graph.solve_only_max_rotation_delta_deg")
    parser.add_argument("--apply-max-translation-delta", type=float, default=None, help="Override PIPE.pose_graph.apply_max_translation_delta")
    parser.add_argument("--apply-max-rotation-delta", type=float, default=None, help="Override PIPE.pose_graph.apply_max_rotation_delta_deg")
    parser.add_argument("--apply-max-sequential-translation-residual", type=float, default=None, help="Override PIPE.pose_graph.apply_max_sequential_translation_residual")
    parser.add_argument("--apply-max-sequential-rotation-residual", type=float, default=None, help="Override PIPE.pose_graph.apply_max_sequential_rotation_residual_deg")
    parser.add_argument("--apply-min-translation-delta", type=float, default=None, help="Skip loop-candidate apply updates whose largest proposed submap translation is below this many meters")
    parser.add_argument("--apply-min-loop-constraints", type=int, default=None, help="Skip loop-candidate apply updates with fewer accepted loop constraints")
    parser.add_argument("--apply-min-loop-submap-span", type=int, default=None, help="Skip loop-candidate apply updates unless at least one loop spans this many submaps")
    parser.add_argument("--apply-min-loop-consistent-observations", type=int, default=None, help="Skip loop-candidate apply updates if any accepted loop has fewer consistent observations")
    parser.add_argument("--apply-min-loop-icp-inliers", type=int, default=None, help="Skip loop-candidate apply updates if any accepted loop has fewer ICP inliers")
    parser.add_argument("--apply-max-loop-score", type=float, default=None, help="Skip loop-candidate apply updates if any accepted loop has a higher verification score")
    parser.add_argument("--apply-support-policy", choices=["strict", "adaptive_small_loop", "high_confidence_single_loop"], default=None, help="Support policy for loop-candidate apply gates")
    parser.add_argument("--apply-adaptive-min-loop-constraints", type=int, default=None, help="Adaptive support: minimum loop constraints required to bypass strict support gates")
    parser.add_argument("--apply-adaptive-min-loop-submap-span", type=int, default=None, help="Adaptive support: minimum loop submap span required to bypass strict support gates")
    parser.add_argument("--apply-adaptive-min-loop-consistent-observations", type=int, default=None, help="Adaptive support: minimum consistent observations required to bypass strict support gates")
    parser.add_argument("--apply-adaptive-min-loop-icp-inliers", type=int, default=None, help="Adaptive support: minimum ICP inliers required to bypass strict support gates")
    parser.add_argument("--apply-adaptive-max-loop-score", type=float, default=None, help="Adaptive support: maximum loop verification score allowed to bypass strict support gates")
    parser.add_argument("--apply-high-confidence-min-loop-constraints", type=int, default=None, help="High-confidence support: minimum loop constraints required to bypass strict support gates")
    parser.add_argument("--apply-high-confidence-min-loop-submap-span", type=int, default=None, help="High-confidence support: minimum loop submap span required to bypass strict support gates")
    parser.add_argument("--apply-high-confidence-min-loop-consistent-observations", type=int, default=None, help="High-confidence support: minimum consistent observations required to bypass strict support gates")
    parser.add_argument("--apply-high-confidence-min-loop-icp-inliers", type=int, default=None, help="High-confidence support: minimum ICP inliers required to bypass strict support gates")
    parser.add_argument("--apply-high-confidence-max-loop-score", type=float, default=None, help="High-confidence support: maximum loop verification score allowed to bypass strict support gates")
    parser.add_argument("--apply-high-confidence-max-icp-rmse", type=float, default=None, help="High-confidence support: maximum ICP RMSE in meters allowed to bypass strict support gates")
    parser.add_argument("--apply-max-frame-relative-translation-change", type=float, default=None, help="Skip loop-candidate apply updates if any submap-boundary frame pair changes relative translation by more than this many meters; 0 disables")
    parser.add_argument("--apply-max-frame-relative-rotation-change", type=float, default=None, help="Skip loop-candidate apply updates if any submap-boundary frame pair changes relative rotation by more than this many degrees; 0 disables")
    parser.add_argument(
        "--apply-loop-selection-policy",
        choices=["all", "greedy_sequential_gate", "best_compatible_subset", "pcm_consistency"],
        default=None,
        help="Filter loop constraints before apply by trial-solving bounded subsets",
    )
    parser.add_argument("--apply-loop-selection-max-candidates", type=int, default=None, help="Maximum accepted loop candidates to trial during greedy apply selection; 0 tests all")
    parser.add_argument("--apply-pcm-min-clique-size", type=int, default=None, help="Minimum mutually consistent loop constraints required by pcm_consistency selection")
    parser.add_argument("--apply-pcm-max-trial-cliques", type=int, default=None, help="Maximum PCM cliques to trial-solve after pairwise consistency ranking")
    parser.add_argument("--apply-pcm-event-cluster-radius", type=int, default=None, help="Collapse PCM candidates whose normalized loop endpoints are within this many submaps")
    parser.add_argument("--apply-pcm-min-distinct-events", type=int, default=None, help="Minimum deduplicated revisit events required by pcm_consistency selection")
    parser.add_argument("--apply-pcm-min-long-span", type=int, default=None, help="Require at least one selected PCM loop event to span this many submaps; 0 disables")
    parser.add_argument("--apply-pcm-prefer-long-span", choices=["auto", "true", "false"], default="auto", help="Rank PCM cliques by max loop span before clique size")
    parser.add_argument("--apply-pcm-scale-threshold-by-cycle-length", choices=["auto", "true", "false"], default="auto", help="Scale PCM residual thresholds by sqrt of the pairwise cycle sequential-edge count")
    parser.add_argument("--apply-pcm-translation-threshold", type=float, default=None, help="Maximum pairwise PCM cycle translation residual in meters")
    parser.add_argument("--apply-pcm-rotation-threshold", type=float, default=None, help="Maximum pairwise PCM cycle rotation residual in degrees")
    parser.add_argument("--apply-skip-repeated-loop-pairs", choices=["auto", "true", "false"], default="auto", help="Skip reapplying the same loop pair until the pose graph has grown")
    parser.add_argument("--apply-repeat-min-new-nodes", type=int, default=None, help="Require this many new pose-graph nodes before reusing an already-applied loop pair")
    parser.add_argument("--apply-correction-scale", type=float, default=None, help="Scale accepted pose-graph transform corrections before live application; 1.0 applies the full graph solution")
    parser.add_argument("--apply-scale-backoff", choices=["auto", "true", "false"], default="auto", help="Retry smaller pose-graph correction scales when the final apply proposal fails bounded gates")
    parser.add_argument("--apply-scale-backoff-min-scale", type=float, default=None, help="Smallest correction scale considered by apply-scale backoff")
    parser.add_argument("--apply-scale-backoff-factor", type=float, default=None, help="Multiplicative factor for each apply-scale backoff retry; must be in (0, 1)")
    parser.add_argument("--apply-scale-backoff-max-trials", type=int, default=None, help="Maximum apply-scale trials including the configured scale")
    parser.add_argument("--apply-observation-gate", choices=["auto", "true", "false"], default="auto", help="Reject pose-graph apply proposals that worsen sampled live RGB-D/SDF raycast consistency")
    parser.add_argument("--apply-observation-gate-in-selection", choices=["auto", "true", "false"], default="auto", help="Also run the expensive observation gate during greedy loop selection trials")
    parser.add_argument("--apply-observation-gate-max-samples", type=int, default=None, help="Maximum stored loop observations raycast for the apply observation gate; 0 disables the gate")
    parser.add_argument("--apply-observation-gate-min-samples", type=int, default=None, help="Minimum valid sampled observations required by the apply observation gate")
    parser.add_argument("--apply-observation-gate-sample-policy", choices=["submap_round_robin", "recent", "resident_recent", "loop_aware"], default=None, help="Observation-gate sample order: exact legacy submap round-robin, newest submaps, resident maps before newest submaps, or loop endpoints/correction magnitude before recent maps")
    parser.add_argument("--apply-observation-max-render-maps-per-sample", type=int, default=None, help="Bound TSDF maps rendered per observation-gate sample; 0 keeps normal combined render selection")
    parser.add_argument("--apply-observation-max-pinned-maps", type=int, default=None, help="Bound TSDF maps pinned across an observation-gate before/after comparison; 0 keeps all selected sample maps")
    parser.add_argument("--apply-observation-defer-nonresident", choices=["auto", "true", "false"], default="auto", help="When observation-gate samples need paged-out SDF maps, prefetch and defer the apply check instead of synchronously paging them in")
    parser.add_argument("--apply-observation-defer-retry-frames", type=int, default=None, help="How many subsequent frames may retry a deferred observation-gate apply after prefetching the needed SDF maps")
    parser.add_argument("--apply-observation-pin-deferred-maps", choices=["auto", "true", "false"], default="auto", help="Temporarily protect deferred observation SDF maps from normal residency eviction during the retry window")
    parser.add_argument("--apply-observation-defer-stops-scale-backoff", choices=["auto", "true", "false"], default="auto", help="Treat nonresident observation maps as a retry condition instead of trying smaller correction scales")
    parser.add_argument("--apply-observation-terminal-sync-page-in", choices=["auto", "true", "false"], default="auto", help="For end-of-sequence apply checks, synchronously page observation maps instead of deferring with no future retry window")
    parser.add_argument("--apply-observation-skip-resident-trim", choices=["auto", "true", "false"], default="auto", help="Skip observation-gate residency trimming when all pinned SDF maps are already resident")
    parser.add_argument("--apply-observation-prefetch-loop-endpoints", choices=["auto", "true", "false"], default="auto", help="Asynchronously prefetch SDF maps touched by accepted loop endpoints before observation-gate apply")
    parser.add_argument("--apply-observation-prefetch-max-maps", type=int, default=None, help="Maximum SDF endpoint maps to prefetch for loop-observation gates; 0 uses the observation pin cap or two maps")
    parser.add_argument("--apply-observation-gate-tolerance-policy", choices=["either", "both", "combined", "catastrophe_depth"], default=None, help="Combine observation-gate tolerances, or use a depth-only catastrophe veto for online loop apply")
    parser.add_argument("--apply-observation-max-color-mse-ratio", type=float, default=None, help="Maximum allowed after/before RGB MSE ratio for the apply observation gate")
    parser.add_argument("--apply-observation-max-color-mse-increase", type=float, default=None, help="Maximum allowed absolute RGB MSE increase for the apply observation gate")
    parser.add_argument("--apply-observation-max-depth-rmse-ratio", type=float, default=None, help="Maximum allowed after/before depth RMSE ratio for the apply observation gate")
    parser.add_argument("--apply-observation-max-depth-rmse-increase", type=float, default=None, help="Maximum allowed absolute depth RMSE increase in meters for the apply observation gate")
    parser.add_argument("--apply-observation-depth-rmse-ratio-floor", type=float, default=None, help="Minimum absolute depth RMSE increase required before a catastrophe-depth ratio regression rejects an apply")
    parser.add_argument("--apply-observation-max-depth-rmse-absolute-increase", type=float, default=None, help="Absolute depth RMSE increase that always rejects a catastrophe-depth apply")
    parser.add_argument("--apply-observation-max-valid-fraction-drop", type=float, default=None, help="Maximum allowed fractional drop in valid SDF depth overlap for catastrophe-depth apply gating")
    parser.add_argument("--apply-max-transform-updates-per-frame", type=int, default=None, help="Cap accepted pose-graph submap transform mutations per foreground frame; 0 keeps immediate apply-all behavior")
    parser.add_argument("--apply-health-gate", choices=["auto", "true", "false"], default="auto", help="Skip loop-candidate transform application when raw tracking health is poor")
    parser.add_argument("--apply-health-max-raw-failed-ratio", type=float, default=None, help="Maximum cumulative raw failed tracking ratio allowed before loop apply is observe-only")
    parser.add_argument("--apply-health-max-raw-non-good-ratio", type=float, default=None, help="Maximum cumulative raw non-good tracking ratio allowed before loop apply is observe-only")
    parser.add_argument("--apply-health-max-raw-failed-streak", type=int, default=None, help="Maximum raw failed tracking streak allowed before loop apply is observe-only")
    parser.add_argument("--apply-health-min-frames", type=int, default=None, help="Minimum processed frames before health ratios can block loop apply")
    parser.add_argument("--dry-run-use-estimated-loop-transforms", choices=["auto", "true", "false"], default="auto", help="Override PIPE.pose_graph.dry_run_use_estimated_loop_transforms")
    parser.add_argument("--solve-only-use-estimated-loop-transforms", choices=["auto", "true", "false"], default="auto", help="Override PIPE.pose_graph.solve_only_use_estimated_loop_transforms")
    parser.add_argument("--apply-use-estimated-loop-transforms", choices=["auto", "true", "false"], default="auto", help="Override PIPE.pose_graph.apply_use_estimated_loop_transforms")
    parser.add_argument("--pose-graph-sequential-weight", type=float, default=None, help="Override PIPE.pose_graph.sequential_constraint_weight")
    parser.add_argument("--pose-graph-loop-weight", type=float, default=None, help="Override PIPE.pose_graph.loop_constraint_weight")
    parser.add_argument("--pose-graph-information-policy", choices=["scalar", "normalized_diagonal", "confidence_diagonal"], default=None, help="Override PIPE.pose_graph.information_policy")
    parser.add_argument("--pose-graph-loop-robust-kernel", choices=["none", "huber", "cauchy"], default=None, help="Override PIPE.pose_graph.loop_robust_kernel")
    parser.add_argument("--pose-graph-loop-robust-scale", type=float, default=None, help="Override PIPE.pose_graph.loop_robust_scale")
    parser.add_argument("--pose-graph-loop-information-min-confidence", type=float, default=None, help="Override PIPE.pose_graph.loop_information_min_confidence")
    parser.add_argument("--pose-graph-loop-information-max-confidence", type=float, default=None, help="Override PIPE.pose_graph.loop_information_max_confidence")
    parser.add_argument(
        "--pose-graph-anchor-policy",
        choices=["first", "active"],
        default=None,
        help="Override PIPE.pose_graph.anchor_policy; active keeps the current primary submap fixed during online apply",
    )
    parser.add_argument("--pose-graph-online-updates", action="store_true", help="Run bounded pose graph updates during the live frame loop")
    parser.add_argument("--pose-graph-candidate-interval", type=int, default=None, help="Override PIPE.pose_graph.candidate_interval_frames")
    parser.add_argument(
        "--pose-graph-submap-events",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override PIPE.pose_graph.run_on_submap_events for live pose-graph scheduling",
    )
    parser.add_argument("--pose-graph-disable-submap-events", action="store_true", help="Do not trigger online pose graph updates immediately on submap-count changes")
    parser.add_argument("--pose-graph-online-min-submaps", type=int, default=None, help="Minimum logical submaps before online pose graph updates may run")
    parser.add_argument("--pose-graph-online-min-submap-growth", type=int, default=None, help="Minimum new logical submaps since the previous online update before a submap-event update may run")
    parser.add_argument("--pose-graph-max-updates", type=int, default=None, help="Override PIPE.pose_graph.max_updates_per_run")
    parser.add_argument("--persistent-loop-pool", choices=["auto", "true", "false"], default="auto", help="Retain verified loop constraints across bounded online candidate scans")
    parser.add_argument("--verified-loop-pool-max", type=int, default=None, help="Maximum retained verified loop constraints; 0 disables retention")
    parser.add_argument("--skip-verified-loop-pairs", choices=["auto", "true", "false"], default="auto", help="Skip expensive re-verification for submap pairs already retained in the verified loop pool")
    parser.add_argument("--tag", default="", help="Extra suffix for generated configs and output directories")
    parser.add_argument("--save-mesh", action="store_true", help="Keep TSDF mesh export enabled; disabled by default for TUM ATE/RPE runs")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue the generated run matrix after a failed trainer command and summarize failures at the end")
    parser.add_argument(
        "--trainer-timeout-sec",
        type=float,
        default=0.0,
        help="Abort an individual slam_trainer run after this many seconds. 0 disables the timeout.",
    )
    parser.add_argument("--min-free-gb", type=float, default=40.0, help="Abort before each run if output-root free space is below this threshold")
    parser.add_argument("--largest-output-dirs", type=int, default=8, help="Print this many largest existing output dirs before running; 0 disables")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional post-run CSV summary path; defaults under output-root when --run is used")
    parser.add_argument("--skip-summary", action="store_true", help="Do not run scripts/summarize_tum_runs.py after generated runs finish")
    parser.add_argument(
        "--quiet-run-logs",
        action="store_true",
        help="Redirect trainer and summary stdout/stderr to per-run log files instead of the terminal.",
    )
    parser.add_argument(
        "--run-log-dir",
        type=Path,
        default=None,
        help="Directory for --quiet-run-logs output. Defaults to <output-root>/run_logs.",
    )
    parser.add_argument(
        "--max-estimated-submaps",
        type=int,
        default=0,
        help=(
            "Abort generated MultiEngine configs whose simple frame-count estimate exceeds this many logical "
            "submaps. 0 disables the guard; useful while >32-map stress paths are being hardened."
        ),
    )
    parser.add_argument(
        "--estimated-submap-margin",
        type=int,
        default=4,
        help="Extra submaps added to the frame-count estimate for capacity/visibility-triggered boundaries.",
    )
    parser.add_argument(
        "--max-estimated-cpu-page-cache-gb",
        type=float,
        default=16.0,
        help=(
            "Abort CPU-backed TSDF paging runs whose estimated passive CPU page cache exceeds this many GB. "
            "0 disables the guard."
        ),
    )
    parser.add_argument(
        "--estimated-disk-page-margin-gb",
        type=float,
        default=2.0,
        help="Extra free-space margin kept after estimated disk-backed TSDF pages when --run is used.",
    )
    parser.add_argument(
        "--disable-disk-page-fit-guard",
        action="store_true",
        help="Allow disk-backed TSDF paging runs even when the conservative page-size estimate does not fit.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Generate and optionally run this many repeats per sequence/variant/config")
    parser.add_argument(
        "--repeat-start-index",
        type=int,
        default=1,
        help="First repeat index used in generated config/output suffixes",
    )
    parser.add_argument(
        "--repeat-seed-step",
        type=int,
        default=0,
        help=(
            "Increment PIPE.random_seed by this amount per repeat. The default 0 keeps the same "
            "seed and tests lifecycle nondeterminism rather than seed sensitivity."
        ),
    )
    parser.add_argument("--full-sequence", action="store_true", help="Use each sequence's full prepared frame range instead of --max-frames")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.tracker_config is not None and args.tracking_profile not in {"auto", "legacy"}:
        parser.error("--tracker-config cannot be combined with a non-legacy --tracking-profile")
    try:
        validate_path_token(args.tag, "--tag", allow_empty=True)
        validate_known_labels(args.sequences, set(SEQUENCES), "--sequences")
        validate_known_labels(args.variants, VARIANTS, "--variants")
    except ValueError as exc:
        parser.error(str(exc))

    if args.trainer is not None:
        args.trainer = (
            (PROJECT_ROOT / args.trainer).resolve()
            if not args.trainer.is_absolute()
            else args.trainer.resolve()
        )
    benchmark_role = ""
    benchmark_binary_profile = infer_binary_hash_profile(args.trainer)
    if args.benchmark_protocol != PROTOCOL_NONE:
        if (
            protocol_release_status(args.benchmark_protocol) != "publication_locked"
            and not args.allow_unpromoted_protocol
        ):
            parser.error(
                f"{args.benchmark_protocol} is unpromoted and not paper-eligible; "
                "use --allow-unpromoted-protocol only to reproduce its rejected screen"
            )
        if len(args.variants) != 1:
            parser.error("a locked TUM benchmark requires exactly one --variants value")
        try:
            benchmark_role = tum_role_for_variant(args.variants[0], args.benchmark_protocol)
        except ValueError as exc:
            parser.error(str(exc))
        if args.benchmark_role != "auto" and args.benchmark_role != benchmark_role:
            parser.error(
                f"--benchmark-role={args.benchmark_role} conflicts with "
                f"--variants={args.variants[0]} ({benchmark_role})"
            )
        if not args.full_sequence or args.repeats < 5:
            parser.error("locked TUM paper benchmarks require --full-sequence and --repeats >= 5")
        required_loop_profile = (
            "recommended_longspan_robust_streaming"
            if args.benchmark_protocol == TUM_DEPLOYED_V1
            else "recommended_longspan_robust"
        )
        if benchmark_role in {"modified", "integrated"} and args.loop_profile != required_loop_profile:
            parser.error(
                "the locked proposed-method benchmark requires "
                f"--loop-profile {required_loop_profile}"
            )
        if benchmark_role in {"modified", "integrated"} and args.tsdf_lifecycle_profile != "paper_parity":
            parser.error(
                "the locked proposed-method benchmark requires "
                "--tsdf-lifecycle-profile paper_parity"
            )
        if (
            args.benchmark_protocol == TUM_DEPLOYED_V1
            and args.tsdf_residency_profile != "thor_streaming_bounded"
        ):
            parser.error(
                "the locked deployed benchmark requires "
                "--tsdf-residency-profile thor_streaming_bounded"
            )
        if args.benchmark_protocol == TUM_DEPLOYED_V1:
            if args.tracking_profile != "robust_rpe":
                parser.error(
                    "the locked deployed benchmark requires --tracking-profile robust_rpe"
                )
        elif args.tracking_profile not in {"auto", "legacy"}:
            parser.error("locked comparisons use the same legacy tracker; use --tracking-profile auto")
        if args.run and args.trainer is None:
            parser.error("locked benchmark execution requires an explicit --trainer binary")
        if benchmark_role == "baseline" and benchmark_binary_profile != "legacy_large":
            parser.error(
                "the faithful TUM baseline requires build_legacy/slam_trainer "
                f"(legacy_large); detected {benchmark_binary_profile!r}"
            )
        if args.run:
            try:
                require_max_performance_state()
            except RuntimeError as exc:
                parser.error(str(exc))

    tum_root = args.tum_root
    if tum_root is None:
        tum_root = DEFAULT_TUM_ROOT
    args.output_root = choose_output_root(args.output_root, args.min_free_gb)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    path_storage_summary("tum_root", tum_root)
    output_storage_summary(args.output_root, max(0, args.largest_output_dirs))

    base = base_config(args.template)
    base["PIPE"]["bypass_libtorch_atexit"] = args.bypass_libtorch_atexit == "true"
    if args.tsdf_input_mode is not None:
        base["PIPE"]["TSDF"]["input_mode"] = args.tsdf_input_mode
        base["READER"]["preload_images"] = args.tsdf_input_mode == "preload"
    if args.tsdf_input_prefetch != "default":
        base["PIPE"]["TSDF"]["input_prefetch"] = args.tsdf_input_prefetch == "on"
    base["PIPE"]["TSDF"]["input_tensor_bridge"] = args.tsdf_input_tensor_bridge
    base["PIPE"]["TSDF"]["input_tensor_bridge_validate"] = bool(
        args.tsdf_input_bridge_validate
    )
    if args.submap_stats_interval_frames is not None:
        base["PIPE"]["submap_stats_interval_frames"] = max(
            0, args.submap_stats_interval_frames
        )
    elif args.tsdf_residency_profile == "thor_cpu_bounded":
        base["PIPE"]["submap_stats_interval_frames"] = 10
    if args.basic_nonfinite_recovery:
        base["PIPE"]["TSDF"]["reject_nonfinite_tracking_pose"] = True
    if args.basic_debug_tracking_start_frame is not None:
        base["PIPE"]["TSDF"]["basic_debug_tracking_start_frame"] = max(0, args.basic_debug_tracking_start_frame)
    if args.basic_debug_tracking_end_frame is not None:
        base["PIPE"]["TSDF"]["basic_debug_tracking_end_frame"] = max(0, args.basic_debug_tracking_end_frame)
    if args.eval_after_train:
        base["PIPE"]["eval_after_train"] = True
    if args.eval_split is not None:
        base["PIPE"]["eval_split"] = args.eval_split
    if args.eval_stride is not None:
        base["PIPE"]["eval_stride"] = max(1, args.eval_stride)
    if args.eval_max_frames is not None:
        base["PIPE"]["eval_max_frames"] = max(0, args.eval_max_frames)
    if args.eval_start_offset is not None:
        base["PIPE"]["eval_start_offset"] = max(0, args.eval_start_offset)
    if args.save_mesh:
        base["PIPE"]["TSDF"]["saved_mesh"] = "tsdf_mesh.ply"
    generated = []
    generated_run_dirs = []
    generated_benchmark_metadata: list[tuple[str, dict, Path]] = []
    repeat_count = max(1, args.repeats)
    repeat_start_index = max(0, args.repeat_start_index)
    for seq in args.sequences:
        seq_cfg = configure_sequence(base, tum_root, args.output_root, seq, None if args.full_sequence else args.max_frames)
        for variant in args.variants:
            cfg = apply_loop_profile(configure_variant(seq_cfg, variant), args.loop_profile)
            cfg = apply_tsdf_lifecycle_profile(cfg, args.tsdf_lifecycle_profile)
            disable_inert_online_pose_graph(cfg)
            suffix_parts = ["full" if args.full_sequence else f"{args.max_frames}f"]
            if args.tsdf_input_mode is not None:
                suffix_parts.append(f"input{args.tsdf_input_mode}")
            if args.tsdf_input_prefetch != "default":
                suffix_parts.append(
                    f"prefetch{1 if args.tsdf_input_prefetch == 'on' else 0}"
                )
            if args.tsdf_input_tensor_bridge != "direct":
                suffix_parts.append(f"bridge{args.tsdf_input_tensor_bridge}")
            if args.tsdf_input_bridge_validate:
                suffix_parts.append("bridgevalidate")
            if args.submap_stats_interval_frames is not None:
                suffix_parts.append(f"stats{max(0, args.submap_stats_interval_frames)}")
            if args.basic_nonfinite_recovery and variant.startswith("basic_"):
                suffix_parts.append("basicfinite")
            if (
                variant.startswith("basic_")
                and args.basic_debug_tracking_start_frame is not None
                and args.basic_debug_tracking_end_frame is not None
            ):
                suffix_parts.append(
                    f"bdiag{max(0, args.basic_debug_tracking_start_frame)}to{max(0, args.basic_debug_tracking_end_frame)}"
                )
            if args.loop_profile != "default":
                suffix_parts.append(f"lprof{args.loop_profile}")
            if args.tsdf_lifecycle_profile != "default":
                suffix_parts.append(f"life{args.tsdf_lifecycle_profile}")
            if args.tracking_failure_mode is not None:
                cfg["PIPE"]["TSDF"]["tracking_failure_mode"] = args.tracking_failure_mode
                suffix_parts.append(f"fail{args.tracking_failure_mode.replace('_', '')}")
            if args.tracker_config is not None:
                cfg["PIPE"]["TSDF"]["tracker_config"] = args.tracker_config
                suffix_parts.append("trackercfg")
            if args.tracking_profile != "auto":
                cfg["PIPE"]["TSDF"]["tracking_profile"] = args.tracking_profile
                suffix_parts.append(f"trackprof{args.tracking_profile}")
            if args.use_bilateral_filter != "auto":
                cfg["PIPE"]["TSDF"]["use_bilateral_filter"] = args.use_bilateral_filter == "true"
                suffix_parts.append(f"bilateral{args.use_bilateral_filter}")
            if args.use_approximate_raycast != "auto":
                cfg["PIPE"]["TSDF"]["use_approximate_raycast"] = args.use_approximate_raycast == "true"
                suffix_parts.append(f"approxray{args.use_approximate_raycast}")
            if args.random_seed is not None:
                seed = int(args.random_seed)
                cfg["PIPE"]["random_seed"] = seed
                suffix_parts.append(f"seed{seed}")
            if args.no_save_after_train:
                cfg["PIPE"]["save_after_train"] = False
                suffix_parts.append("nosave")
            if args.keep_recon_save and variant in {"basic_sdf", "multi_sdf", "multi_sdf_dry_solve", "multi_sdf_orb_dry_solve", "multi_sdf_orb_apply"}:
                cfg["PIPE"]["save_after_train"] = True
            if args.eval_after_train:
                suffix_parts.append(f"eval{cfg['PIPE']['eval_split']}")
                eval_stride = int(cfg["PIPE"].get("eval_stride", 1))
                eval_max_frames = int(cfg["PIPE"].get("eval_max_frames", 0))
                eval_start_offset = int(cfg["PIPE"].get("eval_start_offset", 0))
                if eval_stride != 1:
                    suffix_parts.append(f"estr{eval_stride}")
                if eval_max_frames > 0:
                    suffix_parts.append(f"emax{eval_max_frames}")
                if eval_start_offset > 0:
                    suffix_parts.append(f"eoff{eval_start_offset}")
            if args.tsdf_residency_profile != "manual":
                deep_update(
                    cfg["PIPE"]["TSDF"]["multi"],
                    tsdf_residency_profile_defaults(args.tsdf_residency_profile),
                )
                suffix_parts.append(f"rsp{args.tsdf_residency_profile}")
            if args.hash_profile is not None:
                cfg["PIPE"]["TSDF"]["multi"]["hash_profile"] = args.hash_profile
                if args.tsdf_paging == "auto" and args.hash_profile == "legacy_large":
                    cfg["PIPE"]["TSDF"]["multi"]["paging"] = False
                suffix_parts.append(args.hash_profile)
            if args.force_single_submap:
                cfg["PIPE"]["TSDF"]["multi"]["force_single_submap"] = True
                suffix_parts.append("single")
            if args.tsdf_paging != "auto":
                cfg["PIPE"]["TSDF"]["multi"]["paging"] = args.tsdf_paging == "true"
                suffix_parts.append(f"paging{args.tsdf_paging}")
            if args.tsdf_paging_storage is not None:
                cfg["PIPE"]["TSDF"]["multi"]["paging_storage"] = args.tsdf_paging_storage
                suffix_parts.append(f"pgstore{args.tsdf_paging_storage}")
            if args.resident_gpu_maps is not None:
                cfg["PIPE"]["TSDF"]["multi"]["resident_gpu_maps"] = str(args.resident_gpu_maps)
                suffix_parts.append(f"res{str(args.resident_gpu_maps).replace('.', 'p')}")
            if args.tsdf_async_page_out != "auto":
                async_page_out = args.tsdf_async_page_out == "true"
                cfg["PIPE"]["TSDF"]["multi"]["async_page_out"] = async_page_out
                suffix_parts.append(f"asyncpage{int(async_page_out)}")
            if args.tsdf_max_pending_page_snapshots is not None:
                pending = max(0, args.tsdf_max_pending_page_snapshots)
                cfg["PIPE"]["TSDF"]["multi"]["max_pending_page_snapshots"] = pending
                suffix_parts.append(f"pend{pending}")
            if args.tsdf_max_pending_page_snapshot_mb is not None:
                pending_mb = max(0, args.tsdf_max_pending_page_snapshot_mb)
                cfg["PIPE"]["TSDF"]["multi"]["max_pending_page_snapshot_mb"] = pending_mb
                suffix_parts.append(f"pendmb{pending_mb}")
            if args.tsdf_async_page_in != "auto":
                async_page_in = args.tsdf_async_page_in == "true"
                cfg["PIPE"]["TSDF"]["multi"]["async_page_in"] = async_page_in
                suffix_parts.append(f"asyncpin{int(async_page_in)}")
            if args.tsdf_max_pending_page_ins is not None:
                pending = max(0, args.tsdf_max_pending_page_ins)
                cfg["PIPE"]["TSDF"]["multi"]["max_pending_page_ins"] = pending
                suffix_parts.append(f"pin{pending}")
            if args.tsdf_max_pending_page_in_mb is not None:
                pending_mb = max(0, args.tsdf_max_pending_page_in_mb)
                cfg["PIPE"]["TSDF"]["multi"]["max_pending_page_in_mb"] = pending_mb
                suffix_parts.append(f"pinmb{pending_mb}")
            if args.tsdf_cpu_page_cache_limit_mb is not None:
                limit_mb = max(0, args.tsdf_cpu_page_cache_limit_mb)
                cfg["PIPE"]["TSDF"]["multi"]["cpu_page_cache_limit_mb"] = limit_mb
                suffix_parts.append(f"cpucache{limit_mb}")
            if args.tsdf_render_prefetch != "auto":
                render_prefetch = args.tsdf_render_prefetch == "true"
                cfg["PIPE"]["TSDF"]["multi"]["render_prefetch"] = render_prefetch
                suffix_parts.append(f"rpref{int(render_prefetch)}")
            if args.tsdf_render_prefetch_lookahead_frames is not None:
                lookahead_frames = max(0, args.tsdf_render_prefetch_lookahead_frames)
                cfg["PIPE"]["TSDF"]["multi"]["render_prefetch_lookahead_frames"] = lookahead_frames
                suffix_parts.append(f"rlook{lookahead_frames}")
            if args.tsdf_render_prefetch_lookahead_max_maps is not None:
                lookahead_maps = max(0, args.tsdf_render_prefetch_lookahead_max_maps)
                cfg["PIPE"]["TSDF"]["multi"]["render_prefetch_lookahead_max_maps"] = lookahead_maps
                suffix_parts.append(f"rlookmaps{lookahead_maps}")
            if args.tsdf_render_pin_primary != "auto":
                pin_primary = args.tsdf_render_pin_primary == "true"
                cfg["PIPE"]["TSDF"]["multi"]["render_pin_primary_submap"] = pin_primary
                suffix_parts.append(f"pinprimary{int(pin_primary)}")
            if args.tsdf_render_compositor is not None:
                cfg["PIPE"]["TSDF"]["multi"]["render_compositor"] = args.tsdf_render_compositor
                suffix_parts.append(f"tsdfcomp{args.tsdf_render_compositor}")
            if args.tsdf_boundary_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["boundary_policy"] = args.tsdf_boundary_policy
                suffix_parts.append(f"bp{args.tsdf_boundary_policy}")
            if args.allow_visibility_handoff != "auto":
                allow_visibility = args.allow_visibility_handoff == "true"
                cfg["PIPE"]["TSDF"]["multi"]["allow_visibility_handoff"] = allow_visibility
                suffix_parts.append(f"vishandoff{int(allow_visibility)}")
            if args.tsdf_occupancy_threshold_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["occupancy_threshold_policy"] = args.tsdf_occupancy_threshold_policy
                suffix_parts.append(f"occpol{args.tsdf_occupancy_threshold_policy.replace('_', '')}")
            if args.tsdf_adaptive_new_map_occupancy is not None:
                value = max(0.0, min(1.0, args.tsdf_adaptive_new_map_occupancy))
                cfg["PIPE"]["TSDF"]["multi"]["adaptive_new_map_block_occupancy"] = value
                suffix_parts.append(f"adnocc{str(value).replace('.', 'p')}")
            if args.tsdf_adaptive_max_occupancy is not None:
                value = max(0.0, min(1.0, args.tsdf_adaptive_max_occupancy))
                cfg["PIPE"]["TSDF"]["multi"]["adaptive_max_block_occupancy"] = value
                suffix_parts.append(f"admocc{str(value).replace('.', 'p')}")
            if args.tsdf_adaptive_occupancy_min_primary_frames is not None:
                frames = max(0, args.tsdf_adaptive_occupancy_min_primary_frames)
                cfg["PIPE"]["TSDF"]["multi"]["adaptive_occupancy_min_primary_frames"] = frames
                suffix_parts.append(f"adpmin{frames}")
            if args.tsdf_adaptive_occupancy_min_processed_frames is not None:
                frames = max(0, args.tsdf_adaptive_occupancy_min_processed_frames)
                cfg["PIPE"]["TSDF"]["multi"]["adaptive_occupancy_min_processed_frames"] = frames
                suffix_parts.append(f"adrunmin{frames}")
            if args.tsdf_adaptive_occupancy_max_raw_weak_streak is not None:
                frames = max(0, args.tsdf_adaptive_occupancy_max_raw_weak_streak)
                cfg["PIPE"]["TSDF"]["multi"]["adaptive_occupancy_max_raw_weak_streak"] = frames
                suffix_parts.append(f"adweak{frames}")
            if args.tsdf_adaptive_occupancy_max_raw_failed_streak is not None:
                frames = max(0, args.tsdf_adaptive_occupancy_max_raw_failed_streak)
                cfg["PIPE"]["TSDF"]["multi"]["adaptive_occupancy_max_raw_failed_streak"] = frames
                suffix_parts.append(f"adfail{frames}")
            if args.sequential_transform_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["sequential_transform_policy"] = args.sequential_transform_policy
                suffix_parts.append(f"seqtf{args.sequential_transform_policy}")
            if args.sequential_transform_max_translation is not None:
                value = max(0.0, args.sequential_transform_max_translation)
                cfg["PIPE"]["TSDF"]["multi"]["sequential_transform_max_translation"] = value
                suffix_parts.append(f"seqt{str(value).replace('.', 'p')}")
            if args.sequential_transform_max_rotation is not None:
                value = max(0.0, args.sequential_transform_max_rotation)
                cfg["PIPE"]["TSDF"]["multi"]["sequential_transform_max_rotation_deg"] = value
                suffix_parts.append(f"seqr{str(value).replace('.', 'p')}")
            if args.invalid_transform_restart_attempts is not None:
                attempts = max(0, args.invalid_transform_restart_attempts)
                cfg["PIPE"]["TSDF"]["multi"]["invalid_transform_restart_attempts"] = attempts
                suffix_parts.append(f"invr{attempts}")
            if args.handoff_pose_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["handoff_pose_policy"] = args.handoff_pose_policy
                suffix_parts.append(f"handoffpose{args.handoff_pose_policy}")
            if args.new_map_handoff_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["new_map_handoff_policy"] = args.new_map_handoff_policy
                suffix_parts.append(f"newhandoff{args.new_map_handoff_policy}")
            if args.new_map_handoff_min_visible_ratio is not None:
                visible_ratio = max(0.0, min(1.0, args.new_map_handoff_min_visible_ratio))
                cfg["PIPE"]["TSDF"]["multi"]["new_map_handoff_min_visible_ratio"] = visible_ratio
                suffix_parts.append(f"hmv{str(visible_ratio).replace('.', 'p')}")
            if args.new_map_handoff_min_visible_margin is not None:
                margin = max(0.0, min(1.0, args.new_map_handoff_min_visible_margin))
                cfg["PIPE"]["TSDF"]["multi"]["new_map_handoff_min_visible_margin"] = margin
                suffix_parts.append(f"hmm{str(margin).replace('.', 'p')}")
            if args.new_map_handoff_min_accepts is not None:
                accepts = max(0, args.new_map_handoff_min_accepts)
                cfg["PIPE"]["TSDF"]["multi"]["new_map_handoff_min_accepts"] = accepts
            if args.new_map_handoff_emergency_min_accepts is not None:
                accepts = max(-1, args.new_map_handoff_emergency_min_accepts)
                cfg["PIPE"]["TSDF"]["multi"]["new_map_handoff_emergency_min_accepts"] = accepts
                suffix_parts.append(f"ehacc{accepts}")
            if args.candidate_recovery_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["candidate_recovery_policy"] = args.candidate_recovery_policy
                suffix_parts.append(f"crec{args.candidate_recovery_policy}")
            if args.candidate_recovery_min_visible_ratio is not None:
                visible_ratio = max(0.0, min(1.0, args.candidate_recovery_min_visible_ratio))
                cfg["PIPE"]["TSDF"]["multi"]["candidate_recovery_min_visible_ratio"] = visible_ratio
                suffix_parts.append(f"crecv{str(visible_ratio).replace('.', 'p')}")
            if args.candidate_recovery_min_blocks is not None:
                blocks = max(0, args.candidate_recovery_min_blocks)
                cfg["PIPE"]["TSDF"]["multi"]["candidate_recovery_min_blocks"] = blocks
                suffix_parts.append(f"crecb{blocks}")
            if args.candidate_recovery_max_attempts is not None:
                attempts = max(0, args.candidate_recovery_max_attempts)
                cfg["PIPE"]["TSDF"]["multi"]["candidate_recovery_max_attempts"] = attempts
                suffix_parts.append(f"creca{attempts}")
            if args.primary_failure_handoff_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_policy"] = args.primary_failure_handoff_policy
                suffix_parts.append(f"pfh{args.primary_failure_handoff_policy}")
            if args.primary_failure_handoff_min_raw_failed_frames is not None:
                frames = max(1, args.primary_failure_handoff_min_raw_failed_frames)
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_min_raw_failed_frames"] = frames
                suffix_parts.append(f"pfhrf{frames}")
            if args.primary_failure_handoff_raw_gate is not None:
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_raw_gate"] = args.primary_failure_handoff_raw_gate
                suffix_parts.append(f"pfhgate{args.primary_failure_handoff_raw_gate.replace('_', '')}")
            if args.primary_failure_handoff_min_raw_weak_frames is not None:
                frames = max(0, args.primary_failure_handoff_min_raw_weak_frames)
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_min_raw_weak_frames"] = frames
                suffix_parts.append(f"pfhrw{frames}")
            if args.primary_failure_handoff_min_candidate_visible_ratio is not None:
                visible_ratio = max(0.0, min(1.0, args.primary_failure_handoff_min_candidate_visible_ratio))
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_min_candidate_visible_ratio"] = visible_ratio
                suffix_parts.append(f"pfhv{str(visible_ratio).replace('.', 'p')}")
            if args.primary_failure_handoff_min_candidate_blocks is not None:
                blocks = max(0, args.primary_failure_handoff_min_candidate_blocks)
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_min_candidate_blocks"] = blocks
                suffix_parts.append(f"pfhb{blocks}")
            if args.primary_failure_handoff_min_candidate_good_frames is not None:
                frames = max(1, args.primary_failure_handoff_min_candidate_good_frames)
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_min_candidate_good_frames"] = frames
                suffix_parts.append(f"pfhgood{frames}")
            if args.primary_failure_handoff_min_overlap_frames is not None:
                frames = max(0, args.primary_failure_handoff_min_overlap_frames)
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_min_overlap_frames"] = frames
                suffix_parts.append(f"pfhov{frames}")
            if args.primary_failure_handoff_require_pose_continuity != "auto":
                enabled = args.primary_failure_handoff_require_pose_continuity == "true"
                cfg["PIPE"]["TSDF"]["multi"]["primary_failure_handoff_require_pose_continuity"] = enabled
                suffix_parts.append(f"pfhpose{int(enabled)}")
            if args.tracking_boundary_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["tracking_boundary_policy"] = args.tracking_boundary_policy
                suffix_parts.append(f"tb{args.tracking_boundary_policy}")
            if args.raw_weak_boundary_frames is not None:
                frames = max(1, args.raw_weak_boundary_frames)
                cfg["PIPE"]["TSDF"]["multi"]["raw_weak_boundary_frames"] = frames
                suffix_parts.append(f"rw{frames}")
            if args.raw_failed_boundary_frames is not None:
                frames = max(1, args.raw_failed_boundary_frames)
                cfg["PIPE"]["TSDF"]["multi"]["raw_failed_boundary_frames"] = frames
                suffix_parts.append(f"rf{frames}")
            if args.tracking_boundary_min_primary_frames is not None:
                frames = max(0, args.tracking_boundary_min_primary_frames)
                cfg["PIPE"]["TSDF"]["multi"]["tracking_boundary_min_primary_frames"] = frames
                suffix_parts.append(f"tbmin{frames}")
            if args.tracking_boundary_cooldown_frames is not None:
                frames = max(0, args.tracking_boundary_cooldown_frames)
                cfg["PIPE"]["TSDF"]["multi"]["tracking_boundary_cooldown_frames"] = frames
                suffix_parts.append(f"tbcool{frames}")
            if args.tracking_boundary_force_handoff != "auto":
                enabled = args.tracking_boundary_force_handoff == "true"
                cfg["PIPE"]["TSDF"]["multi"]["tracking_boundary_force_handoff"] = enabled
                suffix_parts.append(f"tbhoff{int(enabled)}")
            if args.tracking_boundary_handoff_policy is not None:
                cfg["PIPE"]["TSDF"]["multi"]["tracking_boundary_handoff_policy"] = args.tracking_boundary_handoff_policy
                suffix_parts.append(f"tbhp{args.tracking_boundary_handoff_policy.replace('_', '')}")
            if args.pose_graph_overlap_roundtrip_check != "auto":
                enabled = args.pose_graph_overlap_roundtrip_check == "true"
                cfg["PIPE"]["gaussian_submaps"]["pose_graph_overlap_roundtrip_check"] = enabled
                suffix_parts.append(f"pgrt{int(enabled)}")
            if args.pose_graph_overlap_roundtrip_tolerance is not None:
                tolerance = max(0.0, args.pose_graph_overlap_roundtrip_tolerance)
                cfg["PIPE"]["gaussian_submaps"]["pose_graph_overlap_roundtrip_tolerance"] = tolerance
                suffix_parts.append(f"pgrttol{tolerance:g}")
            if args.require_raw_tracking_success != "auto":
                enabled = args.require_raw_tracking_success == "true"
                cfg["PIPE"]["TSDF"]["require_raw_tracking_success"] = enabled
                suffix_parts.append(f"rawgate{int(enabled)}")
            if args.raw_tracking_failure_grace_frames is not None:
                frames = max(0, args.raw_tracking_failure_grace_frames)
                cfg["PIPE"]["TSDF"]["raw_tracking_failure_grace_frames"] = frames
                suffix_parts.append(f"rawgrace{frames}")
            if args.tsdf_max_frames is not None:
                cfg["PIPE"]["TSDF"]["multi"]["max_frames"] = max(1, args.tsdf_max_frames)
                suffix_parts.append(f"tsdf{max(1, args.tsdf_max_frames)}")
            if args.tsdf_overlap_frames is not None:
                frames = max(1, args.tsdf_overlap_frames)
                cfg["PIPE"]["TSDF"]["multi"]["overlap_frames"] = frames
                suffix_parts.append(f"ov{frames}")
            if args.tsdf_trial_frames is not None:
                frames = max(1, args.tsdf_trial_frames)
                cfg["PIPE"]["TSDF"]["multi"]["trial_frames"] = frames
                suffix_parts.append(f"trial{frames}")
            if args.tsdf_emergency_overlap_frames is not None:
                frames = max(0, args.tsdf_emergency_overlap_frames)
                cfg["PIPE"]["TSDF"]["multi"]["emergency_overlap_frames"] = frames
                suffix_parts.append(f"emov{frames}")
            if args.tsdf_new_map_occupancy is not None:
                new_occ = max(0.0, min(1.0, args.tsdf_new_map_occupancy))
                cfg["PIPE"]["TSDF"]["multi"]["new_map_block_occupancy"] = new_occ
                suffix_parts.append(f"nocc{str(new_occ).replace('.', 'p')}")
            if args.tsdf_max_occupancy is not None:
                max_occ = max(0.0, min(1.0, args.tsdf_max_occupancy))
                cfg["PIPE"]["TSDF"]["multi"]["max_block_occupancy"] = max_occ
                suffix_parts.append(f"mocc{str(max_occ).replace('.', 'p')}")
            if args.tsdf_fusion_reserve_blocks is not None:
                reserve_blocks = max(0, args.tsdf_fusion_reserve_blocks)
                cfg["PIPE"]["TSDF"]["multi"]["fusion_reserve_blocks"] = reserve_blocks
                suffix_parts.append(f"reserve{reserve_blocks}")
            if args.tsdf_occupancy_growth_lookahead_frames is not None:
                lookahead_frames = max(0, args.tsdf_occupancy_growth_lookahead_frames)
                cfg["PIPE"]["TSDF"]["multi"]["occupancy_growth_lookahead_frames"] = lookahead_frames
                suffix_parts.append(f"og{lookahead_frames}")
            if args.tsdf_occupancy_growth_min_primary_frames is not None:
                min_primary_frames = max(0, args.tsdf_occupancy_growth_min_primary_frames)
                cfg["PIPE"]["TSDF"]["multi"]["occupancy_growth_min_primary_frames"] = min_primary_frames
                suffix_parts.append(f"ogmin{min_primary_frames}")
            if args.tsdf_occupancy_growth_sample_frames is not None:
                sample_frames = max(1, args.tsdf_occupancy_growth_sample_frames)
                cfg["PIPE"]["TSDF"]["multi"]["occupancy_growth_sample_frames"] = sample_frames
                suffix_parts.append(f"ogs{sample_frames}")
            if args.skip_primary_fusion_on_capacity_handoff != "auto":
                enabled = args.skip_primary_fusion_on_capacity_handoff == "true"
                cfg["PIPE"]["TSDF"]["multi"]["skip_primary_fusion_on_capacity_handoff"] = enabled
                suffix_parts.append(f"skipcap{int(enabled)}")
            if variant_uses_gaussians(variant):
                if args.new_gs_sample_ratio is not None:
                    ratio = max(0.0, args.new_gs_sample_ratio)
                    cfg["PIPE"]["new_gs_sample_ratio"] = ratio
                    suffix_parts.append(f"gsr{suffix_value(ratio)}")
                if args.local_opt_iters is not None:
                    iters = max(0, args.local_opt_iters)
                    cfg["PIPE"]["local_opt_iters"] = iters
                    suffix_parts.append(f"opt{iters}")
                if args.local_opt_interval is not None:
                    interval = max(1, args.local_opt_interval)
                    cfg["PIPE"]["local_opt_interval"] = interval
                    suffix_parts.append(f"opint{interval}")
                if args.gaussian_schedule != "auto":
                    schedule = cfg["PIPE"].setdefault("gaussian_schedule", {})
                    schedule["mode"] = args.gaussian_schedule
                    suffix_parts.append(f"gsch{args.gaussian_schedule}")
                if args.gaussian_opt_iters_per_frame is not None:
                    per_frame = max(1, args.gaussian_opt_iters_per_frame)
                    schedule = cfg["PIPE"].setdefault("gaussian_schedule", {})
                    schedule["iterations_per_frame"] = per_frame
                    suffix_parts.append(f"gspf{per_frame}")
                if args.keyframe_select_max is not None:
                    keyframes = max(0, args.keyframe_select_max)
                    cfg["PIPE"]["keyframe_select_max"] = keyframes
                    suffix_parts.append(f"kf{keyframes}")
                if args.submap_keyframe_selection_policy != "auto":
                    cfg["PIPE"]["gaussian_submaps"]["keyframe_selection_policy"] = args.submap_keyframe_selection_policy
                    suffix_parts.append(f"kfsel{args.submap_keyframe_selection_policy.replace('_', '')}")
                if args.model_sh_degree is not None:
                    degree = max(0, args.model_sh_degree)
                    cfg["MODEL"]["sh_degree"] = degree
                    suffix_parts.append(f"sh{degree}")
            if args.export_online_end_trajectory != "auto":
                export_online = args.export_online_end_trajectory == "true"
                cfg["PIPE"]["export_online_end_trajectory"] = export_online
                suffix_parts.append(f"onlineend{int(export_online)}")
            if args.ges_residual_scale is not None:
                residual_scale = max(0.0, args.ges_residual_scale)
                cfg["MODEL"]["ges_residual_scale"] = residual_scale
                suffix_parts.append(f"gres{str(residual_scale).replace('.', 'p')}")
            if args.ges_base_color_weight is not None:
                base_weight = max(1e-6, args.ges_base_color_weight)
                cfg["MODEL"]["ges_base_color_weight"] = base_weight
                suffix_parts.append(f"gbw{str(base_weight).replace('.', 'p')}")
            if args.gaussian_max_frames is not None:
                cfg["PIPE"]["gaussian_submaps"]["max_frames"] = max(1, args.gaussian_max_frames)
                suffix_parts.append(f"gs{max(1, args.gaussian_max_frames)}")
            if args.gaussian_max_gaussians is not None:
                cfg["PIPE"]["gaussian_submaps"]["max_gaussians"] = max(1, args.gaussian_max_gaussians)
                suffix_parts.append(f"gcap{max(1, args.gaussian_max_gaussians)}")
            if args.gaussian_max_new_per_frame is not None:
                cfg["PIPE"]["gaussian_submaps"]["max_new_gaussians_per_frame"] = max(0, args.gaussian_max_new_per_frame)
                suffix_parts.append(f"gnew{max(0, args.gaussian_max_new_per_frame)}")
            if args.page_frozen_gaussians != "auto":
                page_frozen = args.page_frozen_gaussians == "true"
                cfg["PIPE"]["gaussian_submaps"]["page_frozen_gaussians"] = page_frozen
                suffix_parts.append(f"gpage{int(page_frozen)}")
            if args.resident_gaussian_submaps is not None:
                resident_gaussians = max(1, args.resident_gaussian_submaps)
                cfg["PIPE"]["gaussian_submaps"]["resident_gaussian_submaps"] = resident_gaussians
                suffix_parts.append(f"gres{resident_gaussians}")
            if args.gaussian_insert_selection != "auto":
                cfg["PIPE"]["gaussian_submaps"]["gaussian_insert_selection"] = args.gaussian_insert_selection
                suffix_parts.append(f"gsel{args.gaussian_insert_selection.replace('_', '')}")
            if args.gaussian_insert_tile_size is not None:
                tile_size = max(1, args.gaussian_insert_tile_size)
                cfg["PIPE"]["gaussian_submaps"]["gaussian_insert_tile_size"] = tile_size
                suffix_parts.append(f"gts{tile_size}")
            if args.gaussian_insert_tile_max_pixels is not None:
                tile_max = max(0, args.gaussian_insert_tile_max_pixels)
                cfg["PIPE"]["gaussian_submaps"]["gaussian_insert_tile_max_pixels"] = tile_max
                suffix_parts.append(f"gtmax{tile_max}")
            if args.gaussian_insert_min_age_frames is not None:
                age = max(0, args.gaussian_insert_min_age_frames)
                cfg["PIPE"]["gaussian_submaps"]["gaussian_insert_min_age_frames"] = age
                suffix_parts.append(f"gminage{age}")
            if args.gaussian_insert_min_sdf_blocks is not None:
                blocks = max(0, args.gaussian_insert_min_sdf_blocks)
                cfg["PIPE"]["gaussian_submaps"]["gaussian_insert_min_sdf_blocks"] = blocks
                suffix_parts.append(f"gminblk{blocks}")
            if args.candidate_reject_retire != "auto":
                reject_retire = args.candidate_reject_retire == "true"
                cfg["PIPE"]["gaussian_submaps"]["candidate_reject_retire"] = reject_retire
                suffix_parts.append(f"crejret{int(reject_retire)}")
            if args.candidate_reject_retire_min_rejects is not None:
                rejects = max(1, args.candidate_reject_retire_min_rejects)
                cfg["PIPE"]["gaussian_submaps"]["candidate_reject_retire_min_rejects"] = rejects
                suffix_parts.append(f"crej{rejects}")
            if args.candidate_reject_retire_min_observed_frames is not None:
                observed = max(0, args.candidate_reject_retire_min_observed_frames)
                cfg["PIPE"]["gaussian_submaps"]["candidate_reject_retire_min_observed_frames"] = observed
                suffix_parts.append(f"crejobs{observed}")
            if args.candidate_reject_retire_min_gaussians is not None:
                min_gs = max(0, args.candidate_reject_retire_min_gaussians)
                cfg["PIPE"]["gaussian_submaps"]["candidate_reject_retire_min_gaussians"] = min_gs
                suffix_parts.append(f"crejgs{min_gs}")
            if args.candidate_reject_retire_max_inliers is not None:
                max_inliers = max(0, args.candidate_reject_retire_max_inliers)
                cfg["PIPE"]["gaussian_submaps"]["candidate_reject_retire_max_inliers"] = max_inliers
                suffix_parts.append(f"crejinl{max_inliers}")
            if args.candidate_reject_retire_cooldown_frames is not None:
                cooldown = max(0, args.candidate_reject_retire_cooldown_frames)
                cfg["PIPE"]["gaussian_submaps"]["candidate_reject_retire_cooldown_frames"] = cooldown
                suffix_parts.append(f"crejcool{cooldown}")
            if args.candidate_gaussian_insert_min_frames is not None:
                candidate_frames = max(0, args.candidate_gaussian_insert_min_frames)
                cfg["PIPE"]["gaussian_submaps"]["candidate_gaussian_insert_min_frames"] = candidate_frames
                suffix_parts.append(f"cgmin{candidate_frames}")
            if args.candidate_immature_new_gs_sample_ratio is not None:
                immature_ratio = max(-1.0, args.candidate_immature_new_gs_sample_ratio)
                cfg["PIPE"]["gaussian_submaps"]["candidate_immature_new_gs_sample_ratio"] = immature_ratio
                suffix_parts.append(f"cimgr{suffix_value(immature_ratio)}")
            if args.candidate_immature_max_new_per_frame is not None:
                immature_cap = max(0, args.candidate_immature_max_new_per_frame)
                cfg["PIPE"]["gaussian_submaps"]["candidate_immature_max_new_gaussians_per_frame"] = immature_cap
                suffix_parts.append(f"cimmax{immature_cap}")
            if args.candidate_extend_while_overlap != "auto":
                extend = args.candidate_extend_while_overlap == "true"
                cfg["PIPE"]["gaussian_submaps"]["candidate_extend_while_overlap"] = extend
                suffix_parts.append(f"cext{int(extend)}")
            if args.candidate_max_warm_frames is not None:
                max_warm = max(0, args.candidate_max_warm_frames)
                cfg["PIPE"]["gaussian_submaps"]["candidate_max_warm_frames"] = max_warm
                suffix_parts.append(f"cmax{max_warm}")
            if args.candidate_handoff_max_opt_passes is not None:
                candidate_passes = max(-1, args.candidate_handoff_max_opt_passes)
                cfg["PIPE"]["gaussian_submaps"]["candidate_handoff_max_opt_passes"] = candidate_passes
                suffix_parts.append(f"chop{candidate_passes}")
            if args.handoff_frames is not None:
                frames = max(0, args.handoff_frames)
                cfg["PIPE"]["gaussian_submaps"]["handoff_frames"] = frames
                suffix_parts.append(f"hof{frames}")
            if args.handoff_opt_iters is not None:
                iters = max(0, args.handoff_opt_iters)
                cfg["PIPE"]["gaussian_submaps"]["handoff_opt_iters"] = iters
                suffix_parts.append(f"hoi{iters}")
            if args.handoff_max_opt_passes is not None:
                passes = max(0, args.handoff_max_opt_passes)
                cfg["PIPE"]["gaussian_submaps"]["handoff_max_opt_passes"] = passes
                suffix_parts.append(f"hop{passes}")
            if args.handoff_add_outgoing_gaussians != "auto":
                add_outgoing = args.handoff_add_outgoing_gaussians == "true"
                cfg["PIPE"]["gaussian_submaps"]["handoff_add_gaussians_to_outgoing"] = add_outgoing
                suffix_parts.append(f"hoadd{int(add_outgoing)}")
            if args.graduation_optimize != "auto":
                graduation = args.graduation_optimize == "true"
                cfg["PIPE"]["gaussian_submaps"]["graduation_optimize"] = graduation
                suffix_parts.append(f"grad{int(graduation)}")
            if args.graduation_opt_iters is not None:
                grad_iters = max(0, args.graduation_opt_iters)
                cfg["PIPE"]["gaussian_submaps"]["graduation_opt_iters"] = grad_iters
                suffix_parts.append(f"grediters{grad_iters}")
            if args.graduation_keyframes is not None:
                grad_keyframes = max(0, args.graduation_keyframes)
                cfg["PIPE"]["gaussian_submaps"]["graduation_keyframes"] = grad_keyframes
                suffix_parts.append(f"gradkf{grad_keyframes}")
            if args.graduation_max_opt_passes is not None:
                grad_passes = max(0, args.graduation_max_opt_passes)
                cfg["PIPE"]["gaussian_submaps"]["graduation_max_opt_passes"] = grad_passes
                suffix_parts.append(f"gradpasses{grad_passes}")
            if args.graduation_keyframe_policy != "auto":
                cfg["PIPE"]["gaussian_submaps"]["graduation_keyframe_policy"] = args.graduation_keyframe_policy
                suffix_parts.append(f"gradpol{args.graduation_keyframe_policy.replace('_', '')}")
            if args.candidate_warm_interval is not None:
                interval = max(1, args.candidate_warm_interval)
                cfg["PIPE"]["gaussian_submaps"]["candidate_warm_interval"] = interval
                suffix_parts.append(f"cwi{interval}")
            if args.candidate_warming != "auto":
                enabled = args.candidate_warming == "true"
                cfg["PIPE"]["gaussian_submaps"]["candidate_warming"] = enabled
                suffix_parts.append(f"cw{int(enabled)}")
            if args.boundary_observation_handoff != "auto":
                enabled = args.boundary_observation_handoff == "true"
                cfg["PIPE"]["gaussian_submaps"]["boundary_observation_handoff"] = enabled
                suffix_parts.append(f"boh{int(enabled)}")
            if args.render_policy is not None:
                cfg["PIPE"]["gaussian_submaps"]["render_policy"] = args.render_policy
                suffix_parts.append(f"rp{args.render_policy}")
            if args.render_max_visible_gaussians is not None:
                max_visible_gs = max(0, args.render_max_visible_gaussians)
                cfg["PIPE"]["gaussian_submaps"]["render_max_visible_gaussians"] = max_visible_gs
                suffix_parts.append(f"rgs{max_visible_gs}")
            if args.render_include_handoff_submaps != "auto":
                include_handoff = args.render_include_handoff_submaps == "true"
                cfg["PIPE"]["gaussian_submaps"]["render_include_handoff_submaps"] = include_handoff
                suffix_parts.append(f"rhandoff{int(include_handoff)}")
            if args.render_single_submap_passthrough != "auto":
                passthrough = args.render_single_submap_passthrough == "true"
                cfg["PIPE"]["gaussian_submaps"]["render_single_submap_passthrough"] = passthrough
                suffix_parts.append(f"rsingle{int(passthrough)}")
            if args.allow_multi_logical_single_passthrough:
                cfg["PIPE"]["gaussian_submaps"]["render_single_submap_passthrough_requires_single_logical_map"] = False
                suffix_parts.append("rsinglemulti")
            if args.require_single_logical_single_passthrough:
                cfg["PIPE"]["gaussian_submaps"]["render_single_submap_passthrough_requires_single_logical_map"] = True
                suffix_parts.append("rsinglesafe")
            if args.disable_local_sdf_reference_render:
                cfg["PIPE"]["gaussian_submaps"]["render_use_local_sdf_reference"] = False
                suffix_parts.append("globref")
            if args.render_fill_require_sdf_hole != "auto":
                require_hole = args.render_fill_require_sdf_hole == "true"
                cfg["PIPE"]["gaussian_submaps"]["render_fill_require_sdf_hole"] = require_hole
                suffix_parts.append(f"rhole{int(require_hole)}")
            if args.render_fill_blend_weight is not None:
                blend = min(1.0, max(0.0, args.render_fill_blend_weight))
                cfg["PIPE"]["gaussian_submaps"]["render_fill_blend_weight"] = blend
                suffix_parts.append(f"rfbw{str(blend).replace('.', 'p')}")
            if args.render_owner_alpha_fill_thres is not None:
                alpha = max(0.0, args.render_owner_alpha_fill_thres)
                cfg["PIPE"]["gaussian_submaps"]["render_owner_alpha_fill_thres"] = alpha
                suffix_parts.append(f"roa{str(alpha).replace('.', 'p')}")
            if args.render_min_owner_age_frames is not None:
                min_age = max(0, args.render_min_owner_age_frames)
                cfg["PIPE"]["gaussian_submaps"]["render_min_owner_age_frames"] = min_age
                suffix_parts.append(f"rminage{min_age}")
            if args.render_min_ready_frames is not None:
                min_ready = max(0, args.render_min_ready_frames)
                cfg["PIPE"]["gaussian_submaps"]["render_min_ready_frames"] = min_ready
                suffix_parts.append(f"rready{min_ready}")
            if args.render_owner_min_ready_frames is not None:
                min_ready = max(0, args.render_owner_min_ready_frames)
                cfg["PIPE"]["gaussian_submaps"]["render_owner_min_ready_frames"] = min_ready
                suffix_parts.append(f"rownerready{min_ready}")
            if args.render_neighbor_min_ready_frames is not None:
                min_ready = max(0, args.render_neighbor_min_ready_frames)
                cfg["PIPE"]["gaussian_submaps"]["render_neighbor_min_ready_frames"] = min_ready
                suffix_parts.append(f"rneighready{min_ready}")
            if args.render_past_frame_margin is not None:
                margin = max(0, args.render_past_frame_margin)
                cfg["PIPE"]["gaussian_submaps"]["render_past_frame_margin"] = margin
                suffix_parts.append(f"rpast{margin}")
            if args.render_future_frame_margin is not None:
                margin = max(0, args.render_future_frame_margin)
                cfg["PIPE"]["gaussian_submaps"]["render_future_frame_margin"] = margin
                suffix_parts.append(f"rfuture{margin}")
            if args.render_min_neighbor_gaussians is not None:
                min_gs = max(0, args.render_min_neighbor_gaussians)
                cfg["PIPE"]["gaussian_submaps"]["render_min_neighbor_gaussians"] = min_gs
                suffix_parts.append(f"rmng{min_gs}")
            if args.render_blend_alpha_thres is not None:
                alpha = max(0.0, args.render_blend_alpha_thres)
                cfg["PIPE"]["gaussian_submaps"]["render_blend_alpha_thres"] = alpha
                suffix_parts.append(f"rba{str(alpha).replace('.', 'p')}")
            if args.render_blend_depth_thres is not None:
                depth = max(0.0, args.render_blend_depth_thres)
                cfg["PIPE"]["gaussian_submaps"]["render_blend_depth_thres"] = depth
                suffix_parts.append(f"rbd{str(depth).replace('.', 'p')}")
            if args.render_blend_max_weight is not None:
                weight = min(1.0, max(0.0, args.render_blend_max_weight))
                cfg["PIPE"]["gaussian_submaps"]["render_blend_max_weight"] = weight
                suffix_parts.append(f"rbmw{str(weight).replace('.', 'p')}")
            if args.render_blend_weight_scale is not None:
                weight = max(0.0, args.render_blend_weight_scale)
                cfg["PIPE"]["gaussian_submaps"]["render_blend_weight_scale"] = weight
                suffix_parts.append(f"rbws{str(weight).replace('.', 'p')}")
            if args.render_neighbor_alpha_fill_thres is not None:
                alpha = max(0.0, args.render_neighbor_alpha_fill_thres)
                cfg["PIPE"]["gaussian_submaps"]["render_neighbor_alpha_fill_thres"] = alpha
                suffix_parts.append(f"rna{str(alpha).replace('.', 'p')}")
            if args.loop_min_separation is not None:
                cfg["PIPE"]["pose_graph"]["loop_min_submap_separation"] = max(1, args.loop_min_separation)
                suffix_parts.append(f"sep{max(1, args.loop_min_separation)}")
            if args.loop_max_distance is not None:
                cfg["PIPE"]["pose_graph"]["loop_max_distance"] = max(0.0, args.loop_max_distance)
                suffix_parts.append(f"ld{str(max(0.0, args.loop_max_distance)).replace('.', 'p')}")
            if args.loop_max_rotation is not None:
                cfg["PIPE"]["pose_graph"]["loop_max_rotation_deg"] = max(0.0, args.loop_max_rotation)
                suffix_parts.append(f"lr{str(max(0.0, args.loop_max_rotation)).replace('.', 'p')}")
            if args.loop_max_candidates is not None:
                loop_max_candidates = max(0, args.loop_max_candidates)
                cfg["PIPE"]["pose_graph"]["loop_max_candidates"] = loop_max_candidates
                suffix_parts.append(f"lmax{loop_max_candidates}")
            if args.loop_candidate_export_policy is not None:
                cfg["PIPE"]["pose_graph"]["loop_candidate_export_policy"] = args.loop_candidate_export_policy
                suffix_parts.append(f"lexp{args.loop_candidate_export_policy}")
            if args.loop_candidate_export_span_reserve_fraction is not None:
                fraction = min(1.0, max(0.0, args.loop_candidate_export_span_reserve_fraction))
                cfg["PIPE"]["pose_graph"]["loop_candidate_export_span_reserve_fraction"] = fraction
                suffix_parts.append(f"lexpf{str(fraction).replace('.', 'p')}")
            if args.loop_candidate_export_min_span is not None:
                min_span = max(0, args.loop_candidate_export_min_span)
                cfg["PIPE"]["pose_graph"]["loop_candidate_export_min_span"] = min_span
                suffix_parts.append(f"lexps{min_span}")
            if args.max_scanned_pairs is not None:
                scanned = max(0, args.max_scanned_pairs)
                cfg["PIPE"]["pose_graph"]["max_scanned_pairs_per_check"] = scanned
                suffix_parts.append(f"scan{scanned}")
            if args.final_candidate_scan != "auto":
                final_scan = args.final_candidate_scan == "true"
                cfg["PIPE"]["pose_graph"]["final_candidate_scan"] = final_scan
                suffix_parts.append(f"fscanon{int(final_scan)}")
            if args.terminal_online_scan != "auto":
                terminal_scan = args.terminal_online_scan == "true"
                cfg["PIPE"]["pose_graph"]["terminal_online_scan"] = terminal_scan
                suffix_parts.append(f"termonscan{int(terminal_scan)}")
            if args.final_max_scanned_pairs is not None:
                final_scanned = max(0, args.final_max_scanned_pairs)
                cfg["PIPE"]["pose_graph"]["final_max_scanned_pairs"] = final_scanned
                suffix_parts.append(f"fscan{final_scanned}")
            if args.final_rank_all_candidates != "auto":
                enabled = args.final_rank_all_candidates == "true"
                cfg["PIPE"]["pose_graph"]["final_rank_all_candidates"] = enabled
                suffix_parts.append(f"frankall{int(enabled)}")
            if args.incremental_candidate_scan != "auto":
                incremental = args.incremental_candidate_scan == "true"
                cfg["PIPE"]["pose_graph"]["incremental_candidate_scan"] = incremental
                suffix_parts.append(f"incscan{int(incremental)}")
            if args.max_candidate_pairs is not None:
                cfg["PIPE"]["pose_graph"]["max_candidate_pairs_per_check"] = max(0, args.max_candidate_pairs)
                suffix_parts.append(f"pairs{max(0, args.max_candidate_pairs)}")
            if args.candidate_accept_target is not None:
                target = max(0, args.candidate_accept_target)
                cfg["PIPE"]["pose_graph"]["candidate_accept_target_per_check"] = target
                suffix_parts.append(f"acct{target}")
            if args.candidate_order is not None:
                cfg["PIPE"]["pose_graph"]["candidate_order"] = args.candidate_order
                suffix_parts.append(f"order{args.candidate_order}")
            if args.span_reserve_pairs is not None:
                pairs = max(0, args.span_reserve_pairs)
                cfg["PIPE"]["pose_graph"]["span_reserve_pairs"] = pairs
                suffix_parts.append(f"sreserve{pairs}")
            if args.span_reserve_fraction is not None:
                fraction = min(1.0, max(0.0, args.span_reserve_fraction))
                cfg["PIPE"]["pose_graph"]["span_reserve_fraction"] = fraction
                suffix_parts.append(f"sfrac{str(fraction).replace('.', 'p')}")
            if args.candidate_verification is not None:
                cfg["PIPE"]["pose_graph"]["candidate_verification"] = args.candidate_verification
                suffix_parts.append(f"verify{args.candidate_verification}")
            if args.appearance_max_observation_pairs is not None:
                pairs = max(0, args.appearance_max_observation_pairs)
                cfg["PIPE"]["pose_graph"]["appearance_max_observation_pairs"] = pairs
                suffix_parts.append(f"apairs{pairs}")
            if args.appearance_min_orb_matches is not None:
                matches = max(0, args.appearance_min_orb_matches)
                cfg["PIPE"]["pose_graph"]["appearance_min_orb_matches"] = matches
                suffix_parts.append(f"amatch{matches}")
            if args.loop_observation_interval is not None:
                loop_obs_interval = max(0, args.loop_observation_interval)
                cfg["PIPE"]["pose_graph"]["loop_observation_interval_frames"] = loop_obs_interval
                suffix_parts.append(f"lobsint{loop_obs_interval}")
            if args.loop_observation_max is not None:
                loop_obs_max = max(0, args.loop_observation_max)
                cfg["PIPE"]["pose_graph"]["loop_observation_max_per_submap"] = loop_obs_max
                suffix_parts.append(f"lobsmax{loop_obs_max}")
            if args.include_keyframes_with_loop_observations != "auto":
                include = args.include_keyframes_with_loop_observations == "true"
                cfg["PIPE"]["pose_graph"]["include_keyframes_with_loop_observations"] = include
                suffix_parts.append(f"lobskf{int(include)}")
            if args.keyframe_observation_max is not None:
                keyframe_obs_max = max(0, args.keyframe_observation_max)
                cfg["PIPE"]["pose_graph"]["keyframe_observation_max_per_submap"] = keyframe_obs_max
                suffix_parts.append(f"kobsmax{keyframe_obs_max}")
            if args.orb_feature_cache != "auto":
                enabled = args.orb_feature_cache == "true"
                cfg["PIPE"]["pose_graph"]["orb_feature_cache"] = enabled
                suffix_parts.append(f"orbcache{int(enabled)}")
            if args.orb_feature_cache_limit is not None:
                limit = max(0, args.orb_feature_cache_limit)
                cfg["PIPE"]["pose_graph"]["orb_feature_cache_limit"] = limit
                suffix_parts.append(f"orbcachelim{limit}")
            if args.warm_loop_observations != "auto":
                enabled = args.warm_loop_observations == "true"
                cfg["PIPE"]["pose_graph"]["warm_loop_observations"] = enabled
                suffix_parts.append(f"warmorb{int(enabled)}")
            if args.warm_max_per_frame is not None:
                warm_max = max(0, args.warm_max_per_frame)
                cfg["PIPE"]["pose_graph"]["warm_max_per_frame"] = warm_max
                suffix_parts.append(f"warmmax{warm_max}")
            if args.warm_skip_update_frames != "auto":
                enabled = args.warm_skip_update_frames == "true"
                cfg["PIPE"]["pose_graph"]["warm_skip_update_frames"] = enabled
                suffix_parts.append(f"warmskip{int(enabled)}")
            if args.warm_async != "auto":
                enabled = args.warm_async == "true"
                cfg["PIPE"]["pose_graph"]["warm_async"] = enabled
                suffix_parts.append(f"warmasync{int(enabled)}")
            if args.warm_queue_limit is not None:
                queue_limit = max(0, args.warm_queue_limit)
                cfg["PIPE"]["pose_graph"]["warm_queue_limit"] = queue_limit
                suffix_parts.append(f"warmq{queue_limit}")
            if args.orb_max_features is not None:
                max_features = max(16, args.orb_max_features)
                cfg["PIPE"]["pose_graph"]["orb_max_features"] = max_features
                suffix_parts.append(f"orbfeat{max_features}")
            if args.orb_ratio is not None:
                ratio = min(1.0, max(0.0, args.orb_ratio))
                cfg["PIPE"]["pose_graph"]["orb_ratio"] = ratio
                suffix_parts.append(f"orbr{str(ratio).replace('.', 'p')}")
            if args.orb_match_policy is not None:
                cfg["PIPE"]["pose_graph"]["orb_match_policy"] = args.orb_match_policy
                suffix_parts.append(f"orbmatch{args.orb_match_policy}")
            if args.orb_min_quadrants is not None:
                quadrants = min(4, max(1, args.orb_min_quadrants))
                cfg["PIPE"]["pose_graph"]["orb_min_quadrants"] = quadrants
                suffix_parts.append(f"orbq{quadrants}")
            if args.orb_depth_search_radius is not None:
                radius = max(0, args.orb_depth_search_radius)
                cfg["PIPE"]["pose_graph"]["orb_depth_search_radius"] = radius
                suffix_parts.append(f"orbd{radius}")
            if args.orb_pnp_fallback != "auto":
                enabled = args.orb_pnp_fallback == "true"
                cfg["PIPE"]["pose_graph"]["orb_pnp_fallback"] = enabled
                suffix_parts.append(f"pnp{int(enabled)}")
            if args.orb_pnp_iterations is not None:
                iterations = max(1, args.orb_pnp_iterations)
                cfg["PIPE"]["pose_graph"]["orb_pnp_iterations"] = iterations
                suffix_parts.append(f"pnpit{iterations}")
            if args.orb_pnp_reprojection_error is not None:
                reproj = max(0.1, args.orb_pnp_reprojection_error)
                cfg["PIPE"]["pose_graph"]["orb_pnp_reprojection_error"] = reproj
                suffix_parts.append(f"pnperr{str(reproj).replace('.', 'p')}")
            if args.orb_pnp_confidence is not None:
                confidence = min(0.999, max(0.0, args.orb_pnp_confidence))
                cfg["PIPE"]["pose_graph"]["orb_pnp_confidence"] = confidence
                suffix_parts.append(f"pnpconf{str(confidence).replace('.', 'p')}")
            if args.bow_word_bits is not None:
                bits = min(16, max(4, args.bow_word_bits))
                cfg["PIPE"]["pose_graph"]["bow_word_bits"] = bits
                suffix_parts.append(f"bowbits{bits}")
            if args.bow_max_observations is not None:
                observations = max(0, args.bow_max_observations)
                cfg["PIPE"]["pose_graph"]["bow_max_observations_per_submap"] = observations
                suffix_parts.append(f"bowobs{observations}")
            if args.bow_min_score is not None:
                score = max(0.0, args.bow_min_score)
                cfg["PIPE"]["pose_graph"]["bow_min_score"] = score
                suffix_parts.append(f"bows{str(score).replace('.', 'p')}")
            if args.bow_defer_appearance != "auto":
                defer = args.bow_defer_appearance == "true"
                cfg["PIPE"]["pose_graph"]["bow_defer_appearance"] = defer
                suffix_parts.append(f"bowdefer{int(defer)}")
            if args.bow_histogram_cache != "auto":
                enabled = args.bow_histogram_cache == "true"
                cfg["PIPE"]["pose_graph"]["bow_histogram_cache"] = enabled
                suffix_parts.append(f"bowcache{int(enabled)}")
            if args.verify_icp_min_inliers is not None:
                cfg["PIPE"]["pose_graph"]["verify_icp_min_inliers"] = max(3, args.verify_icp_min_inliers)
                suffix_parts.append(f"inl{max(3, args.verify_icp_min_inliers)}")
            if args.verify_icp_max_rmse is not None:
                cfg["PIPE"]["pose_graph"]["verify_icp_max_rmse"] = max(0.0, args.verify_icp_max_rmse)
                suffix_parts.append(f"irmse{str(max(0.0, args.verify_icp_max_rmse)).replace('.', 'p')}")
            if args.verify_icp_max_translation_delta is not None:
                cfg["PIPE"]["pose_graph"]["verify_icp_max_translation_delta"] = max(0.0, args.verify_icp_max_translation_delta)
                suffix_parts.append(f"it{str(max(0.0, args.verify_icp_max_translation_delta)).replace('.', 'p')}")
            if args.verify_icp_max_rotation_delta is not None:
                cfg["PIPE"]["pose_graph"]["verify_icp_max_rotation_delta_deg"] = max(0.0, args.verify_icp_max_rotation_delta)
                suffix_parts.append(f"ir{str(max(0.0, args.verify_icp_max_rotation_delta)).replace('.', 'p')}")
            if args.verify_icp_ransac != "auto":
                cfg["PIPE"]["pose_graph"]["verify_icp_ransac"] = args.verify_icp_ransac == "true"
                suffix_parts.append(f"ransac{args.verify_icp_ransac}")
            if args.verify_icp_ransac_iterations is not None:
                cfg["PIPE"]["pose_graph"]["verify_icp_ransac_iterations"] = max(1, args.verify_icp_ransac_iterations)
                suffix_parts.append(f"rit{max(1, args.verify_icp_ransac_iterations)}")
            if args.verify_icp_ransac_threshold is not None:
                threshold = max(0.0, args.verify_icp_ransac_threshold)
                cfg["PIPE"]["pose_graph"]["verify_icp_ransac_threshold"] = threshold
                suffix_parts.append(f"rth{str(threshold).replace('.', 'p')}")
            if args.verify_icp_ransac_min_inlier_ratio is not None:
                ratio = min(1.0, max(0.0, args.verify_icp_ransac_min_inlier_ratio))
                cfg["PIPE"]["pose_graph"]["verify_icp_ransac_min_inlier_ratio"] = ratio
                suffix_parts.append(f"rr{str(ratio).replace('.', 'p')}")
            if args.verify_icp_weighted_kabsch != "auto":
                enabled = args.verify_icp_weighted_kabsch == "true"
                cfg["PIPE"]["pose_graph"]["verify_icp_weighted_kabsch"] = enabled
                suffix_parts.append(f"wk{1 if enabled else 0}")
            if args.verify_icp_depth_adaptive_ransac != "auto":
                enabled = args.verify_icp_depth_adaptive_ransac == "true"
                cfg["PIPE"]["pose_graph"]["verify_icp_depth_adaptive_ransac"] = enabled
                suffix_parts.append(f"dar{1 if enabled else 0}")
            if args.verify_icp_depth_noise_scale is not None:
                scale = max(1e-6, args.verify_icp_depth_noise_scale)
                cfg["PIPE"]["pose_graph"]["verify_icp_depth_noise_scale"] = scale
                suffix_parts.append(f"dns{str(scale).replace('.', 'p')}")
            if args.verify_icp_depth_adaptive_threshold_scale is not None:
                scale = max(0.1, args.verify_icp_depth_adaptive_threshold_scale)
                cfg["PIPE"]["pose_graph"]["verify_icp_depth_adaptive_threshold_scale"] = scale
                suffix_parts.append(f"dats{str(scale).replace('.', 'p')}")
            if args.verify_icp_max_score is not None:
                score = max(0.0, args.verify_icp_max_score)
                cfg["PIPE"]["pose_graph"]["verify_icp_max_score"] = score
                suffix_parts.append(f"iscore{str(score).replace('.', 'p')}")
            if args.verify_icp_drift_aware_delta != "auto":
                enabled = args.verify_icp_drift_aware_delta == "true"
                cfg["PIPE"]["pose_graph"]["verify_icp_drift_aware_delta"] = enabled
                suffix_parts.append(f"idrift{1 if enabled else 0}")
            if args.verify_icp_drift_reference_span is not None:
                span = max(1.0, args.verify_icp_drift_reference_span)
                cfg["PIPE"]["pose_graph"]["verify_icp_drift_reference_span"] = span
                suffix_parts.append(f"idref{str(span).replace('.', 'p')}")
            if args.verify_icp_drift_max_scale is not None:
                scale = max(1.0, args.verify_icp_drift_max_scale)
                cfg["PIPE"]["pose_graph"]["verify_icp_drift_max_scale"] = scale
                suffix_parts.append(f"idmax{str(scale).replace('.', 'p')}")
            if args.verify_icp_drift_score_margin is not None:
                margin = max(0.0, args.verify_icp_drift_score_margin)
                cfg["PIPE"]["pose_graph"]["verify_icp_drift_score_margin"] = margin
                suffix_parts.append(f"idscore{str(margin).replace('.', 'p')}")
            if args.verify_orb_aggregate_pairs != "auto":
                aggregate = args.verify_orb_aggregate_pairs == "true"
                cfg["PIPE"]["pose_graph"]["verify_orb_aggregate_pairs"] = aggregate
                suffix_parts.append(f"oagg{int(aggregate)}")
            if args.verify_orb_aggregate_max_pairs is not None:
                aggregate_pairs = max(1, args.verify_orb_aggregate_max_pairs)
                cfg["PIPE"]["pose_graph"]["verify_orb_aggregate_max_pairs"] = aggregate_pairs
                suffix_parts.append(f"oaggp{aggregate_pairs}")
            if args.verify_max_observation_pairs is not None:
                max_obs_pairs = max(0, args.verify_max_observation_pairs)
                cfg["PIPE"]["pose_graph"]["verify_max_observation_pairs"] = max_obs_pairs
                suffix_parts.append(f"vpair{max_obs_pairs}")
            if args.verify_pair_selection is not None:
                cfg["PIPE"]["pose_graph"]["verify_pair_selection"] = args.verify_pair_selection
                suffix_parts.append(f"vsel{args.verify_pair_selection}")
            if args.verify_min_observations is not None:
                min_obs = max(1, args.verify_min_observations)
                cfg["PIPE"]["pose_graph"]["verify_min_observations"] = min_obs
                suffix_parts.append(f"vobs{min_obs}")
            if args.verify_observation_max_translation_delta is not None:
                obs_t = max(0.0, args.verify_observation_max_translation_delta)
                cfg["PIPE"]["pose_graph"]["verify_observation_max_translation_delta"] = obs_t
                suffix_parts.append(f"vobsT{str(obs_t).replace('.', 'p')}")
            if args.verify_observation_max_rotation_delta is not None:
                obs_r = max(0.0, args.verify_observation_max_rotation_delta)
                cfg["PIPE"]["pose_graph"]["verify_observation_max_rotation_delta_deg"] = obs_r
                suffix_parts.append(f"vobsR{str(obs_r).replace('.', 'p')}")
            if args.solve_only_max_translation_delta is not None:
                solve_t = max(0.0, args.solve_only_max_translation_delta)
                cfg["PIPE"]["pose_graph"]["solve_only_max_translation_delta"] = solve_t
                suffix_parts.append(f"smaxT{str(solve_t).replace('.', 'p')}")
            if args.solve_only_max_rotation_delta is not None:
                solve_r = max(0.0, args.solve_only_max_rotation_delta)
                cfg["PIPE"]["pose_graph"]["solve_only_max_rotation_delta_deg"] = solve_r
                suffix_parts.append(f"smaxR{str(solve_r).replace('.', 'p')}")
            if args.apply_max_translation_delta is not None:
                apply_t = max(0.0, args.apply_max_translation_delta)
                cfg["PIPE"]["pose_graph"]["apply_max_translation_delta"] = apply_t
                suffix_parts.append(f"amaxT{str(apply_t).replace('.', 'p')}")
            if args.apply_max_rotation_delta is not None:
                apply_r = max(0.0, args.apply_max_rotation_delta)
                cfg["PIPE"]["pose_graph"]["apply_max_rotation_delta_deg"] = apply_r
                suffix_parts.append(f"amaxR{str(apply_r).replace('.', 'p')}")
            if args.apply_max_sequential_translation_residual is not None:
                seq_t = max(0.0, args.apply_max_sequential_translation_residual)
                cfg["PIPE"]["pose_graph"]["apply_max_sequential_translation_residual"] = seq_t
                suffix_parts.append(f"aseqT{str(seq_t).replace('.', 'p')}")
            if args.apply_max_sequential_rotation_residual is not None:
                seq_r = max(0.0, args.apply_max_sequential_rotation_residual)
                cfg["PIPE"]["pose_graph"]["apply_max_sequential_rotation_residual_deg"] = seq_r
                suffix_parts.append(f"aseqR{str(seq_r).replace('.', 'p')}")
            if args.apply_min_translation_delta is not None:
                min_t = max(0.0, args.apply_min_translation_delta)
                cfg["PIPE"]["pose_graph"]["apply_min_translation_delta"] = min_t
                suffix_parts.append(f"aminT{str(min_t).replace('.', 'p')}")
            if args.apply_min_loop_constraints is not None:
                min_loop = max(0, args.apply_min_loop_constraints)
                cfg["PIPE"]["pose_graph"]["apply_min_loop_constraints"] = min_loop
                suffix_parts.append(f"aminL{min_loop}")
            if args.apply_min_loop_submap_span is not None:
                min_span = max(0, args.apply_min_loop_submap_span)
                cfg["PIPE"]["pose_graph"]["apply_min_loop_submap_span"] = min_span
                suffix_parts.append(f"aminSpan{min_span}")
            if args.apply_min_loop_consistent_observations is not None:
                min_obs = max(0, args.apply_min_loop_consistent_observations)
                cfg["PIPE"]["pose_graph"]["apply_min_loop_consistent_observations"] = min_obs
                suffix_parts.append(f"aminObs{min_obs}")
            if args.apply_min_loop_icp_inliers is not None:
                min_inl = max(0, args.apply_min_loop_icp_inliers)
                cfg["PIPE"]["pose_graph"]["apply_min_loop_icp_inliers"] = min_inl
                suffix_parts.append(f"aminInl{min_inl}")
            if args.apply_max_loop_score is not None:
                max_score = max(0.0, args.apply_max_loop_score)
                cfg["PIPE"]["pose_graph"]["apply_max_loop_score"] = max_score
                suffix_parts.append(f"amaxScore{str(max_score).replace('.', 'p')}")
            if args.apply_support_policy is not None:
                cfg["PIPE"]["pose_graph"]["apply_support_policy"] = args.apply_support_policy
                suffix_parts.append(f"asupport{args.apply_support_policy}")
            if args.apply_adaptive_min_loop_constraints is not None:
                min_loop = max(0, args.apply_adaptive_min_loop_constraints)
                cfg["PIPE"]["pose_graph"]["apply_adaptive_min_loop_constraints"] = min_loop
                suffix_parts.append(f"aaminL{min_loop}")
            if args.apply_adaptive_min_loop_submap_span is not None:
                min_span = max(0, args.apply_adaptive_min_loop_submap_span)
                cfg["PIPE"]["pose_graph"]["apply_adaptive_min_loop_submap_span"] = min_span
                suffix_parts.append(f"aaminSpan{min_span}")
            if args.apply_adaptive_min_loop_consistent_observations is not None:
                min_obs = max(0, args.apply_adaptive_min_loop_consistent_observations)
                cfg["PIPE"]["pose_graph"]["apply_adaptive_min_loop_consistent_observations"] = min_obs
                suffix_parts.append(f"aaminObs{min_obs}")
            if args.apply_adaptive_min_loop_icp_inliers is not None:
                min_inl = max(0, args.apply_adaptive_min_loop_icp_inliers)
                cfg["PIPE"]["pose_graph"]["apply_adaptive_min_loop_icp_inliers"] = min_inl
                suffix_parts.append(f"aaminInl{min_inl}")
            if args.apply_adaptive_max_loop_score is not None:
                max_score = max(0.0, args.apply_adaptive_max_loop_score)
                cfg["PIPE"]["pose_graph"]["apply_adaptive_max_loop_score"] = max_score
                suffix_parts.append(f"aamaxScore{str(max_score).replace('.', 'p')}")
            if args.apply_high_confidence_min_loop_constraints is not None:
                min_loop = max(0, args.apply_high_confidence_min_loop_constraints)
                cfg["PIPE"]["pose_graph"]["apply_high_confidence_min_loop_constraints"] = min_loop
                suffix_parts.append(f"hcminL{min_loop}")
            if args.apply_high_confidence_min_loop_submap_span is not None:
                min_span = max(0, args.apply_high_confidence_min_loop_submap_span)
                cfg["PIPE"]["pose_graph"]["apply_high_confidence_min_loop_submap_span"] = min_span
                suffix_parts.append(f"hcminSpan{min_span}")
            if args.apply_high_confidence_min_loop_consistent_observations is not None:
                min_obs = max(0, args.apply_high_confidence_min_loop_consistent_observations)
                cfg["PIPE"]["pose_graph"]["apply_high_confidence_min_loop_consistent_observations"] = min_obs
                suffix_parts.append(f"hcminObs{min_obs}")
            if args.apply_high_confidence_min_loop_icp_inliers is not None:
                min_inl = max(0, args.apply_high_confidence_min_loop_icp_inliers)
                cfg["PIPE"]["pose_graph"]["apply_high_confidence_min_loop_icp_inliers"] = min_inl
                suffix_parts.append(f"hcminInl{min_inl}")
            if args.apply_high_confidence_max_loop_score is not None:
                max_score = max(0.0, args.apply_high_confidence_max_loop_score)
                cfg["PIPE"]["pose_graph"]["apply_high_confidence_max_loop_score"] = max_score
                suffix_parts.append(f"hcmaxScore{str(max_score).replace('.', 'p')}")
            if args.apply_high_confidence_max_icp_rmse is not None:
                max_rmse = max(0.0, args.apply_high_confidence_max_icp_rmse)
                cfg["PIPE"]["pose_graph"]["apply_high_confidence_max_icp_rmse"] = max_rmse
                suffix_parts.append(f"hcmaxRmse{str(max_rmse).replace('.', 'p')}")
            if args.apply_max_frame_relative_translation_change is not None:
                max_rel_t = max(0.0, args.apply_max_frame_relative_translation_change)
                cfg["PIPE"]["pose_graph"]["apply_max_frame_relative_translation_change"] = max_rel_t
                suffix_parts.append(f"arelT{str(max_rel_t).replace('.', 'p')}")
            if args.apply_max_frame_relative_rotation_change is not None:
                max_rel_r = max(0.0, args.apply_max_frame_relative_rotation_change)
                cfg["PIPE"]["pose_graph"]["apply_max_frame_relative_rotation_change_deg"] = max_rel_r
                suffix_parts.append(f"arelR{str(max_rel_r).replace('.', 'p')}")
            if args.apply_loop_selection_policy is not None:
                cfg["PIPE"]["pose_graph"]["apply_loop_selection_policy"] = args.apply_loop_selection_policy
                suffix_parts.append(f"asel{args.apply_loop_selection_policy}")
            if args.apply_loop_selection_max_candidates is not None:
                max_candidates = max(0, args.apply_loop_selection_max_candidates)
                cfg["PIPE"]["pose_graph"]["apply_loop_selection_max_candidates"] = max_candidates
                suffix_parts.append(f"aselmax{max_candidates}")
            if args.apply_pcm_min_clique_size is not None:
                clique_size = max(1, args.apply_pcm_min_clique_size)
                cfg["PIPE"]["pose_graph"]["apply_pcm_min_clique_size"] = clique_size
                suffix_parts.append(f"pcmmin{clique_size}")
            if args.apply_pcm_max_trial_cliques is not None:
                trial_cliques = max(1, args.apply_pcm_max_trial_cliques)
                cfg["PIPE"]["pose_graph"]["apply_pcm_max_trial_cliques"] = trial_cliques
                suffix_parts.append(f"pcmtrials{trial_cliques}")
            if args.apply_pcm_event_cluster_radius is not None:
                radius = max(0, args.apply_pcm_event_cluster_radius)
                cfg["PIPE"]["pose_graph"]["apply_pcm_event_cluster_radius"] = radius
                suffix_parts.append(f"pcmrad{radius}")
            if args.apply_pcm_min_distinct_events is not None:
                events = max(1, args.apply_pcm_min_distinct_events)
                cfg["PIPE"]["pose_graph"]["apply_pcm_min_distinct_events"] = events
                suffix_parts.append(f"pcmevents{events}")
            if args.apply_pcm_min_long_span is not None:
                span = max(0, args.apply_pcm_min_long_span)
                cfg["PIPE"]["pose_graph"]["apply_pcm_min_long_span"] = span
                suffix_parts.append(f"pcmlong{span}")
            if args.apply_pcm_prefer_long_span != "auto":
                enabled = args.apply_pcm_prefer_long_span == "true"
                cfg["PIPE"]["pose_graph"]["apply_pcm_prefer_long_span"] = enabled
                suffix_parts.append(f"pcmprefspan{int(enabled)}")
            if args.apply_pcm_scale_threshold_by_cycle_length != "auto":
                enabled = args.apply_pcm_scale_threshold_by_cycle_length == "true"
                cfg["PIPE"]["pose_graph"]["apply_pcm_scale_threshold_by_cycle_length"] = enabled
                suffix_parts.append(f"pcmscale{int(enabled)}")
            if args.apply_pcm_translation_threshold is not None:
                threshold = max(0.0, args.apply_pcm_translation_threshold)
                cfg["PIPE"]["pose_graph"]["apply_pcm_translation_threshold"] = threshold
                suffix_parts.append(f"pcmt{str(threshold).replace('.', 'p')}")
            if args.apply_pcm_rotation_threshold is not None:
                threshold = max(0.0, args.apply_pcm_rotation_threshold)
                cfg["PIPE"]["pose_graph"]["apply_pcm_rotation_threshold_deg"] = threshold
                suffix_parts.append(f"pcmr{str(threshold).replace('.', 'p')}")
            if args.apply_skip_repeated_loop_pairs != "auto":
                enabled = args.apply_skip_repeated_loop_pairs == "true"
                cfg["PIPE"]["pose_graph"]["apply_skip_repeated_loop_pairs"] = enabled
                suffix_parts.append(f"askiprep{int(enabled)}")
            if args.apply_repeat_min_new_nodes is not None:
                min_nodes = max(1, args.apply_repeat_min_new_nodes)
                cfg["PIPE"]["pose_graph"]["apply_repeat_min_new_nodes"] = min_nodes
                suffix_parts.append(f"arepnodes{min_nodes}")
            if args.apply_correction_scale is not None:
                scale = min(1.0, max(0.0, args.apply_correction_scale))
                cfg["PIPE"]["pose_graph"]["apply_correction_scale"] = scale
                suffix_parts.append(f"ascale{str(scale).replace('.', 'p')}")
            if args.apply_scale_backoff != "auto":
                enabled = args.apply_scale_backoff == "true"
                cfg["PIPE"]["pose_graph"]["apply_scale_backoff"] = enabled
                suffix_parts.append(f"asback{int(enabled)}")
            if args.apply_scale_backoff_min_scale is not None:
                min_scale = min(1.0, max(0.0, args.apply_scale_backoff_min_scale))
                cfg["PIPE"]["pose_graph"]["apply_scale_backoff_min_scale"] = min_scale
                suffix_parts.append(f"asbmin{str(min_scale).replace('.', 'p')}")
            if args.apply_scale_backoff_factor is not None:
                factor = min(0.999, max(0.001, args.apply_scale_backoff_factor))
                cfg["PIPE"]["pose_graph"]["apply_scale_backoff_factor"] = factor
                suffix_parts.append(f"asbf{str(factor).replace('.', 'p')}")
            if args.apply_scale_backoff_max_trials is not None:
                trials = max(1, args.apply_scale_backoff_max_trials)
                cfg["PIPE"]["pose_graph"]["apply_scale_backoff_max_trials"] = trials
                suffix_parts.append(f"asbt{trials}")
            if args.apply_observation_gate != "auto":
                cfg["PIPE"]["pose_graph"]["apply_observation_gate"] = args.apply_observation_gate == "true"
                suffix_parts.append(f"aobs{1 if args.apply_observation_gate == 'true' else 0}")
            if args.apply_observation_gate_in_selection != "auto":
                cfg["PIPE"]["pose_graph"]["apply_observation_gate_in_selection"] = args.apply_observation_gate_in_selection == "true"
                suffix_parts.append(f"aobssel{1 if args.apply_observation_gate_in_selection == 'true' else 0}")
            if args.apply_observation_gate_max_samples is not None:
                max_samples = max(0, args.apply_observation_gate_max_samples)
                cfg["PIPE"]["pose_graph"]["apply_observation_gate_max_samples"] = max_samples
                suffix_parts.append(f"aobsmax{max_samples}")
            if args.apply_observation_gate_min_samples is not None:
                min_samples = max(0, args.apply_observation_gate_min_samples)
                cfg["PIPE"]["pose_graph"]["apply_observation_gate_min_samples"] = min_samples
                suffix_parts.append(f"aobsmin{min_samples}")
            if args.apply_observation_gate_sample_policy is not None:
                cfg["PIPE"]["pose_graph"]["apply_observation_gate_sample_policy"] = args.apply_observation_gate_sample_policy
                suffix_parts.append(f"aobspol{args.apply_observation_gate_sample_policy}")
            if args.apply_observation_max_render_maps_per_sample is not None:
                maps = max(0, args.apply_observation_max_render_maps_per_sample)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_render_maps_per_sample"] = maps
                suffix_parts.append(f"aobsmap{maps}")
            if args.apply_observation_max_pinned_maps is not None:
                maps = max(0, args.apply_observation_max_pinned_maps)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_pinned_maps"] = maps
                suffix_parts.append(f"aobspin{maps}")
            if args.apply_observation_defer_nonresident != "auto":
                defer_nonresident = args.apply_observation_defer_nonresident == "true"
                cfg["PIPE"]["pose_graph"]["apply_observation_defer_nonresident"] = defer_nonresident
                suffix_parts.append(f"aobsdefer{int(defer_nonresident)}")
            if args.apply_observation_defer_retry_frames is not None:
                retry_frames = max(1, args.apply_observation_defer_retry_frames)
                cfg["PIPE"]["pose_graph"]["apply_observation_defer_retry_frames"] = retry_frames
                suffix_parts.append(f"aobsretry{retry_frames}")
            if args.apply_observation_pin_deferred_maps != "auto":
                enabled = args.apply_observation_pin_deferred_maps == "true"
                cfg["PIPE"]["pose_graph"]["apply_observation_pin_deferred_maps"] = enabled
                suffix_parts.append(f"aobspindefer{int(enabled)}")
            if args.apply_observation_defer_stops_scale_backoff != "auto":
                enabled = args.apply_observation_defer_stops_scale_backoff == "true"
                cfg["PIPE"]["pose_graph"]["apply_observation_defer_stops_scale_backoff"] = enabled
                suffix_parts.append(f"aobsdeferstop{int(enabled)}")
            if args.apply_observation_terminal_sync_page_in != "auto":
                enabled = args.apply_observation_terminal_sync_page_in == "true"
                cfg["PIPE"]["pose_graph"]["apply_observation_terminal_sync_page_in"] = enabled
                suffix_parts.append(f"aobstermsync{int(enabled)}")
            if args.apply_observation_skip_resident_trim != "auto":
                enabled = args.apply_observation_skip_resident_trim == "true"
                cfg["PIPE"]["pose_graph"]["apply_observation_skip_resident_trim"] = enabled
                suffix_parts.append(f"aobsskiptrim{int(enabled)}")
            if args.apply_observation_prefetch_loop_endpoints != "auto":
                enabled = args.apply_observation_prefetch_loop_endpoints == "true"
                cfg["PIPE"]["pose_graph"]["apply_observation_prefetch_loop_endpoints"] = enabled
                suffix_parts.append(f"aobspref{int(enabled)}")
            if args.apply_observation_prefetch_max_maps is not None:
                maps = max(0, args.apply_observation_prefetch_max_maps)
                cfg["PIPE"]["pose_graph"]["apply_observation_prefetch_max_maps"] = maps
                suffix_parts.append(f"aobsprefmax{maps}")
            if args.apply_observation_gate_tolerance_policy is not None:
                cfg["PIPE"]["pose_graph"]["apply_observation_gate_tolerance_policy"] = args.apply_observation_gate_tolerance_policy
                suffix_parts.append(f"aobstol{args.apply_observation_gate_tolerance_policy}")
            if args.apply_observation_max_color_mse_ratio is not None:
                ratio = max(0.0, args.apply_observation_max_color_mse_ratio)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_color_mse_ratio"] = ratio
                suffix_parts.append(f"aobscr{str(ratio).replace('.', 'p')}")
            if args.apply_observation_max_color_mse_increase is not None:
                increase = max(0.0, args.apply_observation_max_color_mse_increase)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_color_mse_increase"] = increase
                suffix_parts.append(f"aobsci{str(increase).replace('.', 'p')}")
            if args.apply_observation_max_depth_rmse_ratio is not None:
                ratio = max(0.0, args.apply_observation_max_depth_rmse_ratio)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_depth_rmse_ratio"] = ratio
                suffix_parts.append(f"aobsdr{str(ratio).replace('.', 'p')}")
            if args.apply_observation_max_depth_rmse_increase is not None:
                increase = max(0.0, args.apply_observation_max_depth_rmse_increase)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_depth_rmse_increase"] = increase
                suffix_parts.append(f"aobsdi{str(increase).replace('.', 'p')}")
            if args.apply_observation_depth_rmse_ratio_floor is not None:
                increase = max(0.0, args.apply_observation_depth_rmse_ratio_floor)
                cfg["PIPE"]["pose_graph"]["apply_observation_depth_rmse_ratio_floor"] = increase
                suffix_parts.append(f"aobsdrfloor{str(increase).replace('.', 'p')}")
            if args.apply_observation_max_depth_rmse_absolute_increase is not None:
                increase = max(0.0, args.apply_observation_max_depth_rmse_absolute_increase)
                cfg["PIPE"]["pose_graph"]["apply_observation_max_depth_rmse_absolute_increase"] = increase
                suffix_parts.append(f"aobsdabs{str(increase).replace('.', 'p')}")
            if args.apply_observation_max_valid_fraction_drop is not None:
                drop = min(1.0, max(0.0, args.apply_observation_max_valid_fraction_drop))
                cfg["PIPE"]["pose_graph"]["apply_observation_max_valid_fraction_drop"] = drop
                suffix_parts.append(f"aobsvdrop{str(drop).replace('.', 'p')}")
            if args.apply_max_transform_updates_per_frame is not None:
                max_updates = max(0, args.apply_max_transform_updates_per_frame)
                cfg["PIPE"]["pose_graph"]["apply_max_transform_updates_per_frame"] = max_updates
                suffix_parts.append(f"appupd{max_updates}")
            if args.apply_health_gate != "auto":
                enabled = args.apply_health_gate == "true"
                cfg["PIPE"]["pose_graph"]["apply_health_gate"] = enabled
                suffix_parts.append(f"ahgate{int(enabled)}")
            if args.apply_health_max_raw_failed_ratio is not None:
                ratio = max(0.0, args.apply_health_max_raw_failed_ratio)
                cfg["PIPE"]["pose_graph"]["apply_health_max_raw_failed_ratio"] = ratio
                suffix_parts.append(f"ahfr{str(ratio).replace('.', 'p')}")
            if args.apply_health_max_raw_non_good_ratio is not None:
                ratio = max(0.0, args.apply_health_max_raw_non_good_ratio)
                cfg["PIPE"]["pose_graph"]["apply_health_max_raw_non_good_ratio"] = ratio
                suffix_parts.append(f"ahwr{str(ratio).replace('.', 'p')}")
            if args.apply_health_max_raw_failed_streak is not None:
                streak = max(0, args.apply_health_max_raw_failed_streak)
                cfg["PIPE"]["pose_graph"]["apply_health_max_raw_failed_streak"] = streak
                suffix_parts.append(f"ahfs{streak}")
            if args.apply_health_min_frames is not None:
                frames = max(0, args.apply_health_min_frames)
                cfg["PIPE"]["pose_graph"]["apply_health_min_frames"] = frames
                suffix_parts.append(f"ahmin{frames}")
            if args.dry_run_use_estimated_loop_transforms != "auto":
                enabled = args.dry_run_use_estimated_loop_transforms == "true"
                cfg["PIPE"]["pose_graph"]["dry_run_use_estimated_loop_transforms"] = enabled
                suffix_parts.append(f"dryest{int(enabled)}")
            if args.solve_only_use_estimated_loop_transforms != "auto":
                enabled = args.solve_only_use_estimated_loop_transforms == "true"
                cfg["PIPE"]["pose_graph"]["solve_only_use_estimated_loop_transforms"] = enabled
                suffix_parts.append(f"solvest{int(enabled)}")
            if args.apply_use_estimated_loop_transforms != "auto":
                enabled = args.apply_use_estimated_loop_transforms == "true"
                cfg["PIPE"]["pose_graph"]["apply_use_estimated_loop_transforms"] = enabled
                suffix_parts.append(f"appest{int(enabled)}")
            if args.pose_graph_sequential_weight is not None:
                seq_w = max(0.001, args.pose_graph_sequential_weight)
                cfg["PIPE"]["pose_graph"]["sequential_constraint_weight"] = seq_w
                suffix_parts.append(f"pgsw{str(seq_w).replace('.', 'p')}")
            if args.pose_graph_loop_weight is not None:
                loop_w = max(0.001, args.pose_graph_loop_weight)
                cfg["PIPE"]["pose_graph"]["loop_constraint_weight"] = loop_w
                suffix_parts.append(f"pglw{str(loop_w).replace('.', 'p')}")
            if args.pose_graph_information_policy is not None:
                cfg["PIPE"]["pose_graph"]["information_policy"] = args.pose_graph_information_policy
                suffix_parts.append(f"pginfo{args.pose_graph_information_policy.replace('_', '')}")
            if args.pose_graph_loop_robust_kernel is not None:
                cfg["PIPE"]["pose_graph"]["loop_robust_kernel"] = args.pose_graph_loop_robust_kernel
                suffix_parts.append(f"pgrob{args.pose_graph_loop_robust_kernel}")
            if args.pose_graph_loop_robust_scale is not None:
                robust_scale = max(1e-6, args.pose_graph_loop_robust_scale)
                cfg["PIPE"]["pose_graph"]["loop_robust_scale"] = robust_scale
                suffix_parts.append(f"pgrobsc{str(robust_scale).replace('.', 'p')}")
            if args.pose_graph_loop_information_min_confidence is not None:
                min_conf = max(1e-6, args.pose_graph_loop_information_min_confidence)
                cfg["PIPE"]["pose_graph"]["loop_information_min_confidence"] = min_conf
                suffix_parts.append(f"pginfomin{str(min_conf).replace('.', 'p')}")
            if args.pose_graph_loop_information_max_confidence is not None:
                max_conf = max(1e-6, args.pose_graph_loop_information_max_confidence)
                cfg["PIPE"]["pose_graph"]["loop_information_max_confidence"] = max_conf
                suffix_parts.append(f"pginfomax{str(max_conf).replace('.', 'p')}")
            if args.pose_graph_anchor_policy is not None:
                cfg["PIPE"]["pose_graph"]["anchor_policy"] = args.pose_graph_anchor_policy
                suffix_parts.append(f"anchor{args.pose_graph_anchor_policy}")
            if args.pose_graph_online_updates:
                pose_cfg = cfg["PIPE"]["pose_graph"]
                has_live_pose_graph_op = (
                    pose_cfg.get("candidate_detection", False) or
                    pose_cfg.get("dry_run", False) or
                    pose_cfg.get("solve_only", False) or
                    pose_cfg.get("apply_mode", "disabled") != "disabled"
                )
                if has_live_pose_graph_op:
                    pose_cfg["online_updates"] = True
                    suffix_parts.append("online")
            if args.pose_graph_candidate_interval is not None:
                interval = max(0, args.pose_graph_candidate_interval)
                cfg["PIPE"]["pose_graph"]["candidate_interval_frames"] = interval
                suffix_parts.append(f"pgint{interval}")
            if args.pose_graph_submap_events != "auto":
                submap_events = args.pose_graph_submap_events == "true"
                cfg["PIPE"]["pose_graph"]["run_on_submap_events"] = submap_events
                suffix_parts.append(f"pgsubev{int(submap_events)}")
            if args.pose_graph_disable_submap_events:
                cfg["PIPE"]["pose_graph"]["run_on_submap_events"] = False
                suffix_parts.append("pgnosubev")
            if args.pose_graph_online_min_submaps is not None:
                min_submaps = max(2, args.pose_graph_online_min_submaps)
                cfg["PIPE"]["pose_graph"]["online_min_submaps"] = min_submaps
                suffix_parts.append(f"pgminsub{min_submaps}")
            if args.pose_graph_online_min_submap_growth is not None:
                min_growth = max(1, args.pose_graph_online_min_submap_growth)
                cfg["PIPE"]["pose_graph"]["online_min_submap_growth"] = min_growth
                suffix_parts.append(f"pggrowth{min_growth}")
            if args.pose_graph_max_updates is not None:
                max_updates = max(0, args.pose_graph_max_updates)
                cfg["PIPE"]["pose_graph"]["max_updates_per_run"] = max_updates
                suffix_parts.append(f"pgmax{max_updates}")
            if args.persistent_loop_pool != "auto":
                enabled = args.persistent_loop_pool == "true"
                cfg["PIPE"]["pose_graph"]["persistent_loop_pool"] = enabled
                suffix_parts.append(f"lpool{int(enabled)}")
            if args.verified_loop_pool_max is not None:
                pool_max = max(0, args.verified_loop_pool_max)
                cfg["PIPE"]["pose_graph"]["verified_loop_pool_max"] = pool_max
                suffix_parts.append(f"lpoolmax{pool_max}")
            if args.skip_verified_loop_pairs != "auto":
                enabled = args.skip_verified_loop_pairs == "true"
                cfg["PIPE"]["pose_graph"]["skip_verified_loop_pairs"] = enabled
                suffix_parts.append(f"skippool{int(enabled)}")
            if str(cfg["PIPE"]["TSDF"].get("engine", "")).lower() == "multi":
                accepts = max(0, int(cfg["PIPE"]["TSDF"]["multi"].get("new_map_handoff_min_accepts", 1)))
                suffix_parts.append(f"hacc{accepts}")
            if args.tag:
                suffix_parts.append(args.tag)
            if args.benchmark_protocol != PROTOCOL_NONE:
                try:
                    apply_protocol(
                        cfg,
                        args.benchmark_protocol,
                        benchmark_role,
                        benchmark_binary_profile,
                    )
                except ValueError as exc:
                    parser.error(str(exc))
                suffix_parts.append(f"bench{args.benchmark_protocol.removeprefix('gps_slam_')}")
            # Locked protocols may override memory-relevant settings such as the
            # binary-matched TSDF hash profile and paging policy.  Estimate the
            # final effective configuration, never the pre-protocol template.
            if str(cfg["PIPE"]["TSDF"].get("engine", "")).lower() == "multi":
                enforce_multi_run_safety(cfg, seq, variant, args)
            experiment_group_key = f"{seq}_{variant}_{'_'.join(suffix_parts)}"

            repeat_indices = [None] if repeat_count == 1 else [
                repeat_start_index + offset for offset in range(repeat_count)
            ]
            for repeat_offset, repeat_index in enumerate(repeat_indices):
                repeat_cfg = copy.deepcopy(cfg)
                repeat_suffix_parts = list(suffix_parts)
                repeat_cfg["PIPE"]["experiment_profile_name"] = args.loop_profile
                if repeat_index is not None:
                    base_seed = int(repeat_cfg["PIPE"].get("random_seed", 42))
                    if args.repeat_seed_step != 0:
                        repeat_seed = base_seed + repeat_offset * int(args.repeat_seed_step)
                        repeat_cfg["PIPE"]["random_seed"] = repeat_seed
                        repeat_suffix_parts.append(f"rseed{repeat_seed}")
                    repeat_cfg["PIPE"]["experiment_repeat_index"] = repeat_index
                    repeat_cfg["PIPE"]["experiment_repeat_count"] = repeat_count
                    repeat_cfg["PIPE"]["experiment_repeat_seed_step"] = int(args.repeat_seed_step)
                    repeat_cfg["PIPE"]["experiment_group_key"] = experiment_group_key
                    repeat_suffix_parts.append(f"rep{repeat_index:02d}")
                fingerprint = profile_fingerprint(repeat_cfg)
                repeat_cfg["PIPE"]["experiment_config_fingerprint"] = fingerprint
                repeat_cfg["PIPE"]["experiment_profile_fingerprint"] = fingerprint
                repeat_cfg["PIPE"]["experiment_profile_summary"] = profile_summary(repeat_cfg)

                suffix = "_".join(repeat_suffix_parts)
                workspace_stem = f"{seq}_{variant}_{suffix}"
                if len(workspace_stem) > 180:
                    digest = hashlib.sha1(workspace_stem.encode("utf-8")).hexdigest()[:12]
                    repeat_label = f"_rep{repeat_index:02d}" if repeat_index is not None else ""
                    workspace_stem = f"{seq}_{variant}_{digest}{repeat_label}"
                repeat_cfg["workspace_dir"] = str(args.output_root / workspace_stem)
                generated_run_dirs.append(Path(repeat_cfg["workspace_dir"]))
                config_stem = f"{seq}_{variant}_{suffix}"
                if len(config_stem) > 180:
                    digest = hashlib.sha1(config_stem.encode("utf-8")).hexdigest()[:12]
                    repeat_label = f"_rep{repeat_index:02d}" if repeat_index is not None else ""
                    config_stem = f"{seq}_{variant}_{digest}{repeat_label}"
                path = args.config_dir / f"{config_stem}.yaml"
                path.write_text(yaml.safe_dump(repeat_cfg, sort_keys=False))
                if args.thor_preflight:
                    run_cmd(
                        [sys.executable, str(PROJECT_ROOT / "scripts/thor_preflight.py"), str(path)],
                        PROJECT_ROOT,
                    )
                generated.append(path)
                if args.benchmark_protocol != PROTOCOL_NONE:
                    dataset_root = Path(repeat_cfg["READER"]["input_dir"])
                    generated_benchmark_metadata.append(
                        (benchmark_role, copy.deepcopy(repeat_cfg), dataset_root)
                    )
                    write_manifest(
                        path.with_suffix(".benchmark.json"),
                        protocol=args.benchmark_protocol,
                        role=benchmark_role,
                        cfg=repeat_cfg,
                        config_path=path,
                        binary=args.trainer,
                        dataset_root=dataset_root,
                        dataset_hash_mode=args.benchmark_dataset_hash,
                        extra={
                            "state": "generated",
                            "sequence": seq,
                            "variant": variant,
                            "repeat_index": repeat_index,
                        },
                    )
                print(f"[TUM] generated {path}")

    if args.run:
        trainer = args.trainer
        if trainer is None:
            trainer = PROJECT_ROOT / "build_compact/slam_trainer"
            if not trainer.exists():
                trainer = PROJECT_ROOT / "build/slam_trainer"
        if not trainer.exists():
            raise FileNotFoundError(f"Could not find slam_trainer binary: {trainer}")
        failures: list[tuple[Path, int | str]] = []
        run_log_dir = args.run_log_dir or (args.output_root / "run_logs")
        metric_after_run = args.sampled_quality_metrics == "true" or (
            args.sampled_quality_metrics == "auto" and args.eval_after_train
        )
        trainer_timeout_sec = args.trainer_timeout_sec if args.trainer_timeout_sec > 0 else None
        for generated_index, (cfg_path, run_dir) in enumerate(zip(generated, generated_run_dirs)):
            free_gb = shutil.disk_usage(args.output_root).free / 1024**3
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"Abort: only {free_gb:.1f} GB free under {args.output_root}; "
                    f"threshold is --min-free-gb={args.min_free_gb:.1f}"
                )
            try:
                run_cmd(
                    [str(trainer), str(cfg_path)],
                    PROJECT_ROOT,
                    run_log_dir / f"{cfg_path.stem}.log" if args.quiet_run_logs else None,
                    timeout_sec=trainer_timeout_sec,
                )
                if args.analyze_realtime or args.benchmark_protocol == TUM_DEPLOYED_V1:
                    if args.benchmark_protocol != PROTOCOL_NONE:
                        _, _, dataset_root = generated_benchmark_metadata[generated_index]
                    else:
                        dataset_root = Path(yaml.safe_load(cfg_path.read_text())["READER"]["input_dir"])
                    run_cmd(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts/analyze_realtime_trace.py"),
                            str(run_dir),
                            "--metadata",
                            str(dataset_root / "metadata.json"),
                            "--rate-hz",
                            "30",
                        ],
                        PROJECT_ROOT,
                        (
                            run_log_dir / f"{cfg_path.stem}_realtime.log"
                            if args.quiet_run_logs
                            else None
                        ),
                    )
                if metric_after_run:
                    run_sampled_quality_metrics(
                        run_dir,
                        run_log_dir if args.quiet_run_logs else None,
                    )
                if args.benchmark_protocol != PROTOCOL_NONE:
                    role, manifest_cfg, dataset_root = generated_benchmark_metadata[generated_index]
                    artifact_paths = sorted(run_dir.glob("val*/render/results.json"))
                    artifact_paths.extend(sorted(run_dir.glob("val*/raycast_color/results.json")))
                    artifact_paths.extend(sorted(run_dir.glob("val*/pose_eval.txt")))
                    artifact_paths.extend(
                        path
                        for path in (
                            run_dir / "realtime_trace.csv",
                            run_dir / "realtime_summary.json",
                            run_dir / "runtime_resources.json",
                            run_dir / "pose_graph_live/online_end.json",
                        )
                        if path.is_file()
                    )
                    write_manifest(
                        run_dir / "benchmark_manifest.json",
                        protocol=args.benchmark_protocol,
                        role=role,
                        cfg=manifest_cfg,
                        config_path=cfg_path,
                        binary=trainer,
                        dataset_root=dataset_root,
                        dataset_hash_mode=args.benchmark_dataset_hash,
                        extra={
                            "state": "completed",
                            "status": "OK",
                            "evaluation_artifact_sha256": {
                                str(path.resolve()): sha256_file(path)
                                for path in artifact_paths
                                if path.is_file()
                            },
                        },
                    )
            except subprocess.CalledProcessError as exc:
                if not args.continue_on_error:
                    raise
                failures.append((cfg_path, exc.returncode))
                print(f"[TUM] failed {cfg_path} exit={exc.returncode}; continuing")
            except subprocess.TimeoutExpired:
                if not args.continue_on_error:
                    raise
                failures.append((cfg_path, "timeout"))
                print(f"[TUM] timed out {cfg_path}; continuing")
        if failures:
            print("[TUM] failed runs:")
            for cfg_path, returncode in failures:
                print(f"  exit={returncode} {cfg_path}")
            if not args.continue_on_error:
                raise RuntimeError(f"{len(failures)} TUM run(s) failed")
        if not args.skip_summary and generated_run_dirs:
            summary_output = args.summary_output
            if summary_output is None:
                summary_name = args.tag.strip() or "tum_loop_validation"
                summary_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in summary_name)
                summary_output = args.output_root / f"{summary_name}_summary.csv"
            summary_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/summarize_tum_runs.py"),
                *[str(path) for path in generated_run_dirs],
                "--output",
                str(summary_output),
            ]
            if not args.continue_on_error:
                summary_cmd.append("--fail-on-blockers")
            run_cmd(
                summary_cmd,
                PROJECT_ROOT,
                run_log_dir / f"{summary_output.stem}_summary.log" if args.quiet_run_logs else None,
            )
            print(f"[TUM] summary saved to {summary_output}")
            if repeat_count > 1:
                detail_output = summary_output.with_name(f"{summary_output.stem}_stability.csv")
                aggregate_output = summary_output.with_name(f"{summary_output.stem}_stability_groups.csv")
                json_output = summary_output.with_name(f"{summary_output.stem}_stability.json")
                run_cmd(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/compare_tum_run_stability.py"),
                        str(summary_output),
                        "--output-csv",
                        str(detail_output),
                        "--aggregate-csv",
                        str(aggregate_output),
                        "--output-json",
                        str(json_output),
                    ],
                    PROJECT_ROOT,
                    run_log_dir / f"{summary_output.stem}_stability.log" if args.quiet_run_logs else None,
                )
                print(f"[TUM] stability details saved to {detail_output}")
                print(f"[TUM] stability groups saved to {aggregate_output}")


if __name__ == "__main__":
    main()
