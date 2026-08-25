#!/usr/bin/env python3
"""Locked benchmark protocols and provenance capture for GPS-SLAM.

This module changes configuration only when a caller explicitly selects a
benchmark protocol.  It does not tune the SLAM algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from benchmark_freeze import require_clean_freeze


PROTOCOL_NONE = "none"
REPLICA_PAPER_V1 = "gps_slam_replica_paper_v1"
REPLICA_DEPLOYED_QUALITY_V1 = "gps_slam_replica_deployed_quality_v1"
TUM_TRACKING_V1 = "gps_slam_tum_tracking_v1"
TUM_INTEGRATED_V1 = "gps_slam_tum_integrated_v1"
TUM_DEPLOYED_V1 = "gps_slam_tum_deployed_v1"
PROTOCOLS = (
    PROTOCOL_NONE,
    REPLICA_PAPER_V1,
    REPLICA_DEPLOYED_QUALITY_V1,
    TUM_TRACKING_V1,
    TUM_INTEGRATED_V1,
    TUM_DEPLOYED_V1,
)
PUBLICATION_PROTOCOLS = (REPLICA_PAPER_V1, TUM_TRACKING_V1, TUM_INTEGRATED_V1)
UNPROMOTED_PROTOCOLS = (REPLICA_DEPLOYED_QUALITY_V1, TUM_DEPLOYED_V1)
ROLES = ("auto", "baseline", "modified", "integrated")


def protocol_release_status(protocol: str) -> str:
    if protocol in PUBLICATION_PROTOCOLS:
        return "publication_locked"
    if protocol in UNPROMOTED_PROTOCOLS:
        return "unpromoted_screen_only"
    if protocol == PROTOCOL_NONE:
        return "unlocked"
    raise ValueError(f"unknown benchmark protocol: {protocol}")


@lru_cache(maxsize=1)
def benchmark_freeze_provenance() -> dict[str, Any]:
    """Verify the immutable comparison envelope once per runner process."""
    report = require_clean_freeze()
    return {
        "freeze_id": report["freeze_id"],
        "manifest_path": report["manifest_path"],
        "manifest_sha256": report["manifest_sha256"],
        "functional_digest": report["functional_digest"],
        "functional_file_count": report["functional_file_count"],
        "verified_clean": report["clean"],
    }


def nested_get(cfg: dict, path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def nested_set(cfg: dict, path: tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[path[-1]] = value


def infer_binary_hash_profile(binary: Path | None) -> str:
    if binary is None:
        return "unknown"
    cache = binary.resolve().parent / "CMakeCache.txt"
    if not cache.exists():
        return "unknown"
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("GPS_SLAM_TSDF_HASH_PROFILE:") and "=" in line:
            return line.split("=", 1)[1].strip() or "unknown"
    return "unknown"


def replica_role_for_label(label: str) -> str:
    if label == "basic_unified_gaussian":
        return "baseline"
    if label == "multi_hybrid_warm_fill":
        return "modified"
    raise ValueError(
        f"Replica benchmark label {label!r} is not locked; use "
        "basic_unified_gaussian or multi_hybrid_warm_fill"
    )


def tum_role_for_variant(variant: str, protocol: str) -> str:
    if protocol == TUM_TRACKING_V1:
        if variant == "basic_sdf":
            return "baseline"
        if variant == "multi_sdf_orb_apply":
            return "modified"
    elif protocol in {TUM_INTEGRATED_V1, TUM_DEPLOYED_V1} and variant == "multi_hybrid_orb_apply":
        return "integrated"
    raise ValueError(f"TUM benchmark variant {variant!r} is not valid for {protocol}")


def _set_many(cfg: dict, values: dict[tuple[str, ...], Any]) -> None:
    for path, value in values.items():
        nested_set(cfg, path, value)


def _disable_offline_gaussian_work(cfg: dict) -> None:
    gaussian = cfg.setdefault("PIPE", {}).setdefault("gaussian_submaps", {})
    for key in (
        "quality_finalize_enable",
        "quality_finalize_insert_enable",
        "frozen_refinement_enable",
        "graduation_optimize",
        "render_overlap_diagnostics",
        "gaussian_insert_diagnostics",
    ):
        gaussian[key] = False


def _apply_deployed_paging(cfg: dict) -> None:
    _set_many(
        cfg,
        {
            ("PIPE", "TSDF", "multi", "paging"): True,
            ("PIPE", "TSDF", "multi", "paging_storage"): "cpu",
            ("PIPE", "TSDF", "multi", "resident_gpu_maps"): 8,
            ("PIPE", "TSDF", "multi", "cpu_page_cache_limit_mb"): 12288,
            ("PIPE", "TSDF", "multi", "async_page_out"): True,
            ("PIPE", "TSDF", "multi", "max_pending_page_snapshots"): 3,
            ("PIPE", "TSDF", "multi", "max_pending_page_snapshot_mb"): 2048,
            ("PIPE", "TSDF", "multi", "async_page_in"): True,
            ("PIPE", "TSDF", "multi", "max_pending_page_ins"): 2,
            ("PIPE", "TSDF", "multi", "max_pending_page_in_mb"): 1024,
        },
    )


def apply_protocol(
    cfg: dict,
    protocol: str,
    role: str,
    binary_hash_profile: str = "unknown",
) -> dict:
    """Apply the immutable comparison envelope and return ``cfg``."""
    if protocol == PROTOCOL_NONE:
        return cfg
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown benchmark protocol: {protocol}")
    if role not in ROLES[1:]:
        raise ValueError(f"benchmark role must be baseline, modified, or integrated; got {role}")

    _set_many(
        cfg,
        {
            ("PIPE", "benchmark_protocol"): protocol,
            ("PIPE", "benchmark_role"): role,
            ("READER", "frame_step"): 1,
            ("READER", "test_split_interval"): -1,
            ("PIPE", "eval_split"): "train",
            ("PIPE", "eval_stride"): 1,
            ("PIPE", "eval_max_frames"): 0,
            ("PIPE", "eval_start_offset"): 0,
            ("PIPE", "TSDF", "input_tensor_bridge"): "direct",
            ("PIPE", "TSDF", "input_tensor_bridge_validate"): False,
            ("PIPE", "TSDF", "input_prefetch"): False,
            ("PIPE", "TSDF", "multi", "console_diagnostics"): False,
            ("PIPE", "submap_stats_interval_frames"): 0,
            ("PIPE", "gaussian_schedule", "mode"): "immediate",
            ("PIPE", "gaussian_schedule", "iterations_per_frame"): 1,
            ("PIPE", "export_online_end_trajectory"): False,
        },
    )

    if protocol == REPLICA_PAPER_V1:
        _set_many(
            cfg,
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): True,
                ("READER", "start_frame"): 0,
                ("READER", "end_frame"): 1999,
                ("READER", "image_shape"): [1200, 680],
                ("READER", "downscale_factor"): 1,
                ("PIPE", "eval_after_train"): False,
                ("PIPE", "save_after_train"): True,
                ("PIPE", "new_gs_sample_ratio"): 0.25,
                ("PIPE", "local_opt_iters"): 20,
                ("PIPE", "local_opt_interval"): 10,
                ("PIPE", "keyframe_select_max"): 7,
                ("PIPE", "weight_configs", "ssim_weight"): 0.0,
                ("PIPE", "weight_configs", "depth_weight"): 0.0,
                ("PIPE", "TSDF", "input_mode"): "preload",
                ("PIPE", "TSDF", "use_gt_pose"): True,
                ("PIPE", "TSDF", "voxel_size"): 0.005,
                ("PIPE", "TSDF", "tracking_failure_mode"): "legacy_ignore",
                ("PIPE", "TSDF", "tracking_profile"): "legacy",
                ("PIPE", "TSDF", "use_bilateral_filter"): False,
                ("PIPE", "TSDF", "use_approximate_raycast"): False,
                ("MODEL", "sh_degree"): 3,
            },
        )
        gaussian = cfg.setdefault("PIPE", {}).setdefault("gaussian_submaps", {})
        pose_graph = cfg["PIPE"].setdefault("pose_graph", {})
        for key in (
            "quality_finalize_enable",
            "quality_finalize_insert_enable",
            "frozen_refinement_enable",
            "graduation_optimize",
            "render_overlap_diagnostics",
            "gaussian_insert_diagnostics",
        ):
            gaussian[key] = False
        _set_many(
            cfg,
            {
                ("PIPE", "pose_graph", "candidate_detection"): False,
                ("PIPE", "pose_graph", "optimize"): False,
                ("PIPE", "pose_graph", "dry_run"): False,
                ("PIPE", "pose_graph", "solve_only"): False,
                ("PIPE", "pose_graph", "online_updates"): False,
                ("PIPE", "pose_graph", "apply_mode"): "disabled",
            },
        )
        if role == "baseline":
            _set_many(
                cfg,
                {
                    ("PIPE", "TSDF", "engine"): "basic",
                    ("PIPE", "gaussian_submaps", "enable"): False,
                    ("PIPE", "gaussian_submaps", "ownership"): "legacy",
                    ("PIPE", "gaussian_submaps", "coordinate_frame"): "global",
                },
            )
        elif role == "modified":
            _set_many(
                cfg,
                {
                    ("PIPE", "TSDF", "engine"): "multi",
                    ("PIPE", "gaussian_submaps", "enable"): True,
                    ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                    ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                    ("PIPE", "gaussian_submaps", "debug"): False,
                },
            )
        else:
            raise ValueError(f"{protocol} does not use role {role}")
    elif protocol == REPLICA_DEPLOYED_QUALITY_V1:
        if role != "modified":
            raise ValueError(f"{protocol} requires role modified")
        _set_many(
            cfg,
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): False,
                ("READER", "start_frame"): 0,
                ("READER", "end_frame"): 1999,
                ("READER", "image_shape"): [1200, 680],
                ("READER", "downscale_factor"): 1,
                ("PIPE", "eval_after_train"): False,
                ("PIPE", "save_after_train"): True,
                ("PIPE", "new_gs_sample_ratio"): 0.25,
                ("PIPE", "local_opt_iters"): 20,
                ("PIPE", "local_opt_interval"): 10,
                ("PIPE", "keyframe_select_max"): 7,
                ("PIPE", "weight_configs", "ssim_weight"): 0.0,
                ("PIPE", "weight_configs", "depth_weight"): 0.0,
                ("PIPE", "gaussian_schedule", "mode"): "sliced",
                ("PIPE", "gaussian_schedule", "iterations_per_frame"): 2,
                ("PIPE", "TSDF", "input_mode"): "on_demand",
                ("PIPE", "TSDF", "input_prefetch"): True,
                ("PIPE", "TSDF", "use_gt_pose"): True,
                ("PIPE", "TSDF", "voxel_size"): 0.005,
                ("PIPE", "TSDF", "tracking_failure_mode"): "legacy_ignore",
                ("PIPE", "TSDF", "tracking_profile"): "legacy",
                ("PIPE", "TSDF", "use_bilateral_filter"): False,
                ("PIPE", "TSDF", "use_approximate_raycast"): False,
                ("PIPE", "TSDF", "engine"): "multi",
                ("PIPE", "gaussian_submaps", "enable"): True,
                ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                ("PIPE", "gaussian_submaps", "debug"): False,
                ("MODEL", "sh_degree"): 3,
                ("PIPE", "pose_graph", "candidate_detection"): False,
                ("PIPE", "pose_graph", "online_updates"): False,
                ("PIPE", "pose_graph", "final_candidate_scan"): False,
                ("PIPE", "pose_graph", "terminal_online_scan"): False,
                ("PIPE", "pose_graph", "apply_mode"): "disabled",
            },
        )
        _apply_deployed_paging(cfg)
        _disable_offline_gaussian_work(cfg)
    elif protocol in {TUM_TRACKING_V1, TUM_INTEGRATED_V1}:
        _set_many(
            cfg,
            {
                ("READER", "preload_images"): True,
                ("PIPE", "TSDF", "input_mode"): "preload",
                ("PIPE", "TSDF", "use_gt_pose"): False,
                ("PIPE", "TSDF", "tracking_profile"): "legacy",
                ("PIPE", "TSDF", "tracking_failure_mode"): "legacy_ignore",
                ("PIPE", "TSDF", "use_bilateral_filter"): False,
                ("PIPE", "TSDF", "use_approximate_raycast"): False,
                ("PIPE", "TSDF", "require_raw_tracking_success"): False,
                ("PIPE", "eval_after_train"): False,
            },
        )
        if protocol == TUM_TRACKING_V1:
            _set_many(
                cfg,
                {
                    ("work_mode",): "recon",
                    ("PIPE", "save_after_train"): False,
                    ("PIPE", "gaussian_submaps", "enable"): False,
                },
            )
            if role == "baseline":
                _set_many(
                    cfg,
                    {
                        ("PIPE", "TSDF", "engine"): "basic",
                        ("PIPE", "pose_graph", "candidate_detection"): False,
                        ("PIPE", "pose_graph", "online_updates"): False,
                        ("PIPE", "pose_graph", "apply_mode"): "disabled",
                    },
                )
            elif role == "modified":
                nested_set(cfg, ("PIPE", "TSDF", "engine"), "multi")
            else:
                raise ValueError(f"{protocol} does not use role {role}")
        else:
            if role != "integrated":
                raise ValueError(f"{protocol} requires role integrated")
            _set_many(
                cfg,
                {
                    ("work_mode",): "train",
                    ("PIPE", "save_after_train"): True,
                    ("PIPE", "TSDF", "engine"): "multi",
                    ("PIPE", "gaussian_submaps", "enable"): True,
                    ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                    ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                    ("PIPE", "gaussian_submaps", "debug"): False,
                    ("PIPE", "new_gs_sample_ratio"): 0.25,
                    ("PIPE", "local_opt_iters"): 20,
                    ("PIPE", "local_opt_interval"): 10,
                    ("PIPE", "keyframe_select_max"): 7,
                    ("PIPE", "weight_configs", "ssim_weight"): 0.0,
                    ("PIPE", "weight_configs", "depth_weight"): 0.0,
                    ("MODEL", "sh_degree"): 3,
                },
            )
            gaussian = cfg["PIPE"].setdefault("gaussian_submaps", {})
            for key in (
                "quality_finalize_enable",
                "quality_finalize_insert_enable",
                "frozen_refinement_enable",
                "graduation_optimize",
                "render_overlap_diagnostics",
                "gaussian_insert_diagnostics",
            ):
                gaussian[key] = False
    elif protocol == TUM_DEPLOYED_V1:
        if role != "integrated":
            raise ValueError(f"{protocol} requires role integrated")
        _set_many(
            cfg,
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): False,
                ("PIPE", "save_after_train"): True,
                ("PIPE", "eval_after_train"): False,
                ("PIPE", "TSDF", "engine"): "multi",
                ("PIPE", "TSDF", "input_mode"): "on_demand",
                ("PIPE", "TSDF", "input_prefetch"): True,
                ("PIPE", "TSDF", "use_gt_pose"): False,
                ("PIPE", "TSDF", "tracking_profile"): "robust_rpe",
                ("PIPE", "TSDF", "tracking_failure_mode"): "contain_failed",
                ("PIPE", "TSDF", "reject_nonfinite_tracking_pose"): True,
                ("PIPE", "TSDF", "use_bilateral_filter"): False,
                ("PIPE", "TSDF", "use_approximate_raycast"): False,
                ("PIPE", "TSDF", "require_raw_tracking_success"): False,
                ("PIPE", "gaussian_submaps", "enable"): True,
                ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                ("PIPE", "gaussian_submaps", "debug"): False,
                ("PIPE", "new_gs_sample_ratio"): 0.25,
                ("PIPE", "local_opt_iters"): 20,
                ("PIPE", "local_opt_interval"): 10,
                ("PIPE", "keyframe_select_max"): 7,
                ("PIPE", "weight_configs", "ssim_weight"): 0.0,
                ("PIPE", "weight_configs", "depth_weight"): 0.0,
                ("PIPE", "gaussian_schedule", "mode"): "sliced",
                ("PIPE", "gaussian_schedule", "iterations_per_frame"): 2,
                ("PIPE", "export_online_end_trajectory"): True,
                ("PIPE", "pose_graph", "online_updates"): True,
                ("PIPE", "pose_graph", "run_on_submap_events"): False,
                ("PIPE", "pose_graph", "candidate_interval_frames"): 60,
                ("PIPE", "pose_graph", "max_scanned_pairs_per_check"): 8,
                ("PIPE", "pose_graph", "max_candidate_pairs_per_check"): 1,
                ("PIPE", "pose_graph", "span_reserve_pairs"): 1,
                ("PIPE", "pose_graph", "orb_feature_cache"): True,
                ("PIPE", "pose_graph", "orb_feature_cache_limit"): 512,
                ("PIPE", "pose_graph", "final_candidate_scan"): False,
                ("PIPE", "pose_graph", "terminal_online_scan"): False,
                ("PIPE", "pose_graph", "final_rank_all_candidates"): False,
                ("MODEL", "sh_degree"): 3,
            },
        )
        _apply_deployed_paging(cfg)
        _disable_offline_gaussian_work(cfg)

    if binary_hash_profile != "unknown" and role in {"modified", "integrated"}:
        nested_set(cfg, ("PIPE", "TSDF", "multi", "hash_profile"), binary_hash_profile)
    validate_protocol(cfg, protocol, role, binary_hash_profile)
    return cfg


def _expected_fields(protocol: str, role: str) -> dict[tuple[str, ...], Any]:
    common: dict[tuple[str, ...], Any] = {
        ("PIPE", "benchmark_protocol"): protocol,
        ("PIPE", "benchmark_role"): role,
        ("READER", "frame_step"): 1,
        ("READER", "test_split_interval"): -1,
        ("PIPE", "TSDF", "input_tensor_bridge"): "direct",
        ("PIPE", "TSDF", "input_tensor_bridge_validate"): False,
        ("PIPE", "gaussian_schedule", "mode"): "immediate",
        ("PIPE", "gaussian_schedule", "iterations_per_frame"): 1,
        ("PIPE", "export_online_end_trajectory"): False,
    }
    if protocol == REPLICA_PAPER_V1:
        common.update(
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): True,
                ("READER", "start_frame"): 0,
                ("READER", "end_frame"): 1999,
                ("READER", "image_shape"): [1200, 680],
                ("READER", "downscale_factor"): 1,
                ("PIPE", "eval_split"): "train",
                ("PIPE", "new_gs_sample_ratio"): 0.25,
                ("PIPE", "local_opt_iters"): 20,
                ("PIPE", "local_opt_interval"): 10,
                ("PIPE", "keyframe_select_max"): 7,
                ("PIPE", "weight_configs", "ssim_weight"): 0.0,
                ("PIPE", "weight_configs", "depth_weight"): 0.0,
                ("PIPE", "TSDF", "use_gt_pose"): True,
                ("PIPE", "TSDF", "voxel_size"): 0.005,
                ("PIPE", "TSDF", "tracking_failure_mode"): "legacy_ignore",
                ("PIPE", "TSDF", "tracking_profile"): "legacy",
                ("PIPE", "TSDF", "use_bilateral_filter"): False,
                ("PIPE", "TSDF", "use_approximate_raycast"): False,
                ("MODEL", "sh_degree"): 3,
                ("PIPE", "pose_graph", "apply_mode"): "disabled",
                ("PIPE", "gaussian_submaps", "quality_finalize_enable"): False,
                ("PIPE", "gaussian_submaps", "quality_finalize_insert_enable"): False,
                ("PIPE", "gaussian_submaps", "frozen_refinement_enable"): False,
                ("PIPE", "gaussian_submaps", "graduation_optimize"): False,
            }
        )
        if role == "baseline":
            common.update(
                {
                    ("PIPE", "TSDF", "engine"): "basic",
                    ("PIPE", "gaussian_submaps", "enable"): False,
                }
            )
        elif role == "modified":
            common.update(
                {
                    ("PIPE", "TSDF", "engine"): "multi",
                    ("PIPE", "gaussian_submaps", "enable"): True,
                    ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                    ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                    ("PIPE", "gaussian_submaps", "debug"): False,
                }
            )
    elif protocol == REPLICA_DEPLOYED_QUALITY_V1:
        common.update(
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): False,
                ("READER", "start_frame"): 0,
                ("READER", "end_frame"): 1999,
                ("READER", "image_shape"): [1200, 680],
                ("READER", "downscale_factor"): 1,
                ("PIPE", "TSDF", "engine"): "multi",
                ("PIPE", "TSDF", "input_mode"): "on_demand",
                ("PIPE", "TSDF", "input_prefetch"): True,
                ("PIPE", "TSDF", "use_gt_pose"): True,
                ("PIPE", "gaussian_schedule", "mode"): "sliced",
                ("PIPE", "gaussian_schedule", "iterations_per_frame"): 2,
                ("PIPE", "gaussian_submaps", "enable"): True,
                ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                ("PIPE", "gaussian_submaps", "debug"): False,
                ("PIPE", "pose_graph", "apply_mode"): "disabled",
                ("PIPE", "pose_graph", "final_candidate_scan"): False,
                ("PIPE", "TSDF", "multi", "resident_gpu_maps"): 8,
                ("PIPE", "TSDF", "multi", "paging"): True,
                ("PIPE", "TSDF", "multi", "paging_storage"): "cpu",
                ("PIPE", "TSDF", "multi", "cpu_page_cache_limit_mb"): 12288,
                ("PIPE", "TSDF", "multi", "async_page_out"): True,
                ("PIPE", "TSDF", "multi", "max_pending_page_snapshots"): 3,
                ("PIPE", "TSDF", "multi", "max_pending_page_snapshot_mb"): 2048,
                ("PIPE", "TSDF", "multi", "async_page_in"): True,
                ("PIPE", "TSDF", "multi", "max_pending_page_ins"): 2,
                ("PIPE", "TSDF", "multi", "max_pending_page_in_mb"): 1024,
            }
        )
    elif protocol == TUM_TRACKING_V1:
        common.update(
            {
                ("work_mode",): "recon",
                ("READER", "preload_images"): True,
                ("PIPE", "TSDF", "use_gt_pose"): False,
                ("PIPE", "TSDF", "tracking_failure_mode"): "legacy_ignore",
                ("PIPE", "TSDF", "tracking_profile"): "legacy",
                ("PIPE", "TSDF", "use_bilateral_filter"): False,
                ("PIPE", "TSDF", "use_approximate_raycast"): False,
                ("PIPE", "gaussian_submaps", "enable"): False,
            }
        )
        if role == "baseline":
            common.update(
                {
                    ("PIPE", "TSDF", "engine"): "basic",
                    ("PIPE", "pose_graph", "apply_mode"): "disabled",
                }
            )
        elif role == "modified":
            common[("PIPE", "TSDF", "engine")] = "multi"
    elif protocol == TUM_INTEGRATED_V1:
        common.update(
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): True,
                ("PIPE", "TSDF", "engine"): "multi",
                ("PIPE", "TSDF", "use_gt_pose"): False,
                ("PIPE", "gaussian_submaps", "enable"): True,
                ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                ("PIPE", "gaussian_submaps", "debug"): False,
                ("PIPE", "gaussian_submaps", "quality_finalize_enable"): False,
                ("PIPE", "gaussian_submaps", "quality_finalize_insert_enable"): False,
                ("PIPE", "gaussian_submaps", "frozen_refinement_enable"): False,
            }
        )
    elif protocol == TUM_DEPLOYED_V1:
        common.update(
            {
                ("work_mode",): "train",
                ("READER", "preload_images"): False,
                ("PIPE", "TSDF", "engine"): "multi",
                ("PIPE", "TSDF", "input_mode"): "on_demand",
                ("PIPE", "TSDF", "input_prefetch"): True,
                ("PIPE", "TSDF", "use_gt_pose"): False,
                ("PIPE", "TSDF", "tracking_profile"): "robust_rpe",
                ("PIPE", "TSDF", "tracking_failure_mode"): "contain_failed",
                ("PIPE", "TSDF", "reject_nonfinite_tracking_pose"): True,
                ("PIPE", "gaussian_schedule", "mode"): "sliced",
                ("PIPE", "gaussian_schedule", "iterations_per_frame"): 2,
                ("PIPE", "export_online_end_trajectory"): True,
                ("PIPE", "gaussian_submaps", "enable"): True,
                ("PIPE", "gaussian_submaps", "ownership"): "tsdf",
                ("PIPE", "gaussian_submaps", "coordinate_frame"): "local",
                ("PIPE", "gaussian_submaps", "debug"): False,
                ("PIPE", "pose_graph", "online_updates"): True,
                ("PIPE", "pose_graph", "run_on_submap_events"): False,
                ("PIPE", "pose_graph", "candidate_interval_frames"): 60,
                ("PIPE", "pose_graph", "max_scanned_pairs_per_check"): 8,
                ("PIPE", "pose_graph", "max_candidate_pairs_per_check"): 1,
                ("PIPE", "pose_graph", "span_reserve_pairs"): 1,
                ("PIPE", "pose_graph", "orb_feature_cache_limit"): 512,
                ("PIPE", "pose_graph", "final_candidate_scan"): False,
                ("PIPE", "pose_graph", "terminal_online_scan"): False,
                ("PIPE", "pose_graph", "final_rank_all_candidates"): False,
                ("PIPE", "TSDF", "multi", "resident_gpu_maps"): 8,
                ("PIPE", "TSDF", "multi", "paging"): True,
                ("PIPE", "TSDF", "multi", "paging_storage"): "cpu",
                ("PIPE", "TSDF", "multi", "cpu_page_cache_limit_mb"): 12288,
                ("PIPE", "TSDF", "multi", "async_page_out"): True,
                ("PIPE", "TSDF", "multi", "max_pending_page_snapshots"): 3,
                ("PIPE", "TSDF", "multi", "max_pending_page_snapshot_mb"): 2048,
                ("PIPE", "TSDF", "multi", "async_page_in"): True,
                ("PIPE", "TSDF", "multi", "max_pending_page_ins"): 2,
                ("PIPE", "TSDF", "multi", "max_pending_page_in_mb"): 1024,
            }
        )
    return common


def validate_protocol(
    cfg: dict,
    protocol: str,
    role: str,
    binary_hash_profile: str = "unknown",
) -> None:
    issues: list[str] = []
    if protocol == REPLICA_PAPER_V1 and role not in {"baseline", "modified"}:
        issues.append(f"role {role!r} is invalid for {protocol}")
    if protocol == REPLICA_DEPLOYED_QUALITY_V1 and role != "modified":
        issues.append(f"role {role!r} is invalid for {protocol}")
    if protocol == TUM_TRACKING_V1 and role not in {"baseline", "modified"}:
        issues.append(f"role {role!r} is invalid for {protocol}")
    if protocol == TUM_INTEGRATED_V1 and role != "integrated":
        issues.append(f"role {role!r} is invalid for {protocol}")
    if protocol == TUM_DEPLOYED_V1 and role != "integrated":
        issues.append(f"role {role!r} is invalid for {protocol}")
    for path, expected in _expected_fields(protocol, role).items():
        actual = nested_get(cfg, path)
        if actual != expected:
            issues.append(f"{'.'.join(path)} expected {expected!r}, got {actual!r}")
    if role == "baseline" and binary_hash_profile not in {"unknown", "legacy_large"}:
        issues.append(
            "faithful baseline requires a legacy_large binary; "
            f"detected {binary_hash_profile!r}"
        )
    if role in {"modified", "integrated"}:
        replica_protocol = protocol in {REPLICA_PAPER_V1, REPLICA_DEPLOYED_QUALITY_V1}
        if not nested_get(cfg, ("PIPE", "pose_graph", "candidate_detection"), False) and not replica_protocol:
            issues.append("modified TUM protocol requires pose-graph candidate detection")
        if nested_get(cfg, ("PIPE", "pose_graph", "apply_mode"), "disabled") == "disabled" and not replica_protocol:
            issues.append("modified TUM protocol requires a non-disabled pose-graph apply mode")
        configured_profile = nested_get(cfg, ("PIPE", "TSDF", "multi", "hash_profile"), "unknown")
        if binary_hash_profile != "unknown" and configured_profile != binary_hash_profile:
            issues.append(
                "config/binary TSDF hash mismatch: "
                f"config={configured_profile!r}, binary={binary_hash_profile!r}"
            )
    if issues:
        raise ValueError("benchmark protocol validation failed:\n- " + "\n- ".join(issues))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(root: Path, mode: str = "inventory") -> dict[str, Any]:
    root = root.resolve()
    if mode == "none":
        return {"mode": mode, "root": str(root)}
    if mode not in {"inventory", "full"}:
        raise ValueError(f"unknown dataset hash mode: {mode}")
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        if mode == "full":
            digest.update(sha256_file(path).encode("ascii"))
            digest.update(b"\0")
        count += 1
        total_bytes += size
    return {
        "mode": mode,
        "root": str(root),
        "sha256": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total_bytes,
    }


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (completed.stdout + completed.stderr).strip()
    return output[:8000]


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return ""


def _clock_state() -> dict[str, Any]:
    """Read the host-visible CPU/GPU clock constraints from sysfs."""
    state: dict[str, Any] = {}
    frequency_groups: list[dict[str, Any]] = []
    cpu_roots = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"))
    # CPU policies are often shared. Deduplicate identical resolved cpufreq dirs.
    seen: set[str] = set()
    for root in cpu_roots:
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        frequency_groups.append(
            {
                "name": root.parent.name,
                "kind": "cpu",
                "governor": _read_optional(root / "scaling_governor"),
                "current_hz": _read_optional(root / "scaling_cur_freq"),
                "min_hz": _read_optional(root / "scaling_min_freq"),
                "max_hz": _read_optional(root / "scaling_max_freq"),
            }
        )
    for root in sorted(Path("/sys/class/devfreq").glob("gpu-*")):
        frequency_groups.append(
            {
                "name": root.name,
                "kind": "gpu",
                "governor": _read_optional(root / "governor"),
                "current_hz": _read_optional(root / "cur_freq"),
                "min_hz": _read_optional(root / "min_freq"),
                "max_hz": _read_optional(root / "max_freq"),
            }
        )
    state["frequency_groups"] = frequency_groups
    constrained = [
        group
        for group in frequency_groups
        if group.get("min_hz") and group.get("max_hz")
    ]
    state["all_reported_clocks_locked"] = bool(constrained) and all(
        group["min_hz"] == group["max_hz"] for group in constrained
    )
    return state


def runtime_provenance(binary: Path | None = None) -> dict[str, Any]:
    nvpmodel_state = os.environ.get("GPS_SLAM_HOST_NVPMODEL_STATE") or _command_output(
        ["nvpmodel", "-q"]
    )
    clocks = _clock_state()
    info: dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "jetson_model": _read_optional(Path("/proc/device-tree/model")),
        "l4t_release": _read_optional(Path("/etc/nv_tegra_release")),
        "container_image": os.environ.get("GPS_SLAM_CONTAINER_IMAGE", ""),
        "container_image_id": os.environ.get("GPS_SLAM_CONTAINER_IMAGE_ID", ""),
        "nvpmodel": nvpmodel_state,
        "jetson_clocks": _command_output(["jetson_clocks", "--show"]),
        "clock_state": clocks,
        "max_performance_state": (
            "MAXN" in nvpmodel_state.upper()
            and bool(clocks.get("all_reported_clocks_locked"))
        ),
        "nvcc": _command_output(["nvcc", "--version"]),
    }
    if binary is not None:
        resolved = binary.resolve()
        info["binary"] = str(resolved)
        info["binary_exists"] = resolved.is_file()
        info["binary_sha256"] = sha256_file(resolved) if resolved.is_file() else ""
        info["binary_hash_profile"] = infer_binary_hash_profile(resolved)
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else []
    except Exception as exc:  # pragma: no cover - depends on host/container
        info["torch_error"] = repr(exc)
    return info


def require_max_performance_state() -> None:
    """Reject headline execution unless MAXN and locked clocks are observable."""
    nvpmodel_state = os.environ.get("GPS_SLAM_HOST_NVPMODEL_STATE") or _command_output(
        ["nvpmodel", "-q"]
    )
    clocks = _clock_state()
    issues: list[str] = []
    if "MAXN" not in nvpmodel_state.upper():
        issues.append(f"nvpmodel is not MAXN: {nvpmodel_state!r}")
    if not clocks.get("all_reported_clocks_locked"):
        issues.append("CPU/GPU sysfs min/max frequencies are not locked")
    if issues:
        raise RuntimeError(
            "headline benchmark requires MAXN with locked clocks:\n- "
            + "\n- ".join(issues)
        )


def write_manifest(
    output: Path,
    *,
    protocol: str,
    role: str,
    cfg: dict,
    config_path: Path,
    binary: Path | None,
    dataset_root: Path,
    dataset_hash_mode: str = "inventory",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_protocol(cfg, protocol, role, infer_binary_hash_profile(binary))
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "protocol": protocol,
        "protocol_release_status": protocol_release_status(protocol),
        "role": role,
        "code_freeze": benchmark_freeze_provenance(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset": dataset_fingerprint(dataset_root, dataset_hash_mode),
        "runtime": runtime_provenance(binary),
    }
    if extra:
        manifest["run"] = extra
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS[1:], required=True)
    parser.add_argument("--role", choices=ROLES[1:], required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-hash", choices=["none", "inventory", "full"], default="inventory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    validate_protocol(cfg, args.protocol, args.role, infer_binary_hash_profile(args.binary))
    write_manifest(
        args.output,
        protocol=args.protocol,
        role=args.role,
        cfg=cfg,
        config_path=args.config,
        binary=args.binary,
        dataset_root=args.dataset_root,
        dataset_hash_mode=args.dataset_hash,
    )
    print(f"benchmark manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
