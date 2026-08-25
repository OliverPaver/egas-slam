#!/usr/bin/env python3
"""
Canonical smoke/regression runner for hybrid MultiEngine Gaussian submaps.

It generates matched configs from the release office0 config and runs small
gated experiments before any long run is attempted.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from benchmark_protocol import (  # noqa: E402
    PROTOCOL_NONE,
    REPLICA_DEPLOYED_QUALITY_V1,
    REPLICA_PAPER_V1,
    apply_protocol,
    protocol_release_status,
    replica_role_for_label,
    require_max_performance_state,
    sha256_file,
    write_manifest,
)
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "hybrid_validation"
BASE_CONFIG = PROJECT_ROOT / "configs" / "release" / "replica" / "office0.yaml"
ATE_SCRIPT = PROJECT_ROOT / "scripts" / "ate_general.py"
METRIC_SCRIPT = PROJECT_ROOT / "scripts" / "metric_general.py"
SAMPLED_METRIC_SCRIPT = PROJECT_ROOT / "scripts" / "metric_sampled_eval.py"
GEO_SCRIPT = PROJECT_ROOT / "scripts" / "geo_general.py"
GT_HALF = PROJECT_ROOT / "data" / "replica" / "office0_half_clean"
GT_FULL = PROJECT_ROOT / "data" / "replica" / "office0"
REPLICA_PREPARED_ROOTS = [
    Path("/media/oliver/B6AC1273AC122DFD/GPS-SLAM-data/datasets/replica_prepared"),
    Path("/media/oliver/New Volume/GPS-SLAM-data/datasets/replica_prepared"),
    PROJECT_ROOT / "data" / "replica",
]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
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


def parse_requested_labels(raw_labels: str, available_labels: set[str]) -> set[str]:
    """Parse and validate the optional comma-separated experiment selection."""
    if not raw_labels.strip():
        return set()
    parts = raw_labels.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError("--labels contains an empty label")
    labels = {
        validate_path_token(part.strip(), "--labels")
        for part in parts
    }
    unknown = sorted(labels - available_labels)
    if unknown:
        raise ValueError(
            "unknown --labels value(s): "
            + ", ".join(unknown)
            + "; choose from: "
            + ", ".join(sorted(available_labels))
        )
    return labels


def validation_exit_failures(
    rows: list[dict], gates: dict
) -> tuple[list[str], list[str]]:
    """Return failed run names and failed hard-gate names."""
    failed_runs = [
        str(row.get("name", "<unnamed>"))
        for row in rows
        if row.get("status") not in {"OK", "GENERATED"}
    ]
    failed_gates: set[str] = set()
    failed_row_gate_names: set[str] = set()
    for row in rows:
        run_name = str(row.get("name", "<unnamed>"))
        for name, value in row.items():
            if name.endswith("_gate") and str(value).upper() == "FAIL":
                failed_row_gate_names.add(name)
                failed_gates.add(f"{run_name}:{name}")
    for name, value in gates.items():
        if (
            name.endswith("_gate")
            and str(value).upper() == "FAIL"
            and not any(name.endswith(row_gate_name) for row_gate_name in failed_row_gate_names)
        ):
            failed_gates.add(str(name))
    return failed_runs, sorted(failed_gates)


def infer_binary_hash_profile(binary: Path) -> str:
    """Infer the build-time TSDF hash profile for a slam_trainer binary.

    The hash table capacity is compiled into InfiniTAM and affects BasicEngine
    as well as MultiEngine. Recording it prevents reduced-capacity diagnostic
    runs from being mistaken for original-capacity BasicEngine baselines.
    """
    resolved = binary.resolve()
    cache_path = resolved.parent / "CMakeCache.txt"
    if cache_path.exists():
        text = cache_path.read_text(errors="ignore")
        match = re.search(r"^GPS_SLAM_TSDF_HASH_PROFILE:STRING=(\S+)\s*$", text, re.MULTILINE)
        if match:
            return match.group(1)

    lowered = str(resolved).lower()
    if "compact" in lowered:
        return "compact_64k"
    if "medium" in lowered:
        return "medium_128k"
    if "legacy" in lowered:
        return "legacy_large"
    return "unknown"


def metric_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "_comp" not in path.name.lower()
    ]


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def dataset_metric_root_matches_render(gt_dir: Path, render_dir: Path) -> bool:
    """Return true when the original dataset root can be scored directly.

    `metric_general.py` is the historical Replica metric path and expects
    `camera/` and `depth/` under the GT root. Explicit full-resolution inputs
    can still render half-resolution eval images, so we only use that path when
    the image dimensions match. Otherwise the runner uses eval-produced sampled
    GT images with the same dimensions as the renders.
    """
    camera_dir = gt_dir / "camera"
    depth_dir = gt_dir / "depth"
    if not camera_dir.exists() or not depth_dir.exists():
        return False
    render_files = metric_image_files(render_dir)
    gt_files = metric_image_files(camera_dir)
    if not render_files or not gt_files:
        return False
    return image_size(render_files[0]) == image_size(gt_files[0])


def suffix_value(value) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def safe_experiment_name(name: str, max_len: int = 180) -> str:
    """Keep generated config/workspace names below common filename limits."""
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    keep = max(16, max_len - len(digest) - 1)
    return f"{name[:keep]}_{digest}"


def deep_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def apply_reader_preload_policy(cfg: dict, policy: str) -> None:
    """Keep generated reader residency consistent with the TSDF input path.

    The release Replica config preloads frames because it is optimized for a
    desktop paper run. Thor-style generated configs use TSDF on-demand input,
    so retaining READER.preload_images=True gives a false memory profile.
    """
    if policy == "preserve":
        return

    reader_cfg = cfg.setdefault("READER", {})
    if policy == "true":
        reader_cfg["preload_images"] = True
        return
    if policy == "false":
        reader_cfg["preload_images"] = False
        return

    tsdf_mode = str(cfg.get("PIPE", {}).get("TSDF", {}).get("input_mode", "preload")).lower()
    reader_cfg["preload_images"] = tsdf_mode != "on_demand"


def resolve_replica_input_dir(scene_name: str, half: bool, explicit_input_dir: Path | None) -> Path:
    """Resolve the dataset directory used by a Replica scene run.

    The runner started as an office0-only validation helper, so changing only
    --scene-name used to change labels without changing input data. For fair
    cross-scene experiments, scene names now resolve to the prepared Replica
    dataset root or fail loudly.
    """
    if explicit_input_dir is not None:
        return explicit_input_dir.resolve()

    if scene_name == "office0":
        return GT_HALF if half else GT_FULL

    relative_names = [f"{scene_name}_half_clean"] if half else [scene_name]
    for root in REPLICA_PREPARED_ROOTS:
        for relative_name in relative_names:
            if root.name == "replica":
                candidate = root / relative_name
            else:
                candidate = root / ("half" if half else "full") / relative_name
            if candidate.exists():
                return candidate.resolve()

    resolution = "half" if half else "full"
    searched = []
    for root in REPLICA_PREPARED_ROOTS:
        for relative_name in relative_names:
            if root.name == "replica":
                searched.append(str(root / relative_name))
            else:
                searched.append(str(root / resolution / relative_name))
    raise FileNotFoundError(
        f"Could not resolve Replica {resolution}-resolution scene '{scene_name}'. "
        "Pass --input-dir explicitly or prepare the scene under one of: "
        + ", ".join(searched)
    )


def load_base(
    frame_count: int,
    half: bool,
    name: str,
    output_root: Path,
    gaussian_profile: str = "fast",
    heldout_interval: int = -1,
    save_after_train: bool = True,
    input_dir: Path | None = None,
    scene_tag: str = "office0",
    start_frame: int = 0,
    frame_step: int = 1,
) -> dict:
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    start_frame = max(0, int(start_frame))
    frame_step = max(1, int(frame_step))
    reader_input_dir = input_dir if input_dir is not None else (GT_HALF if half else GT_FULL)
    cfg["workspace_dir"] = str(output_root / name)
    cfg["work_mode"] = "train"
    cfg["READER"]["input_dir"] = str(reader_input_dir)
    cfg["READER"]["end_frame"] = start_frame + (max(1, frame_count) - 1) * frame_step
    cfg["READER"]["start_frame"] = start_frame
    cfg["READER"]["frame_step"] = frame_step
    cfg["READER"]["test_split_interval"] = heldout_interval if heldout_interval > 0 else -1
    if half:
        cfg["READER"]["intrinsics"] = [300.0, 300.0, 299.75, 169.75]
        cfg["READER"]["image_shape"] = [600, 340]
        path_hint = str(reader_input_dir).lower()
        cfg["READER"]["downscale_factor"] = 1 if "half" in path_hint else 2
    else:
        cfg["READER"]["downscale_factor"] = 1
    cfg["PIPE"]["eval_after_train"] = False
    cfg["PIPE"]["eval_split"] = "train"
    cfg["PIPE"]["save_after_train"] = save_after_train
    cfg["PIPE"]["log_slam_state"] = False
    cfg["PIPE"]["TSDF"]["saved_mesh"] = ""
    cfg["PIPE"]["TSDF"]["saved_images"] = "raycasted"
    thor_quality_profiles = {
        "thor_quality",
        "thor_quality_adaptive",
        "thor_quality_candidate_handoff2",
        "thor_quality_loss_guided",
        "thor_quality_refine",
        "thor_quality_xnovelty",
        "thor_quality_xnovelty_refine",
        "thor_quality_reconciled",
        "thor_quality_reconciled_perceptual",
        "thor_quality_reconciled_finalize",
        "thor_quality_reconciled_spatial64",
        "thor_quality_reconciled_adaptive_spatial64",
        "thor_quality_reconciled_frozen_refine",
        "thor_quality_reconciled_frozen_guarded",
        "thor_quality_reconciled_xboundary",
        "thor_quality_reconciled_xboundary_accept",
        "thor_quality_reconciled_xboundary_readiness_accept",
        "thor_quality_reconciled_xboundary_sdferror_accept",
    }
    if gaussian_profile in {"fast", *thor_quality_profiles}:
        cfg["PIPE"]["new_gs_sample_ratio"] = 0.2
        cfg["PIPE"]["local_opt_iters"] = 12
        cfg["PIPE"]["local_opt_interval"] = 10
        cfg["PIPE"]["keyframe_select_max"] = 4
        cfg["MODEL"]["sh_degree"] = 1
        if gaussian_profile in thor_quality_profiles:
            cfg["PIPE"]["new_gs_sample_ratio"] = 0.25
            cfg["PIPE"]["local_opt_iters"] = 24
            cfg["PIPE"]["keyframe_select_max"] = 7
        if gaussian_profile == "thor_quality_reconciled_perceptual":
            cfg["PIPE"].setdefault("weight_configs", {})["ssim_weight"] = 0.2
    elif gaussian_profile != "quality":
        raise ValueError(f"Unknown Gaussian profile: {gaussian_profile}")
    cfg["scene_name"] = scene_tag
    return cfg


def hybrid_submap_lifecycle_defaults(frame_count: int, profile: str) -> dict:
    """Return MultiEngine lifecycle defaults for hybrid Gaussian validation.

    Normal validation should not accidentally use forced-boundary stress
    settings. Transition and stress profiles remain available for explicit
    submap-boundary and residency coverage.
    """
    if profile == "normal":
        return {
            "max_frames": 750,
            "overlap_frames": 20,
            "trial_frames": 30,
        }
    if profile == "transition":
        return {
            "max_frames": 60 if frame_count <= 180 else 120,
            "overlap_frames": 8,
            "trial_frames": 12,
        }
    if profile == "stress":
        return {
            "max_frames": 25,
            "overlap_frames": 3,
            "trial_frames": 6,
        }
    raise ValueError(f"Unknown hybrid submap profile: {profile}")


def tsdf_residency_profile_defaults(profile: str) -> dict:
    """Named TSDF residency policies for reproducible long-run comparisons.

    These presets only affect MultiEngine configs. Low-level CLI overrides are
    applied afterwards, so experiments can still adjust one field without
    copying every residency knob into a command line.
    """
    if profile == "manual":
        return {}
    if profile == "thor_cpu_bounded":
        return {
            "paging": True,
            "paging_storage": "cpu",
            "resident_gpu_maps": 8,
            "async_page_out": True,
            "max_pending_page_snapshots": 2,
            "max_pending_page_snapshot_mb": 1200,
            "async_page_in": True,
            "max_pending_page_ins": 1,
            "max_pending_page_in_mb": 512,
            "cpu_page_cache_limit_mb": 8192,
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


def experiment_matrix(frame_count: int, half: bool, gaussian_profile: str, hybrid_submap_profile: str) -> list[tuple[str, dict]]:
    hybrid_lifecycle = hybrid_submap_lifecycle_defaults(frame_count, hybrid_submap_profile)
    basic_unified_pipe = {
        "TSDF": {"engine": "basic", "input_mode": "on_demand", "input_prefetch": True, "use_gt_pose": True, "saved_engine": "tsdf_engine/"},
        "gaussian_submaps": {"enable": False},
    }
    multi_hybrid_pipe = {
        "TSDF": {
            "engine": "multi",
            "input_mode": "on_demand",
            "input_prefetch": True,
            "use_gt_pose": True,
            "saved_engine": "tsdf_engine/",
            "multi": {
                "force_single_submap": False,
                "loop_closure": False,
                "global_adjustment": False,
                "overlap_frames": hybrid_lifecycle["overlap_frames"],
                "trial_frames": hybrid_lifecycle["trial_frames"],
                "sampled_original_blocks": 300,
                "max_frames": hybrid_lifecycle["max_frames"],
                "new_map_block_occupancy": 0.86,
                "max_block_occupancy": 0.92,
                "occupancy_growth_lookahead_frames": 30,
                "occupancy_growth_min_primary_frames": 20,
                "occupancy_growth_sample_frames": 5,
                "paging": True,
                "resident_gpu_maps": "auto",
                "max_render_maps": 4,
                "render_compositor": "handoff_aware",
            },
        },
        "gaussian_submaps": {
            "enable": True,
            "debug": True,
            "ownership": "tsdf",
            "coordinate_frame": "local",
            "max_new_gaussians_per_frame": 1000,
            "bootstrap_max_new_gaussians_per_frame": 0,
            "bootstrap_until_gaussians": 0,
            "bootstrap_until_age_frames": 0,
            "gaussian_insert_diagnostics": False,
            "gaussian_insert_min_age_frames": 0,
            "gaussian_insert_min_sdf_blocks": 0,
            "handoff_optimize": True,
            "candidate_warming": True,
            "boundary_observation_handoff": True,
            "gaussian_insert_selection": "error_topk",
            "candidate_warm_interval": 1,
            "retire_unaccepted_candidates": True,
            "candidate_reject_retire": True,
            "candidate_reject_retire_min_rejects": 4,
            "candidate_reject_retire_min_observed_frames": 16,
            "candidate_reject_retire_min_gaussians": 1,
            "candidate_reject_retire_max_inliers": 1,
            "candidate_reject_retire_cooldown_frames": 30,
            "candidate_gaussian_insert_min_frames": 0,
            "candidate_immature_new_gs_sample_ratio": -1.0,
            "candidate_immature_max_new_gaussians_per_frame": 0,
            "candidate_extend_while_overlap": True,
            "candidate_max_warm_frames": hybrid_lifecycle["overlap_frames"] + hybrid_lifecycle["trial_frames"] + 8,
            "candidate_handoff_max_opt_passes": 6,
            "handoff_frames": 8,
            "handoff_opt_iters": 1,
            "handoff_max_opt_passes": 2,
            "candidate_new_gs_sample_ratio": -1.0,
            "candidate_max_new_gaussians_per_frame": 1000,
            "render_policy": "owner_neighbor_fill",
            "render_past_frame_margin": 20,
            "render_future_frame_margin": 0,
            "render_min_ready_frames": 16,
            "render_owner_min_ready_frames": 0,
            "render_neighbor_min_ready_frames": 5,
            "render_owner_alpha_fill_thres": 0.25,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_adaptive_kernel": 5,
            "render_adaptive_alpha_scale": 2.0,
            "render_adaptive_min_weight": 0.02,
            "render_adaptive_boundary_feather": 0.25,
            "render_adaptive_depth_thres": 0.05,
            "render_blend_alpha_thres": 0.1,
            "render_blend_depth_thres": 0.05,
            "render_blend_max_weight": 0.2,
            "render_blend_weight_scale": 0.2,
            "render_min_neighbor_gaussians": 0,
            "render_single_submap_passthrough": True,
            "render_single_submap_passthrough_requires_single_logical_map": False,
            "render_use_local_sdf_reference": True,
            "render_overlap_diagnostics": False,
            "page_frozen_gaussians": True,
            "resident_gaussian_submaps": 2,
            "save_global_merged_model": False,
        },
    }
    thor_quality_profiles = {
        "thor_quality",
        "thor_quality_adaptive",
        "thor_quality_candidate_handoff2",
        "thor_quality_loss_guided",
        "thor_quality_refine",
        "thor_quality_xnovelty",
        "thor_quality_xnovelty_refine",
        "thor_quality_reconciled",
        "thor_quality_reconciled_perceptual",
        "thor_quality_reconciled_finalize",
        "thor_quality_reconciled_spatial64",
        "thor_quality_reconciled_adaptive_spatial64",
        "thor_quality_reconciled_frozen_refine",
        "thor_quality_reconciled_frozen_guarded",
        "thor_quality_reconciled_xboundary",
        "thor_quality_reconciled_xboundary_accept",
        "thor_quality_reconciled_xboundary_readiness_accept",
        "thor_quality_reconciled_xboundary_sdferror_accept",
    }
    if gaussian_profile in thor_quality_profiles:
        multi_hybrid_pipe["gaussian_submaps"].update({
            "bootstrap_max_new_gaussians_per_frame": 2500,
            "bootstrap_until_gaussians": 15000,
            "bootstrap_until_age_frames": 24,
            "handoff_frames": 12,
            "handoff_max_opt_passes": 3,
            "graduation_optimize": True,
            "graduation_opt_iters": 10,
            "graduation_keyframes": 12,
            "graduation_max_opt_passes": 1,
            "graduation_keyframe_policy": "loss_temporal_mix",
            "training_observation_max_per_submap": 16,
            "training_observation_interval_frames": 10,
            "render_selection_policy": "sdf_bounds",
            "render_max_visible_submaps": 8,
            "render_bounds_inflate": 0.05,
            "render_bounds_min_projected_area": 0.0,
            "resident_gaussian_submaps": 3,
        })
    if gaussian_profile in {"thor_quality_refine", "thor_quality_xnovelty_refine"}:
        multi_hybrid_pipe["gaussian_submaps"].update({
            "frozen_refinement_enable": True,
            "frozen_refinement_interval_frames": 30,
            "frozen_refinement_opt_iters": 1,
            "frozen_refinement_keyframes": 6,
            "frozen_refinement_max_passes_per_submap": 1,
            "frozen_refinement_max_submaps_per_frame": 1,
            "frozen_refinement_min_new_observations": 2,
        })
    if gaussian_profile == "thor_quality_adaptive":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "graduation_adaptive": True,
            "graduation_max_opt_passes": 2,
            "graduation_adaptive_loss_sample_cameras": 6,
            "graduation_adaptive_target_loss": 0.0,
            "graduation_adaptive_min_abs_drop": 0.0001,
            "graduation_adaptive_min_rel_drop": 0.005,
        })
    if gaussian_profile == "thor_quality_loss_guided":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "graduation_observation_policy": "loss_temporal_mix",
        })
    if gaussian_profile == "thor_quality_candidate_handoff2":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "candidate_handoff_opt_iters": 2,
        })
    if gaussian_profile in {"thor_quality_xnovelty", "thor_quality_xnovelty_refine"}:
        multi_hybrid_pipe["gaussian_submaps"].update({
            "cross_submap_novelty_gate": True,
            "cross_novelty_alpha_thres": 0.7,
            "cross_novelty_depth_abs_thres": 0.03,
            "cross_novelty_depth_rel_thres": 0.01,
            "cross_novelty_photometric_thres": 0.2,
            "cross_novelty_max_reference_submaps": 4,
            "cross_novelty_previous_only": True,
            "cross_novelty_resident_only": True,
            "cross_novelty_erode_kernel": 3,
            "gaussian_insert_diagnostics": True,
        })
    if gaussian_profile in {
        "thor_quality_reconciled",
        "thor_quality_reconciled_perceptual",
        "thor_quality_reconciled_finalize",
        "thor_quality_reconciled_spatial64",
        "thor_quality_reconciled_adaptive_spatial64",
        "thor_quality_reconciled_frozen_refine",
        "thor_quality_reconciled_frozen_guarded",
        "thor_quality_reconciled_xboundary",
        "thor_quality_reconciled_xboundary_accept",
        "thor_quality_reconciled_xboundary_readiness_accept",
        "thor_quality_reconciled_xboundary_sdferror_accept",
    }:
        multi_hybrid_pipe["gaussian_submaps"].update({
            "render_selection_policy": "sdf_bounds",
            "render_max_visible_submaps": 8,
            "render_max_visible_gaussians": 280000,
            "render_recompose_with_global_sdf_base": True,
            "render_overlap_depth_gate": True,
            "render_overlap_depth_tolerance_mode": "scaled_max",
            "render_overlap_depth_abs_thres": 0.05,
            "render_overlap_depth_floor": 0.015,
            "render_overlap_depth_voxel_scale": 0.75,
            "render_overlap_depth_quadratic_scale": 0.005,
            "render_overlap_alpha_thres": 0.05,
            "render_overlap_diagnostics": True,
            "render_selection_hysteresis_frames": 5,
            "render_selection_hysteresis_area_bonus": 0.05,
            "cross_submap_novelty_gate": True,
            "cross_novelty_reference_policy": "asymmetric_handoff",
            "cross_novelty_depth_tolerance_mode": "scaled_max",
            "cross_novelty_alpha_thres": 0.7,
            "cross_novelty_depth_floor": 0.015,
            "cross_novelty_depth_voxel_scale": 0.75,
            "cross_novelty_depth_quadratic_scale": 0.005,
            "cross_novelty_photometric_thres": 0.2,
            "cross_novelty_max_reference_submaps": 4,
            "cross_novelty_previous_only": True,
            "cross_novelty_resident_only": True,
            "cross_novelty_erode_kernel": 3,
            "gaussian_insert_diagnostics": True,
            "gaussian_budget_policy": "sdf_block_density",
            "gaussian_budget_base": 8000,
            "gaussian_budget_blocks_density": 0.40,
            "gaussian_budget_min": 8000,
            "gaussian_budget_max": 90000,
            "gaussian_budget_pressure_ratio": 0.90,
            "gaussian_budget_prune_interval_frames": 10,
            "graduation_readiness_enable": True,
            "graduation_readiness_sample_cameras": 5,
            "graduation_readiness_skip_gain": 0.0015,
            "graduation_readiness_min_abs_drop": 0.0001,
            "graduation_max_opt_passes": 2,
        })
    if gaussian_profile in {
        "thor_quality_reconciled_finalize",
        "thor_quality_reconciled_spatial64",
        "thor_quality_reconciled_adaptive_spatial64",
        "thor_quality_reconciled_frozen_refine",
        "thor_quality_reconciled_frozen_guarded",
        "thor_quality_reconciled_xboundary",
        "thor_quality_reconciled_xboundary_accept",
        "thor_quality_reconciled_xboundary_readiness_accept",
        "thor_quality_reconciled_xboundary_sdferror_accept",
    }:
        multi_hybrid_pipe["TSDF"]["multi"].update({
            "new_map_visible_ratio": 0.25,
            "sampled_original_blocks": 2000,
        })
        multi_hybrid_pipe["gaussian_submaps"].update({
            "quality_finalize_enable": True,
            "quality_finalize_include_frozen": True,
            "quality_finalize_include_active": True,
            "quality_finalize_resident_only": True,
            "quality_finalize_max_submaps": 2,
            "quality_finalize_max_observed_frames": 256,
            "quality_finalize_min_keyframes": 8,
            "quality_finalize_insert_enable": False,
            "quality_finalize_insert_include_frozen": False,
            "quality_finalize_insert_min_sdf_submaps": 2,
            "quality_finalize_insert_frames": 3,
            "quality_finalize_insert_max_new_gaussians_per_frame": 2000,
            "quality_finalize_insert_acceptance_enable": True,
            "quality_finalize_insert_acceptance_cameras": 5,
            "quality_finalize_insert_min_readiness_delta": 0.0,
            "graduation_readiness_skip_min_observed_frames": 300,
            "graduation_readiness_skip_min_keyframes": 8,
        })
    if gaussian_profile == "thor_quality_reconciled_spatial64":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "gaussian_insert_selection": "tile_error_topk",
            "gaussian_insert_tile_size": 64,
            "gaussian_insert_tile_max_pixels": 0,
        })
    if gaussian_profile == "thor_quality_reconciled_adaptive_spatial64":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "gaussian_insert_selection": "adaptive_tile_error_topk",
            "gaussian_insert_tile_size": 64,
            "gaussian_insert_tile_max_pixels": 0,
            "gaussian_insert_adaptive_tile_min_candidate_pixels": 1024,
            "gaussian_insert_adaptive_tile_min_target_insertions": 256,
            "gaussian_insert_adaptive_tile_min_submap_gaussians": 0,
            "gaussian_insert_adaptive_tile_min_observed_frames": 0,
        })
    if gaussian_profile == "thor_quality_reconciled_frozen_refine":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "training_observation_max_per_submap": 24,
            "frozen_refinement_enable": True,
            "frozen_refinement_interval_frames": 30,
            "frozen_refinement_opt_iters": 1,
            "frozen_refinement_keyframes": 8,
            "frozen_refinement_max_passes_per_submap": 2,
            "frozen_refinement_max_submaps_per_frame": 1,
            "frozen_refinement_min_new_observations": 2,
            "cross_boundary_refinement_enable": False,
        })
    if gaussian_profile == "thor_quality_reconciled_frozen_guarded":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "frozen_refinement_enable": True,
            "frozen_refinement_interval_frames": 60,
            "frozen_refinement_opt_iters": 1,
            "frozen_refinement_keyframes": 8,
            "frozen_refinement_max_passes_per_submap": 1,
            "frozen_refinement_max_submaps_per_frame": 1,
            "frozen_refinement_min_new_observations": 8,
            "frozen_refinement_readiness_guard_enable": True,
            "frozen_refinement_readiness_use_cached_graduation": True,
            "frozen_refinement_readiness_sample_cameras": 5,
            "frozen_refinement_readiness_max_pre_gain": 0.0,
            "frozen_refinement_readiness_min_valid_cameras": 2,
            "cross_boundary_refinement_enable": False,
        })
    if gaussian_profile in {
        "thor_quality_reconciled_xboundary",
        "thor_quality_reconciled_xboundary_accept",
        "thor_quality_reconciled_xboundary_readiness_accept",
        "thor_quality_reconciled_xboundary_sdferror_accept",
    }:
        multi_hybrid_pipe["gaussian_submaps"].update({
            "training_observation_max_per_submap": 24,
            "frozen_refinement_enable": True,
            "frozen_refinement_interval_frames": 30,
            "frozen_refinement_opt_iters": 1,
            "frozen_refinement_keyframes": 8,
            "frozen_refinement_max_passes_per_submap": 2,
            "frozen_refinement_max_submaps_per_frame": 1,
            "frozen_refinement_min_new_observations": 2,
            "cross_boundary_refinement_enable": True,
            "cross_boundary_observation_interval_frames": 10,
            "cross_boundary_max_submaps_per_frame": 1,
            "cross_boundary_min_owner_age_frames": 8,
            "cross_boundary_min_neighbor_gaussians": 512,
            "cross_boundary_frozen_only": True,
            "cross_boundary_resident_only": True,
            "cross_boundary_sequential_only": True,
            "cross_boundary_depth_gate": True,
            "cross_boundary_min_projected_ratio": 0.02,
            "cross_boundary_min_depth_consistent_ratio": 0.55,
            "cross_boundary_depth_tolerance_mode": "scaled_max",
            "cross_boundary_depth_abs_thres": 0.05,
            "cross_boundary_depth_floor": 0.015,
            "cross_boundary_depth_voxel_scale": 0.75,
            "cross_boundary_depth_quadratic_scale": 0.005,
        })
    if gaussian_profile in {
        "thor_quality_reconciled_xboundary_accept",
        "thor_quality_reconciled_xboundary_readiness_accept",
        "thor_quality_reconciled_xboundary_sdferror_accept",
    }:
        multi_hybrid_pipe["gaussian_submaps"].update({
            "frozen_refinement_acceptance_enable": True,
            "frozen_refinement_acceptance_min_delta": 0.0,
        })
    if gaussian_profile == "thor_quality_reconciled_xboundary_readiness_accept":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "frozen_refinement_observation_policy": "readiness_temporal_mix",
            "frozen_refinement_observation_selection_max_eval": 12,
        })
    if gaussian_profile == "thor_quality_reconciled_xboundary_sdferror_accept":
        multi_hybrid_pipe["gaussian_submaps"].update({
            "frozen_refinement_observation_policy": "sdf_error_temporal_mix",
            "frozen_refinement_observation_selection_max_eval": 8,
        })
    return [
        ("basic_sdf", {
            "work_mode": "recon",
            "PIPE": {
                "train_mode": "ges",
                "TSDF": {"engine": "basic", "input_mode": "on_demand", "input_prefetch": True, "use_gt_pose": False, "saved_engine": ""},
                "gaussian_submaps": {"enable": False},
            },
        }),
        ("basic_sdf_gt", {
            "work_mode": "recon",
            "PIPE": {
                "train_mode": "ges",
                "TSDF": {"engine": "basic", "input_mode": "on_demand", "input_prefetch": True, "use_gt_pose": True, "saved_engine": ""},
                "gaussian_submaps": {"enable": False},
            },
        }),
        ("basic_unified_gaussian", {
            "PIPE": basic_unified_pipe,
        }),
        ("multi_sdf", {
            "work_mode": "recon",
            "PIPE": {
                "TSDF": {
                    "engine": "multi",
                    "input_mode": "on_demand",
                    "input_prefetch": True,
                    "use_gt_pose": False,
                    "saved_engine": "",
                    "multi": {
                        "force_single_submap": False,
                        "loop_closure": False,
                        "global_adjustment": False,
                        "overlap_frames": 3,
                        "trial_frames": 6,
                        "sampled_original_blocks": 300,
                        "max_frames": 750,
                        "new_map_block_occupancy": 0.86,
                        "max_block_occupancy": 0.92,
                        "occupancy_growth_lookahead_frames": 30,
                        "occupancy_growth_min_primary_frames": 20,
                        "occupancy_growth_sample_frames": 5,
                        "paging": True,
                        "resident_gpu_maps": 2,
                        "max_render_maps": 4,
                    },
                },
                "gaussian_submaps": {"enable": False},
            },
        }),
        ("multi_hybrid_warm_fill", {
            "PIPE": multi_hybrid_pipe,
        }),
    ]


def parse_time_log(path: Path) -> dict:
    stats = {}
    if not path.exists():
        return stats
    for line in path.read_text(errors="ignore").splitlines():
        if "[PIPELINE AVG TIME]" not in line:
            continue
        for part in line.split(","):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.replace("[PIPELINE AVG TIME]", "").strip().replace(" ", "_")
            try:
                stats[key] = float(value.strip())
            except ValueError:
                pass
    return stats


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def summarize_timing(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    def values(column: str) -> list[float]:
        result = []
        for row in rows:
            try:
                result.append(float(row.get(column, "") or 0.0))
            except ValueError:
                pass
        return result

    total_ms = values("total_measured_ms")
    fusion_ms = values("fusion_ms")
    gaussian_ms = values("gaussianOps_ms")
    return {
        "timing_rows": len(rows),
        "frame_total_p95_ms": percentile(total_ms, 95),
        "frame_total_p99_ms": percentile(total_ms, 99),
        "fusion_p95_ms": percentile(fusion_ms, 95),
        "fusion_p99_ms": percentile(fusion_ms, 99),
        "gaussian_ops_p95_ms": percentile(gaussian_ms, 95),
        "gaussian_ops_p99_ms": percentile(gaussian_ms, 99),
    }


def summarize_paging(path: Path) -> dict:
    if not path.exists():
        return {}
    page_in_ms = []
    page_in_prefetch_ms = []
    page_out_ms = []
    page_out_backpressure_ms = []
    storage_modes = set()
    max_resident = None
    max_pending = None
    max_pending_pageins = None
    max_resident_estimated_bytes = 0
    max_pending_estimated_bytes = 0
    max_pending_pagein_estimated_bytes = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                duration = float(row.get("duration_ms", "") or 0.0)
            except ValueError:
                duration = 0.0
            action = row.get("action", "")
            if action == "page_in":
                page_in_ms.append(duration)
            elif action == "page_in_prefetch":
                page_in_prefetch_ms.append(duration)
            elif action == "page_out":
                page_out_ms.append(duration)
            elif action == "page_out_backpressure_wait":
                page_out_backpressure_ms.append(duration)
            storage_mode = row.get("storage_mode", "").strip()
            if storage_mode:
                storage_modes.add(storage_mode)
            try:
                resident = int(row.get("resident_count", "") or 0)
                max_resident = resident if max_resident is None else max(max_resident, resident)
            except ValueError:
                pass
            try:
                pending = int(row.get("pending_snapshots", "") or 0)
                max_pending = pending if max_pending is None else max(max_pending, pending)
            except ValueError:
                pass
            try:
                pending_pageins = int(row.get("pending_pageins", "") or 0)
                max_pending_pageins = pending_pageins if max_pending_pageins is None else max(max_pending_pageins, pending_pageins)
            except ValueError:
                pass
            try:
                max_resident_estimated_bytes = max(
                    max_resident_estimated_bytes,
                    int(float(row.get("resident_estimated_bytes", "") or 0)),
                )
            except ValueError:
                pass
            try:
                max_pending_estimated_bytes = max(
                    max_pending_estimated_bytes,
                    int(float(row.get("pending_estimated_bytes", "") or 0)),
                )
            except ValueError:
                pass
            try:
                max_pending_pagein_estimated_bytes = max(
                    max_pending_pagein_estimated_bytes,
                    int(float(row.get("pending_pagein_estimated_bytes", "") or 0)),
                )
            except ValueError:
                pass
    return {
        "page_in_count": len(page_in_ms),
        "page_in_prefetch_count": len(page_in_prefetch_ms),
        "page_out_count": len(page_out_ms),
        "page_out_backpressure_count": len(page_out_backpressure_ms),
        "page_in_p95_ms": percentile(page_in_ms, 95),
        "page_in_prefetch_p95_ms": percentile(page_in_prefetch_ms, 95),
        "page_out_p95_ms": percentile(page_out_ms, 95),
        "page_out_backpressure_p95_ms": percentile(page_out_backpressure_ms, 95),
        "page_in_max_ms": max(page_in_ms) if page_in_ms else None,
        "page_in_prefetch_max_ms": max(page_in_prefetch_ms) if page_in_prefetch_ms else None,
        "page_out_max_ms": max(page_out_ms) if page_out_ms else None,
        "page_out_backpressure_max_ms": max(page_out_backpressure_ms) if page_out_backpressure_ms else None,
        "paging_storage_modes": "|".join(sorted(storage_modes)) if storage_modes else "",
        "paging_max_resident": max_resident,
        "paging_max_pending_snapshots": max_pending,
        "paging_max_pending_pageins": max_pending_pageins,
        "paging_max_resident_estimated_mb": max_resident_estimated_bytes / (1024 * 1024) if max_resident_estimated_bytes else None,
        "paging_max_pending_estimated_mb": max_pending_estimated_bytes / (1024 * 1024) if max_pending_estimated_bytes else None,
        "paging_max_pending_pagein_estimated_mb": max_pending_pagein_estimated_bytes / (1024 * 1024) if max_pending_pagein_estimated_bytes else None,
    }


def config_fingerprint(cfg: dict) -> dict:
    pipe_cfg = cfg.get("PIPE", {}) if isinstance(cfg, dict) else {}
    weight_cfg = pipe_cfg.get("weight_configs", {}) if isinstance(pipe_cfg, dict) else {}
    tsdf_cfg = pipe_cfg.get("TSDF", {}) if isinstance(pipe_cfg, dict) else {}
    multi_cfg = tsdf_cfg.get("multi", {}) if isinstance(tsdf_cfg, dict) else {}
    gaussian_cfg = pipe_cfg.get("gaussian_submaps", {}) if isinstance(pipe_cfg, dict) else {}

    def get_multi(key: str):
        return multi_cfg.get(key) if isinstance(multi_cfg, dict) else None

    def get_gaussian(key: str):
        return gaussian_cfg.get(key) if isinstance(gaussian_cfg, dict) else None

    return {
        "cfg_random_seed": pipe_cfg.get("random_seed") if isinstance(pipe_cfg, dict) else None,
        "cfg_reader_start_frame": cfg.get("READER", {}).get("start_frame") if isinstance(cfg.get("READER", {}), dict) else None,
        "cfg_reader_end_frame": cfg.get("READER", {}).get("end_frame") if isinstance(cfg.get("READER", {}), dict) else None,
        "cfg_reader_frame_step": cfg.get("READER", {}).get("frame_step") if isinstance(cfg.get("READER", {}), dict) else None,
        "cfg_reader_preload_images": cfg.get("READER", {}).get("preload_images") if isinstance(cfg.get("READER", {}), dict) else None,
        "cfg_pipe_save_after_train": pipe_cfg.get("save_after_train") if isinstance(pipe_cfg, dict) else None,
        "cfg_pipe_ssim_weight": weight_cfg.get("ssim_weight") if isinstance(weight_cfg, dict) else None,
        "cfg_pipe_depth_weight": weight_cfg.get("depth_weight") if isinstance(weight_cfg, dict) else None,
        "cfg_pipe_localframe_cam_window_length": pipe_cfg.get("localframe_cam_window_length") if isinstance(pipe_cfg, dict) else None,
        "cfg_pipe_localframe_cam_window_interval": pipe_cfg.get("localframe_cam_window_interval") if isinstance(pipe_cfg, dict) else None,
        "cfg_pipe_local_opt_iters": pipe_cfg.get("local_opt_iters") if isinstance(pipe_cfg, dict) else None,
        "cfg_pipe_local_opt_interval": pipe_cfg.get("local_opt_interval") if isinstance(pipe_cfg, dict) else None,
        "cfg_pipe_keyframe_select_max": pipe_cfg.get("keyframe_select_max") if isinstance(pipe_cfg, dict) else None,
        "cfg_tsdf_engine": tsdf_cfg.get("engine") if isinstance(tsdf_cfg, dict) else None,
        "cfg_tsdf_input_mode": tsdf_cfg.get("input_mode") if isinstance(tsdf_cfg, dict) else None,
        "cfg_multi_boundary_policy": get_multi("boundary_policy"),
        "cfg_multi_new_map_visible_ratio": get_multi("new_map_visible_ratio"),
        "cfg_multi_sampled_original_blocks": get_multi("sampled_original_blocks"),
        "cfg_multi_overlap_frames": get_multi("overlap_frames"),
        "cfg_multi_trial_frames": get_multi("trial_frames"),
        "cfg_multi_max_frames": get_multi("max_frames"),
        "cfg_multi_new_map_handoff_policy": get_multi("new_map_handoff_policy"),
        "cfg_multi_new_map_handoff_min_visible_ratio": get_multi("new_map_handoff_min_visible_ratio"),
        "cfg_multi_new_map_handoff_min_visible_margin": get_multi("new_map_handoff_min_visible_margin"),
        "cfg_multi_new_map_handoff_min_accepts": get_multi("new_map_handoff_min_accepts"),
        "cfg_multi_hash_profile": get_multi("hash_profile"),
        "cfg_multi_resident_gpu_maps": get_multi("resident_gpu_maps"),
        "cfg_multi_paging": get_multi("paging"),
        "cfg_multi_paging_storage": get_multi("paging_storage"),
        "cfg_multi_async_page_out": get_multi("async_page_out"),
        "cfg_multi_max_pending_page_snapshots": get_multi("max_pending_page_snapshots"),
        "cfg_multi_max_pending_page_snapshot_mb": get_multi("max_pending_page_snapshot_mb"),
        "cfg_multi_async_page_in": get_multi("async_page_in"),
        "cfg_multi_max_pending_page_ins": get_multi("max_pending_page_ins"),
        "cfg_multi_max_pending_page_in_mb": get_multi("max_pending_page_in_mb"),
        "cfg_gaussian_submaps_enable": get_gaussian("enable"),
        "cfg_gaussian_submaps_ownership": get_gaussian("ownership"),
        "cfg_gaussian_submaps_render_selection_policy": get_gaussian("render_selection_policy"),
        "cfg_gaussian_submaps_insert_selection": get_gaussian("gaussian_insert_selection"),
        "cfg_gaussian_submaps_insert_tile_size": get_gaussian("gaussian_insert_tile_size"),
        "cfg_gaussian_submaps_insert_tile_max_pixels": get_gaussian("gaussian_insert_tile_max_pixels"),
        "cfg_gaussian_submaps_insert_adaptive_tile_min_candidate_pixels": get_gaussian("gaussian_insert_adaptive_tile_min_candidate_pixels"),
        "cfg_gaussian_submaps_insert_adaptive_tile_min_target_insertions": get_gaussian("gaussian_insert_adaptive_tile_min_target_insertions"),
        "cfg_gaussian_submaps_insert_adaptive_tile_min_submap_gaussians": get_gaussian("gaussian_insert_adaptive_tile_min_submap_gaussians"),
        "cfg_gaussian_submaps_insert_adaptive_tile_min_observed_frames": get_gaussian("gaussian_insert_adaptive_tile_min_observed_frames"),
        "cfg_gaussian_submaps_quality_finalize_enable": get_gaussian("quality_finalize_enable"),
        "cfg_gaussian_submaps_quality_finalize_resident_only": get_gaussian("quality_finalize_resident_only"),
        "cfg_gaussian_submaps_quality_finalize_max_hybrid_l1": get_gaussian("quality_finalize_max_hybrid_l1"),
        "cfg_gaussian_submaps_quality_finalize_insert_enable": get_gaussian("quality_finalize_insert_enable"),
        "cfg_gaussian_submaps_quality_finalize_insert_acceptance_enable": get_gaussian(
            "quality_finalize_insert_acceptance_enable"
        ),
        "cfg_gaussian_submaps_quality_finalize_insert_acceptance_cameras": get_gaussian(
            "quality_finalize_insert_acceptance_cameras"
        ),
        "cfg_gaussian_submaps_quality_finalize_insert_min_readiness_delta": get_gaussian(
            "quality_finalize_insert_min_readiness_delta"
        ),
        "cfg_gaussian_submaps_budget_policy": get_gaussian("gaussian_budget_policy"),
        "cfg_gaussian_submaps_frozen_refinement_enable": get_gaussian("frozen_refinement_enable"),
        "cfg_gaussian_submaps_frozen_refinement_interval_frames": get_gaussian("frozen_refinement_interval_frames"),
        "cfg_gaussian_submaps_frozen_refinement_keyframes": get_gaussian("frozen_refinement_keyframes"),
        "cfg_gaussian_submaps_frozen_refinement_readiness_guard_enable": get_gaussian("frozen_refinement_readiness_guard_enable"),
        "cfg_gaussian_submaps_frozen_refinement_readiness_use_cached_graduation": get_gaussian("frozen_refinement_readiness_use_cached_graduation"),
        "cfg_gaussian_submaps_frozen_refinement_readiness_max_pre_gain": get_gaussian("frozen_refinement_readiness_max_pre_gain"),
        "cfg_gaussian_submaps_frozen_refinement_readiness_min_valid_cameras": get_gaussian("frozen_refinement_readiness_min_valid_cameras"),
        "cfg_gaussian_submaps_frozen_refinement_acceptance_enable": get_gaussian("frozen_refinement_acceptance_enable"),
        "cfg_gaussian_submaps_frozen_refinement_acceptance_min_delta": get_gaussian("frozen_refinement_acceptance_min_delta"),
        "cfg_gaussian_submaps_frozen_refinement_observation_policy": get_gaussian("frozen_refinement_observation_policy"),
        "cfg_gaussian_submaps_frozen_refinement_observation_selection_max_eval": get_gaussian("frozen_refinement_observation_selection_max_eval"),
        "cfg_gaussian_submaps_cross_boundary_refinement_enable": get_gaussian("cross_boundary_refinement_enable"),
        "cfg_gaussian_submaps_cross_boundary_interval_frames": get_gaussian("cross_boundary_observation_interval_frames"),
        "cfg_gaussian_submaps_cross_boundary_depth_gate": get_gaussian("cross_boundary_depth_gate"),
        "cfg_gaussian_submaps_cross_boundary_min_projected_ratio": get_gaussian("cross_boundary_min_projected_ratio"),
        "cfg_gaussian_submaps_cross_boundary_min_depth_consistent_ratio": get_gaussian("cross_boundary_min_depth_consistent_ratio"),
    }


def summarize_submap_diagnostics(out_dir: Path) -> dict:
    result = {}
    stats_path = out_dir / "submap_stats.csv"
    if stats_path.exists():
        accepted_sdf_ids = set()
        max_logged_sdf_count = 0
        max_training_observations = 0
        max_frozen_refinement_passes = 0
        max_cross_boundary_observations = 0
        max_cross_boundary_rejections = 0
        cross_boundary_observed_submaps = set()
        max_last_budget_cap = -1
        max_budget_pressure_prunes = 0
        max_budget_cap_hits = 0
        budget_cap_hit_submaps = set()
        with stats_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    accepted_sdf_ids.add(int(row.get("submap_id", "") or -1))
                except ValueError:
                    pass
                try:
                    max_logged_sdf_count = max(max_logged_sdf_count, len(accepted_sdf_ids))
                except ValueError:
                    pass
                try:
                    max_training_observations = max(
                        max_training_observations,
                        int(row.get("training_observations", "") or 0),
                    )
                except ValueError:
                    pass
                try:
                    max_frozen_refinement_passes = max(
                        max_frozen_refinement_passes,
                        int(row.get("frozen_refinement_passes", "") or 0),
                    )
                except ValueError:
                    pass
                try:
                    cross_obs = int(row.get("cross_boundary_observations", "") or 0)
                    max_cross_boundary_observations = max(max_cross_boundary_observations, cross_obs)
                    if cross_obs > 0:
                        cross_boundary_observed_submaps.add(int(row.get("gaussian_submap_id", "") or -1))
                except ValueError:
                    pass
                try:
                    max_cross_boundary_rejections = max(
                        max_cross_boundary_rejections,
                        int(row.get("cross_boundary_rejections", "") or 0),
                    )
                except ValueError:
                    pass
                try:
                    max_last_budget_cap = max(
                        max_last_budget_cap,
                        int(row.get("last_budget_cap", "") or -1),
                    )
                except ValueError:
                    pass
                try:
                    pressure_prunes = int(row.get("budget_pressure_prunes", "") or 0)
                    max_budget_pressure_prunes = max(max_budget_pressure_prunes, pressure_prunes)
                except ValueError:
                    pass
                try:
                    cap_hits = int(row.get("budget_cap_hits", "") or 0)
                    max_budget_cap_hits = max(max_budget_cap_hits, cap_hits)
                    if cap_hits > 0:
                        budget_cap_hit_submaps.add(int(row.get("gaussian_submap_id", "") or -1))
                except ValueError:
                    pass
        result["accepted_sdf_submap_count"] = len([idx for idx in accepted_sdf_ids if idx >= 0])
        result["max_logged_sdf_submap_count"] = max_logged_sdf_count
        result["max_training_observations_per_submap"] = max_training_observations
        result["max_frozen_refinement_passes_per_submap"] = max_frozen_refinement_passes
        result["max_cross_boundary_observations_per_submap"] = max_cross_boundary_observations
        result["max_cross_boundary_rejections_per_submap"] = max_cross_boundary_rejections
        result["cross_boundary_observed_submap_count"] = len([idx for idx in cross_boundary_observed_submaps if idx >= 0])
        result["max_last_gaussian_budget_cap"] = max_last_budget_cap if max_last_budget_cap >= 0 else None
        result["max_budget_pressure_prunes_per_submap"] = max_budget_pressure_prunes
        result["max_budget_cap_hits_per_submap"] = max_budget_cap_hits
        result["budget_cap_hit_submap_count"] = len([idx for idx in budget_cap_hit_submaps if idx >= 0])

    assignments_path = out_dir / "frame_assignments.csv"
    if assignments_path.exists():
        owner_ids = set()
        max_active = 0
        missing_active = 0
        with assignments_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    owner_ids.add(int(row.get("primary_submap", "") or -1))
                except ValueError:
                    pass
                active = [part for part in row.get("active_submaps", "").split("|") if part.strip()]
                max_active = max(max_active, len(active))
                if not active:
                    missing_active += 1
        result["unique_primary_submap_count"] = len([idx for idx in owner_ids if idx >= 0])
        result["max_active_submaps_per_frame"] = max_active
        result["missing_active_submap_rows"] = missing_active

    events_path = out_dir / "submap_events.jsonl"
    if events_path.exists():
        event_counts = {}
        gaussian_insert_added_total = 0
        gaussian_insert_candidate_pixels_total = 0
        gaussian_insert_target_total = 0
        gaussian_insert_max_candidate_pixels = 0
        gaussian_insert_max_added = 0
        cross_novelty_reference_frames = 0
        cross_novelty_raw_suppressed_total = 0
        cross_novelty_suppressed_total = 0
        cross_novelty_candidate_before_total = 0
        cross_novelty_max_reference_submaps = 0
        cross_novelty_skipped_nonresident_gaussian_total = 0
        cross_novelty_skipped_nonresident_tsdf_total = 0
        tile_insert_frames = 0
        tile_insert_active_tiles_total = 0
        tile_insert_preclip_pixels_total = 0
        adaptive_tile_insert_frames = 0
        adaptive_tile_insert_enabled_frames = 0
        adaptive_tile_insert_reasons = {}
        graduation_start_count = 0
        graduation_end_count = 0
        graduation_skip_count = 0
        graduation_selected_cameras_total = 0
        graduation_iters_total = 0
        graduation_completed_count = 0
        graduation_adaptive_end_count = 0
        graduation_adaptive_loss_drop_total = 0.0
        graduation_adaptive_loss_rel_drop_total = 0.0
        graduation_adaptive_extra_pass_count = 0
        graduation_readiness_end_count = 0
        graduation_readiness_skip_count = 0
        graduation_readiness_delta_before_total = 0.0
        graduation_readiness_delta_after_total = 0.0
        graduation_readiness_improving_count = 0
        graduation_readiness_target_met_count = 0
        quality_finalize_candidate_count = 0
        quality_finalize_start_count = 0
        quality_finalize_end_count = 0
        quality_finalize_ran_count = 0
        quality_finalize_max_candidates = 0
        quality_finalize_insert_attempt_count = 0
        quality_finalize_insert_count = 0
        quality_finalize_insert_rejected_count = 0
        quality_finalize_insert_restored_count = 0
        quality_finalize_insert_added_total = 0
        quality_finalize_insert_attempted_added_total = 0
        quality_finalize_insert_selected_cameras_total = 0
        quality_finalize_skip_count = 0
        quality_finalize_skip_nonresident_count = 0
        frozen_refinement_start_count = 0
        frozen_refinement_end_count = 0
        frozen_refinement_skip_count = 0
        frozen_refinement_readiness_skip_count = 0
        frozen_refinement_readiness_invalid_skip_count = 0
        frozen_refinement_readiness_positive_skip_count = 0
        frozen_refinement_cached_ready_skip_count = 0
        frozen_refinement_readiness_before_total = 0.0
        frozen_refinement_readiness_before_count = 0
        frozen_refinement_readiness_after_total = 0.0
        frozen_refinement_readiness_after_count = 0
        frozen_refinement_readiness_change_total = 0.0
        frozen_refinement_readiness_change_count = 0
        frozen_refinement_acceptance_enabled_count = 0
        frozen_refinement_acceptance_valid_count = 0
        frozen_refinement_acceptance_rejected_count = 0
        frozen_refinement_acceptance_restored_count = 0
        frozen_refinement_acceptance_invalid_skip_count = 0
        frozen_refinement_observation_selection_count = 0
        frozen_refinement_observation_selection_evaluated_total = 0
        frozen_refinement_observation_selection_valid_scores_total = 0
        frozen_refinement_observation_selection_valid_readiness_total = 0
        frozen_refinement_observation_selection_score_total = 0.0
        frozen_refinement_observation_selection_score_count = 0
        frozen_refinement_observation_selection_score_max = None
        budget_pressure_prune_count = 0
        budget_cap_skip_count = 0
        budget_cap_skip_gaussians_total = 0
        max_effective_budget_cap = -1
        cross_boundary_scan_count = 0
        cross_boundary_scan_considered_total = 0
        cross_boundary_scan_stored_total = 0
        cross_boundary_scan_rejected_depth_total = 0
        cross_boundary_scan_raycast_failures_total = 0
        cross_boundary_observation_events = 0
        cross_boundary_observation_stored_total = 0
        cross_boundary_rejected_events = 0
        with events_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    event = payload.get("event", "")
                except json.JSONDecodeError:
                    event = "malformed_event"
                    payload = {}
                if event:
                    event_counts[event] = event_counts.get(event, 0) + 1
                if event == "gaussian_insert":
                    added = int(payload.get("added", 0) or 0)
                    candidate_pixels = int(payload.get("candidate_pixels", 0) or 0)
                    target_insertions = int(payload.get("target_insertions", 0) or 0)
                    gaussian_insert_added_total += added
                    gaussian_insert_candidate_pixels_total += max(0, candidate_pixels)
                    gaussian_insert_target_total += max(0, target_insertions)
                    gaussian_insert_max_candidate_pixels = max(gaussian_insert_max_candidate_pixels, candidate_pixels)
                    gaussian_insert_max_added = max(gaussian_insert_max_added, added)
                    max_effective_budget_cap = max(
                        max_effective_budget_cap,
                        int(payload.get("effective_submap_gaussian_cap", -1) or -1),
                    )
                    ref_count = int(payload.get("cross_novelty_reference_submaps", 0) or 0)
                    raw_suppressed = int(payload.get("cross_novelty_raw_suppressed_pixels", -1) or -1)
                    suppressed = int(payload.get("cross_novelty_suppressed_pixels", -1) or -1)
                    before = int(payload.get("cross_novelty_candidate_pixels_before", -1) or -1)
                    if ref_count > 0:
                        cross_novelty_reference_frames += 1
                        cross_novelty_max_reference_submaps = max(cross_novelty_max_reference_submaps, ref_count)
                    if suppressed >= 0:
                        cross_novelty_suppressed_total += suppressed
                    if raw_suppressed >= 0:
                        cross_novelty_raw_suppressed_total += raw_suppressed
                    if before >= 0:
                        cross_novelty_candidate_before_total += before
                    cross_novelty_skipped_nonresident_gaussian_total += int(
                        payload.get("cross_novelty_skipped_nonresident_gaussian", 0) or 0
                    )
                    cross_novelty_skipped_nonresident_tsdf_total += int(
                        payload.get("cross_novelty_skipped_nonresident_tsdf", 0) or 0
                    )
                    active_tiles = int(payload.get("tile_insert_active_tiles", -1) or -1)
                    preclip_pixels = int(payload.get("tile_insert_preclip_pixels", -1) or -1)
                    if active_tiles >= 0:
                        tile_insert_frames += 1
                        tile_insert_active_tiles_total += active_tiles
                    if preclip_pixels >= 0:
                        tile_insert_preclip_pixels_total += preclip_pixels
                    if payload.get("insert_selection") == "adaptive_tile_error_topk":
                        adaptive_tile_insert_frames += 1
                        reason = str(payload.get("adaptive_tile_insert_reason", "unknown") or "unknown")
                        adaptive_tile_insert_reasons[reason] = adaptive_tile_insert_reasons.get(reason, 0) + 1
                        adaptive_tile_insert_enabled_frames += int(
                            payload.get("adaptive_tile_insert_enabled", 0) or 0
                        )
                elif event == "gaussian_submap_graduation_start":
                    graduation_start_count += 1
                    graduation_selected_cameras_total += int(payload.get("selected_cameras", 0) or 0)
                    graduation_iters_total += int(payload.get("iters", 0) or 0)
                elif event == "gaussian_submap_graduation_end":
                    graduation_end_count += 1
                    graduation_completed_count += int(payload.get("complete", 0) or 0)
                    if int(payload.get("readiness_enable", 0) or 0):
                        before = float(payload.get("readiness_delta_before", -999.0) or -999.0)
                        after = float(payload.get("readiness_delta_after", -999.0) or -999.0)
                        if before > -998.0 and after > -998.0:
                            graduation_readiness_end_count += 1
                            graduation_readiness_delta_before_total += before
                            graduation_readiness_delta_after_total += after
                            graduation_readiness_improving_count += int(payload.get("readiness_improving", 0) or 0)
                            graduation_readiness_target_met_count += int(payload.get("readiness_target_met", 0) or 0)
                    if int(payload.get("adaptive", 0) or 0):
                        graduation_adaptive_end_count += 1
                        graduation_adaptive_loss_drop_total += float(payload.get("loss_drop", 0.0) or 0.0)
                        graduation_adaptive_loss_rel_drop_total += float(payload.get("loss_rel_drop", 0.0) or 0.0)
                        if int(payload.get("passes", 0) or 0) > 1:
                            graduation_adaptive_extra_pass_count += 1
                elif event == "gaussian_submap_graduation_skipped":
                    graduation_skip_count += 1
                    if payload.get("skip_reason") == "readiness_already_met":
                        graduation_readiness_skip_count += 1
                elif event == "gaussian_submap_quality_finalize_candidates":
                    quality_finalize_candidate_count += 1
                    quality_finalize_max_candidates = max(
                        quality_finalize_max_candidates,
                        int(payload.get("candidate_count", 0) or 0),
                    )
                elif event == "gaussian_submap_quality_finalize_start":
                    quality_finalize_start_count += 1
                elif event == "gaussian_submap_quality_finalize_end":
                    quality_finalize_end_count += 1
                    quality_finalize_ran_count += int(payload.get("ran", 0) or 0)
                elif event == "gaussian_submap_quality_finalize_insert":
                    quality_finalize_insert_attempt_count += 1
                    accepted = int(payload.get("accepted", 1) or 0) != 0
                    if accepted:
                        quality_finalize_insert_count += 1
                    else:
                        quality_finalize_insert_rejected_count += 1
                    if int(payload.get("acceptance_restored", 0) or 0) != 0:
                        quality_finalize_insert_restored_count += 1
                    quality_finalize_insert_added_total += int(payload.get("added", 0) or 0)
                    quality_finalize_insert_attempted_added_total += int(
                        payload.get("attempted_added", payload.get("added", 0)) or 0
                    )
                    quality_finalize_insert_selected_cameras_total += int(payload.get("selected_cameras", 0) or 0)
                elif event == "gaussian_submap_quality_finalize_skipped":
                    quality_finalize_skip_count += 1
                    if payload.get("skip_reason") == "nonresident":
                        quality_finalize_skip_nonresident_count += 1
                elif event == "gaussian_submap_frozen_refinement_start":
                    frozen_refinement_start_count += 1
                    before = float(payload.get("readiness_delta_before", -999.0) or -999.0)
                    if before > -998.0:
                        frozen_refinement_readiness_before_total += before
                        frozen_refinement_readiness_before_count += 1
                elif event == "gaussian_submap_frozen_refinement_end":
                    frozen_refinement_end_count += 1
                    if int(payload.get("acceptance_enabled", 0) or 0):
                        frozen_refinement_acceptance_enabled_count += 1
                        if int(payload.get("acceptance_valid", 0) or 0):
                            frozen_refinement_acceptance_valid_count += 1
                        if int(payload.get("accepted", 1) or 0) == 0:
                            frozen_refinement_acceptance_rejected_count += 1
                        if int(payload.get("acceptance_restored", 0) or 0):
                            frozen_refinement_acceptance_restored_count += 1
                    after = float(payload.get("readiness_delta_after", -999.0) or -999.0)
                    if after > -998.0:
                        frozen_refinement_readiness_after_total += after
                        frozen_refinement_readiness_after_count += 1
                    change = float(payload.get("readiness_delta_change", -999.0) or -999.0)
                    if change > -998.0:
                        frozen_refinement_readiness_change_total += change
                        frozen_refinement_readiness_change_count += 1
                elif event == "gaussian_submap_frozen_refinement_skipped":
                    frozen_refinement_skip_count += 1
                    skip_reason = str(payload.get("skip_reason", ""))
                    if skip_reason.startswith("readiness_"):
                        frozen_refinement_readiness_skip_count += 1
                    if skip_reason.startswith("acceptance_"):
                        frozen_refinement_acceptance_invalid_skip_count += 1
                    if skip_reason == "readiness_invalid":
                        frozen_refinement_readiness_invalid_skip_count += 1
                    if skip_reason == "readiness_already_positive":
                        frozen_refinement_readiness_positive_skip_count += 1
                    if skip_reason == "cached_graduation_ready":
                        frozen_refinement_cached_ready_skip_count += 1
                    before = float(payload.get("readiness_delta_before", -999.0) or -999.0)
                    if before > -998.0:
                        frozen_refinement_readiness_before_total += before
                        frozen_refinement_readiness_before_count += 1
                elif event == "gaussian_submap_frozen_refinement_observation_selection":
                    frozen_refinement_observation_selection_count += 1
                    frozen_refinement_observation_selection_evaluated_total += int(
                        payload.get("evaluated", 0) or 0
                    )
                    frozen_refinement_observation_selection_valid_scores_total += int(
                        payload.get("valid_scores", 0) or 0
                    )
                    frozen_refinement_observation_selection_valid_readiness_total += int(
                        payload.get("valid_readiness", 0) or 0
                    )
                    score = float(payload.get("avg_selection_score", -999.0) or -999.0)
                    if score > -998.0:
                        frozen_refinement_observation_selection_score_total += score
                        frozen_refinement_observation_selection_score_count += 1
                    max_score = float(payload.get("max_selection_score", -999.0) or -999.0)
                    if max_score > -998.0:
                        if frozen_refinement_observation_selection_score_max is None:
                            frozen_refinement_observation_selection_score_max = max_score
                        else:
                            frozen_refinement_observation_selection_score_max = max(
                                frozen_refinement_observation_selection_score_max,
                                max_score,
                            )
                elif event == "gaussian_budget_pressure_prune":
                    budget_pressure_prune_count += 1
                    max_effective_budget_cap = max(max_effective_budget_cap, int(payload.get("cap", -1) or -1))
                elif event == "gaussian_insert_skipped_cap":
                    budget_cap_skip_count += 1
                    budget_cap_skip_gaussians_total += int(payload.get("gaussians", 0) or 0)
                    max_effective_budget_cap = max(max_effective_budget_cap, int(payload.get("cap", -1) or -1))
                elif event == "gaussian_cross_boundary_scan":
                    cross_boundary_scan_count += 1
                    cross_boundary_scan_considered_total += int(payload.get("considered", 0) or 0)
                    cross_boundary_scan_stored_total += int(payload.get("stored", 0) or 0)
                    cross_boundary_scan_rejected_depth_total += int(payload.get("rejected_depth", 0) or 0)
                    cross_boundary_scan_raycast_failures_total += int(payload.get("raycast_failures", 0) or 0)
                elif event == "gaussian_cross_boundary_observation":
                    cross_boundary_observation_events += 1
                    cross_boundary_observation_stored_total += int(payload.get("stored", 0) or 0)
                elif event == "gaussian_cross_boundary_observation_rejected":
                    cross_boundary_rejected_events += 1
        for event, count in event_counts.items():
            result[f"submap_event_{event}_count"] = count
        if graduation_start_count or graduation_end_count or graduation_skip_count:
            result["graduation_start_count"] = graduation_start_count
            result["graduation_end_count"] = graduation_end_count
            result["graduation_skip_count"] = graduation_skip_count
            result["graduation_selected_cameras_total"] = graduation_selected_cameras_total
            result["graduation_iters_total"] = graduation_iters_total
            result["graduation_completed_count"] = graduation_completed_count
            if graduation_adaptive_end_count:
                result["graduation_adaptive_end_count"] = graduation_adaptive_end_count
                result["graduation_adaptive_extra_pass_count"] = graduation_adaptive_extra_pass_count
                result["graduation_adaptive_loss_drop_avg"] = graduation_adaptive_loss_drop_total / graduation_adaptive_end_count
                result["graduation_adaptive_loss_rel_drop_avg"] = graduation_adaptive_loss_rel_drop_total / graduation_adaptive_end_count
            if graduation_readiness_end_count:
                result["graduation_readiness_end_count"] = graduation_readiness_end_count
                result["graduation_readiness_delta_before_avg"] = (
                    graduation_readiness_delta_before_total / graduation_readiness_end_count
                )
                result["graduation_readiness_delta_after_avg"] = (
                    graduation_readiness_delta_after_total / graduation_readiness_end_count
                )
                result["graduation_readiness_improving_count"] = graduation_readiness_improving_count
                result["graduation_readiness_target_met_count"] = graduation_readiness_target_met_count
            result["graduation_readiness_skip_count"] = graduation_readiness_skip_count
        if quality_finalize_candidate_count or quality_finalize_start_count or quality_finalize_end_count or quality_finalize_skip_count:
            result["quality_finalize_candidate_events"] = quality_finalize_candidate_count
            result["quality_finalize_max_candidates"] = quality_finalize_max_candidates
            result["quality_finalize_start_count"] = quality_finalize_start_count
            result["quality_finalize_end_count"] = quality_finalize_end_count
            result["quality_finalize_ran_count"] = quality_finalize_ran_count
            result["quality_finalize_insert_attempt_count"] = quality_finalize_insert_attempt_count
            result["quality_finalize_insert_count"] = quality_finalize_insert_count
            result["quality_finalize_insert_rejected_count"] = quality_finalize_insert_rejected_count
            result["quality_finalize_insert_restored_count"] = quality_finalize_insert_restored_count
            result["quality_finalize_insert_added_total"] = quality_finalize_insert_added_total
            result["quality_finalize_insert_attempted_added_total"] = quality_finalize_insert_attempted_added_total
            result["quality_finalize_insert_selected_cameras_total"] = quality_finalize_insert_selected_cameras_total
            result["quality_finalize_skip_count"] = quality_finalize_skip_count
            result["quality_finalize_skip_nonresident_count"] = quality_finalize_skip_nonresident_count
        if frozen_refinement_start_count or frozen_refinement_end_count or frozen_refinement_skip_count:
            result["frozen_refinement_start_count"] = frozen_refinement_start_count
            result["frozen_refinement_end_count"] = frozen_refinement_end_count
            result["frozen_refinement_skip_count"] = frozen_refinement_skip_count
            result["frozen_refinement_readiness_skip_count"] = frozen_refinement_readiness_skip_count
            result["frozen_refinement_readiness_invalid_skip_count"] = frozen_refinement_readiness_invalid_skip_count
            result["frozen_refinement_readiness_positive_skip_count"] = frozen_refinement_readiness_positive_skip_count
            result["frozen_refinement_cached_ready_skip_count"] = frozen_refinement_cached_ready_skip_count
            if frozen_refinement_readiness_before_count:
                result["frozen_refinement_readiness_delta_before_avg"] = (
                    frozen_refinement_readiness_before_total / frozen_refinement_readiness_before_count
                )
            if frozen_refinement_readiness_after_count:
                result["frozen_refinement_readiness_delta_after_avg"] = (
                    frozen_refinement_readiness_after_total / frozen_refinement_readiness_after_count
                )
            if frozen_refinement_readiness_change_count:
                result["frozen_refinement_readiness_delta_change_avg"] = (
                    frozen_refinement_readiness_change_total / frozen_refinement_readiness_change_count
                )
            if frozen_refinement_acceptance_enabled_count or frozen_refinement_acceptance_invalid_skip_count:
                result["frozen_refinement_acceptance_enabled_count"] = (
                    frozen_refinement_acceptance_enabled_count
                )
                result["frozen_refinement_acceptance_valid_count"] = (
                    frozen_refinement_acceptance_valid_count
                )
                result["frozen_refinement_acceptance_rejected_count"] = (
                    frozen_refinement_acceptance_rejected_count
                )
                result["frozen_refinement_acceptance_restored_count"] = (
                    frozen_refinement_acceptance_restored_count
                )
                result["frozen_refinement_acceptance_invalid_skip_count"] = (
                    frozen_refinement_acceptance_invalid_skip_count
                )
            if frozen_refinement_observation_selection_count:
                result["frozen_refinement_observation_selection_count"] = (
                    frozen_refinement_observation_selection_count
                )
                result["frozen_refinement_observation_selection_evaluated_total"] = (
                    frozen_refinement_observation_selection_evaluated_total
                )
                result["frozen_refinement_observation_selection_valid_scores_total"] = (
                    frozen_refinement_observation_selection_valid_scores_total
                )
                result["frozen_refinement_observation_selection_valid_readiness_total"] = (
                    frozen_refinement_observation_selection_valid_readiness_total
                )
                if frozen_refinement_observation_selection_score_count:
                    result["frozen_refinement_observation_selection_score_avg"] = (
                        frozen_refinement_observation_selection_score_total /
                        frozen_refinement_observation_selection_score_count
                    )
                result["frozen_refinement_observation_selection_score_max"] = (
                    frozen_refinement_observation_selection_score_max
                )
        if budget_pressure_prune_count or budget_cap_skip_count or max_effective_budget_cap >= 0:
            result["gaussian_budget_pressure_prune_count"] = budget_pressure_prune_count
            result["gaussian_budget_cap_skip_count"] = budget_cap_skip_count
            result["gaussian_budget_cap_skip_gaussians_total"] = budget_cap_skip_gaussians_total
            result["gaussian_budget_effective_cap_max"] = max_effective_budget_cap if max_effective_budget_cap >= 0 else None
        if cross_boundary_scan_count or cross_boundary_observation_events or cross_boundary_rejected_events:
            result["cross_boundary_scan_count"] = cross_boundary_scan_count
            result["cross_boundary_scan_considered_total"] = cross_boundary_scan_considered_total
            result["cross_boundary_scan_stored_total"] = cross_boundary_scan_stored_total
            result["cross_boundary_scan_rejected_depth_total"] = cross_boundary_scan_rejected_depth_total
            result["cross_boundary_scan_raycast_failures_total"] = cross_boundary_scan_raycast_failures_total
            result["cross_boundary_observation_events"] = cross_boundary_observation_events
            result["cross_boundary_observation_stored_total"] = cross_boundary_observation_stored_total
            result["cross_boundary_rejected_events"] = cross_boundary_rejected_events
        if event_counts.get("gaussian_insert", 0):
            result["gaussian_insert_added_total"] = gaussian_insert_added_total
            result["gaussian_insert_candidate_pixels_total"] = gaussian_insert_candidate_pixels_total
            result["gaussian_insert_target_total"] = gaussian_insert_target_total
            result["gaussian_insert_max_candidate_pixels"] = gaussian_insert_max_candidate_pixels
            result["gaussian_insert_max_added"] = gaussian_insert_max_added
            result["cross_novelty_reference_frames"] = cross_novelty_reference_frames
            result["cross_novelty_raw_suppressed_total"] = cross_novelty_raw_suppressed_total
            result["cross_novelty_suppressed_total"] = cross_novelty_suppressed_total
            result["cross_novelty_candidate_before_total"] = cross_novelty_candidate_before_total
            result["cross_novelty_max_reference_submaps"] = cross_novelty_max_reference_submaps
            result["cross_novelty_skipped_nonresident_gaussian_total"] = cross_novelty_skipped_nonresident_gaussian_total
            result["cross_novelty_skipped_nonresident_tsdf_total"] = cross_novelty_skipped_nonresident_tsdf_total
            result["tile_insert_frames"] = tile_insert_frames
            result["tile_insert_active_tiles_total"] = tile_insert_active_tiles_total
            result["tile_insert_preclip_pixels_total"] = tile_insert_preclip_pixels_total
            result["adaptive_tile_insert_frames"] = adaptive_tile_insert_frames
            result["adaptive_tile_insert_enabled_frames"] = adaptive_tile_insert_enabled_frames
            if adaptive_tile_insert_frames:
                result["adaptive_tile_insert_enabled_ratio"] = (
                    adaptive_tile_insert_enabled_frames / adaptive_tile_insert_frames
                )
                for reason, count in sorted(adaptive_tile_insert_reasons.items()):
                    result[f"adaptive_tile_insert_reason_{reason}_count"] = count
            if cross_novelty_candidate_before_total > 0:
                result["cross_novelty_raw_suppressed_ratio"] = (
                    cross_novelty_raw_suppressed_total / cross_novelty_candidate_before_total
                )
                result["cross_novelty_suppressed_ratio"] = (
                    cross_novelty_suppressed_total / cross_novelty_candidate_before_total
                )

    transitions_path = out_dir / "submap_transitions.csv"
    if transitions_path.exists():
        transition_counts = {}
        rejected_pairs = set()
        handoff_without_graph_observation = 0
        transition_rows = []
        with transitions_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transition_rows.append(row)
                event = row.get("event", "")
                if event:
                    transition_counts[event] = transition_counts.get(event, 0) + 1
                if event == "sequential_transform_reject":
                    rejected_pairs.add((
                        row.get("frame", ""),
                        row.get("primary_local_map", ""),
                        row.get("new_local_map", ""),
                    ))
        for row in transition_rows:
            if row.get("event") != "visibility_handoff":
                continue
            key = (
                row.get("frame", ""),
                row.get("primary_local_map", ""),
                row.get("new_local_map", ""),
            )
            if key in rejected_pairs:
                handoff_without_graph_observation += 1
        for event, count in transition_counts.items():
            result[f"transition_{event}_count"] = count
        result["sequential_transform_reject_count"] = len(rejected_pairs)
        result["handoff_without_graph_observation_count"] = handoff_without_graph_observation

    time_log = out_dir / "time_log.txt"
    if time_log.exists():
        for line in time_log.read_text(errors="ignore").splitlines():
            count_match = re.search(
                r"\[SUBMAP COUNTS\]\s+accepted_sdf_submaps:\s+(-?\d+),\s+hybrid_registry_submaps:\s+(\d+),\s+retired_hybrid_submaps:\s+(\d+),\s+active_hybrid_submap:\s+(-?\d+)",
                line,
            )
            if count_match:
                result["accepted_sdf_submap_count"] = int(count_match.group(1))
                result["hybrid_registry_submap_count"] = int(count_match.group(2))
                result["retired_hybrid_submap_count"] = int(count_match.group(3))
                result["active_hybrid_submap"] = int(count_match.group(4))
            match = re.search(r"\[SUBMAP AVG\]\s+submaps:\s+(\d+),\s+total keyframes:\s+(\d+),\s+active submap:\s+(-?\d+)", line)
            if match:
                result.setdefault("hybrid_registry_submap_count", int(match.group(1)))
                result["hybrid_total_keyframes"] = int(match.group(2))
                result.setdefault("active_hybrid_submap", int(match.group(3)))
    return result


def summarize_render_debug(path: Path) -> dict:
    if not path.exists():
        return {}
    prefetch_ms = []
    raycast_ms = []
    select_ms = []
    render_ms = []
    total_ms = []
    resident_counts = []
    selected_counts = []
    selected_gaussian_counts = []
    overlap_counts = []
    render_selection_candidates = []
    render_selection_truncated_submaps = []
    render_selection_truncated_gaussians = []
    render_selection_owner_oversized = []
    render_selection_sticky_count = []
    numeric_fields = {
        "render_diag_contributions": [],
        "render_diag_sdf_hole_ratio": [],
        "render_diag_owner_alpha_mean": [],
        "render_diag_owner_alpha_coverage": [],
        "render_diag_owner_weak_ratio": [],
        "render_diag_neighbor_alpha_coverage": [],
        "render_diag_neighbor_depth_consistent_ratio": [],
        "render_diag_final_alpha_mean": [],
        "render_diag_final_alpha_coverage": [],
        "render_diag_fill_mask_ratio": [],
        "render_diag_final_depth_disagreement_ratio": [],
        "render_diag_hybrid_rgb_l1": [],
        "render_diag_sdf_rgb_l1": [],
        "render_diag_overlap_depth_gate_enabled": [],
        "render_diag_overlap_depth_gate_candidate_pixel_ratio": [],
        "render_diag_overlap_depth_gate_rejected_pixel_ratio": [],
        "render_diag_overlap_depth_gate_rejected_weight_ratio": [],
    }
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prefetch_ms.append(float(row.get("tsdf_prefetch_ms", "") or 0.0))
            except ValueError:
                pass
            try:
                raycast_ms.append(float(row.get("raycast_ms", "") or 0.0))
            except ValueError:
                pass
            for key, values in [
                ("select_ms", select_ms),
                ("render_ms", render_ms),
                ("total_ms", total_ms),
            ]:
                try:
                    values.append(float(row.get(key, "") or 0.0))
                except ValueError:
                    pass
            try:
                resident_counts.append(int(row.get("tsdf_resident_after_prefetch", "") or 0))
            except ValueError:
                pass
            try:
                selected_gaussian_counts.append(int(float(row.get("selected_gaussians", "") or 0)))
            except ValueError:
                pass
            for key, values in [
                ("render_selection_candidates", render_selection_candidates),
                ("render_selection_truncated_submaps", render_selection_truncated_submaps),
                ("render_selection_truncated_gaussians", render_selection_truncated_gaussians),
                ("render_selection_owner_oversized", render_selection_owner_oversized),
                ("render_selection_sticky_count", render_selection_sticky_count),
            ]:
                try:
                    values.append(int(float(row.get(key, "") or 0)))
                except ValueError:
                    pass
            selected = [part for part in row.get("selected_submaps", "").split("|") if part != ""]
            overlap = [part for part in row.get("overlap_submaps", "").split("|") if part != ""]
            selected_counts.append(len(selected))
            overlap_counts.append(len(overlap))
            for key, values in numeric_fields.items():
                try:
                    value = float(row.get(key, ""))
                except ValueError:
                    continue
                if value >= 0:
                    values.append(value)
    result = {
        "eval_tsdf_prefetch_p95_ms": percentile(prefetch_ms, 95),
        "eval_tsdf_prefetch_p99_ms": percentile(prefetch_ms, 99),
        "eval_tsdf_prefetch_max_ms": max(prefetch_ms) if prefetch_ms else None,
        "eval_raycast_p95_ms": percentile(raycast_ms, 95),
        "eval_raycast_p99_ms": percentile(raycast_ms, 99),
        "eval_raycast_max_ms": max(raycast_ms) if raycast_ms else None,
        "eval_tsdf_resident_after_prefetch_max": max(resident_counts) if resident_counts else None,
    }
    for name, values in [
        ("select", select_ms),
        ("render", render_ms),
        ("total", total_ms),
    ]:
        if not values:
            continue
        result[f"eval_{name}_avg_ms"] = sum(values) / len(values)
        result[f"eval_{name}_p95_ms"] = percentile(values, 95)
        result[f"eval_{name}_p99_ms"] = percentile(values, 99)
        result[f"eval_{name}_max_ms"] = max(values)
    if selected_counts:
        result["eval_render_selected_submaps_avg"] = sum(selected_counts) / len(selected_counts)
        result["eval_render_selected_submaps_max"] = max(selected_counts)
    if selected_gaussian_counts:
        result["eval_render_selected_gaussians_avg"] = sum(selected_gaussian_counts) / len(selected_gaussian_counts)
        result["eval_render_selected_gaussians_p95"] = percentile(selected_gaussian_counts, 95)
        result["eval_render_selected_gaussians_max"] = max(selected_gaussian_counts)
    if render_selection_candidates:
        result["eval_render_selection_candidates_avg"] = sum(render_selection_candidates) / len(render_selection_candidates)
        result["eval_render_selection_candidates_max"] = max(render_selection_candidates)
    if render_selection_truncated_submaps:
        result["eval_render_selection_truncated_submaps_total"] = sum(render_selection_truncated_submaps)
        result["eval_render_selection_truncated_submaps_max"] = max(render_selection_truncated_submaps)
    if render_selection_truncated_gaussians:
        result["eval_render_selection_truncated_gaussians_total"] = sum(render_selection_truncated_gaussians)
        result["eval_render_selection_truncated_gaussians_max"] = max(render_selection_truncated_gaussians)
    if render_selection_owner_oversized:
        result["eval_render_selection_owner_oversized_frames"] = sum(1 for value in render_selection_owner_oversized if value > 0)
    if render_selection_sticky_count:
        result["eval_render_selection_sticky_total"] = sum(render_selection_sticky_count)
        result["eval_render_selection_sticky_max"] = max(render_selection_sticky_count)
    if overlap_counts:
        result["eval_render_overlap_submaps_avg"] = sum(overlap_counts) / len(overlap_counts)
        result["eval_render_overlap_submaps_max"] = max(overlap_counts)
    for key, values in numeric_fields.items():
        if not values:
            continue
        suffix = key.removeprefix("render_diag_")
        result[f"eval_render_diag_{suffix}_avg"] = sum(values) / len(values)
        result[f"eval_render_diag_{suffix}_p95"] = percentile(values, 95)
        result[f"eval_render_diag_{suffix}_max"] = max(values)
    hybrid_l1 = numeric_fields["render_diag_hybrid_rgb_l1"]
    sdf_l1 = numeric_fields["render_diag_sdf_rgb_l1"]
    if hybrid_l1 and sdf_l1 and len(hybrid_l1) == len(sdf_l1):
        result["eval_render_diag_rgb_l1_delta_avg"] = (
            sum(hybrid_l1) / len(hybrid_l1) - sum(sdf_l1) / len(sdf_l1)
        )
    return result


def summarize_tsdf_pages(out_dir: Path) -> dict:
    page_root = out_dir / "tsdf_pages"
    if not page_root.exists():
        return {"tsdf_page_count": 0, "tsdf_pages_bytes": 0, "tsdf_page_max_bytes": 0}
    page_sizes = []
    for submap_dir in sorted(page_root.glob("submap_*")):
        if not submap_dir.is_dir():
            continue
        total = 0
        for file_path in submap_dir.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        page_sizes.append(total)
    return {
        "tsdf_page_count": len(page_sizes),
        "tsdf_pages_bytes": sum(page_sizes),
        "tsdf_page_max_bytes": max(page_sizes) if page_sizes else 0,
    }


def summarize_runtime_resources(out_dir: Path) -> dict:
    path = out_dir / "runtime_resources.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        return {"runtime_resources_error": str(exc)}
    return {
        f"runtime_{key}": value
        for key, value in payload.items()
        if isinstance(value, (bool, int, float, str))
    }


def tail_log(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict | None = None,
    log_path: Path | None = None,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess:
    run_env = None
    if env is not None:
        import os
        run_env = os.environ.copy()
        run_env.update(env)
    if log_path is None:
        return subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            env=run_env,
            timeout=timeout_sec,
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("[RUN] " + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=run_env,
            timeout=timeout_sec,
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, "", "")


def evaluate_ate(out_dir: Path, gt_dir: Path) -> float | None:
    pose_dir = out_dir / "val" / "pose"
    if not pose_dir.exists():
        return None
    proc = run_cmd(["python", str(ATE_SCRIPT), "--gt_path", str(gt_dir / "camera"), "--est_path", str(pose_dir)], PROJECT_ROOT)
    if proc.returncode != 0:
        return None
    eval_file = pose_dir.parent / "pose_eval.txt"
    if not eval_file.exists():
        return None
    for line in eval_file.read_text().splitlines():
        if line.startswith("ATE actual RMSE:"):
            return float(line.split(":", 1)[1].strip())
    return None


def scan_pose_finiteness(out_dir: Path) -> dict:
    pose_dir = out_dir / "val" / "pose"
    if not pose_dir.exists():
        return {"invalid_pose_count": None, "first_invalid_pose": ""}
    invalid = []
    for pose_path in sorted(pose_dir.glob("*.txt")):
        try:
            pose = np.loadtxt(pose_path)
            if not np.isfinite(pose).all():
                invalid.append(pose_path.name)
        except Exception:
            invalid.append(pose_path.name)
        if len(invalid) >= 20:
            break
    return {
        "invalid_pose_count": len(invalid),
        "first_invalid_pose": invalid[0] if invalid else "",
    }


def run_quality_eval(
    binary: Path,
    cfg: dict,
    cfg_path: Path,
    out_dir: Path,
    gt_dir: Path,
    name: str,
    log_root: Path,
    eval_split: str = "train",
    quiet_run_logs: bool = False,
    eval_dir_name: str = "val",
    metric_prefix: str = "",
    eval_config_mutator=None,
) -> dict:
    eval_cfg = deepcopy(cfg)
    eval_cfg["work_mode"] = "eval"
    eval_cfg["PIPE"]["eval_after_train"] = False
    eval_cfg["PIPE"]["save_after_train"] = False
    eval_cfg["PIPE"]["eval_split"] = eval_split
    eval_cfg["PIPE"]["eval_path"] = f"/{eval_dir_name}"
    if eval_config_mutator is not None:
        eval_config_mutator(eval_cfg)
    eval_suffix = "_eval" if eval_dir_name == "val" else f"_eval_{eval_dir_name}"
    eval_cfg_path = cfg_path.with_name(
        safe_experiment_name(cfg_path.stem + eval_suffix, max_len=180) + ".yaml"
    )
    eval_cfg_path.write_text(yaml.dump(eval_cfg, sort_keys=False))
    eval_dir = out_dir / eval_dir_name
    for stale_dir in [eval_dir / "render", eval_dir / "raycast_color", eval_dir / "raycast_depth"]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    eval_log = log_root / (safe_experiment_name(name + eval_suffix, max_len=180) + ".log")
    proc = run_cmd(
        [str(binary), str(eval_cfg_path)],
        PROJECT_ROOT,
        log_path=eval_log if quiet_run_logs else None,
    )
    if not quiet_run_logs:
        eval_log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    prefix = f"{metric_prefix}_" if metric_prefix else ""
    quality = {
        f"{prefix}eval_status": "OK" if proc.returncode == 0 else "CRASH",
        f"{prefix}eval_log": str(eval_log),
    }
    if proc.returncode != 0:
        return quality

    eval_gt_dir = eval_dir / "gt"
    candidate_render_dir = eval_dir / "render"
    if not candidate_render_dir.exists():
        candidate_render_dir = eval_dir / "raycast_color"
    use_dataset_depth_metrics = dataset_metric_root_matches_render(gt_dir, candidate_render_dir)
    if use_dataset_depth_metrics:
        metric_gt_dir = gt_dir
        metric_mode = "dataset_depth"
    elif eval_gt_dir.exists() and any(eval_gt_dir.iterdir()):
        metric_gt_dir = eval_gt_dir
        metric_mode = "sampled_eval"
    else:
        metric_gt_dir = gt_dir
        use_dataset_depth_metrics = (metric_gt_dir / "camera").exists() and (metric_gt_dir / "depth").exists()
        metric_mode = "dataset_depth" if use_dataset_depth_metrics else "sampled_eval"
    quality[f"{prefix}metric_gt_dir"] = str(metric_gt_dir)
    quality[f"{prefix}metric_mode"] = metric_mode

    for label, render_dir in {
        "hybrid": eval_dir / "render",
        "sdf": eval_dir / "raycast_color",
    }.items():
        if not render_dir.exists():
            continue
        for stale_metric_file in [render_dir / "results.json", render_dir / "per_view.json"]:
            if stale_metric_file.exists():
                stale_metric_file.unlink()
        if use_dataset_depth_metrics:
            metric_cmd = [
                "python",
                str(METRIC_SCRIPT),
                "--gt_path",
                str(metric_gt_dir),
                "--render_path",
                str(render_dir),
            ]
        else:
            metric_cmd = [
                "python",
                str(SAMPLED_METRIC_SCRIPT),
                "--gt_path",
                str(metric_gt_dir),
                "--render_path",
                str(render_dir),
            ]
        metric_proc = run_cmd(
            metric_cmd,
            PROJECT_ROOT,
            env={"MKL_THREADING_LAYER": "GNU"},
        )
        metric_log = log_root / (
            safe_experiment_name(
                f"{name}_{metric_prefix + '_' if metric_prefix else ''}{label}_metric",
                max_len=180,
            ) + ".log"
        )
        metric_log.write_text(metric_proc.stdout + "\n--- STDERR ---\n" + metric_proc.stderr)
        quality[f"{prefix}{label}_metric_status"] = "OK" if metric_proc.returncode == 0 else "CRASH"
        quality[f"{prefix}{label}_metric_log"] = str(metric_log)
        results_path = render_dir / "results.json"
        if not results_path.exists():
            quality[f"{prefix}{label}_metric_status"] = "NO_RESULTS"
            continue
        results = json.loads(results_path.read_text())
        for metric_name, metric_value in results.items():
            quality[f"{prefix}{label}_{metric_name}"] = metric_value
    return quality


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--full-res", action="store_true")
    parser.add_argument("--scene-name", default="office0", help="scene tag used in generated configs and summaries")
    parser.add_argument("--input-dir", type=Path, default=None, help="override READER.input_dir; useful for prepared scenes on external drives")
    parser.add_argument("--gt-dir", type=Path, default=None, help="override metric/ATE ground-truth directory; defaults to --input-dir when provided")
    parser.add_argument("--start-frame", type=int, default=0, help="override READER.start_frame for nonzero-start smoke tests")
    parser.add_argument("--frame-step", type=int, default=1, help="override READER.frame_step; --frames remains the processed frame count")
    parser.add_argument("--binary", type=Path, default=PROJECT_ROOT / "build_compact" / "slam_trainer")
    parser.add_argument(
        "--benchmark-protocol",
        choices=[PROTOCOL_NONE, REPLICA_PAPER_V1, REPLICA_DEPLOYED_QUALITY_V1],
        default=PROTOCOL_NONE,
        help="lock the paper-parity Replica comparison envelope and write provenance",
    )
    parser.add_argument(
        "--benchmark-role",
        choices=["auto", "baseline", "modified"],
        default="auto",
        help="comparison role; auto derives it from the single selected label",
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="directory for generated run outputs and summaries")
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument(
        "--trainer-timeout-sec",
        type=float,
        default=0.0,
        help="abort an individual slam_trainer run after this many seconds; 0 disables the timeout",
    )
    parser.add_argument(
        "--thor-preflight",
        action="store_true",
        help="run scripts/thor_preflight.py on every generated config before execution",
    )
    parser.add_argument("--tag", default="", help="extra suffix for generated configs and output directories")
    parser.add_argument("--random-seed", type=int, default=None, help="override PIPE.random_seed for reproducible matched validation runs")
    parser.add_argument("--only-generate", action="store_true")
    parser.add_argument("--quality", action="store_true", help="run eval+PSNR/SSIM/LPIPS for hybrid runs")
    parser.add_argument(
        "--quality-native-full",
        action="store_true",
        help=(
            "after half-resolution training, load the saved map and render the same "
            "evaluation split against native full-resolution Replica images"
        ),
    )
    parser.add_argument(
        "--quality-final-map",
        action="store_true",
        help=(
            "also run a final-map quality eval for hybrid submaps: the already-built live submap graph is rendered "
            "with bounded future-visible SDF/Gaussian submaps, without offline reintegration or retraining"
        ),
    )
    parser.add_argument(
        "--final-map-tsdf-render-compositor",
        choices=["owner_priority", "handoff_aware", "nearest_depth"],
        default="owner_priority",
        help="PIPE.TSDF.multi.render_compositor used only by --quality-final-map",
    )
    parser.add_argument(
        "--final-map-tsdf-render-order-policy",
        choices=["inferred", "adjacent_first", "continuity"],
        default="adjacent_first",
        help=(
            "PIPE.TSDF.multi.render_order_policy used only by --quality-final-map. "
            "Default keeps the owner first, then owner-adjacent maps, which preserved the room1 late-frame fix "
            "without the room0 regression seen from full continuity ordering."
        ),
    )
    parser.add_argument("--final-map-max-render-maps", type=int, default=8, help="PIPE.TSDF.multi.max_render_maps for --quality-final-map")
    parser.add_argument("--final-map-render-max-visible-submaps", type=int, default=8, help="Gaussian submap visible-map cap for --quality-final-map")
    parser.add_argument("--final-map-render-max-visible-gaussians", type=int, default=280000, help="Gaussian count cap for --quality-final-map; 0 disables the cap")
    parser.add_argument(
        "--final-map-use-local-sdf-reference",
        action="store_true",
        help=(
            "for --quality-final-map, render each local Gaussian submap against its own local SDF reference. "
            "Default is off because the global SDF base already supplies the reference in final-map recomposition "
            "and local references can force extra TSDF maps resident on memory-constrained GPUs."
        ),
    )
    parser.add_argument(
        "--final-map-align-tsdf-to-gaussian-selection",
        action="store_true",
        help="for --quality-final-map, render the TSDF base from the selected Gaussian submap owners instead of the broad inferred overlap set",
    )
    parser.add_argument(
        "--final-map-tsdf-residency-profile",
        choices=["inherit", "thor_cpu_bounded", "eval_resident64", "disk_stress"],
        default="inherit",
        help=(
            "TSDF residency profile used only by --quality-final-map eval. "
            "Use thor_cpu_bounded to avoid measuring avoidable eval-time disk page faults while leaving the train run unchanged."
        ),
    )
    parser.add_argument(
        "--final-map-render-selection-policy",
        choices=["sdf_bounds", "sdf_bounds_continuity"],
        default="sdf_bounds",
        help="Gaussian submap render-selection policy used only by --quality-final-map",
    )
    parser.add_argument(
        "--final-map-render-margin",
        type=int,
        default=-1,
        help="past/future frame margin for --quality-final-map; -1 uses the run frame count",
    )
    parser.add_argument(
        "--gaussian-profile",
        choices=[
            "fast",
            "quality",
            "thor_quality",
            "thor_quality_adaptive",
            "thor_quality_candidate_handoff2",
            "thor_quality_loss_guided",
            "thor_quality_refine",
            "thor_quality_xnovelty",
            "thor_quality_xnovelty_refine",
            "thor_quality_reconciled",
            "thor_quality_reconciled_perceptual",
            "thor_quality_reconciled_finalize",
            "thor_quality_reconciled_spatial64",
            "thor_quality_reconciled_adaptive_spatial64",
            "thor_quality_reconciled_frozen_refine",
            "thor_quality_reconciled_frozen_guarded",
            "thor_quality_reconciled_xboundary",
            "thor_quality_reconciled_xboundary_accept",
            "thor_quality_reconciled_xboundary_readiness_accept",
            "thor_quality_reconciled_xboundary_sdferror_accept",
        ],
        default="fast",
        help=(
            "fast uses Thor-style low-cost Gaussian settings; quality keeps the release Gaussian settings; "
            "thor_quality adds the validated bounded bootstrap/graduation/bounds submap quality policy; "
            "thor_quality_refine adds bounded visible-frozen submap refinement; "
            "thor_quality_xnovelty also enables cross-submap duplicate-suppression diagnostics; "
            "thor_quality_reconciled adds SDF-bounds hysteresis, asymmetric novelty, readiness probes, "
            "and SDF-density Gaussian budgets; thor_quality_reconciled_perceptual adds the validated 0.2 "
            "SSIM objective while preserving the baseline profile; thor_quality_reconciled_finalize adds a capped end-of-sequence "
            "quality pass for short or low-keyframe submaps; thor_quality_reconciled_spatial64 adds "
            "64px tile-balanced error top-k insertion for better texture coverage; "
            "thor_quality_reconciled_adaptive_spatial64 uses that tiled path only when the insertion window "
            "is large enough to justify the extra spatial pass; thor_quality_reconciled_frozen_refine adds "
            "bounded visible-frozen refinement; thor_quality_reconciled_frozen_guarded only runs that "
            "refinement when SDF-vs-hybrid readiness says it is needed; thor_quality_reconciled_xboundary "
            "keeps the stricter depth-gated cross-boundary observation policy for diagnostics; "
            "thor_quality_reconciled_xboundary_accept rolls back frozen-refinement passes whose held-out "
            "readiness does not improve; thor_quality_reconciled_xboundary_readiness_accept also selects "
            "frozen-refinement observations by worst held-out hybrid-vs-SDF readiness before filling "
            "with temporal coverage"
        ),
    )
    parser.add_argument(
        "--hybrid-submap-profile",
        choices=["normal", "transition", "stress"],
        default="normal",
        help=(
            "normal uses production-like lifecycle limits; transition forces mature two-submap coverage; "
            "stress forces frequent boundaries for corner-case testing"
        ),
    )
    parser.add_argument("--heldout-interval", type=int, default=-1, help="set READER.test_split_interval and render PIPE.eval_split=val for quality eval")
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated experiment labels to run, e.g. basic_sdf,multi_sdf,multi_hybrid_warm_fill",
    )
    parser.add_argument(
        "--no-save-after-train",
        action="store_true",
        help="skip checkpoint/model/mesh saving for long stability runs; poses and timing are still written",
    )
    parser.add_argument(
        "--quiet-run-logs",
        action="store_true",
        help="write trainer/eval stdout and stderr directly to log files instead of buffering it in Python memory",
    )
    parser.add_argument("--multi-max-frames", type=int, default=None, help="override PIPE.TSDF.multi.max_frames for MultiEngine experiments")
    parser.add_argument("--multi-overlap-frames", type=int, default=None, help="override PIPE.TSDF.multi.overlap_frames for MultiEngine experiments")
    parser.add_argument("--multi-trial-frames", type=int, default=None, help="override PIPE.TSDF.multi.trial_frames for MultiEngine experiments")
    parser.add_argument("--multi-sampled-original-blocks", type=int, default=None, help="override PIPE.TSDF.multi.sampled_original_blocks for visibility boundary/handoff decisions")
    parser.add_argument("--multi-new-map-visible-ratio", type=float, default=None, help="override PIPE.TSDF.multi.new_map_visible_ratio for visibility boundary decisions")
    parser.add_argument("--multi-new-map-occupancy", type=float, default=None, help="override PIPE.TSDF.multi.new_map_block_occupancy for MultiEngine experiments")
    parser.add_argument("--multi-max-occupancy", type=float, default=None, help="override PIPE.TSDF.multi.max_block_occupancy for MultiEngine experiments")
    parser.add_argument(
        "--multi-hash-profile",
        choices=["default", "legacy_large", "medium_128k", "compact_64k"],
        default="default",
        help="override PIPE.TSDF.multi.hash_profile; must match the selected binary build profile",
    )
    parser.add_argument("--multi-occupancy-growth-lookahead-frames", type=int, default=None, help="override PIPE.TSDF.multi.occupancy_growth_lookahead_frames")
    parser.add_argument("--multi-occupancy-growth-min-primary-frames", type=int, default=None, help="override PIPE.TSDF.multi.occupancy_growth_min_primary_frames")
    parser.add_argument("--multi-occupancy-growth-sample-frames", type=int, default=None, help="override PIPE.TSDF.multi.occupancy_growth_sample_frames")
    parser.add_argument("--multi-paging", choices=["default", "on", "off"], default="default", help="override PIPE.TSDF.multi.paging")
    parser.add_argument(
        "--tsdf-input-mode",
        choices=["default", "preload", "on_demand"],
        default="default",
        help="override PIPE.TSDF.input_mode for generated experiments",
    )
    parser.add_argument(
        "--reader-preload-images",
        choices=["auto", "preserve", "true", "false"],
        default="auto",
        help=(
            "control READER.preload_images in generated configs; auto mirrors "
            "PIPE.TSDF.input_mode so on_demand runs do not keep desktop preloading"
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
    parser.add_argument("--tsdf-paging-storage", choices=["disk", "cpu"], default=None, help="override PIPE.TSDF.multi.paging_storage")
    parser.add_argument("--tsdf-async-page-out", choices=["default", "on", "off"], default="default", help="override PIPE.TSDF.multi.async_page_out")
    parser.add_argument("--tsdf-max-pending-page-snapshots", type=int, default=None, help="override PIPE.TSDF.multi.max_pending_page_snapshots")
    parser.add_argument("--tsdf-max-pending-page-snapshot-mb", type=int, default=None, help="override PIPE.TSDF.multi.max_pending_page_snapshot_mb")
    parser.add_argument("--resident-gpu-maps", default="", help="override PIPE.TSDF.multi.resident_gpu_maps")
    parser.add_argument(
        "--tsdf-residency-profile",
        choices=["manual", "thor_cpu_bounded", "eval_resident64", "disk_stress"],
        default="manual",
        help="apply a named MultiEngine residency policy before low-level TSDF residency overrides",
    )
    parser.add_argument("--tsdf-render-prefetch", action="store_true", help="enable PIPE.TSDF.multi.render_prefetch for MultiEngine eval runs")
    parser.add_argument("--tsdf-render-prefetch-lookahead-frames", type=int, default=None, help="override PIPE.TSDF.multi.render_prefetch_lookahead_frames")
    parser.add_argument("--tsdf-render-prefetch-lookahead-max-maps", type=int, default=None, help="override PIPE.TSDF.multi.render_prefetch_lookahead_max_maps; 0 means unlimited")
    parser.add_argument("--tsdf-render-pin-primary", choices=["default", "on", "off"], default="default", help="override PIPE.TSDF.multi.render_pin_primary_submap")
    parser.add_argument("--tsdf-async-page-in", action="store_true", help="enable CPU-staged async TSDF page-in prefetch for MultiEngine eval/render runs")
    parser.add_argument("--tsdf-max-pending-page-ins", type=int, default=None, help="override PIPE.TSDF.multi.max_pending_page_ins")
    parser.add_argument("--tsdf-max-pending-page-in-mb", type=int, default=None, help="override PIPE.TSDF.multi.max_pending_page_in_mb")
    parser.add_argument("--tsdf-render-compositor", choices=["nearest_depth", "owner_priority", "handoff_aware"], default=None, help="override PIPE.TSDF.multi.render_compositor for bounded MultiEngine raycasts")
    parser.add_argument("--sequential-transform-max-translation", type=float, default=None, help="override PIPE.TSDF.multi.sequential_transform_max_translation for MultiEngine runs")
    parser.add_argument("--sequential-transform-max-rotation-deg", type=float, default=None, help="override PIPE.TSDF.multi.sequential_transform_max_rotation_deg for MultiEngine runs")
    parser.add_argument("--invalid-transform-restart-attempts", type=int, default=None, help="override PIPE.TSDF.multi.invalid_transform_restart_attempts for MultiEngine runs")
    parser.add_argument("--new-map-handoff-policy", choices=["visibility", "accepted", "validated_visibility"], default=None, help="override PIPE.TSDF.multi.new_map_handoff_policy for MultiEngine runs")
    parser.add_argument("--new-map-handoff-min-visible-ratio", type=float, default=None, help="minimum candidate visible-original-block ratio before promoting a NEW_LOCAL_MAP")
    parser.add_argument("--new-map-handoff-min-visible-margin", type=float, default=None, help="minimum candidate-visible minus primary-visible margin before promoting a NEW_LOCAL_MAP")
    parser.add_argument("--new-map-handoff-min-accepts", type=int, default=None, help="accepted overlap windows required before promoting a NEW_LOCAL_MAP")
    parser.add_argument("--new-map-handoff-emergency-min-accepts", type=int, default=None, help="emergency/capacity accepted overlap windows required before promoting a NEW_LOCAL_MAP; -1 inherits the normal minimum")
    parser.add_argument("--multi-force-single", action="store_true", help="force MultiEngine to keep one local SDF map")
    parser.add_argument("--multi-use-gt-pose", action="store_true", help="use GT poses for selected MultiEngine experiments")
    parser.add_argument(
        "--basic-unified-use-estimated-pose",
        action="store_true",
        help="run the BasicEngine unified-Gaussian label with estimated tracker poses instead of its GT-pose quality default",
    )
    parser.add_argument("--hybrid-use-estimated-pose", action="store_true", help="force hybrid MultiEngine experiments to use estimated tracking poses instead of the GT-pose isolation default")
    parser.add_argument("--new-gs-sample-ratio", type=float, default=None, help="override PIPE.new_gs_sample_ratio")
    parser.add_argument("--local-opt-iters", type=int, default=None, help="override PIPE.local_opt_iters")
    parser.add_argument("--local-opt-interval", type=int, default=None, help="override PIPE.local_opt_interval")
    parser.add_argument(
        "--gaussian-schedule",
        choices=["auto", "immediate", "sliced"],
        default="auto",
        help="override PIPE.gaussian_schedule.mode",
    )
    parser.add_argument(
        "--gaussian-opt-iters-per-frame",
        type=int,
        default=None,
        help="iterations per frame for sliced Gaussian optimization",
    )
    parser.add_argument("--localframe-window-length", type=int, default=None, help="override PIPE.localframe_cam_window_length")
    parser.add_argument("--localframe-window-interval", type=int, default=None, help="override PIPE.localframe_cam_window_interval")
    parser.add_argument("--ssim-weight", type=float, default=None, help="override PIPE.weight_configs.ssim_weight in [0,1]")
    parser.add_argument("--depth-weight", type=float, default=None, help="override PIPE.weight_configs.depth_weight; 0 disables Gaussian depth loss")
    parser.add_argument("--color-error-thres", type=float, default=None, help="override PIPE.color_error_thres")
    parser.add_argument("--model-default-opacities", type=float, default=None, help="override MODEL.default_opacities")
    parser.add_argument("--model-min-init-scale", type=float, default=None, help="override MODEL.min_init_scale")
    parser.add_argument("--model-max-init-scale", type=float, default=None, help="override MODEL.max_init_scale")
    parser.add_argument("--model-delta-depth", type=float, default=None, help="override MODEL.delta_depth")
    parser.add_argument("--model-sh-degree", type=int, default=None, help="override MODEL.sh_degree")
    parser.add_argument("--gaussian-render-policy", default="", help="override PIPE.gaussian_submaps.render_policy for hybrid runs")
    parser.add_argument(
        "--gaussian-render-selection-policy",
        choices=["auto", "temporal", "sdf_bounds", "sdf_bounds_continuity"],
        default="auto",
        help="override PIPE.gaussian_submaps.render_selection_policy for hybrid runs",
    )
    parser.add_argument("--render-max-visible-submaps", type=int, default=None, help="override PIPE.gaussian_submaps.render_max_visible_submaps; 0 means no cap")
    parser.add_argument("--render-max-visible-gaussians", type=int, default=None, help="override PIPE.gaussian_submaps.render_max_visible_gaussians; 0 means no Gaussian-count cap")
    parser.add_argument("--render-bounds-inflate", type=float, default=None, help="inflate SDF local-map bounds by this many meters for frustum selection")
    parser.add_argument("--render-bounds-min-projected-area", type=float, default=None, help="minimum projected screen-area fraction for SDF-bounds render selection")
    parser.add_argument("--render-selection-hysteresis-frames", type=int, default=None, help="keep recently selected SDF-bounds Gaussian submaps slightly sticky for this many frames")
    parser.add_argument("--render-selection-hysteresis-area-bonus", type=float, default=None, help="relative projected-area bonus for recently selected SDF-bounds Gaussian submaps")
    parser.add_argument("--render-selection-continuity-frame-scale", type=int, default=None, help="frame-distance decay scale for sdf_bounds_continuity render selection")
    parser.add_argument("--render-selection-continuity-min-factor", type=float, default=None, help="minimum temporal-decay multiplier for sdf_bounds_continuity render selection")
    parser.add_argument("--render-selection-adjacent-bonus", type=float, default=None, help="relative score bonus for submaps adjacent to the owner in sdf_bounds_continuity selection")
    parser.add_argument("--render-selection-active-overlap-bonus", type=float, default=None, help="relative score bonus for recorded active-overlap submaps in sdf_bounds_continuity selection")
    parser.add_argument("--cross-submap-novelty-gate", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.cross_submap_novelty_gate")
    parser.add_argument("--cross-novelty-reference-policy", choices=["auto", "legacy_previous", "asymmetric_handoff"], default="auto", help="reference-submap policy for cross-submap novelty gating")
    parser.add_argument("--cross-novelty-depth-tolerance-mode", choices=["auto", "legacy_additive", "scaled_max"], default="auto", help="depth tolerance mode for cross-submap novelty gating")
    parser.add_argument("--cross-novelty-alpha-thres", type=float, default=None, help="predecessor alpha/weight threshold for suppressing duplicate Gaussian insertion")
    parser.add_argument("--cross-novelty-depth-abs-thres", type=float, default=None, help="absolute sensor-depth agreement threshold in meters for cross-submap novelty")
    parser.add_argument("--cross-novelty-depth-rel-thres", type=float, default=None, help="quadratic depth tolerance coefficient for cross-submap novelty")
    parser.add_argument("--cross-novelty-depth-floor", type=float, default=None, help="minimum depth tolerance in meters for scaled_max novelty mode")
    parser.add_argument("--cross-novelty-depth-voxel-scale", type=float, default=None, help="voxel-size multiplier for scaled_max novelty depth tolerance")
    parser.add_argument("--cross-novelty-depth-quadratic-scale", type=float, default=None, help="quadratic depth coefficient for scaled_max novelty tolerance")
    parser.add_argument("--cross-novelty-photometric-thres", type=float, default=None, help="loose per-pixel RGB L1 threshold for cross-submap novelty")
    parser.add_argument("--cross-novelty-max-reference-submaps", type=int, default=None, help="cap reference predecessor submaps used by cross-submap novelty; 0 means no cap")
    parser.add_argument("--cross-novelty-previous-only", choices=["auto", "on", "off"], default="auto", help="only use earlier submaps as cross-submap novelty references")
    parser.add_argument("--cross-novelty-resident-only", choices=["auto", "on", "off"], default="auto", help="only use already-resident SDF/Gaussian maps for cross-submap novelty checks")
    parser.add_argument("--cross-novelty-erode-kernel", type=int, default=None, help="odd kernel size for eroding duplicate-suppression masks; 0 or 1 disables erosion")
    parser.add_argument("--gaussian-max-new-per-frame", type=int, default=None, help="override PIPE.gaussian_submaps.max_new_gaussians_per_frame")
    parser.add_argument("--gaussian-bootstrap-max-new-per-frame", type=int, default=None, help="temporarily lift PIPE.gaussian_submaps.max_new_gaussians_per_frame while a submap is underfilled; 0 disables")
    parser.add_argument("--gaussian-bootstrap-until-gaussians", type=int, default=None, help="keep the bootstrap insertion cap active until the owner submap reaches this Gaussian count; 0 disables the count condition")
    parser.add_argument("--gaussian-bootstrap-until-age-frames", type=int, default=None, help="keep the bootstrap insertion cap active only through this many observed submap frames; 0 disables the age condition")
    parser.add_argument("--gaussian-budget-policy", choices=["auto", "fixed", "sdf_block_density"], default="auto", help="per-submap Gaussian budget policy")
    parser.add_argument("--gaussian-budget-base", type=int, default=None, help="base Gaussian cap for sdf_block_density budget policy")
    parser.add_argument("--gaussian-budget-blocks-density", type=float, default=None, help="additional Gaussian cap per allocated SDF block")
    parser.add_argument("--gaussian-budget-min", type=int, default=None, help="minimum per-submap Gaussian cap for sdf_block_density budget policy")
    parser.add_argument("--gaussian-budget-max", type=int, default=None, help="maximum per-submap Gaussian cap for sdf_block_density budget policy")
    parser.add_argument("--gaussian-budget-pressure-ratio", type=float, default=None, help="run redundant pruning when a submap reaches this fraction of its cap")
    parser.add_argument("--gaussian-budget-prune-interval-frames", type=int, default=None, help="minimum frames between budget-pressure pruning attempts per submap")
    parser.add_argument("--gaussian-insert-diagnostics", choices=["auto", "on", "off"], default="auto", help="log per-insertion candidate pixels and added Gaussian counts for hybrid runs")
    parser.add_argument(
        "--gaussian-submaps-debug",
        choices=["auto", "on", "off"],
        default="auto",
        help="override PIPE.gaussian_submaps.debug; use off for uncontaminated throughput runs",
    )
    parser.add_argument("--gaussian-insert-selection", choices=["auto", "random", "error_topk", "tile_error_topk", "adaptive_tile_error_topk"], default="auto", help="override PIPE.gaussian_submaps.gaussian_insert_selection")
    parser.add_argument("--gaussian-insert-tile-size", type=int, default=None, help="tile size in pixels for tile_error_topk insertion")
    parser.add_argument("--gaussian-insert-tile-max-pixels", type=int, default=None, help="max selected pixels per tile for tile_error_topk; 0 derives a balanced cap")
    parser.add_argument("--gaussian-insert-adaptive-tile-min-candidate-pixels", type=int, default=None, help="minimum candidate pixels before adaptive_tile_error_topk uses tiled insertion")
    parser.add_argument("--gaussian-insert-adaptive-tile-min-target-insertions", type=int, default=None, help="minimum target insertions before adaptive_tile_error_topk uses tiled insertion")
    parser.add_argument("--gaussian-insert-adaptive-tile-min-submap-gaussians", type=int, default=None, help="minimum existing submap Gaussians before adaptive_tile_error_topk uses tiled insertion")
    parser.add_argument("--gaussian-insert-adaptive-tile-min-observed-frames", type=int, default=None, help="minimum observed owner/candidate frames before adaptive_tile_error_topk uses tiled insertion")
    parser.add_argument("--submap-keyframe-selection-policy", choices=["auto", "random", "temporal_spread", "loss_temporal_mix"], default="auto", help="override PIPE.gaussian_submaps.keyframe_selection_policy")
    parser.add_argument("--graduation-readiness-enable", choices=["auto", "on", "off"], default="auto", help="enable held-out SDF-vs-hybrid readiness probes before Gaussian submap graduation")
    parser.add_argument("--graduation-readiness-sample-cameras", type=int, default=None, help="held-out cameras used by graduation readiness probes")
    parser.add_argument("--graduation-readiness-skip-gain", type=float, default=None, help="skip graduation optimization if hybrid RGB L1 already beats SDF RGB L1 by this amount")
    parser.add_argument("--graduation-readiness-min-abs-drop", type=float, default=None, help="minimum readiness improvement for an extra capped graduation pass")
    parser.add_argument("--graduation-readiness-skip-min-observed-frames", type=int, default=None, help="require at least this many observed frames before readiness may skip graduation; 0 disables the guard")
    parser.add_argument("--graduation-readiness-skip-min-keyframes", type=int, default=None, help="require at least this many keyframes before readiness may skip graduation; 0 disables the guard")
    parser.add_argument("--quality-finalize-enable", choices=["auto", "on", "off"], default="auto", help="run a capped end-of-sequence quality pass on short or low-keyframe Gaussian submaps")
    parser.add_argument("--quality-finalize-resident-only", choices=["auto", "on", "off"], default="auto", help="skip sequence-end Gaussian quality finalization when its Gaussian or TSDF owner is not already resident")
    parser.add_argument("--quality-finalize-max-submaps", type=int, default=None, help="maximum Gaussian submaps finalized at sequence end")
    parser.add_argument("--quality-finalize-max-observed-frames", type=int, default=None, help="finalize submaps with observed-frame count at or below this threshold; 0 disables this criterion")
    parser.add_argument("--quality-finalize-min-keyframes", type=int, default=None, help="finalize submaps with fewer than this many keyframes; 0 disables this criterion")
    parser.add_argument("--quality-finalize-max-hybrid-l1", type=float, default=None, help="finalize submaps whose cached live-observation hybrid RGB L1 is above this threshold; 0 disables this criterion")
    parser.add_argument("--quality-finalize-insert-enable", choices=["auto", "on", "off"], default="auto", help="perform a bounded Gaussian insertion top-up before sequence-end quality optimization")
    parser.add_argument("--quality-finalize-insert-include-frozen", choices=["auto", "on", "off"], default="auto", help="allow sequence-end insertion top-up into already-frozen submaps")
    parser.add_argument("--quality-finalize-insert-min-sdf-submaps", type=int, default=None, help="minimum logical SDF submaps before sequence-end insertion top-up is allowed")
    parser.add_argument("--quality-finalize-insert-frames", type=int, default=None, help="stored observations used for sequence-end Gaussian insertion top-up")
    parser.add_argument("--quality-finalize-insert-max-new-per-frame", type=int, default=None, help="temporary per-frame insertion cap used only by sequence-end top-up")
    parser.add_argument("--quality-finalize-insert-acceptance", choices=["auto", "on", "off"], default="auto", help="rollback sequence-end insertion top-up if held-out readiness regresses")
    parser.add_argument("--quality-finalize-insert-acceptance-cameras", type=int, default=None, help="held-out cameras used to accept or reject sequence-end insertion top-up")
    parser.add_argument("--quality-finalize-insert-min-readiness-delta", type=float, default=None, help="minimum readiness gain required to keep inserted sequence-end Gaussians")
    parser.add_argument("--keyframe-select-max", type=int, default=None, help="override PIPE.keyframe_select_max for Gaussian optimization")
    parser.add_argument("--resident-gaussian-submaps", type=int, default=None, help="override PIPE.gaussian_submaps.resident_gaussian_submaps")
    parser.add_argument("--gaussian-insert-min-age-frames", type=int, default=None, help="delay Gaussian insertion until the active hybrid submap has this many processed frames")
    parser.add_argument("--gaussian-insert-min-sdf-blocks", type=int, default=None, help="delay Gaussian insertion until the owner TSDF submap has this many allocated voxel blocks")
    parser.add_argument("--ges-residual-scale", type=float, default=None, help="override MODEL.ges_residual_scale; 1.0 preserves the original GES blend")
    parser.add_argument("--ges-base-color-weight", type=float, default=None, help="override MODEL.ges_base_color_weight; 1.0 preserves the original GES blend")
    parser.add_argument("--candidate-reject-retire", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.candidate_reject_retire")
    parser.add_argument("--candidate-reject-retire-min-rejects", type=int, default=None, help="minimum repeated TSDF link rejects before a warming Gaussian candidate can be retired")
    parser.add_argument("--candidate-reject-retire-min-observed-frames", type=int, default=None, help="minimum observed frames before TSDF-reject retirement can remove a warming Gaussian candidate")
    parser.add_argument("--candidate-reject-retire-min-gaussians", type=int, default=None, help="minimum Gaussian count before TSDF-reject retirement is allowed")
    parser.add_argument("--candidate-reject-retire-max-inliers", type=int, default=None, help="maximum link-reject inliers still considered weak enough for TSDF-reject retirement")
    parser.add_argument("--candidate-reject-retire-cooldown-frames", type=int, default=None, help="frames to suppress recreating a Gaussian candidate after TSDF-reject retirement")
    parser.add_argument("--candidate-gaussian-insert-min-frames", type=int, default=None, help="delay warming-candidate Gaussian insertion until this many overlap frames have been observed")
    parser.add_argument("--candidate-immature-new-gs-sample-ratio", type=float, default=None, help="sample ratio for warming candidates before candidate_gaussian_insert_min_frames; -1 preserves the old full-skip behavior unless an immature cap is set")
    parser.add_argument("--candidate-immature-max-new-per-frame", type=int, default=None, help="per-frame insertion cap for warming candidates before candidate_gaussian_insert_min_frames; 0 leaves the ramp disabled")
    parser.add_argument("--candidate-extend-while-overlap", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.candidate_extend_while_overlap")
    parser.add_argument("--candidate-max-warm-frames", type=int, default=None, help="override PIPE.gaussian_submaps.candidate_max_warm_frames; 0 disables the cap")
    parser.add_argument("--candidate-handoff-max-opt-passes", type=int, default=None, help="override PIPE.gaussian_submaps.candidate_handoff_max_opt_passes; -1 inherits handoff_max_opt_passes and 0 disables the candidate cap")
    parser.add_argument("--candidate-handoff-opt-iters", type=int, default=None, help="override PIPE.gaussian_submaps.candidate_handoff_opt_iters; -1 inherits handoff_opt_iters and 0 disables candidate handoff optimization")
    parser.add_argument("--handoff-frames", type=int, default=None, help="override PIPE.gaussian_submaps.handoff_frames for outgoing submap finalization")
    parser.add_argument("--handoff-opt-iters", type=int, default=None, help="override PIPE.gaussian_submaps.handoff_opt_iters")
    parser.add_argument("--handoff-max-opt-passes", type=int, default=None, help="override PIPE.gaussian_submaps.handoff_max_opt_passes; 0 disables outgoing handoff optimization")
    parser.add_argument("--handoff-add-gaussians-to-outgoing", choices=["auto", "on", "off"], default="auto", help="allow bounded Gaussian insertion into outgoing submaps during handoff optimization")
    parser.add_argument("--graduation-optimize", choices=["auto", "on", "off"], default="auto", help="run a bounded final Gaussian optimization pass before freezing outgoing submaps")
    parser.add_argument("--graduation-opt-iters", type=int, default=None, help="override PIPE.gaussian_submaps.graduation_opt_iters")
    parser.add_argument("--graduation-keyframes", type=int, default=None, help="override PIPE.gaussian_submaps.graduation_keyframes")
    parser.add_argument("--graduation-max-opt-passes", type=int, default=None, help="override PIPE.gaussian_submaps.graduation_max_opt_passes")
    parser.add_argument("--graduation-keyframe-policy", choices=["auto", "random", "temporal_spread", "loss_temporal_mix"], default="auto", help="override PIPE.gaussian_submaps.graduation_keyframe_policy")
    parser.add_argument("--graduation-observation-policy", choices=["auto", "temporal_spread", "loss_temporal_mix"], default="auto", help="override PIPE.gaussian_submaps.graduation_observation_policy")
    parser.add_argument("--graduation-adaptive", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.graduation_adaptive")
    parser.add_argument("--graduation-adaptive-loss-sample-cameras", type=int, default=None, help="override PIPE.gaussian_submaps.graduation_adaptive_loss_sample_cameras")
    parser.add_argument("--graduation-adaptive-target-loss", type=float, default=None, help="override PIPE.gaussian_submaps.graduation_adaptive_target_loss")
    parser.add_argument("--graduation-adaptive-min-abs-drop", type=float, default=None, help="override PIPE.gaussian_submaps.graduation_adaptive_min_abs_drop")
    parser.add_argument("--graduation-adaptive-min-rel-drop", type=float, default=None, help="override PIPE.gaussian_submaps.graduation_adaptive_min_rel_drop")
    parser.add_argument("--training-observation-max-per-submap", type=int, default=None, help="override PIPE.gaussian_submaps.training_observation_max_per_submap")
    parser.add_argument("--training-observation-interval-frames", type=int, default=None, help="override PIPE.gaussian_submaps.training_observation_interval_frames")
    parser.add_argument("--frozen-refinement-enable", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.frozen_refinement_enable")
    parser.add_argument("--frozen-refinement-interval-frames", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_interval_frames")
    parser.add_argument("--frozen-refinement-opt-iters", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_opt_iters")
    parser.add_argument("--frozen-refinement-keyframes", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_keyframes")
    parser.add_argument("--frozen-refinement-max-passes-per-submap", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_max_passes_per_submap")
    parser.add_argument("--frozen-refinement-max-submaps-per-frame", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_max_submaps_per_frame")
    parser.add_argument("--frozen-refinement-min-new-observations", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_min_new_observations")
    parser.add_argument("--frozen-refinement-readiness-guard", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.frozen_refinement_readiness_guard_enable")
    parser.add_argument("--frozen-refinement-readiness-use-cached", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.frozen_refinement_readiness_use_cached_graduation")
    parser.add_argument("--frozen-refinement-readiness-sample-cameras", type=int, default=None, help="override PIPE.gaussian_submaps.frozen_refinement_readiness_sample_cameras")
    parser.add_argument("--frozen-refinement-readiness-max-pre-gain", type=float, default=None, help="run frozen refinement only when readiness_delta_before is at or below this value")
    parser.add_argument("--frozen-refinement-readiness-min-valid-cameras", type=int, default=None, help="minimum valid readiness-probe cameras required before guarded refinement may run")
    parser.add_argument("--frozen-refinement-acceptance", choices=["auto", "on", "off"], default="auto", help="rollback frozen-refinement optimization when held-out readiness regresses")
    parser.add_argument("--frozen-refinement-acceptance-min-delta", type=float, default=None, help="minimum readiness gain required to keep a frozen-refinement pass")
    parser.add_argument(
        "--frozen-refinement-observation-policy",
        choices=["auto", "temporal_spread", "readiness_temporal_mix", "sdf_error_temporal_mix"],
        default="auto",
        help="override PIPE.gaussian_submaps.frozen_refinement_observation_policy",
    )
    parser.add_argument(
        "--frozen-refinement-observation-selection-max-eval",
        type=int,
        default=None,
        help="maximum stored observations evaluated by readiness-based frozen-refinement selection; 0 evaluates all",
    )
    parser.add_argument("--cross-boundary-refinement", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.cross_boundary_refinement_enable")
    parser.add_argument("--cross-boundary-interval-frames", type=int, default=None, help="override PIPE.gaussian_submaps.cross_boundary_observation_interval_frames")
    parser.add_argument("--cross-boundary-max-submaps-per-frame", type=int, default=None, help="override PIPE.gaussian_submaps.cross_boundary_max_submaps_per_frame")
    parser.add_argument("--cross-boundary-min-owner-age-frames", type=int, default=None, help="override PIPE.gaussian_submaps.cross_boundary_min_owner_age_frames")
    parser.add_argument("--cross-boundary-min-neighbor-gaussians", type=int, default=None, help="override PIPE.gaussian_submaps.cross_boundary_min_neighbor_gaussians")
    parser.add_argument("--cross-boundary-frozen-only", choices=["auto", "on", "off"], default="auto", help="only collect cross-boundary observations for accepted frozen maps")
    parser.add_argument("--cross-boundary-resident-only", choices=["auto", "on", "off"], default="auto", help="skip nonresident SDF/Gaussian maps during online cross-boundary observation")
    parser.add_argument("--cross-boundary-sequential-only", choices=["auto", "on", "off"], default="auto", help="skip non-adjacent submaps during online cross-boundary observation")
    parser.add_argument("--cross-boundary-depth-gate", choices=["auto", "on", "off"], default="auto", help="require SDF raycast depth agreement before storing cross-boundary observations")
    parser.add_argument("--cross-boundary-min-projected-ratio", type=float, default=None, help="minimum candidate SDF projected pixel ratio for cross-boundary observations")
    parser.add_argument("--cross-boundary-min-depth-consistent-ratio", type=float, default=None, help="minimum depth-consistent comparable pixel ratio for cross-boundary observations")
    parser.add_argument("--cross-boundary-depth-tolerance-mode", choices=["auto", "fixed", "scaled_max"], default="auto", help="depth tolerance policy for cross-boundary observations")
    parser.add_argument("--cross-boundary-depth-abs-thres", type=float, default=None, help="fixed cross-boundary depth tolerance in meters")
    parser.add_argument("--cross-boundary-depth-floor", type=float, default=None, help="minimum cross-boundary scaled depth tolerance in meters")
    parser.add_argument("--cross-boundary-depth-voxel-scale", type=float, default=None, help="voxel-size multiplier for cross-boundary scaled depth tolerance")
    parser.add_argument("--cross-boundary-depth-quadratic-scale", type=float, default=None, help="quadratic depth coefficient for cross-boundary scaled depth tolerance")
    parser.add_argument("--candidate-new-gs-sample-ratio", type=float, default=None, help="override PIPE.gaussian_submaps.candidate_new_gs_sample_ratio; -1 inherits PIPE.new_gs_sample_ratio")
    parser.add_argument("--candidate-max-new-per-frame", type=int, default=None, help="override PIPE.gaussian_submaps.candidate_max_new_gaussians_per_frame")
    parser.add_argument("--render-min-ready-frames", type=int, default=None, help="override PIPE.gaussian_submaps.render_min_ready_frames")
    parser.add_argument("--render-owner-min-ready-frames", type=int, default=None, help="override PIPE.gaussian_submaps.render_owner_min_ready_frames")
    parser.add_argument("--render-neighbor-min-ready-frames", type=int, default=None, help="override PIPE.gaussian_submaps.render_neighbor_min_ready_frames")
    parser.add_argument("--render-past-frame-margin", type=int, default=None, help="override PIPE.gaussian_submaps.render_past_frame_margin")
    parser.add_argument("--render-future-frame-margin", type=int, default=None, help="override PIPE.gaussian_submaps.render_future_frame_margin")
    parser.add_argument("--render-min-neighbor-gaussians", type=int, default=None, help="override PIPE.gaussian_submaps.render_min_neighbor_gaussians")
    parser.add_argument("--render-overlap-depth-gate", choices=["auto", "on", "off"], default="auto", help="override PIPE.gaussian_submaps.render_overlap_depth_gate")
    parser.add_argument("--render-overlap-depth-mode", choices=["auto", "fixed", "scaled_max"], default="auto", help="override PIPE.gaussian_submaps.render_overlap_depth_tolerance_mode")
    parser.add_argument("--render-overlap-depth-thres", type=float, default=None, help="override PIPE.gaussian_submaps.render_overlap_depth_abs_thres")
    parser.add_argument("--render-overlap-alpha-thres", type=float, default=None, help="override PIPE.gaussian_submaps.render_overlap_alpha_thres")
    parser.add_argument("--render-hole-requires-boundary", action="store_true", help="require nearby SDF boundary evidence before Gaussian hole fill")
    parser.add_argument("--render-include-handoff-submaps", action="store_true", help="include handoff submaps during Gaussian render selection")
    parser.add_argument("--disable-single-submap-passthrough", action="store_true", help="force single-submap renders through the overlap compositor for ablation")
    parser.add_argument("--allow-multi-logical-single-passthrough", action="store_true", help="allow passthrough for a single selected Gaussian submap even when the saved graph has multiple logical submaps; kept for compatibility because this is now the default")
    parser.add_argument("--require-single-logical-single-passthrough", action="store_true", help="only allow single-submap passthrough when the whole saved graph has one logical Gaussian submap")
    parser.add_argument("--disable-local-sdf-reference-render", action="store_true", help="render local Gaussian submaps against the combined global SDF reference for ablation")
    parser.add_argument("--render-overlap-diagnostics", action="store_true", help="write scalar owner/neighbor/fill coverage diagnostics into submap_render_debug.csv")
    parser.add_argument(
        "--disable-render-overlap-diagnostics",
        action="store_true",
        help="force live PIPE.gaussian_submaps.render_overlap_diagnostics off for throughput runs",
    )
    parser.add_argument("--pose-graph-identity-check", action="store_true", help="enable reversible identity pose-graph readiness check for hybrid runs")
    parser.add_argument("--pose-graph-synthetic-check", action="store_true", help="enable reversible synthetic pose-graph transform check for hybrid runs")
    parser.add_argument("--pose-graph-render-identity-check", action="store_true", help="enable render-output identity check for hybrid submap eval runs")
    parser.add_argument("--pose-graph-render-synthetic-check", action="store_true", help="enable render-output synthetic correction check for hybrid submap eval runs")
    parser.add_argument("--pose-graph-dry-run", action="store_true", help="validate exported hybrid pose-graph residuals without applying corrections")
    parser.add_argument("--pose-graph-dry-run-include-loops", action="store_true", help="include exported loop candidates in pose-graph dry-run residual checks")
    parser.add_argument("--pose-graph-solve-only", action="store_true", help="run pose-graph optimizer on copied data and report proposed deltas without applying corrections")
    parser.add_argument("--pose-graph-solve-only-include-loops", action="store_true", help="include exported loop candidates in solve-only pose-graph optimization")
    parser.add_argument(
        "--pose-graph-solve-only-max-translation-delta",
        type=float,
        default=None,
        help="override PIPE.pose_graph.solve_only_max_translation_delta",
    )
    parser.add_argument(
        "--pose-graph-solve-only-max-rotation-delta-deg",
        type=float,
        default=None,
        help="override PIPE.pose_graph.solve_only_max_rotation_delta_deg",
    )
    parser.add_argument("--pose-graph-candidates", action="store_true", help="export CPU-only pose graph loop candidates for hybrid submap runs")
    parser.add_argument("--pose-graph-loop-max-distance", type=float, default=None, help="override PIPE.pose_graph.loop_max_distance")
    parser.add_argument("--pose-graph-loop-max-rotation-deg", type=float, default=None, help="override PIPE.pose_graph.loop_max_rotation_deg")
    parser.add_argument("--pose-graph-loop-min-separation", type=int, default=None, help="override PIPE.pose_graph.loop_min_submap_separation")
    parser.add_argument("--pose-graph-loop-max-candidates", type=int, default=None, help="override PIPE.pose_graph.loop_max_candidates")
    parser.add_argument(
        "--pose-graph-candidate-verification",
        choices=["none", "keyframe_pose", "keyframe_depth", "keyframe_icp"],
        default=None,
        help="override PIPE.pose_graph.candidate_verification",
    )
    parser.add_argument(
        "--pose-graph-verify-max-keyframe-distance",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_max_keyframe_distance",
    )
    parser.add_argument(
        "--pose-graph-verify-max-keyframe-rotation-deg",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_max_keyframe_rotation_deg",
    )
    parser.add_argument(
        "--pose-graph-verify-max-depth-error",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_max_depth_error",
    )
    parser.add_argument(
        "--pose-graph-verify-min-depth-overlap",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_min_depth_overlap",
    )
    parser.add_argument(
        "--pose-graph-verify-depth-sample-stride",
        type=int,
        default=None,
        help="override PIPE.pose_graph.verify_depth_sample_stride",
    )
    parser.add_argument(
        "--pose-graph-verify-icp-min-inliers",
        type=int,
        default=None,
        help="override PIPE.pose_graph.verify_icp_min_inliers",
    )
    parser.add_argument(
        "--pose-graph-verify-icp-max-rmse",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_icp_max_rmse",
    )
    parser.add_argument(
        "--pose-graph-verify-icp-max-translation-delta",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_icp_max_translation_delta",
    )
    parser.add_argument(
        "--pose-graph-verify-icp-max-rotation-delta-deg",
        type=float,
        default=None,
        help="override PIPE.pose_graph.verify_icp_max_rotation_delta_deg",
    )
    args = parser.parse_args()
    if args.render_overlap_diagnostics and args.disable_render_overlap_diagnostics:
        parser.error(
            "--render-overlap-diagnostics and --disable-render-overlap-diagnostics conflict"
        )
    if args.ssim_weight is not None and (not math.isfinite(args.ssim_weight) or not 0.0 <= args.ssim_weight <= 1.0):
        parser.error("--ssim-weight must be finite and in [0, 1]")
    if args.depth_weight is not None and (not math.isfinite(args.depth_weight) or args.depth_weight < 0.0):
        parser.error("--depth-weight must be finite and non-negative")
    if args.quality_native_full and (not args.quality or args.full_res):
        parser.error("--quality-native-full requires --quality and a half-resolution training run")
    if args.multi_use_gt_pose and args.hybrid_use_estimated_pose:
        parser.error("--multi-use-gt-pose and --hybrid-use-estimated-pose conflict for hybrid labels")
    if not math.isfinite(args.trainer_timeout_sec) or args.trainer_timeout_sec < 0.0:
        parser.error("--trainer-timeout-sec must be finite and non-negative")

    half = not args.full_res
    matrix = experiment_matrix(
        args.frames,
        half,
        args.gaussian_profile,
        args.hybrid_submap_profile,
    )
    available_labels = {label for label, _overrides in matrix}
    try:
        validate_path_token(args.scene_name, "--scene-name")
        validate_path_token(args.tag, "--tag", allow_empty=True)
        requested_labels = parse_requested_labels(args.labels, available_labels)
    except ValueError as exc:
        parser.error(str(exc))

    args.binary = (PROJECT_ROOT / args.binary).resolve() if not args.binary.is_absolute() else args.binary.resolve()
    binary_hash_profile = infer_binary_hash_profile(args.binary)
    benchmark_roles: dict[str, str] = {}
    if args.benchmark_protocol != PROTOCOL_NONE:
        if (
            protocol_release_status(args.benchmark_protocol) != "publication_locked"
            and not args.allow_unpromoted_protocol
        ):
            parser.error(
                f"{args.benchmark_protocol} is unpromoted and not paper-eligible; "
                "use --allow-unpromoted-protocol only to reproduce its rejected screen"
            )
        if not requested_labels or len(requested_labels) != 1:
            parser.error("a locked benchmark requires exactly one explicit --labels value")
        benchmark_label = next(iter(requested_labels))
        try:
            inferred_role = replica_role_for_label(benchmark_label)
        except ValueError as exc:
            parser.error(str(exc))
        if args.benchmark_role != "auto" and args.benchmark_role != inferred_role:
            parser.error(
                f"--benchmark-role={args.benchmark_role} conflicts with "
                f"--labels={benchmark_label} ({inferred_role})"
            )
        benchmark_roles[benchmark_label] = inferred_role
        if args.benchmark_protocol == REPLICA_DEPLOYED_QUALITY_V1 and inferred_role != "modified":
            parser.error(
                "the deployed Replica quality protocol requires --labels multi_hybrid_warm_fill"
            )
        if not args.full_res or args.frames != 2000 or args.start_frame != 0 or args.frame_step != 1:
            parser.error(
                "Replica paper parity requires --full-res --frames 2000 "
                "--start-frame 0 --frame-step 1"
            )
        if args.heldout_interval != -1 or not args.quality or args.quality_native_full:
            parser.error(
                "Replica paper parity requires --heldout-interval -1 and --quality, "
                "without --quality-native-full"
            )
        if inferred_role == "modified" and not args.quality_final_map:
            parser.error("the modified Replica benchmark requires --quality-final-map")
        if inferred_role == "baseline" and binary_hash_profile != "legacy_large":
            parser.error(
                "the faithful baseline requires build_legacy/slam_trainer "
                f"(legacy_large); detected {binary_hash_profile!r}"
            )
        if not args.only_generate:
            try:
                require_max_performance_state()
            except RuntimeError as exc:
                parser.error(str(exc))
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "_logs"
    free_gb = shutil.disk_usage(args.output_root).free / 1024**3
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Refusing to run: {free_gb:.1f} GB free, need {args.min_free_gb:.1f} GB")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = resolve_replica_input_dir(args.scene_name, half, args.input_dir)
    gt_dir = args.gt_dir.resolve() if args.gt_dir is not None else input_dir
    if input_dir is not None and not input_dir.exists():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")
    if gt_dir is not None and not gt_dir.exists():
        raise FileNotFoundError(f"--gt-dir does not exist: {gt_dir}")
    quality_eval_split = "val" if args.heldout_interval > 0 else "train"
    mixed_pose_quality_warning = (
        args.quality
        and args.heldout_interval > 0
        and args.hybrid_use_estimated_pose
    )
    if mixed_pose_quality_warning:
        print(
            "[WARN] Estimated-pose held-out quality is a diagnostic only: "
            "training frames use saved SLAM poses, but held-out frames have no live estimated pose "
            "and fall back to dataset poses during eval."
        )
    basic_labels = {"basic_sdf", "basic_sdf_gt", "basic_unified_gaussian"}
    selected_labels = requested_labels or available_labels
    non_legacy_basic_warning = bool(selected_labels & basic_labels) and binary_hash_profile not in {"legacy_large", "unknown"}
    if non_legacy_basic_warning:
        print(
            "[WARN] BasicEngine labels are using a reduced-capacity TSDF hash profile "
            f"({binary_hash_profile}) from binary {args.binary}. Use build_legacy/slam_trainer "
            "for original-capacity BasicEngine baselines."
        )
    rows = []
    for label, overrides in matrix:
        if requested_labels and label not in requested_labels:
            continue
        suffix_parts = [
            f"{args.frames}f",
            "half" if half else "full",
            args.scene_name,
            f"gp{args.gaussian_profile}",
            f"bin{binary_hash_profile}",
        ]
        if args.heldout_interval > 0:
            suffix_parts.append(f"heldout{args.heldout_interval}")
        if args.random_seed is not None:
            suffix_parts.append(f"seed{int(args.random_seed)}")
        if args.start_frame > 0:
            suffix_parts.append(f"start{max(0, args.start_frame)}")
        if args.frame_step > 1:
            suffix_parts.append(f"step{max(1, args.frame_step)}")
        if label == "multi_hybrid_warm_fill" and args.hybrid_submap_profile != "normal":
            suffix_parts.append(args.hybrid_submap_profile)
        if label.startswith("multi_"):
            suffix_parts.append(f"hsub{args.hybrid_submap_profile}")
        if label.startswith("multi_") and args.tsdf_render_compositor is not None:
            suffix_parts.append(f"tsdfcomp{args.tsdf_render_compositor}")
        if label.startswith("multi_") and args.sequential_transform_max_rotation_deg is not None:
            suffix_parts.append(f"seqrot{suffix_value(max(0.0, args.sequential_transform_max_rotation_deg))}")
        if label.startswith("multi_") and args.sequential_transform_max_translation is not None:
            suffix_parts.append(f"seqtrans{suffix_value(max(0.0, args.sequential_transform_max_translation))}")
        if label.startswith("multi_") and args.invalid_transform_restart_attempts is not None:
            suffix_parts.append(f"seqrestart{max(0, args.invalid_transform_restart_attempts)}")
        if label.startswith("multi_") and args.new_map_handoff_policy is not None:
            suffix_parts.append(f"handoff{args.new_map_handoff_policy}")
        if label.startswith("multi_") and args.new_map_handoff_min_visible_ratio is not None:
            suffix_parts.append(f"hmv{suffix_value(max(0.0, min(1.0, args.new_map_handoff_min_visible_ratio)))}")
        if label.startswith("multi_") and args.new_map_handoff_min_visible_margin is not None:
            suffix_parts.append(f"hmm{suffix_value(max(0.0, min(1.0, args.new_map_handoff_min_visible_margin)))}")
        if label.startswith("multi_") and args.new_map_handoff_min_accepts is not None:
            suffix_parts.append(f"hacc{max(0, args.new_map_handoff_min_accepts)}")
        if label.startswith("multi_") and args.new_map_handoff_emergency_min_accepts is not None:
            suffix_parts.append(f"ehacc{max(-1, args.new_map_handoff_emergency_min_accepts)}")
        if label.startswith("multi_") and args.multi_hash_profile != "default":
            suffix_parts.append(args.multi_hash_profile)
        if label.startswith("multi_") and args.multi_sampled_original_blocks is not None:
            suffix_parts.append(f"sampblocks{max(1, args.multi_sampled_original_blocks)}")
        if label.startswith("multi_") and args.multi_new_map_visible_ratio is not None:
            suffix_parts.append(f"vis{suffix_value(max(0.0, min(1.0, args.multi_new_map_visible_ratio)))}")
        if label == "basic_unified_gaussian" and args.basic_unified_use_estimated_pose:
            suffix_parts.append("estpose")
        if args.tsdf_input_mode != "default":
            suffix_parts.append(f"input{args.tsdf_input_mode.replace('_', '')}")
        if args.tsdf_input_prefetch != "default":
            suffix_parts.append(f"prefetch{1 if args.tsdf_input_prefetch == 'on' else 0}")
        if args.tsdf_input_tensor_bridge != "direct":
            suffix_parts.append(f"bridge{args.tsdf_input_tensor_bridge}")
        if args.tsdf_input_bridge_validate:
            suffix_parts.append("bridgevalidate")
        if args.submap_stats_interval_frames is not None:
            suffix_parts.append(f"stats{max(0, args.submap_stats_interval_frames)}")
        if label.startswith("multi_") and args.tsdf_residency_profile != "manual":
            suffix_parts.append(f"res{args.tsdf_residency_profile}")
        if (
            label.startswith("multi_") and
            args.quality_final_map and
            args.final_map_tsdf_residency_profile != "inherit"
        ):
            suffix_parts.append(f"fmres{args.final_map_tsdf_residency_profile}")
        if args.tsdf_render_pin_primary != "default":
            suffix_parts.append(f"pinprimary{1 if args.tsdf_render_pin_primary == 'on' else 0}")
        if args.tsdf_render_prefetch_lookahead_frames is not None:
            suffix_parts.append(f"rlook{max(0, args.tsdf_render_prefetch_lookahead_frames)}")
        if args.tsdf_render_prefetch_lookahead_max_maps is not None:
            suffix_parts.append(f"rlookmaps{max(0, args.tsdf_render_prefetch_lookahead_max_maps)}")
        if args.new_gs_sample_ratio is not None:
            suffix_parts.append(f"gsr{suffix_value(args.new_gs_sample_ratio)}")
        if args.local_opt_iters is not None:
            suffix_parts.append(f"opt{max(0, args.local_opt_iters)}")
        if args.local_opt_interval is not None:
            suffix_parts.append(f"opint{max(1, args.local_opt_interval)}")
        if args.color_error_thres is not None:
            suffix_parts.append(f"cerr{suffix_value(args.color_error_thres)}")
        if args.model_default_opacities is not None:
            suffix_parts.append(f"opac{suffix_value(args.model_default_opacities)}")
        if args.model_min_init_scale is not None:
            suffix_parts.append(f"mins{suffix_value(args.model_min_init_scale)}")
        if args.model_max_init_scale is not None:
            suffix_parts.append(f"maxs{suffix_value(args.model_max_init_scale)}")
        if args.model_delta_depth is not None:
            suffix_parts.append(f"ddepth{suffix_value(args.model_delta_depth)}")
        if args.model_sh_degree is not None:
            suffix_parts.append(f"sh{max(0, args.model_sh_degree)}")
        if args.ges_residual_scale is not None:
            suffix_parts.append(f"gres{suffix_value(args.ges_residual_scale)}")
        if args.ges_base_color_weight is not None:
            suffix_parts.append(f"gbw{suffix_value(args.ges_base_color_weight)}")
        if args.gaussian_insert_selection != "auto":
            suffix_parts.append(f"gsel{args.gaussian_insert_selection.replace('_', '')}")
        if args.gaussian_insert_tile_size is not None:
            suffix_parts.append(f"gts{max(1, args.gaussian_insert_tile_size)}")
        if args.gaussian_insert_tile_max_pixels is not None:
            suffix_parts.append(f"gtmax{max(0, args.gaussian_insert_tile_max_pixels)}")
        if args.gaussian_insert_adaptive_tile_min_candidate_pixels is not None:
            suffix_parts.append(f"gatcand{max(0, args.gaussian_insert_adaptive_tile_min_candidate_pixels)}")
        if args.gaussian_insert_adaptive_tile_min_target_insertions is not None:
            suffix_parts.append(f"gattgt{max(0, args.gaussian_insert_adaptive_tile_min_target_insertions)}")
        if args.gaussian_insert_adaptive_tile_min_submap_gaussians is not None:
            suffix_parts.append(f"gatgs{max(0, args.gaussian_insert_adaptive_tile_min_submap_gaussians)}")
        if args.gaussian_insert_adaptive_tile_min_observed_frames is not None:
            suffix_parts.append(f"gatobs{max(0, args.gaussian_insert_adaptive_tile_min_observed_frames)}")
        if args.submap_keyframe_selection_policy != "auto":
            suffix_parts.append(f"kfsel{args.submap_keyframe_selection_policy.replace('_', '')}")
        if args.keyframe_select_max is not None:
            suffix_parts.append(f"kfmax{max(0, args.keyframe_select_max)}")
        if args.gaussian_render_selection_policy != "auto":
            suffix_parts.append(f"rsel{args.gaussian_render_selection_policy.replace('_', '')}")
        if args.render_max_visible_submaps is not None:
            suffix_parts.append(f"rvis{max(0, args.render_max_visible_submaps)}")
        if args.render_max_visible_gaussians is not None:
            suffix_parts.append(f"rgs{max(0, args.render_max_visible_gaussians)}")
        if args.render_bounds_inflate is not None:
            suffix_parts.append(f"rbinf{suffix_value(max(0.0, args.render_bounds_inflate))}")
        if args.render_bounds_min_projected_area is not None:
            suffix_parts.append(f"rbarea{suffix_value(max(0.0, args.render_bounds_min_projected_area))}")
        if args.render_selection_hysteresis_frames is not None:
            suffix_parts.append(f"rhyst{max(0, args.render_selection_hysteresis_frames)}")
        if args.render_selection_hysteresis_area_bonus is not None:
            suffix_parts.append(f"rhystb{suffix_value(max(0.0, args.render_selection_hysteresis_area_bonus))}")
        if args.render_selection_continuity_frame_scale is not None:
            suffix_parts.append(f"rcont{max(1, args.render_selection_continuity_frame_scale)}")
        if args.render_selection_continuity_min_factor is not None:
            suffix_parts.append(f"rcontmin{suffix_value(min(1.0, max(0.0, args.render_selection_continuity_min_factor)))}")
        if args.render_selection_adjacent_bonus is not None:
            suffix_parts.append(f"radj{suffix_value(max(0.0, args.render_selection_adjacent_bonus))}")
        if args.render_selection_active_overlap_bonus is not None:
            suffix_parts.append(f"rovlapb{suffix_value(max(0.0, args.render_selection_active_overlap_bonus))}")
        if args.render_overlap_depth_gate != "auto":
            suffix_parts.append(f"rdepth{1 if args.render_overlap_depth_gate == 'on' else 0}")
        if args.render_overlap_depth_mode != "auto":
            suffix_parts.append(f"rdepth{args.render_overlap_depth_mode.replace('_', '')}")
        if args.render_overlap_depth_thres is not None:
            suffix_parts.append(f"rdeptht{suffix_value(max(0.0, args.render_overlap_depth_thres))}")
        if args.render_overlap_alpha_thres is not None:
            suffix_parts.append(f"rdeptha{suffix_value(max(0.0, args.render_overlap_alpha_thres))}")
        if args.cross_submap_novelty_gate != "auto":
            suffix_parts.append(f"xnov{1 if args.cross_submap_novelty_gate == 'on' else 0}")
        if args.cross_novelty_reference_policy != "auto":
            suffix_parts.append(f"xrefpol{args.cross_novelty_reference_policy.replace('_', '')}")
        if args.cross_novelty_depth_tolerance_mode != "auto":
            suffix_parts.append(f"xdtol{args.cross_novelty_depth_tolerance_mode.replace('_', '')}")
        if args.cross_novelty_alpha_thres is not None:
            suffix_parts.append(f"xalpha{suffix_value(max(0.0, args.cross_novelty_alpha_thres))}")
        if args.cross_novelty_depth_abs_thres is not None:
            suffix_parts.append(f"xdepth{suffix_value(max(0.0, args.cross_novelty_depth_abs_thres))}")
        if args.cross_novelty_depth_rel_thres is not None:
            suffix_parts.append(f"xrel{suffix_value(max(0.0, args.cross_novelty_depth_rel_thres))}")
        if args.cross_novelty_depth_floor is not None:
            suffix_parts.append(f"xfloor{suffix_value(max(0.0, args.cross_novelty_depth_floor))}")
        if args.cross_novelty_depth_voxel_scale is not None:
            suffix_parts.append(f"xvox{suffix_value(max(0.0, args.cross_novelty_depth_voxel_scale))}")
        if args.cross_novelty_depth_quadratic_scale is not None:
            suffix_parts.append(f"xquad{suffix_value(max(0.0, args.cross_novelty_depth_quadratic_scale))}")
        if args.cross_novelty_photometric_thres is not None:
            suffix_parts.append(f"xphoto{suffix_value(max(0.0, args.cross_novelty_photometric_thres))}")
        if args.cross_novelty_max_reference_submaps is not None:
            suffix_parts.append(f"xrefs{max(0, args.cross_novelty_max_reference_submaps)}")
        if args.cross_novelty_resident_only != "auto":
            suffix_parts.append(f"xres{1 if args.cross_novelty_resident_only == 'on' else 0}")
        if args.cross_novelty_erode_kernel is not None:
            suffix_parts.append(f"xerode{max(0, args.cross_novelty_erode_kernel)}")
        if args.quality_finalize_resident_only != "auto":
            suffix_parts.append(f"qfres{1 if args.quality_finalize_resident_only == 'on' else 0}")
        if args.quality_finalize_max_hybrid_l1 is not None:
            suffix_parts.append(f"qfl1{suffix_value(max(0.0, args.quality_finalize_max_hybrid_l1))}")
        if args.quality_finalize_insert_acceptance != "auto":
            suffix_parts.append(f"qfiacc{1 if args.quality_finalize_insert_acceptance == 'on' else 0}")
        if args.quality_finalize_insert_acceptance_cameras is not None:
            suffix_parts.append(f"qfiaccn{max(0, args.quality_finalize_insert_acceptance_cameras)}")
        if args.quality_finalize_insert_min_readiness_delta is not None:
            suffix_parts.append(
                f"qfiaccd{suffix_value(max(0.0, args.quality_finalize_insert_min_readiness_delta))}"
            )
        if args.resident_gaussian_submaps is not None:
            suffix_parts.append(f"gresident{max(1, args.resident_gaussian_submaps)}")
        if args.gaussian_bootstrap_max_new_per_frame is not None:
            suffix_parts.append(f"gbootmax{max(0, args.gaussian_bootstrap_max_new_per_frame)}")
        if args.gaussian_bootstrap_until_gaussians is not None:
            suffix_parts.append(f"gbootgs{max(0, args.gaussian_bootstrap_until_gaussians)}")
        if args.gaussian_bootstrap_until_age_frames is not None:
            suffix_parts.append(f"gbootage{max(0, args.gaussian_bootstrap_until_age_frames)}")
        if args.gaussian_budget_policy != "auto":
            suffix_parts.append(f"gbpol{args.gaussian_budget_policy.replace('_', '')}")
        if args.gaussian_budget_base is not None:
            suffix_parts.append(f"gbbase{max(0, args.gaussian_budget_base)}")
        if args.gaussian_budget_blocks_density is not None:
            suffix_parts.append(f"gbdens{suffix_value(max(0.0, args.gaussian_budget_blocks_density))}")
        if args.gaussian_budget_min is not None:
            suffix_parts.append(f"gbmin{max(0, args.gaussian_budget_min)}")
        if args.gaussian_budget_max is not None:
            suffix_parts.append(f"gbmax{max(0, args.gaussian_budget_max)}")
        if args.gaussian_budget_pressure_ratio is not None:
            suffix_parts.append(f"gbpress{suffix_value(max(0.0, args.gaussian_budget_pressure_ratio))}")
        if args.gaussian_budget_prune_interval_frames is not None:
            suffix_parts.append(f"gbpint{max(0, args.gaussian_budget_prune_interval_frames)}")
        if args.candidate_reject_retire != "auto":
            suffix_parts.append(f"crejret{1 if args.candidate_reject_retire == 'on' else 0}")
        if args.candidate_reject_retire_min_rejects is not None:
            suffix_parts.append(f"crej{max(1, args.candidate_reject_retire_min_rejects)}")
        if args.candidate_reject_retire_min_observed_frames is not None:
            suffix_parts.append(f"crejobs{max(0, args.candidate_reject_retire_min_observed_frames)}")
        if args.candidate_reject_retire_min_gaussians is not None:
            suffix_parts.append(f"crejgs{max(0, args.candidate_reject_retire_min_gaussians)}")
        if args.candidate_reject_retire_max_inliers is not None:
            suffix_parts.append(f"crejinl{max(0, args.candidate_reject_retire_max_inliers)}")
        if args.candidate_reject_retire_cooldown_frames is not None:
            suffix_parts.append(f"crejcool{max(0, args.candidate_reject_retire_cooldown_frames)}")
        if args.candidate_gaussian_insert_min_frames is not None:
            suffix_parts.append(f"cginsertmin{max(0, args.candidate_gaussian_insert_min_frames)}")
        if args.candidate_immature_new_gs_sample_ratio is not None:
            suffix_parts.append(f"cimmgsr{suffix_value(max(-1.0, args.candidate_immature_new_gs_sample_ratio))}")
        if args.candidate_immature_max_new_per_frame is not None:
            suffix_parts.append(f"cimmmax{max(0, args.candidate_immature_max_new_per_frame)}")
        if args.graduation_readiness_enable != "auto":
            suffix_parts.append(f"gready{1 if args.graduation_readiness_enable == 'on' else 0}")
        if args.graduation_readiness_sample_cameras is not None:
            suffix_parts.append(f"greadys{max(1, args.graduation_readiness_sample_cameras)}")
        if args.graduation_readiness_skip_gain is not None:
            suffix_parts.append(f"greadygain{suffix_value(max(0.0, args.graduation_readiness_skip_gain))}")
        if args.graduation_readiness_min_abs_drop is not None:
            suffix_parts.append(f"greadydrop{suffix_value(max(0.0, args.graduation_readiness_min_abs_drop))}")
        if args.candidate_handoff_opt_iters is not None:
            suffix_parts.append(f"chiters{max(-1, args.candidate_handoff_opt_iters)}")
        if args.handoff_frames is not None:
            suffix_parts.append(f"hframes{max(0, args.handoff_frames)}")
        if args.handoff_opt_iters is not None:
            suffix_parts.append(f"hiters{max(0, args.handoff_opt_iters)}")
        if args.handoff_max_opt_passes is not None:
            suffix_parts.append(f"hpasses{max(0, args.handoff_max_opt_passes)}")
        if args.handoff_add_gaussians_to_outgoing != "auto":
            suffix_parts.append(f"hoadd{1 if args.handoff_add_gaussians_to_outgoing == 'on' else 0}")
        if args.graduation_optimize != "auto":
            suffix_parts.append(f"grad{1 if args.graduation_optimize == 'on' else 0}")
        if args.graduation_opt_iters is not None:
            suffix_parts.append(f"grediters{max(0, args.graduation_opt_iters)}")
        if args.graduation_keyframes is not None:
            suffix_parts.append(f"gradkf{max(0, args.graduation_keyframes)}")
        if args.graduation_max_opt_passes is not None:
            suffix_parts.append(f"gradpasses{max(0, args.graduation_max_opt_passes)}")
        if args.graduation_keyframe_policy != "auto":
            suffix_parts.append(f"gradpol{args.graduation_keyframe_policy}")
        if args.graduation_observation_policy != "auto":
            suffix_parts.append(f"gradobspol{args.graduation_observation_policy}")
        if args.graduation_adaptive != "auto":
            suffix_parts.append(f"gradadapt{1 if args.graduation_adaptive == 'on' else 0}")
        if args.graduation_adaptive_loss_sample_cameras is not None:
            suffix_parts.append(f"gradadcam{max(1, args.graduation_adaptive_loss_sample_cameras)}")
        if args.graduation_adaptive_target_loss is not None:
            suffix_parts.append(f"gradadtgt{args.graduation_adaptive_target_loss:g}".replace(".", "p"))
        if args.graduation_adaptive_min_abs_drop is not None:
            suffix_parts.append(f"gradadabs{max(0.0, args.graduation_adaptive_min_abs_drop):g}".replace(".", "p"))
        if args.graduation_adaptive_min_rel_drop is not None:
            suffix_parts.append(f"gradadrel{max(0.0, args.graduation_adaptive_min_rel_drop):g}".replace(".", "p"))
        if args.training_observation_max_per_submap is not None:
            suffix_parts.append(f"trobs{max(0, args.training_observation_max_per_submap)}")
        if args.training_observation_interval_frames is not None:
            suffix_parts.append(f"trobsint{max(1, args.training_observation_interval_frames)}")
        if args.frozen_refinement_enable != "auto":
            suffix_parts.append(f"fref{args.frozen_refinement_enable}")
        if args.frozen_refinement_interval_frames is not None:
            suffix_parts.append(f"frefint{max(1, args.frozen_refinement_interval_frames)}")
        if args.frozen_refinement_opt_iters is not None:
            suffix_parts.append(f"frefiters{max(0, args.frozen_refinement_opt_iters)}")
        if args.frozen_refinement_keyframes is not None:
            suffix_parts.append(f"frefkf{max(0, args.frozen_refinement_keyframes)}")
        if args.frozen_refinement_max_passes_per_submap is not None:
            suffix_parts.append(f"frefpasses{max(0, args.frozen_refinement_max_passes_per_submap)}")
        if args.frozen_refinement_max_submaps_per_frame is not None:
            suffix_parts.append(f"frefmaps{max(0, args.frozen_refinement_max_submaps_per_frame)}")
        if args.frozen_refinement_min_new_observations is not None:
            suffix_parts.append(f"frefnew{max(1, args.frozen_refinement_min_new_observations)}")
        if args.frozen_refinement_readiness_guard != "auto":
            suffix_parts.append(f"frefguard{1 if args.frozen_refinement_readiness_guard == 'on' else 0}")
        if args.frozen_refinement_readiness_use_cached != "auto":
            suffix_parts.append(f"frefcache{1 if args.frozen_refinement_readiness_use_cached == 'on' else 0}")
        if args.frozen_refinement_readiness_sample_cameras is not None:
            suffix_parts.append(f"frefreadys{max(1, args.frozen_refinement_readiness_sample_cameras)}")
        if args.frozen_refinement_readiness_max_pre_gain is not None:
            suffix_parts.append(f"frefgain{suffix_value(args.frozen_refinement_readiness_max_pre_gain)}")
        if args.frozen_refinement_readiness_min_valid_cameras is not None:
            suffix_parts.append(f"frefvalid{max(1, args.frozen_refinement_readiness_min_valid_cameras)}")
        if args.frozen_refinement_acceptance != "auto":
            suffix_parts.append(f"frefacc{1 if args.frozen_refinement_acceptance == 'on' else 0}")
        if args.frozen_refinement_acceptance_min_delta is not None:
            suffix_parts.append(f"frefaccdelta{suffix_value(args.frozen_refinement_acceptance_min_delta)}")
        if args.frozen_refinement_observation_policy != "auto":
            suffix_parts.append(f"frefobspol{args.frozen_refinement_observation_policy}")
        if args.frozen_refinement_observation_selection_max_eval is not None:
            suffix_parts.append(
                f"frefobsmaxeval{max(0, args.frozen_refinement_observation_selection_max_eval)}"
            )
        if args.cross_boundary_refinement != "auto":
            suffix_parts.append(f"xbref{args.cross_boundary_refinement}")
        if args.cross_boundary_interval_frames is not None:
            suffix_parts.append(f"xbint{max(0, args.cross_boundary_interval_frames)}")
        if args.cross_boundary_max_submaps_per_frame is not None:
            suffix_parts.append(f"xbmaps{max(0, args.cross_boundary_max_submaps_per_frame)}")
        if args.cross_boundary_min_owner_age_frames is not None:
            suffix_parts.append(f"xbage{max(0, args.cross_boundary_min_owner_age_frames)}")
        if args.cross_boundary_min_neighbor_gaussians is not None:
            suffix_parts.append(f"xbgs{max(0, args.cross_boundary_min_neighbor_gaussians)}")
        if args.cross_boundary_depth_gate != "auto":
            suffix_parts.append(f"xbdgate{1 if args.cross_boundary_depth_gate == 'on' else 0}")
        if args.cross_boundary_min_projected_ratio is not None:
            suffix_parts.append(f"xbproj{suffix_value(max(0.0, args.cross_boundary_min_projected_ratio))}")
        if args.cross_boundary_min_depth_consistent_ratio is not None:
            suffix_parts.append(f"xbcons{suffix_value(max(0.0, args.cross_boundary_min_depth_consistent_ratio))}")
        if args.candidate_new_gs_sample_ratio is not None:
            suffix_parts.append(f"cgsr{suffix_value(max(-1.0, args.candidate_new_gs_sample_ratio))}")
        if args.candidate_max_new_per_frame is not None:
            suffix_parts.append(f"cmax{max(0, args.candidate_max_new_per_frame)}")
        if args.render_owner_min_ready_frames is not None:
            suffix_parts.append(f"rownerready{max(0, args.render_owner_min_ready_frames)}")
        if args.render_neighbor_min_ready_frames is not None:
            suffix_parts.append(f"rneighready{max(0, args.render_neighbor_min_ready_frames)}")
        if args.render_overlap_diagnostics:
            suffix_parts.append("renderdiag")
        if args.disable_render_overlap_diagnostics:
            suffix_parts.append("renderdiag0")
        if args.gaussian_insert_diagnostics == "on":
            suffix_parts.append("insdiag")
        if args.gaussian_submaps_debug != "auto":
            suffix_parts.append(
                f"gdebug{1 if args.gaussian_submaps_debug == 'on' else 0}"
            )
        if args.quality_native_full:
            suffix_parts.append("nativefull")
        if args.ssim_weight is not None:
            suffix_parts.append(f"ssim{suffix_value(args.ssim_weight)}")
        if args.depth_weight is not None:
            suffix_parts.append(f"depthw{suffix_value(args.depth_weight)}")
        if args.localframe_window_length is not None:
            suffix_parts.append(f"lfwin{max(1, args.localframe_window_length)}")
        if args.localframe_window_interval is not None:
            suffix_parts.append(f"lfint{max(1, args.localframe_window_interval)}")
        if args.tag:
            suffix_parts.append(args.tag)
        full_name = f"canonical_{label}_{'_'.join(suffix_parts)}"
        name = safe_experiment_name(full_name)
        cfg = load_base(
            args.frames,
            half,
            name,
            args.output_root,
            args.gaussian_profile,
            args.heldout_interval,
            not args.no_save_after_train,
            input_dir,
            args.scene_name,
            args.start_frame,
            args.frame_step,
        )
        cfg.setdefault("RUNNER", {})["full_experiment_name"] = full_name
        if name != full_name:
            cfg["RUNNER"]["shortened_experiment_name"] = name
            cfg["RUNNER"]["name_hash"] = name.rsplit("_", 1)[-1]
        deep_update(cfg, overrides)
        if args.random_seed is not None:
            cfg["PIPE"]["random_seed"] = int(args.random_seed)
        if args.new_gs_sample_ratio is not None:
            cfg["PIPE"]["new_gs_sample_ratio"] = max(0.0, args.new_gs_sample_ratio)
        if args.local_opt_iters is not None:
            cfg["PIPE"]["local_opt_iters"] = max(0, args.local_opt_iters)
        if args.local_opt_interval is not None:
            cfg["PIPE"]["local_opt_interval"] = max(1, args.local_opt_interval)
        if args.gaussian_schedule != "auto":
            cfg["PIPE"].setdefault("gaussian_schedule", {})["mode"] = args.gaussian_schedule
        if args.gaussian_opt_iters_per_frame is not None:
            cfg["PIPE"].setdefault("gaussian_schedule", {})[
                "iterations_per_frame"
            ] = max(1, args.gaussian_opt_iters_per_frame)
        if args.localframe_window_length is not None:
            cfg["PIPE"]["localframe_cam_window_length"] = max(1, args.localframe_window_length)
        if args.localframe_window_interval is not None:
            cfg["PIPE"]["localframe_cam_window_interval"] = max(1, args.localframe_window_interval)
        if args.ssim_weight is not None:
            cfg["PIPE"].setdefault("weight_configs", {})["ssim_weight"] = args.ssim_weight
        if args.depth_weight is not None:
            cfg["PIPE"].setdefault("weight_configs", {})["depth_weight"] = args.depth_weight
        if args.tsdf_input_mode != "default":
            cfg["PIPE"]["TSDF"]["input_mode"] = args.tsdf_input_mode
        if args.tsdf_input_prefetch != "default":
            cfg["PIPE"]["TSDF"]["input_prefetch"] = args.tsdf_input_prefetch == "on"
        cfg["PIPE"]["TSDF"]["input_tensor_bridge"] = args.tsdf_input_tensor_bridge
        cfg["PIPE"]["TSDF"]["input_tensor_bridge_validate"] = bool(
            args.tsdf_input_bridge_validate
        )
        if args.submap_stats_interval_frames is not None:
            cfg["PIPE"]["submap_stats_interval_frames"] = max(
                0, args.submap_stats_interval_frames
            )
        elif args.tsdf_residency_profile == "thor_cpu_bounded":
            cfg["PIPE"]["submap_stats_interval_frames"] = 10
        if args.gaussian_submaps_debug != "auto":
            cfg["PIPE"].setdefault("gaussian_submaps", {})["debug"] = (
                args.gaussian_submaps_debug == "on"
            )
        if args.color_error_thres is not None:
            cfg["PIPE"]["color_error_thres"] = max(0.0, args.color_error_thres)
        if args.model_default_opacities is not None:
            cfg["MODEL"]["default_opacities"] = max(1e-6, min(1.0 - 1e-6, args.model_default_opacities))
        if args.model_min_init_scale is not None:
            cfg["MODEL"]["min_init_scale"] = max(1e-6, args.model_min_init_scale)
        if args.model_max_init_scale is not None:
            cfg["MODEL"]["max_init_scale"] = max(1e-6, args.model_max_init_scale)
        if args.model_delta_depth is not None:
            cfg["MODEL"]["delta_depth"] = max(0.0, args.model_delta_depth)
        if args.model_sh_degree is not None:
            cfg["MODEL"]["sh_degree"] = max(0, args.model_sh_degree)
        if args.ges_residual_scale is not None:
            cfg["MODEL"]["ges_residual_scale"] = max(0.0, args.ges_residual_scale)
        if args.ges_base_color_weight is not None:
            cfg["MODEL"]["ges_base_color_weight"] = max(1e-6, args.ges_base_color_weight)
        if label.startswith("multi_"):
            if args.multi_use_gt_pose:
                cfg["PIPE"]["TSDF"]["use_gt_pose"] = True
            multi_cfg = cfg["PIPE"]["TSDF"].setdefault("multi", {})
            if args.multi_hash_profile == "default":
                multi_cfg.pop("hash_profile", None)
            if args.multi_max_frames is not None:
                multi_cfg["max_frames"] = args.multi_max_frames
            if args.multi_overlap_frames is not None:
                multi_cfg["overlap_frames"] = max(0, args.multi_overlap_frames)
            if args.multi_trial_frames is not None:
                multi_cfg["trial_frames"] = max(0, args.multi_trial_frames)
            if args.multi_sampled_original_blocks is not None:
                multi_cfg["sampled_original_blocks"] = max(1, args.multi_sampled_original_blocks)
            if args.multi_new_map_visible_ratio is not None:
                multi_cfg["new_map_visible_ratio"] = max(0.0, min(1.0, args.multi_new_map_visible_ratio))
            if args.multi_new_map_occupancy is not None:
                multi_cfg["new_map_block_occupancy"] = max(0.0, min(1.0, args.multi_new_map_occupancy))
            if args.multi_max_occupancy is not None:
                multi_cfg["max_block_occupancy"] = args.multi_max_occupancy
            if args.multi_hash_profile != "default":
                multi_cfg["hash_profile"] = args.multi_hash_profile
            if args.multi_occupancy_growth_lookahead_frames is not None:
                multi_cfg["occupancy_growth_lookahead_frames"] = max(0, args.multi_occupancy_growth_lookahead_frames)
            if args.multi_occupancy_growth_min_primary_frames is not None:
                multi_cfg["occupancy_growth_min_primary_frames"] = max(0, args.multi_occupancy_growth_min_primary_frames)
            if args.multi_occupancy_growth_sample_frames is not None:
                multi_cfg["occupancy_growth_sample_frames"] = max(1, args.multi_occupancy_growth_sample_frames)
            deep_update(multi_cfg, tsdf_residency_profile_defaults(args.tsdf_residency_profile))
            if args.multi_paging != "default":
                multi_cfg["paging"] = args.multi_paging == "on"
            if args.tsdf_paging_storage is not None:
                multi_cfg["paging_storage"] = args.tsdf_paging_storage
            if args.tsdf_async_page_out != "default":
                multi_cfg["async_page_out"] = args.tsdf_async_page_out == "on"
            if args.tsdf_max_pending_page_snapshots is not None:
                multi_cfg["max_pending_page_snapshots"] = max(0, args.tsdf_max_pending_page_snapshots)
            if args.tsdf_max_pending_page_snapshot_mb is not None:
                multi_cfg["max_pending_page_snapshot_mb"] = max(0, args.tsdf_max_pending_page_snapshot_mb)
            if args.resident_gpu_maps:
                try:
                    multi_cfg["resident_gpu_maps"] = int(args.resident_gpu_maps)
                except ValueError:
                    multi_cfg["resident_gpu_maps"] = args.resident_gpu_maps
            if args.tsdf_render_prefetch:
                multi_cfg["render_prefetch"] = True
            if args.tsdf_render_prefetch_lookahead_frames is not None:
                multi_cfg["render_prefetch_lookahead_frames"] = max(0, args.tsdf_render_prefetch_lookahead_frames)
            if args.tsdf_render_prefetch_lookahead_max_maps is not None:
                multi_cfg["render_prefetch_lookahead_max_maps"] = max(0, args.tsdf_render_prefetch_lookahead_max_maps)
            if args.tsdf_render_pin_primary != "default":
                multi_cfg["render_pin_primary_submap"] = args.tsdf_render_pin_primary == "on"
            if args.tsdf_async_page_in:
                multi_cfg["async_page_in"] = True
            if args.tsdf_max_pending_page_ins is not None:
                multi_cfg["max_pending_page_ins"] = max(0, args.tsdf_max_pending_page_ins)
            if args.tsdf_max_pending_page_in_mb is not None:
                multi_cfg["max_pending_page_in_mb"] = max(0, args.tsdf_max_pending_page_in_mb)
            if args.tsdf_render_compositor is not None:
                multi_cfg["render_compositor"] = args.tsdf_render_compositor
            if args.sequential_transform_max_translation is not None:
                multi_cfg["sequential_transform_max_translation"] = max(0.0, args.sequential_transform_max_translation)
            if args.sequential_transform_max_rotation_deg is not None:
                multi_cfg["sequential_transform_max_rotation_deg"] = max(0.0, args.sequential_transform_max_rotation_deg)
            if args.invalid_transform_restart_attempts is not None:
                multi_cfg["invalid_transform_restart_attempts"] = max(0, args.invalid_transform_restart_attempts)
            if args.new_map_handoff_policy is not None:
                multi_cfg["new_map_handoff_policy"] = args.new_map_handoff_policy
            if args.new_map_handoff_min_visible_ratio is not None:
                multi_cfg["new_map_handoff_min_visible_ratio"] = max(0.0, min(1.0, args.new_map_handoff_min_visible_ratio))
            if args.new_map_handoff_min_visible_margin is not None:
                multi_cfg["new_map_handoff_min_visible_margin"] = max(0.0, min(1.0, args.new_map_handoff_min_visible_margin))
            if args.new_map_handoff_min_accepts is not None:
                multi_cfg["new_map_handoff_min_accepts"] = max(0, args.new_map_handoff_min_accepts)
            if args.new_map_handoff_emergency_min_accepts is not None:
                multi_cfg["new_map_handoff_emergency_min_accepts"] = max(-1, args.new_map_handoff_emergency_min_accepts)
            if args.multi_force_single:
                multi_cfg["force_single_submap"] = True
        if label == "basic_unified_gaussian" and args.basic_unified_use_estimated_pose:
            cfg["PIPE"]["TSDF"]["use_gt_pose"] = False
        if args.keyframe_select_max is not None and ("hybrid" in label or label == "basic_unified_gaussian"):
            cfg["PIPE"]["keyframe_select_max"] = max(0, args.keyframe_select_max)
        if "hybrid" in label:
            if args.hybrid_use_estimated_pose:
                cfg["PIPE"]["TSDF"]["use_gt_pose"] = False
            gs_cfg = cfg["PIPE"].setdefault("gaussian_submaps", {})
            if args.gaussian_render_policy:
                gs_cfg["render_policy"] = args.gaussian_render_policy
            if args.gaussian_render_selection_policy != "auto":
                gs_cfg["render_selection_policy"] = args.gaussian_render_selection_policy
            if args.render_max_visible_submaps is not None:
                gs_cfg["render_max_visible_submaps"] = max(0, args.render_max_visible_submaps)
            if args.render_max_visible_gaussians is not None:
                gs_cfg["render_max_visible_gaussians"] = max(0, args.render_max_visible_gaussians)
            if args.render_bounds_inflate is not None:
                gs_cfg["render_bounds_inflate"] = max(0.0, args.render_bounds_inflate)
            if args.render_bounds_min_projected_area is not None:
                gs_cfg["render_bounds_min_projected_area"] = max(0.0, args.render_bounds_min_projected_area)
            if args.render_selection_hysteresis_frames is not None:
                gs_cfg["render_selection_hysteresis_frames"] = max(0, args.render_selection_hysteresis_frames)
            if args.render_selection_hysteresis_area_bonus is not None:
                gs_cfg["render_selection_hysteresis_area_bonus"] = max(0.0, args.render_selection_hysteresis_area_bonus)
            if args.render_selection_continuity_frame_scale is not None:
                gs_cfg["render_selection_continuity_frame_scale"] = max(1, args.render_selection_continuity_frame_scale)
            if args.render_selection_continuity_min_factor is not None:
                gs_cfg["render_selection_continuity_min_factor"] = min(
                    1.0, max(0.0, args.render_selection_continuity_min_factor)
                )
            if args.render_selection_adjacent_bonus is not None:
                gs_cfg["render_selection_adjacent_bonus"] = max(0.0, args.render_selection_adjacent_bonus)
            if args.render_selection_active_overlap_bonus is not None:
                gs_cfg["render_selection_active_overlap_bonus"] = max(0.0, args.render_selection_active_overlap_bonus)
            if args.cross_submap_novelty_gate != "auto":
                gs_cfg["cross_submap_novelty_gate"] = args.cross_submap_novelty_gate == "on"
            if args.cross_novelty_reference_policy != "auto":
                gs_cfg["cross_novelty_reference_policy"] = args.cross_novelty_reference_policy
            if args.cross_novelty_depth_tolerance_mode != "auto":
                gs_cfg["cross_novelty_depth_tolerance_mode"] = args.cross_novelty_depth_tolerance_mode
            if args.cross_novelty_alpha_thres is not None:
                gs_cfg["cross_novelty_alpha_thres"] = max(0.0, args.cross_novelty_alpha_thres)
            if args.cross_novelty_depth_abs_thres is not None:
                gs_cfg["cross_novelty_depth_abs_thres"] = max(0.0, args.cross_novelty_depth_abs_thres)
            if args.cross_novelty_depth_rel_thres is not None:
                gs_cfg["cross_novelty_depth_rel_thres"] = max(0.0, args.cross_novelty_depth_rel_thres)
            if args.cross_novelty_depth_floor is not None:
                gs_cfg["cross_novelty_depth_floor"] = max(0.0, args.cross_novelty_depth_floor)
            if args.cross_novelty_depth_voxel_scale is not None:
                gs_cfg["cross_novelty_depth_voxel_scale"] = max(0.0, args.cross_novelty_depth_voxel_scale)
            if args.cross_novelty_depth_quadratic_scale is not None:
                gs_cfg["cross_novelty_depth_quadratic_scale"] = max(0.0, args.cross_novelty_depth_quadratic_scale)
            if args.cross_novelty_photometric_thres is not None:
                gs_cfg["cross_novelty_photometric_thres"] = max(0.0, args.cross_novelty_photometric_thres)
            if args.cross_novelty_max_reference_submaps is not None:
                gs_cfg["cross_novelty_max_reference_submaps"] = max(0, args.cross_novelty_max_reference_submaps)
            if args.cross_novelty_previous_only != "auto":
                gs_cfg["cross_novelty_previous_only"] = args.cross_novelty_previous_only == "on"
            if args.cross_novelty_resident_only != "auto":
                gs_cfg["cross_novelty_resident_only"] = args.cross_novelty_resident_only == "on"
            if args.cross_novelty_erode_kernel is not None:
                gs_cfg["cross_novelty_erode_kernel"] = max(0, args.cross_novelty_erode_kernel)
            if args.resident_gaussian_submaps is not None:
                gs_cfg["resident_gaussian_submaps"] = max(1, args.resident_gaussian_submaps)
            if args.gaussian_max_new_per_frame is not None:
                gs_cfg["max_new_gaussians_per_frame"] = max(0, args.gaussian_max_new_per_frame)
            if args.gaussian_bootstrap_max_new_per_frame is not None:
                gs_cfg["bootstrap_max_new_gaussians_per_frame"] = max(0, args.gaussian_bootstrap_max_new_per_frame)
            if args.gaussian_bootstrap_until_gaussians is not None:
                gs_cfg["bootstrap_until_gaussians"] = max(0, args.gaussian_bootstrap_until_gaussians)
            if args.gaussian_bootstrap_until_age_frames is not None:
                gs_cfg["bootstrap_until_age_frames"] = max(0, args.gaussian_bootstrap_until_age_frames)
            if args.gaussian_budget_policy != "auto":
                gs_cfg["gaussian_budget_policy"] = args.gaussian_budget_policy
            if args.gaussian_budget_base is not None:
                gs_cfg["gaussian_budget_base"] = max(0, args.gaussian_budget_base)
            if args.gaussian_budget_blocks_density is not None:
                gs_cfg["gaussian_budget_blocks_density"] = max(0.0, args.gaussian_budget_blocks_density)
            if args.gaussian_budget_min is not None:
                gs_cfg["gaussian_budget_min"] = max(0, args.gaussian_budget_min)
            if args.gaussian_budget_max is not None:
                gs_cfg["gaussian_budget_max"] = max(0, args.gaussian_budget_max)
            if args.gaussian_budget_pressure_ratio is not None:
                gs_cfg["gaussian_budget_pressure_ratio"] = max(0.0, args.gaussian_budget_pressure_ratio)
            if args.gaussian_budget_prune_interval_frames is not None:
                gs_cfg["gaussian_budget_prune_interval_frames"] = max(0, args.gaussian_budget_prune_interval_frames)
            if args.gaussian_insert_diagnostics != "auto":
                gs_cfg["gaussian_insert_diagnostics"] = args.gaussian_insert_diagnostics == "on"
            if args.gaussian_insert_selection != "auto":
                gs_cfg["gaussian_insert_selection"] = args.gaussian_insert_selection
            if args.gaussian_insert_tile_size is not None:
                gs_cfg["gaussian_insert_tile_size"] = max(1, args.gaussian_insert_tile_size)
            if args.gaussian_insert_tile_max_pixels is not None:
                gs_cfg["gaussian_insert_tile_max_pixels"] = max(0, args.gaussian_insert_tile_max_pixels)
            if args.gaussian_insert_adaptive_tile_min_candidate_pixels is not None:
                gs_cfg["gaussian_insert_adaptive_tile_min_candidate_pixels"] = max(
                    0, args.gaussian_insert_adaptive_tile_min_candidate_pixels
                )
            if args.gaussian_insert_adaptive_tile_min_target_insertions is not None:
                gs_cfg["gaussian_insert_adaptive_tile_min_target_insertions"] = max(
                    0, args.gaussian_insert_adaptive_tile_min_target_insertions
                )
            if args.gaussian_insert_adaptive_tile_min_submap_gaussians is not None:
                gs_cfg["gaussian_insert_adaptive_tile_min_submap_gaussians"] = max(
                    0, args.gaussian_insert_adaptive_tile_min_submap_gaussians
                )
            if args.gaussian_insert_adaptive_tile_min_observed_frames is not None:
                gs_cfg["gaussian_insert_adaptive_tile_min_observed_frames"] = max(
                    0, args.gaussian_insert_adaptive_tile_min_observed_frames
                )
            if args.submap_keyframe_selection_policy != "auto":
                gs_cfg["keyframe_selection_policy"] = args.submap_keyframe_selection_policy
            if args.graduation_readiness_enable != "auto":
                gs_cfg["graduation_readiness_enable"] = args.graduation_readiness_enable == "on"
            if args.graduation_readiness_sample_cameras is not None:
                gs_cfg["graduation_readiness_sample_cameras"] = max(1, args.graduation_readiness_sample_cameras)
            if args.graduation_readiness_skip_gain is not None:
                gs_cfg["graduation_readiness_skip_gain"] = max(0.0, args.graduation_readiness_skip_gain)
            if args.graduation_readiness_min_abs_drop is not None:
                gs_cfg["graduation_readiness_min_abs_drop"] = max(0.0, args.graduation_readiness_min_abs_drop)
            if args.graduation_readiness_skip_min_observed_frames is not None:
                gs_cfg["graduation_readiness_skip_min_observed_frames"] = max(
                    0, args.graduation_readiness_skip_min_observed_frames
                )
            if args.graduation_readiness_skip_min_keyframes is not None:
                gs_cfg["graduation_readiness_skip_min_keyframes"] = max(
                    0, args.graduation_readiness_skip_min_keyframes
                )
            if args.quality_finalize_enable != "auto":
                gs_cfg["quality_finalize_enable"] = args.quality_finalize_enable == "on"
            if args.quality_finalize_resident_only != "auto":
                gs_cfg["quality_finalize_resident_only"] = args.quality_finalize_resident_only == "on"
            if args.quality_finalize_max_submaps is not None:
                gs_cfg["quality_finalize_max_submaps"] = max(0, args.quality_finalize_max_submaps)
            if args.quality_finalize_max_observed_frames is not None:
                gs_cfg["quality_finalize_max_observed_frames"] = max(0, args.quality_finalize_max_observed_frames)
            if args.quality_finalize_min_keyframes is not None:
                gs_cfg["quality_finalize_min_keyframes"] = max(0, args.quality_finalize_min_keyframes)
            if args.quality_finalize_max_hybrid_l1 is not None:
                gs_cfg["quality_finalize_max_hybrid_l1"] = max(0.0, args.quality_finalize_max_hybrid_l1)
            if args.quality_finalize_insert_enable != "auto":
                gs_cfg["quality_finalize_insert_enable"] = args.quality_finalize_insert_enable == "on"
            if args.quality_finalize_insert_include_frozen != "auto":
                gs_cfg["quality_finalize_insert_include_frozen"] = (
                    args.quality_finalize_insert_include_frozen == "on"
                )
            if args.quality_finalize_insert_min_sdf_submaps is not None:
                gs_cfg["quality_finalize_insert_min_sdf_submaps"] = max(
                    0, args.quality_finalize_insert_min_sdf_submaps
                )
            if args.quality_finalize_insert_frames is not None:
                gs_cfg["quality_finalize_insert_frames"] = max(0, args.quality_finalize_insert_frames)
            if args.quality_finalize_insert_max_new_per_frame is not None:
                gs_cfg["quality_finalize_insert_max_new_gaussians_per_frame"] = max(
                    0, args.quality_finalize_insert_max_new_per_frame
                )
            if args.quality_finalize_insert_acceptance != "auto":
                gs_cfg["quality_finalize_insert_acceptance_enable"] = (
                    args.quality_finalize_insert_acceptance == "on"
                )
            if args.quality_finalize_insert_acceptance_cameras is not None:
                gs_cfg["quality_finalize_insert_acceptance_cameras"] = max(
                    0, args.quality_finalize_insert_acceptance_cameras
                )
            if args.quality_finalize_insert_min_readiness_delta is not None:
                gs_cfg["quality_finalize_insert_min_readiness_delta"] = max(
                    0.0, args.quality_finalize_insert_min_readiness_delta
                )
            if args.gaussian_insert_min_age_frames is not None:
                gs_cfg["gaussian_insert_min_age_frames"] = max(0, args.gaussian_insert_min_age_frames)
            if args.gaussian_insert_min_sdf_blocks is not None:
                gs_cfg["gaussian_insert_min_sdf_blocks"] = max(0, args.gaussian_insert_min_sdf_blocks)
            if args.candidate_reject_retire != "auto":
                gs_cfg["candidate_reject_retire"] = args.candidate_reject_retire == "on"
            if args.candidate_reject_retire_min_rejects is not None:
                gs_cfg["candidate_reject_retire_min_rejects"] = max(1, args.candidate_reject_retire_min_rejects)
            if args.candidate_reject_retire_min_observed_frames is not None:
                gs_cfg["candidate_reject_retire_min_observed_frames"] = max(0, args.candidate_reject_retire_min_observed_frames)
            if args.candidate_reject_retire_min_gaussians is not None:
                gs_cfg["candidate_reject_retire_min_gaussians"] = max(0, args.candidate_reject_retire_min_gaussians)
            if args.candidate_reject_retire_max_inliers is not None:
                gs_cfg["candidate_reject_retire_max_inliers"] = max(0, args.candidate_reject_retire_max_inliers)
            if args.candidate_reject_retire_cooldown_frames is not None:
                gs_cfg["candidate_reject_retire_cooldown_frames"] = max(0, args.candidate_reject_retire_cooldown_frames)
            if args.candidate_gaussian_insert_min_frames is not None:
                gs_cfg["candidate_gaussian_insert_min_frames"] = max(0, args.candidate_gaussian_insert_min_frames)
            if args.candidate_immature_new_gs_sample_ratio is not None:
                gs_cfg["candidate_immature_new_gs_sample_ratio"] = max(-1.0, args.candidate_immature_new_gs_sample_ratio)
            if args.candidate_immature_max_new_per_frame is not None:
                gs_cfg["candidate_immature_max_new_gaussians_per_frame"] = max(0, args.candidate_immature_max_new_per_frame)
            if args.candidate_extend_while_overlap != "auto":
                gs_cfg["candidate_extend_while_overlap"] = args.candidate_extend_while_overlap == "on"
            if args.candidate_max_warm_frames is not None:
                gs_cfg["candidate_max_warm_frames"] = max(0, args.candidate_max_warm_frames)
            if args.candidate_handoff_max_opt_passes is not None:
                gs_cfg["candidate_handoff_max_opt_passes"] = max(-1, args.candidate_handoff_max_opt_passes)
            if args.candidate_handoff_opt_iters is not None:
                gs_cfg["candidate_handoff_opt_iters"] = max(-1, args.candidate_handoff_opt_iters)
            if args.handoff_frames is not None:
                gs_cfg["handoff_frames"] = max(0, args.handoff_frames)
            if args.handoff_opt_iters is not None:
                gs_cfg["handoff_opt_iters"] = max(0, args.handoff_opt_iters)
            if args.handoff_max_opt_passes is not None:
                gs_cfg["handoff_max_opt_passes"] = max(0, args.handoff_max_opt_passes)
            if args.handoff_add_gaussians_to_outgoing != "auto":
                gs_cfg["handoff_add_gaussians_to_outgoing"] = args.handoff_add_gaussians_to_outgoing == "on"
            if args.graduation_optimize != "auto":
                gs_cfg["graduation_optimize"] = args.graduation_optimize == "on"
            if args.graduation_opt_iters is not None:
                gs_cfg["graduation_opt_iters"] = max(0, args.graduation_opt_iters)
            if args.graduation_keyframes is not None:
                gs_cfg["graduation_keyframes"] = max(0, args.graduation_keyframes)
            if args.graduation_max_opt_passes is not None:
                gs_cfg["graduation_max_opt_passes"] = max(0, args.graduation_max_opt_passes)
            if args.graduation_keyframe_policy != "auto":
                gs_cfg["graduation_keyframe_policy"] = args.graduation_keyframe_policy
            if args.graduation_observation_policy != "auto":
                gs_cfg["graduation_observation_policy"] = args.graduation_observation_policy
            if args.graduation_adaptive != "auto":
                gs_cfg["graduation_adaptive"] = args.graduation_adaptive == "on"
            if args.graduation_adaptive_loss_sample_cameras is not None:
                gs_cfg["graduation_adaptive_loss_sample_cameras"] = max(1, args.graduation_adaptive_loss_sample_cameras)
            if args.graduation_adaptive_target_loss is not None:
                gs_cfg["graduation_adaptive_target_loss"] = max(0.0, args.graduation_adaptive_target_loss)
            if args.graduation_adaptive_min_abs_drop is not None:
                gs_cfg["graduation_adaptive_min_abs_drop"] = max(0.0, args.graduation_adaptive_min_abs_drop)
            if args.graduation_adaptive_min_rel_drop is not None:
                gs_cfg["graduation_adaptive_min_rel_drop"] = max(0.0, args.graduation_adaptive_min_rel_drop)
            if args.training_observation_max_per_submap is not None:
                gs_cfg["training_observation_max_per_submap"] = max(0, args.training_observation_max_per_submap)
            if args.training_observation_interval_frames is not None:
                gs_cfg["training_observation_interval_frames"] = max(1, args.training_observation_interval_frames)
            if args.frozen_refinement_enable != "auto":
                gs_cfg["frozen_refinement_enable"] = args.frozen_refinement_enable == "on"
            if args.frozen_refinement_interval_frames is not None:
                gs_cfg["frozen_refinement_interval_frames"] = max(1, args.frozen_refinement_interval_frames)
            if args.frozen_refinement_opt_iters is not None:
                gs_cfg["frozen_refinement_opt_iters"] = max(0, args.frozen_refinement_opt_iters)
            if args.frozen_refinement_keyframes is not None:
                gs_cfg["frozen_refinement_keyframes"] = max(0, args.frozen_refinement_keyframes)
            if args.frozen_refinement_max_passes_per_submap is not None:
                gs_cfg["frozen_refinement_max_passes_per_submap"] = max(0, args.frozen_refinement_max_passes_per_submap)
            if args.frozen_refinement_max_submaps_per_frame is not None:
                gs_cfg["frozen_refinement_max_submaps_per_frame"] = max(0, args.frozen_refinement_max_submaps_per_frame)
            if args.frozen_refinement_min_new_observations is not None:
                gs_cfg["frozen_refinement_min_new_observations"] = max(1, args.frozen_refinement_min_new_observations)
            if args.frozen_refinement_readiness_guard != "auto":
                gs_cfg["frozen_refinement_readiness_guard_enable"] = (
                    args.frozen_refinement_readiness_guard == "on"
                )
            if args.frozen_refinement_readiness_use_cached != "auto":
                gs_cfg["frozen_refinement_readiness_use_cached_graduation"] = (
                    args.frozen_refinement_readiness_use_cached == "on"
                )
            if args.frozen_refinement_readiness_sample_cameras is not None:
                gs_cfg["frozen_refinement_readiness_sample_cameras"] = max(
                    1, args.frozen_refinement_readiness_sample_cameras
                )
            if args.frozen_refinement_readiness_max_pre_gain is not None:
                gs_cfg["frozen_refinement_readiness_max_pre_gain"] = args.frozen_refinement_readiness_max_pre_gain
            if args.frozen_refinement_readiness_min_valid_cameras is not None:
                gs_cfg["frozen_refinement_readiness_min_valid_cameras"] = max(
                    1, args.frozen_refinement_readiness_min_valid_cameras
                )
            if args.frozen_refinement_acceptance != "auto":
                gs_cfg["frozen_refinement_acceptance_enable"] = (
                    args.frozen_refinement_acceptance == "on"
                )
            if args.frozen_refinement_acceptance_min_delta is not None:
                gs_cfg["frozen_refinement_acceptance_min_delta"] = (
                    args.frozen_refinement_acceptance_min_delta
                )
            if args.frozen_refinement_observation_policy != "auto":
                gs_cfg["frozen_refinement_observation_policy"] = args.frozen_refinement_observation_policy
            if args.frozen_refinement_observation_selection_max_eval is not None:
                gs_cfg["frozen_refinement_observation_selection_max_eval"] = max(
                    0, args.frozen_refinement_observation_selection_max_eval
                )
            if args.cross_boundary_refinement != "auto":
                gs_cfg["cross_boundary_refinement_enable"] = args.cross_boundary_refinement == "on"
            if args.cross_boundary_interval_frames is not None:
                gs_cfg["cross_boundary_observation_interval_frames"] = max(0, args.cross_boundary_interval_frames)
            if args.cross_boundary_max_submaps_per_frame is not None:
                gs_cfg["cross_boundary_max_submaps_per_frame"] = max(0, args.cross_boundary_max_submaps_per_frame)
            if args.cross_boundary_min_owner_age_frames is not None:
                gs_cfg["cross_boundary_min_owner_age_frames"] = max(0, args.cross_boundary_min_owner_age_frames)
            if args.cross_boundary_min_neighbor_gaussians is not None:
                gs_cfg["cross_boundary_min_neighbor_gaussians"] = max(0, args.cross_boundary_min_neighbor_gaussians)
            if args.cross_boundary_frozen_only != "auto":
                gs_cfg["cross_boundary_frozen_only"] = args.cross_boundary_frozen_only == "on"
            if args.cross_boundary_resident_only != "auto":
                gs_cfg["cross_boundary_resident_only"] = args.cross_boundary_resident_only == "on"
            if args.cross_boundary_sequential_only != "auto":
                gs_cfg["cross_boundary_sequential_only"] = args.cross_boundary_sequential_only == "on"
            if args.cross_boundary_depth_gate != "auto":
                gs_cfg["cross_boundary_depth_gate"] = args.cross_boundary_depth_gate == "on"
            if args.cross_boundary_min_projected_ratio is not None:
                gs_cfg["cross_boundary_min_projected_ratio"] = min(
                    1.0, max(0.0, args.cross_boundary_min_projected_ratio)
                )
            if args.cross_boundary_min_depth_consistent_ratio is not None:
                gs_cfg["cross_boundary_min_depth_consistent_ratio"] = min(
                    1.0, max(0.0, args.cross_boundary_min_depth_consistent_ratio)
                )
            if args.cross_boundary_depth_tolerance_mode != "auto":
                gs_cfg["cross_boundary_depth_tolerance_mode"] = args.cross_boundary_depth_tolerance_mode
            if args.cross_boundary_depth_abs_thres is not None:
                gs_cfg["cross_boundary_depth_abs_thres"] = max(0.0, args.cross_boundary_depth_abs_thres)
            if args.cross_boundary_depth_floor is not None:
                gs_cfg["cross_boundary_depth_floor"] = max(0.0, args.cross_boundary_depth_floor)
            if args.cross_boundary_depth_voxel_scale is not None:
                gs_cfg["cross_boundary_depth_voxel_scale"] = max(0.0, args.cross_boundary_depth_voxel_scale)
            if args.cross_boundary_depth_quadratic_scale is not None:
                gs_cfg["cross_boundary_depth_quadratic_scale"] = max(0.0, args.cross_boundary_depth_quadratic_scale)
            if args.candidate_new_gs_sample_ratio is not None:
                gs_cfg["candidate_new_gs_sample_ratio"] = max(-1.0, args.candidate_new_gs_sample_ratio)
            if args.candidate_max_new_per_frame is not None:
                gs_cfg["candidate_max_new_gaussians_per_frame"] = max(0, args.candidate_max_new_per_frame)
            if args.render_min_ready_frames is not None:
                gs_cfg["render_min_ready_frames"] = max(0, args.render_min_ready_frames)
            if args.render_owner_min_ready_frames is not None:
                gs_cfg["render_owner_min_ready_frames"] = max(0, args.render_owner_min_ready_frames)
            if args.render_neighbor_min_ready_frames is not None:
                gs_cfg["render_neighbor_min_ready_frames"] = max(0, args.render_neighbor_min_ready_frames)
            if args.render_past_frame_margin is not None:
                gs_cfg["render_past_frame_margin"] = max(0, args.render_past_frame_margin)
            if args.render_future_frame_margin is not None:
                gs_cfg["render_future_frame_margin"] = max(0, args.render_future_frame_margin)
            if args.render_min_neighbor_gaussians is not None:
                gs_cfg["render_min_neighbor_gaussians"] = max(0, args.render_min_neighbor_gaussians)
            if args.render_overlap_depth_gate != "auto":
                gs_cfg["render_overlap_depth_gate"] = args.render_overlap_depth_gate == "on"
            if args.render_overlap_depth_mode != "auto":
                gs_cfg["render_overlap_depth_tolerance_mode"] = args.render_overlap_depth_mode
            if args.render_overlap_depth_thres is not None:
                gs_cfg["render_overlap_depth_abs_thres"] = max(0.0, args.render_overlap_depth_thres)
            if args.render_overlap_alpha_thres is not None:
                gs_cfg["render_overlap_alpha_thres"] = max(0.0, args.render_overlap_alpha_thres)
            if args.render_hole_requires_boundary:
                gs_cfg["render_hole_requires_boundary"] = True
            if args.render_include_handoff_submaps:
                gs_cfg["render_include_handoff_submaps"] = True
            if args.disable_single_submap_passthrough:
                gs_cfg["render_single_submap_passthrough"] = False
            if args.allow_multi_logical_single_passthrough:
                gs_cfg["render_single_submap_passthrough_requires_single_logical_map"] = False
            if args.require_single_logical_single_passthrough:
                gs_cfg["render_single_submap_passthrough_requires_single_logical_map"] = True
            if args.disable_local_sdf_reference_render:
                gs_cfg["render_use_local_sdf_reference"] = False
            if args.render_overlap_diagnostics:
                gs_cfg["render_overlap_diagnostics"] = True
            if args.disable_render_overlap_diagnostics:
                gs_cfg["render_overlap_diagnostics"] = False
            if args.pose_graph_identity_check:
                gs_cfg["pose_graph_identity_check"] = True
            if args.pose_graph_synthetic_check:
                gs_cfg["pose_graph_synthetic_check"] = True
            if args.pose_graph_render_identity_check:
                gs_cfg["pose_graph_render_identity_check"] = True
            if args.pose_graph_render_synthetic_check:
                gs_cfg["pose_graph_render_synthetic_check"] = True
            if args.pose_graph_dry_run or args.pose_graph_solve_only or args.pose_graph_candidates:
                pg_cfg = cfg["PIPE"].setdefault("pose_graph", {})
                if args.pose_graph_dry_run:
                    pg_cfg["dry_run"] = True
                    pg_cfg["dry_run_include_loop_candidates"] = bool(args.pose_graph_dry_run_include_loops)
                    pg_cfg["dry_run_use_estimated_loop_transforms"] = True
                if args.pose_graph_solve_only:
                    pg_cfg["solve_only"] = True
                    pg_cfg["solve_only_include_loop_candidates"] = bool(args.pose_graph_solve_only_include_loops)
                    pg_cfg["solve_only_use_estimated_loop_transforms"] = True
                    if args.pose_graph_solve_only_max_translation_delta is not None:
                        pg_cfg["solve_only_max_translation_delta"] = args.pose_graph_solve_only_max_translation_delta
                    if args.pose_graph_solve_only_max_rotation_delta_deg is not None:
                        pg_cfg["solve_only_max_rotation_delta_deg"] = args.pose_graph_solve_only_max_rotation_delta_deg
                if args.pose_graph_candidates:
                    pg_cfg["candidate_detection"] = True
                    pg_cfg["optimize"] = False
                    if args.pose_graph_loop_max_distance is not None:
                        pg_cfg["loop_max_distance"] = args.pose_graph_loop_max_distance
                    if args.pose_graph_loop_max_rotation_deg is not None:
                        pg_cfg["loop_max_rotation_deg"] = args.pose_graph_loop_max_rotation_deg
                    if args.pose_graph_loop_min_separation is not None:
                        pg_cfg["loop_min_submap_separation"] = args.pose_graph_loop_min_separation
                    if args.pose_graph_loop_max_candidates is not None:
                        pg_cfg["loop_max_candidates"] = args.pose_graph_loop_max_candidates
                    if args.pose_graph_candidate_verification is not None:
                        pg_cfg["candidate_verification"] = args.pose_graph_candidate_verification
                    if args.pose_graph_verify_max_keyframe_distance is not None:
                        pg_cfg["verify_max_keyframe_distance"] = args.pose_graph_verify_max_keyframe_distance
                    if args.pose_graph_verify_max_keyframe_rotation_deg is not None:
                        pg_cfg["verify_max_keyframe_rotation_deg"] = args.pose_graph_verify_max_keyframe_rotation_deg
                    if args.pose_graph_verify_max_depth_error is not None:
                        pg_cfg["verify_max_depth_error"] = args.pose_graph_verify_max_depth_error
                    if args.pose_graph_verify_min_depth_overlap is not None:
                        pg_cfg["verify_min_depth_overlap"] = args.pose_graph_verify_min_depth_overlap
                    if args.pose_graph_verify_depth_sample_stride is not None:
                        pg_cfg["verify_depth_sample_stride"] = args.pose_graph_verify_depth_sample_stride
                    if args.pose_graph_verify_icp_min_inliers is not None:
                        pg_cfg["verify_icp_min_inliers"] = args.pose_graph_verify_icp_min_inliers
                    if args.pose_graph_verify_icp_max_rmse is not None:
                        pg_cfg["verify_icp_max_rmse"] = args.pose_graph_verify_icp_max_rmse
                    if args.pose_graph_verify_icp_max_translation_delta is not None:
                        pg_cfg["verify_icp_max_translation_delta"] = args.pose_graph_verify_icp_max_translation_delta
                    if args.pose_graph_verify_icp_max_rotation_delta_deg is not None:
                        pg_cfg["verify_icp_max_rotation_delta_deg"] = args.pose_graph_verify_icp_max_rotation_delta_deg

        apply_reader_preload_policy(cfg, args.reader_preload_images)
        benchmark_role = benchmark_roles.get(label)
        if benchmark_role is not None:
            try:
                apply_protocol(
                    cfg,
                    args.benchmark_protocol,
                    benchmark_role,
                    binary_hash_profile,
                )
            except ValueError as exc:
                parser.error(str(exc))
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg_path.write_text(yaml.dump(cfg, sort_keys=False))
        if args.thor_preflight:
            preflight = run_cmd(
                ["python3", str(PROJECT_ROOT / "scripts" / "thor_preflight.py"), str(cfg_path)],
                PROJECT_ROOT,
            )
            if preflight.stdout:
                print(preflight.stdout, end="")
            if preflight.returncode != 0:
                if preflight.stderr:
                    print(preflight.stderr, end="", file=sys.stderr)
                raise RuntimeError(f"Thor preflight blocked generated config: {cfg_path}")
        out_dir = Path(cfg["workspace_dir"])
        cfg_summary = config_fingerprint(cfg)
        if args.only_generate:
            if benchmark_role is not None:
                write_manifest(
                    cfg_path.with_suffix(".benchmark.json"),
                    protocol=args.benchmark_protocol,
                    role=benchmark_role,
                    cfg=cfg,
                    config_path=cfg_path,
                    binary=args.binary,
                    dataset_root=input_dir,
                    dataset_hash_mode=args.benchmark_dataset_hash,
                    extra={"state": "generated", "label": label},
                )
            rows.append({
                "name": name,
                "status": "GENERATED",
                "scene_name": args.scene_name,
                "frames": args.frames,
                "resolution": "half" if half else "full",
                "gaussian_profile": args.gaussian_profile,
                "hybrid_submap_profile": args.hybrid_submap_profile,
                "multi_hash_profile": args.multi_hash_profile,
                "tsdf_residency_profile": args.tsdf_residency_profile,
                "binary": str(args.binary),
                "binary_hash_profile": binary_hash_profile,
                "baseline_hash_warning": non_legacy_basic_warning and label in basic_labels,
                "benchmark_protocol": args.benchmark_protocol,
                "benchmark_role": benchmark_role or "",
                "input_dir": str(input_dir),
                "gt_dir": str(gt_dir),
                **cfg_summary,
                "config": str(cfg_path),
            })
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{name}.log"
        timed_out = False
        try:
            proc = run_cmd(
                [str(args.binary), str(cfg_path)],
                PROJECT_ROOT,
                log_path=log_path if args.quiet_run_logs else None,
                timeout_sec=args.trainer_timeout_sec if args.trainer_timeout_sec > 0.0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            timeout_output = exc.stdout or ""
            timeout_error = exc.stderr or ""
            if not args.quiet_run_logs:
                log_path.write_text(
                    f"{timeout_output}\n--- STDERR ---\n{timeout_error}\n"
                    f"[TIMEOUT] exceeded {args.trainer_timeout_sec:g} seconds\n"
                )
            else:
                with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                    log_file.write(f"\n[TIMEOUT] exceeded {args.trainer_timeout_sec:g} seconds\n")
            proc = subprocess.CompletedProcess(
                [str(args.binary), str(cfg_path)], 124, timeout_output, timeout_error
            )
        if not args.quiet_run_logs:
            if not timed_out:
                log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        status = "TIMEOUT" if timed_out else ("OK" if proc.returncode == 0 else "CRASH")
        if benchmark_role is not None:
            manifest_dir = out_dir if out_dir.exists() else cfg_path.parent
            write_manifest(
                manifest_dir / "benchmark_manifest.json",
                protocol=args.benchmark_protocol,
                role=benchmark_role,
                cfg=cfg,
                config_path=cfg_path,
                binary=args.binary,
                dataset_root=input_dir,
                dataset_hash_mode=args.benchmark_dataset_hash,
                extra={
                    "state": "completed" if status == "OK" else "failed",
                    "status": status,
                    "return_code": proc.returncode,
                    "label": label,
                },
            )
        stats = parse_time_log(out_dir / "time_log.txt")
        timing_stats = summarize_timing(out_dir / "timing.csv")
        paging_stats = summarize_paging(out_dir / "paging.csv")
        submap_diag = summarize_submap_diagnostics(out_dir)
        render_debug_stats = summarize_render_debug(out_dir / "val" / "submap_render_debug.csv")
        page_stats = summarize_tsdf_pages(out_dir)
        pose_finiteness = scan_pose_finiteness(out_dir)
        rows.append({
            "name": name,
            "status": status,
            "return_code": proc.returncode,
            "scene_name": args.scene_name,
            "frames": args.frames,
            "resolution": "half" if half else "full",
            "gaussian_profile": args.gaussian_profile,
            "hybrid_submap_profile": args.hybrid_submap_profile,
            "multi_hash_profile": args.multi_hash_profile,
            "tsdf_residency_profile": args.tsdf_residency_profile,
            "binary": str(args.binary),
            "binary_hash_profile": binary_hash_profile,
            "baseline_hash_warning": non_legacy_basic_warning and label in basic_labels,
            "benchmark_protocol": args.benchmark_protocol,
            "benchmark_role": benchmark_role or "",
            "input_dir": str(input_dir),
            "gt_dir": str(gt_dir),
            **cfg_summary,
            "ate_actual_rmse_cm": evaluate_ate(out_dir, gt_dir) if status == "OK" else None,
            **pose_finiteness,
            "fps": stats.get("FPS"),
            "fusion_ms": stats.get("per_frame_fusion_time"),
            "local_opt_ms": stats.get("localOptimize_time"),
            "handoff_ms": stats.get("handoffOptimize_time"),
            "gaussian_ops_ms": stats.get("gaussianOps_time"),
            "gs_num": stats.get("GS_num"),
            **timing_stats,
            **paging_stats,
            **submap_diag,
            **render_debug_stats,
            **page_stats,
            **summarize_runtime_resources(out_dir),
            "config": str(cfg_path),
            "log": str(log_path),
        })
        quality_supported = "hybrid" in label or label == "basic_unified_gaussian"
        if args.quality and status == "OK" and quality_supported:
            rows[-1].update(run_quality_eval(
                args.binary,
                cfg,
                cfg_path,
                out_dir,
                gt_dir,
                name,
                log_root,
                quality_eval_split,
                args.quiet_run_logs,
            ))
            rows[-1].update(summarize_render_debug(out_dir / "val" / "submap_render_debug.csv"))
            if mixed_pose_quality_warning and "hybrid" in label:
                rows[-1]["quality_pose_protocol_warning"] = "estimated_train_poses_with_gt_heldout_eval_poses"
            if "hybrid_PSNR" in rows[-1] and "sdf_PSNR" in rows[-1]:
                rows[-1]["delta_PSNR_hybrid_minus_sdf"] = rows[-1]["hybrid_PSNR"] - rows[-1]["sdf_PSNR"]
            if "hybrid_SSIM" in rows[-1] and "sdf_SSIM" in rows[-1]:
                rows[-1]["delta_SSIM_hybrid_minus_sdf"] = rows[-1]["hybrid_SSIM"] - rows[-1]["sdf_SSIM"]
            if "hybrid_LPIPS" in rows[-1] and "sdf_LPIPS" in rows[-1]:
                rows[-1]["delta_LPIPS_hybrid_minus_sdf"] = rows[-1]["hybrid_LPIPS"] - rows[-1]["sdf_LPIPS"]
            psnr_pass = rows[-1].get("delta_PSNR_hybrid_minus_sdf", -999) >= 0.2
            lpips_pass = rows[-1].get("delta_LPIPS_hybrid_minus_sdf", 999) <= -0.005
            rows[-1]["quality_gate"] = "PASS" if psnr_pass or lpips_pass else "FAIL"
        if args.quality_native_full and status == "OK" and quality_supported:
            native_full_dir = resolve_replica_input_dir(args.scene_name, False, None)

            def apply_native_full_quality_overrides(eval_cfg: dict) -> None:
                reader_cfg = eval_cfg.setdefault("READER", {})
                reader_cfg["input_dir"] = str(native_full_dir)
                reader_cfg["intrinsics"] = [600.0, 600.0, 599.5, 339.5]
                reader_cfg["image_shape"] = [1200, 680]
                reader_cfg["downscale_factor"] = 1
                reader_cfg["preload_images"] = False

            rows[-1].update(run_quality_eval(
                args.binary,
                cfg,
                cfg_path,
                out_dir,
                native_full_dir,
                name,
                log_root,
                quality_eval_split,
                args.quiet_run_logs,
                eval_dir_name="val_native_full",
                metric_prefix="nativefull",
                eval_config_mutator=apply_native_full_quality_overrides,
            ))
            rows[-1]["nativefull_protocol"] = "half_train_native_full_eval"
            rows[-1]["nativefull_input_dir"] = str(native_full_dir)
            for metric in ("PSNR", "SSIM", "LPIPS"):
                hybrid_key = f"nativefull_hybrid_{metric}"
                sdf_key = f"nativefull_sdf_{metric}"
                if hybrid_key in rows[-1] and sdf_key in rows[-1]:
                    rows[-1][f"nativefull_delta_{metric}_hybrid_minus_sdf"] = (
                        rows[-1][hybrid_key] - rows[-1][sdf_key]
                    )
        if args.quality_final_map and status == "OK" and "hybrid" in label:
            final_map_margin = args.final_map_render_margin if args.final_map_render_margin >= 0 else args.frames
            final_map_eval_dir_base = (
                "val_final_map_aligned_tsdf"
                if args.final_map_align_tsdf_to_gaussian_selection
                else "val_final_map"
            )
            final_map_policy_suffix = (
                ""
                if args.final_map_render_selection_policy == "sdf_bounds"
                else f"_{args.final_map_render_selection_policy}"
            )
            final_map_order_suffix = (
                ""
                if args.final_map_tsdf_render_order_policy == "inferred"
                else f"_tsdf_{args.final_map_tsdf_render_order_policy}"
            )
            final_map_reference_suffix = "_localref" if args.final_map_use_local_sdf_reference else "_globalref"
            final_map_residency_suffix = (
                ""
                if args.final_map_tsdf_residency_profile == "inherit"
                else f"_res{args.final_map_tsdf_residency_profile}"
            )
            final_map_eval_dir_name = (
                final_map_eval_dir_base +
                final_map_policy_suffix +
                final_map_order_suffix +
                final_map_reference_suffix +
                final_map_residency_suffix
            )

            def apply_final_map_quality_overrides(eval_cfg: dict) -> None:
                tsdf_multi = eval_cfg.setdefault("PIPE", {}).setdefault("TSDF", {}).setdefault("multi", {})
                if args.final_map_tsdf_residency_profile != "inherit":
                    deep_update(
                        tsdf_multi,
                        tsdf_residency_profile_defaults(args.final_map_tsdf_residency_profile),
                    )
                tsdf_multi["render_compositor"] = args.final_map_tsdf_render_compositor
                tsdf_multi["render_order_policy"] = args.final_map_tsdf_render_order_policy
                tsdf_multi["max_render_maps"] = max(1, args.final_map_max_render_maps)
                if args.final_map_align_tsdf_to_gaussian_selection:
                    tsdf_multi["render_from_residency_pins"] = True
                    tsdf_multi["render_pins_from_gaussian_selection"] = True
                    tsdf_multi["render_pin_primary_submap"] = False
                gs_cfg = eval_cfg.setdefault("PIPE", {}).setdefault("gaussian_submaps", {})
                gs_cfg["render_recompose_with_global_sdf_base"] = True
                gs_cfg["render_use_local_sdf_reference"] = bool(args.final_map_use_local_sdf_reference)
                gs_cfg["render_overlap_depth_gate"] = True
                gs_cfg["render_overlap_depth_tolerance_mode"] = "scaled_max"
                gs_cfg["render_overlap_depth_abs_thres"] = 0.05
                gs_cfg["render_overlap_depth_floor"] = 0.015
                gs_cfg["render_overlap_depth_voxel_scale"] = 0.75
                gs_cfg["render_overlap_depth_quadratic_scale"] = 0.005
                gs_cfg["render_overlap_alpha_thres"] = 0.05
                gs_cfg["render_overlap_diagnostics"] = True
                gs_cfg["render_selection_policy"] = args.final_map_render_selection_policy
                gs_cfg["render_past_frame_margin"] = max(0, final_map_margin)
                gs_cfg["render_future_frame_margin"] = max(0, final_map_margin)
                gs_cfg["render_nonsequential_overlaps"] = True
                gs_cfg["render_max_visible_submaps"] = max(0, args.final_map_render_max_visible_submaps)
                gs_cfg["render_max_visible_gaussians"] = max(0, args.final_map_render_max_visible_gaussians)
                if args.render_selection_continuity_frame_scale is not None:
                    gs_cfg["render_selection_continuity_frame_scale"] = max(1, args.render_selection_continuity_frame_scale)
                if args.render_selection_continuity_min_factor is not None:
                    gs_cfg["render_selection_continuity_min_factor"] = min(
                        1.0, max(0.0, args.render_selection_continuity_min_factor)
                    )
                if args.render_selection_adjacent_bonus is not None:
                    gs_cfg["render_selection_adjacent_bonus"] = max(0.0, args.render_selection_adjacent_bonus)
                if args.render_selection_active_overlap_bonus is not None:
                    gs_cfg["render_selection_active_overlap_bonus"] = max(0.0, args.render_selection_active_overlap_bonus)

            rows[-1].update(run_quality_eval(
                args.binary,
                cfg,
                cfg_path,
                out_dir,
                gt_dir,
                name,
                log_root,
                quality_eval_split,
                args.quiet_run_logs,
                eval_dir_name=final_map_eval_dir_name,
                metric_prefix="finalmap",
                eval_config_mutator=apply_final_map_quality_overrides,
            ))
            rows[-1].update({
                "finalmap_protocol": "bounded_final_live_submap_graph",
                "finalmap_eval_dir": final_map_eval_dir_name,
                "finalmap_tsdf_render_compositor": args.final_map_tsdf_render_compositor,
                "finalmap_tsdf_render_order_policy": args.final_map_tsdf_render_order_policy,
                "finalmap_tsdf_residency_profile": args.final_map_tsdf_residency_profile,
                "finalmap_max_render_maps": max(1, args.final_map_max_render_maps),
                "finalmap_tsdf_render_from_residency_pins": bool(args.final_map_align_tsdf_to_gaussian_selection),
                "finalmap_tsdf_render_pins_from_gaussian_selection": bool(args.final_map_align_tsdf_to_gaussian_selection),
                "finalmap_tsdf_render_pin_primary_submap": (
                    False
                    if args.final_map_align_tsdf_to_gaussian_selection
                    else (
                        False
                        if args.final_map_tsdf_residency_profile in {"thor_cpu_bounded", "eval_resident64", "disk_stress"}
                        else ("" if args.tsdf_render_pin_primary == "default" else args.tsdf_render_pin_primary == "on")
                    )
                ),
                "finalmap_render_max_visible_submaps": max(0, args.final_map_render_max_visible_submaps),
                "finalmap_render_max_visible_gaussians": max(0, args.final_map_render_max_visible_gaussians),
                "finalmap_render_margin": max(0, final_map_margin),
                "finalmap_render_use_local_sdf_reference": bool(args.final_map_use_local_sdf_reference),
                "finalmap_render_selection_policy": args.final_map_render_selection_policy,
                "finalmap_render_selection_continuity_frame_scale": (
                    "" if args.render_selection_continuity_frame_scale is None
                    else max(1, args.render_selection_continuity_frame_scale)
                ),
                "finalmap_render_selection_continuity_min_factor": (
                    "" if args.render_selection_continuity_min_factor is None
                    else min(1.0, max(0.0, args.render_selection_continuity_min_factor))
                ),
                "finalmap_render_selection_adjacent_bonus": (
                    "" if args.render_selection_adjacent_bonus is None
                    else max(0.0, args.render_selection_adjacent_bonus)
                ),
                "finalmap_render_selection_active_overlap_bonus": (
                    "" if args.render_selection_active_overlap_bonus is None
                    else max(0.0, args.render_selection_active_overlap_bonus)
                ),
                "finalmap_render_overlap_depth_gate": True,
                "finalmap_render_overlap_depth_tolerance_mode": "scaled_max",
                "finalmap_render_overlap_depth_abs_thres": 0.05,
                "finalmap_render_overlap_alpha_thres": 0.05,
            })
            rows[-1].update({
                f"finalmap_{key}": value
                for key, value in summarize_render_debug(out_dir / final_map_eval_dir_name / "submap_render_debug.csv").items()
            })
            if "finalmap_hybrid_PSNR" in rows[-1] and "finalmap_sdf_PSNR" in rows[-1]:
                rows[-1]["finalmap_delta_PSNR_hybrid_minus_sdf"] = (
                    rows[-1]["finalmap_hybrid_PSNR"] - rows[-1]["finalmap_sdf_PSNR"]
                )
            if "finalmap_hybrid_SSIM" in rows[-1] and "finalmap_sdf_SSIM" in rows[-1]:
                rows[-1]["finalmap_delta_SSIM_hybrid_minus_sdf"] = (
                    rows[-1]["finalmap_hybrid_SSIM"] - rows[-1]["finalmap_sdf_SSIM"]
                )
            if "finalmap_hybrid_LPIPS" in rows[-1] and "finalmap_sdf_LPIPS" in rows[-1]:
                rows[-1]["finalmap_delta_LPIPS_hybrid_minus_sdf"] = (
                    rows[-1]["finalmap_hybrid_LPIPS"] - rows[-1]["finalmap_sdf_LPIPS"]
                )
            finalmap_psnr_pass = rows[-1].get("finalmap_delta_PSNR_hybrid_minus_sdf", -999) >= 0.2
            finalmap_lpips_pass = rows[-1].get("finalmap_delta_LPIPS_hybrid_minus_sdf", 999) <= -0.005
            rows[-1]["finalmap_quality_gate"] = "PASS" if finalmap_psnr_pass or finalmap_lpips_pass else "FAIL"
        if benchmark_role is not None and status == "OK":
            provenance_files = sorted(cfg_path.parent.glob(f"{cfg_path.stem}_eval*.yaml"))
            provenance_files.extend(sorted(out_dir.glob("val*/render/results.json")))
            provenance_files.extend(sorted(out_dir.glob("val*/raycast_color/results.json")))
            artifact_hashes = {
                str(path.resolve()): sha256_file(path)
                for path in provenance_files
                if path.is_file()
            }
            manifest_path = out_dir / "benchmark_manifest.json"
            write_manifest(
                manifest_path,
                protocol=args.benchmark_protocol,
                role=benchmark_role,
                cfg=cfg,
                config_path=cfg_path,
                binary=args.binary,
                dataset_root=input_dir,
                dataset_hash_mode=args.benchmark_dataset_hash,
                extra={
                    "state": "completed" if status == "OK" else "failed",
                    "status": status,
                    "return_code": proc.returncode,
                    "label": label,
                    "evaluation_artifact_sha256": artifact_hashes,
                },
            )
            rows[-1]["benchmark_manifest"] = str(manifest_path)
        print(json.dumps(rows[-1], indent=2))

    by_label = {row["name"]: row for row in rows}
    basic = next((row for row in rows if "_basic_sdf_" in row["name"]), None)
    multi = next((row for row in rows if "_multi_sdf_" in row["name"]), None)
    hybrid = next((row for row in rows if "_multi_hybrid_" in row["name"]), None)
    gates = {}
    if basic and multi and basic.get("ate_actual_rmse_cm") is not None and multi.get("ate_actual_rmse_cm") is not None:
        gates["multi_sdf_ate_delta_cm"] = multi["ate_actual_rmse_cm"] - basic["ate_actual_rmse_cm"]
        gates["multi_sdf_ate_gate"] = "PASS" if gates["multi_sdf_ate_delta_cm"] <= 0.02 else "FAIL"
    if basic and multi and basic.get("fps") and multi.get("fps"):
        gates["multi_sdf_fps_ratio"] = multi["fps"] / basic["fps"]
        gates["multi_sdf_fps_gate"] = "PASS" if gates["multi_sdf_fps_ratio"] >= 0.85 else "FAIL"
    if hybrid and basic and basic.get("fps") and hybrid.get("fps"):
        gates["hybrid_vs_basic_sdf_fps_ratio"] = hybrid["fps"] / basic["fps"]
    if hybrid and "quality_gate" in hybrid:
        gates["hybrid_quality_gate"] = hybrid["quality_gate"]
    if hybrid and "finalmap_quality_gate" in hybrid:
        gates["hybrid_finalmap_quality_gate"] = hybrid["finalmap_quality_gate"]

    split_suffix = f"_heldout{args.heldout_interval}" if args.heldout_interval > 0 else ""
    summary_stem = safe_experiment_name(
        "_".join([
            "canonical_summary",
            args.scene_name,
            f"{args.frames}f",
            "half" if half else "full",
            args.gaussian_profile,
            args.hybrid_submap_profile,
            f"bin{binary_hash_profile}",
            f"tsdf{args.multi_hash_profile}",
        ] + ([f"seed{int(args.random_seed)}"] if args.random_seed is not None else [])
          + ([f"fmres{args.final_map_tsdf_residency_profile}"]
             if args.quality_final_map and args.final_map_tsdf_residency_profile != "inherit" else [])
          + ([args.tag] if args.tag else []))
    )
    summary_path = args.output_root / f"{summary_stem}{split_suffix}.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")
    gate_stem = summary_stem.replace("canonical_summary", "canonical_gates", 1)
    gate_path = args.output_root / f"{gate_stem}{split_suffix}.json"
    gate_path.write_text(json.dumps(gates, indent=2) + "\n")
    print(f"Gates: {gate_path}")
    print(json.dumps(gates, indent=2))
    failed_runs, failed_gates = validation_exit_failures(rows, gates)
    if failed_runs:
        print("[ERROR] Trainer failure(s): " + ", ".join(failed_runs))
    if failed_gates:
        print("[ERROR] Hard gate failure(s): " + ", ".join(failed_gates))
    return 1 if failed_runs or failed_gates else 0


if __name__ == "__main__":
    raise SystemExit(main())
