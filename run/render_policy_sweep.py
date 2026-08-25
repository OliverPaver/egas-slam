#!/usr/bin/env python3
"""
Eval-only sweep for Gaussian submap render policies on an existing checkpoint.
"""
import argparse
import csv
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
METRIC_SCRIPT = PROJECT_ROOT / "scripts" / "metric_general.py"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)


def parse_int_sweep(value: str) -> list[int]:
    result = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("selection-cap values must be positive integers")
        result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("selection-cap sweep cannot be empty")
    return result


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_numeric(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_avg": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_avg": sum(values) / len(values),
        f"{prefix}_p95": percentile(values, 95.0),
        f"{prefix}_p99": percentile(values, 99.0),
        f"{prefix}_max": max(values),
    }


def selected_submap_count(value: str | None) -> int:
    if value is None:
        return 0
    value = value.strip()
    if not value:
        return 0
    return len([part for part in value.split("|") if part.strip()])


def summarize_render_diagnostics(eval_path: Path) -> dict[str, object]:
    diag_path = eval_path / "submap_render_debug.csv"
    if not diag_path.exists():
        return {"render_diag_rows": 0}

    rows: list[dict[str, str]] = []
    with diag_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"render_diag_rows": 0}

    summary: dict[str, object] = {"render_diag_rows": len(rows)}
    fingerprint_fields = [
        "render_policy",
        "render_selection_policy",
        "render_max_visible_submaps",
        "render_max_visible_gaussians",
        "render_use_local_sdf_reference",
        "render_recompose_with_global_sdf_base",
        "render_overlap_depth_gate",
        "render_overlap_depth_tolerance_mode",
        "tsdf_render_order_policy",
        "tsdf_render_from_residency_pins",
        "tsdf_render_pins_from_gaussian_selection",
    ]
    for field in fingerprint_fields:
        summary[f"diag_{field}"] = rows[0].get(field, "")
    selected_counts = [float(selected_submap_count(row.get("selected_submaps"))) for row in rows]
    summary.update(summarize_numeric(selected_counts, "selected_submaps"))

    numeric_fields = {
        "selected_gaussians": "selected_gaussians",
        "render_selection_candidates": "selection_candidates",
        "render_selection_truncated_submaps": "truncated_submaps",
        "render_selection_truncated_gaussians": "truncated_gaussians",
        "render_selection_budget_gaussians": "budget_gaussians",
        "render_selection_sticky_count": "sticky_submaps",
        "tsdf_resident_after_prefetch": "tsdf_resident_after_prefetch",
        "tsdf_prefetch_ms": "tsdf_prefetch_ms",
        "raycast_ms": "raycast_ms",
        "select_ms": "select_ms",
        "render_ms": "render_ms",
        "total_ms": "total_ms",
        "render_diag_hybrid_rgb_l1": "hybrid_rgb_l1",
        "render_diag_sdf_rgb_l1": "sdf_rgb_l1",
        "render_diag_owner_alpha_coverage": "owner_alpha_coverage",
        "render_diag_neighbor_alpha_coverage": "neighbor_alpha_coverage",
        "render_diag_final_alpha_coverage": "final_alpha_coverage",
        "render_diag_final_depth_disagreement_ratio": "final_depth_disagreement_ratio",
    }
    parsed_columns: dict[str, list[float]] = {prefix: [] for prefix in numeric_fields.values()}
    for row in rows:
        for csv_key, prefix in numeric_fields.items():
            value = parse_float(row.get(csv_key))
            if value is not None:
                parsed_columns[prefix].append(value)
    for prefix, values in parsed_columns.items():
        summary.update(summarize_numeric(values, prefix))

    truncated_submaps = parsed_columns["truncated_submaps"]
    truncated_gaussians = parsed_columns["truncated_gaussians"]
    owner_oversized = [
        parse_float(row.get("render_selection_owner_oversized")) or 0.0
        for row in rows
    ]
    sticky = parsed_columns["sticky_submaps"]
    hybrid_l1 = parsed_columns["hybrid_rgb_l1"]
    sdf_l1 = parsed_columns["sdf_rgb_l1"]
    summary["truncated_submaps_total"] = int(round(sum(truncated_submaps)))
    summary["truncated_submap_frames"] = sum(1 for value in truncated_submaps if value > 0.0)
    summary["truncated_gaussians_total"] = int(round(sum(truncated_gaussians)))
    summary["truncated_gaussian_frames"] = sum(1 for value in truncated_gaussians if value > 0.0)
    summary["owner_oversized_frames"] = sum(1 for value in owner_oversized if value > 0.0)
    summary["sticky_frames"] = sum(1 for value in sticky if value > 0.0)
    if hybrid_l1 and sdf_l1 and len(hybrid_l1) == len(sdf_l1):
        l1_gain = [sdf - hybrid for sdf, hybrid in zip(sdf_l1, hybrid_l1)]
        summary.update(summarize_numeric(l1_gain, "rgb_l1_gain"))
    return summary


def selection_cap_variants(
    caps: list[int],
    render_max_visible_gaussians: int | None,
    render_selection_policy: str | None,
) -> list[dict]:
    rows = []
    for cap in caps:
        suffix_parts = [f"selcap{cap}"]
        row: dict[str, object] = {
            "suffix": "_".join(suffix_parts),
            "render_max_visible_submaps": cap,
        }
        if render_max_visible_gaussians is not None:
            row["render_max_visible_gaussians"] = render_max_visible_gaussians
            row["suffix"] += f"_gcap{render_max_visible_gaussians}"
        if render_selection_policy is not None:
            row["render_selection_policy"] = render_selection_policy
            row["suffix"] += f"_{render_selection_policy}"
        rows.append(row)
    return rows


def variants() -> list[dict]:
    rows = [
        {"suffix": "sdf_guided_fill_a0p001", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.001},
        {"suffix": "sdf_guided_fill_a0p002", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.002},
        {"suffix": "sdf_guided_fill_a0p005", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.005},
        {"suffix": "sdf_guided_fill_a0p010", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.010},
        {"suffix": "sdf_guided_fill_a0p020", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.020},
        {"suffix": "sdf_guided_fill_a0p050", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.050},
        {"suffix": "sdf_guided_fill_a0p100", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.100},
        {"suffix": "sdf_guided_fill_a0p250", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.250},
        {"suffix": "sdf_guided_fill_a0p500", "render_policy": "sdf_guided_fill", "render_owner_alpha_fill_thres": 0.500},
    ]
    for alpha in [0.5, 1.0, 2.0]:
        for depth in [0.01, 0.02, 0.03]:
            for max_weight in [0.03, 0.05, 0.10]:
                rows.append({
                    "suffix": f"depth_blend_a{alpha:g}_d{depth:g}_w{max_weight:g}".replace(".", "p"),
                    "render_policy": "sdf_depth_consistent_blend",
                    "render_blend_alpha_thres": alpha,
                    "render_blend_depth_thres": depth,
                    "render_blend_max_weight": max_weight,
                    "render_blend_weight_scale": max_weight,
                })
    guarded_suffixes = {
        "handoff_aware_no_future",
        "handoff_aware_no_future_min512",
        "adaptive_no_future",
    }
    rows.extend(variant for variant in boundary_variants() if variant["suffix"] in guarded_suffixes)
    return rows


def blend_variants() -> list[dict]:
    rows = []
    for weight in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.50, 1.00]:
        rows.append({
            "suffix": f"sdf_guided_fill_blend{weight:.2f}".replace(".", "p"),
            "render_policy": "sdf_guided_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": weight,
        })
    return rows


def adaptive_variants() -> list[dict]:
    rows = []
    for blend in [0.15, 0.20, 0.30, 0.50]:
        for feather in [0.25, 0.50, 0.75]:
            rows.append({
                "suffix": f"adaptive_b{blend:.2f}_f{feather:.2f}".replace(".", "p"),
                "render_policy": "sdf_guided_adaptive_fill",
                "render_owner_alpha_fill_thres": 0.001,
                "render_fill_blend_weight": blend,
                "render_adaptive_kernel": 5,
                "render_adaptive_alpha_scale": 2.0,
                "render_adaptive_min_weight": 0.02,
                "render_adaptive_boundary_feather": feather,
                "render_adaptive_depth_thres": 0.05,
            })
    return rows


def overhead_variants() -> list[dict]:
    return [
        {
            "suffix": "constant_fill_b0p15",
            "render_policy": "sdf_guided_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.15,
        },
        {
            "suffix": "adaptive_fill_b0p20_f0p25",
            "render_policy": "sdf_guided_adaptive_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_adaptive_kernel": 5,
            "render_adaptive_alpha_scale": 2.0,
            "render_adaptive_min_weight": 0.02,
            "render_adaptive_boundary_feather": 0.25,
            "render_adaptive_depth_thres": 0.05,
        },
    ]


def boundary_variants() -> list[dict]:
    return [
        {
            "suffix": "owner_fill_hole",
            "render_policy": "owner_neighbor_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
        },
        {
            "suffix": "owner_fill_valid",
            "render_policy": "owner_neighbor_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": False,
        },
        {
            "suffix": "owner_fill_hole_lag30",
            "render_policy": "owner_neighbor_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_include_handoff_submaps": True,
            "render_min_owner_age_frames": 30,
        },
        {
            "suffix": "owner_fill_valid_lag30",
            "render_policy": "owner_neighbor_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": False,
            "render_include_handoff_submaps": True,
            "render_min_owner_age_frames": 30,
        },
        {
            "suffix": "owner_conf_hole",
            "render_policy": "owner_confidence_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.05,
            "render_blend_weight_scale": 0.05,
        },
        {
            "suffix": "owner_conf_valid_lag30",
            "render_policy": "owner_confidence_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": False,
            "render_include_handoff_submaps": True,
            "render_min_owner_age_frames": 30,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.05,
            "render_blend_weight_scale": 0.05,
        },
        {
            "suffix": "owner_conf_valid",
            "render_policy": "owner_confidence_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": False,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.05,
            "render_blend_weight_scale": 0.05,
        },
        {
            "suffix": "owner_conf_valid_strict",
            "render_policy": "owner_confidence_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_neighbor_alpha_fill_thres": 0.10,
            "render_fill_blend_weight": 0.15,
            "render_fill_require_sdf_hole": False,
            "render_blend_depth_thres": 0.02,
            "render_blend_max_weight": 0.03,
            "render_blend_weight_scale": 0.03,
        },
        {
            "suffix": "adaptive_fill_default",
            "render_policy": "sdf_guided_adaptive_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_adaptive_kernel": 5,
            "render_adaptive_alpha_scale": 2.0,
            "render_adaptive_min_weight": 0.02,
            "render_adaptive_boundary_feather": 0.25,
            "render_adaptive_depth_thres": 0.05,
        },
        {
            "suffix": "handoff_aware_default",
            "render_policy": "handoff_aware_fill",
            "render_owner_alpha_fill_thres": 0.25,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
        },
        {
            "suffix": "handoff_aware_no_future",
            "render_policy": "handoff_aware_fill",
            "render_owner_alpha_fill_thres": 0.25,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
            "render_future_frame_margin": 0,
        },
        {
            "suffix": "handoff_aware_no_future_min512",
            "render_policy": "handoff_aware_fill",
            "render_owner_alpha_fill_thres": 0.25,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
            "render_future_frame_margin": 0,
            "render_min_neighbor_gaussians": 512,
        },
        {
            "suffix": "handoff_aware_boundary",
            "render_policy": "handoff_aware_fill",
            "render_owner_alpha_fill_thres": 0.25,
            "render_neighbor_alpha_fill_thres": 0.05,
            "render_fill_blend_weight": 0.20,
            "render_fill_require_sdf_hole": True,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
            "render_hole_requires_boundary": True,
            "render_hole_boundary_kernel": 5,
            "render_hole_boundary_min_valid": 1,
            "render_hole_boundary_depth_thres": 0.05,
        },
        {
            "suffix": "depth_blend_default",
            "render_policy": "sdf_depth_consistent_blend",
            "render_blend_alpha_thres": 0.50,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
            "render_fill_blend_weight": 0.20,
        },
        {
            "suffix": "depth_blend_boundary",
            "render_policy": "sdf_depth_consistent_blend",
            "render_blend_alpha_thres": 0.50,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
            "render_fill_blend_weight": 0.20,
            "render_hole_requires_boundary": True,
            "render_hole_boundary_kernel": 5,
            "render_hole_boundary_min_valid": 1,
            "render_hole_boundary_depth_thres": 0.05,
        },
        {
            "suffix": "adaptive_no_future",
            "render_policy": "sdf_guided_adaptive_fill",
            "render_owner_alpha_fill_thres": 0.001,
            "render_fill_blend_weight": 0.20,
            "render_adaptive_kernel": 5,
            "render_adaptive_alpha_scale": 2.0,
            "render_adaptive_min_weight": 0.02,
            "render_adaptive_boundary_feather": 0.25,
            "render_adaptive_depth_thres": 0.05,
            "render_future_frame_margin": 0,
        },
        {
            "suffix": "depth_blend_light",
            "render_policy": "sdf_depth_consistent_blend",
            "render_blend_alpha_thres": 0.10,
            "render_blend_depth_thres": 0.03,
            "render_blend_max_weight": 0.05,
            "render_blend_weight_scale": 0.05,
            "render_fill_blend_weight": 0.20,
        },
        {
            "suffix": "depth_blend_medium",
            "render_policy": "sdf_depth_consistent_blend",
            "render_blend_alpha_thres": 0.10,
            "render_blend_depth_thres": 0.05,
            "render_blend_max_weight": 0.10,
            "render_blend_weight_scale": 0.10,
            "render_fill_blend_weight": 0.20,
        },
        {
            "suffix": "depth_blend_balanced",
            "render_policy": "sdf_depth_consistent_blend",
            "render_blend_alpha_thres": 0.10,
            "render_blend_depth_thres": 0.05,
            "render_blend_max_weight": 0.20,
            "render_blend_weight_scale": 0.20,
            "render_fill_blend_weight": 0.20,
        },
        {
            "suffix": "nearest_depth",
            "render_policy": "nearest_depth",
        },
    ]


def owner_confidence_variants() -> list[dict]:
    return [variant for variant in boundary_variants() if variant["render_policy"] == "owner_confidence_fill"]


def lag_variants() -> list[dict]:
    rows = []
    for lag in [5, 10, 15, 20, 30]:
        for require_hole in [True, False]:
            rows.append({
                "suffix": f"owner_fill_{'hole' if require_hole else 'valid'}_lag{lag}",
                "render_policy": "owner_neighbor_fill",
                "render_owner_alpha_fill_thres": 0.001,
                "render_fill_blend_weight": 0.20,
                "render_fill_require_sdf_hole": require_hole,
                "render_include_handoff_submaps": True,
                "render_min_owner_age_frames": lag,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, default=PROJECT_ROOT / "data" / "replica" / "office0_half_clean")
    parser.add_argument("--binary", type=Path, default=PROJECT_ROOT / "build_compact" / "slam_trainer")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--eval-split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--only-hole-fill", action="store_true")
    parser.add_argument("--blend-sweep", action="store_true")
    parser.add_argument("--adaptive-sweep", action="store_true")
    parser.add_argument("--overhead-benchmark", action="store_true")
    parser.add_argument("--boundary-sweep", action="store_true")
    parser.add_argument("--owner-confidence-sweep", action="store_true")
    parser.add_argument("--lag-sweep", action="store_true")
    parser.add_argument(
        "--selection-cap-sweep",
        type=parse_int_sweep,
        default=None,
        help="Comma-separated render_max_visible_submaps values for an eval-only selection cap sweep",
    )
    parser.add_argument(
        "--render-max-visible-gaussians",
        type=int,
        default=None,
        help="Override PIPE.gaussian_submaps.render_max_visible_gaussians for selection-cap sweeps or all variants",
    )
    parser.add_argument(
        "--render-selection-policy",
        choices=["temporal", "sdf_bounds", "sdf_bounds_continuity"],
        default=None,
        help="Override PIPE.gaussian_submaps.render_selection_policy for selection-cap sweeps or all variants",
    )
    parser.add_argument("--only-suffix", default="", help="Run only variants with this exact suffix")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N variants; useful for smoke tests")
    parser.add_argument("--depth-mask", action="store_true", help="Use the same valid-depth mask as the TUM quality audits")
    parser.add_argument("--render-min-ready-frames", type=int, default=None, help="Skip Gaussian submaps in rendering until they have accumulated this many frames")
    parser.add_argument("--render-owner-min-ready-frames", type=int, default=None, help="Owner submap readiness override; defaults to render_min_ready_frames")
    parser.add_argument("--render-neighbor-min-ready-frames", type=int, default=None, help="Neighbor submap readiness override; defaults to render_min_ready_frames")
    parser.add_argument("--render-past-frame-margin", type=int, default=None, help="Override PIPE.gaussian_submaps.render_past_frame_margin for all variants")
    parser.add_argument("--render-future-frame-margin", type=int, default=None, help="Override PIPE.gaussian_submaps.render_future_frame_margin for all variants")
    nonseq_group = parser.add_mutually_exclusive_group()
    nonseq_group.add_argument(
        "--enable-render-nonsequential-overlaps",
        action="store_true",
        help="allow non-neighbor Gaussian/SDF submaps during render selection",
    )
    nonseq_group.add_argument(
        "--disable-render-nonsequential-overlaps",
        action="store_true",
        help="restrict render selection to owner and sequential neighbors",
    )
    parser.add_argument("--render-min-neighbor-gaussians", type=int, default=None, help="Override PIPE.gaussian_submaps.render_min_neighbor_gaussians for all variants")
    parser.add_argument(
        "--tsdf-render-compositor",
        choices=["nearest_depth", "owner_priority", "handoff_aware"],
        default=None,
        help="Override PIPE.TSDF.multi.render_compositor before eval so TSDF overlap rendering can be isolated",
    )
    parser.add_argument(
        "--tsdf-render-order-policy",
        choices=["inferred", "adjacent_first", "continuity"],
        default=None,
        help="Override PIPE.TSDF.multi.render_order_policy before eval",
    )
    parser.add_argument(
        "--tsdf-max-render-maps",
        type=int,
        default=None,
        help="Override PIPE.TSDF.multi.max_render_maps before eval",
    )
    render_from_pins_group = parser.add_mutually_exclusive_group()
    render_from_pins_group.add_argument(
        "--enable-tsdf-render-from-residency-pins",
        action="store_true",
        help="render the combined TSDF base only from the bounded residency pins",
    )
    render_from_pins_group.add_argument(
        "--disable-tsdf-render-from-residency-pins",
        action="store_true",
        help="render the combined TSDF base from the renderer-selected maps instead of residency pins",
    )
    pins_from_gaussians_group = parser.add_mutually_exclusive_group()
    pins_from_gaussians_group.add_argument(
        "--enable-tsdf-render-pins-from-gaussian-selection",
        action="store_true",
        help="add selected Gaussian submap owners to the bounded TSDF residency pins",
    )
    pins_from_gaussians_group.add_argument(
        "--disable-tsdf-render-pins-from-gaussian-selection",
        action="store_true",
        help="do not derive TSDF residency pins from selected Gaussian submaps",
    )
    primary_pin_group = parser.add_mutually_exclusive_group()
    primary_pin_group.add_argument(
        "--enable-tsdf-render-pin-primary-submap",
        action="store_true",
        help="always pin the frame owner primary submap for TSDF rendering",
    )
    primary_pin_group.add_argument(
        "--disable-tsdf-render-pin-primary-submap",
        action="store_true",
        help="do not automatically pin the frame owner primary submap for TSDF rendering",
    )
    parser.add_argument("--disable-single-submap-passthrough", action="store_true", help="force one selected submap through the selected compositor")
    parser.add_argument("--allow-multi-logical-single-passthrough", action="store_true", help="allow passthrough for one selected submap even when the saved graph has multiple logical submaps; kept for compatibility because this is now the default")
    parser.add_argument("--require-single-logical-single-passthrough", action="store_true", help="only allow single-submap passthrough when the whole saved graph has one logical Gaussian submap")
    local_ref_group = parser.add_mutually_exclusive_group()
    local_ref_group.add_argument("--enable-local-sdf-reference-render", action="store_true", help="render each local Gaussian submap against its owner local SDF reference")
    local_ref_group.add_argument("--disable-local-sdf-reference-render", action="store_true", help="render local Gaussian submaps against the combined global SDF reference for ablation")
    parser.add_argument(
        "--disable-bypass-libtorch-atexit",
        action="store_true",
        help="leave LibTorch normal process teardown enabled for eval sweeps; default bypasses it to avoid false crash rows after completed renders",
    )
    args = parser.parse_args()

    free_gb = shutil.disk_usage(args.workspace).free / 1024**3
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Refusing to run: {free_gb:.1f} GB free, need {args.min_free_gb:.1f} GB")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.base_config.read_text())
    base["work_mode"] = "eval"
    base["workspace_dir"] = str(args.workspace)
    base["PIPE"]["eval_after_train"] = False
    base["PIPE"]["save_after_train"] = False
    base["PIPE"]["eval_split"] = args.eval_split
    if not args.disable_bypass_libtorch_atexit:
        base["PIPE"]["bypass_libtorch_atexit"] = True
    if args.render_min_ready_frames is not None:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_min_ready_frames"] = max(0, args.render_min_ready_frames)
    if args.render_owner_min_ready_frames is not None:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_owner_min_ready_frames"] = max(0, args.render_owner_min_ready_frames)
    if args.render_neighbor_min_ready_frames is not None:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_neighbor_min_ready_frames"] = max(0, args.render_neighbor_min_ready_frames)
    if args.render_past_frame_margin is not None:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_past_frame_margin"] = max(0, args.render_past_frame_margin)
    if args.render_future_frame_margin is not None:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_future_frame_margin"] = max(0, args.render_future_frame_margin)
    if args.enable_render_nonsequential_overlaps:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_nonsequential_overlaps"] = True
    elif args.disable_render_nonsequential_overlaps:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_nonsequential_overlaps"] = False
    if args.render_min_neighbor_gaussians is not None:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_min_neighbor_gaussians"] = max(0, args.render_min_neighbor_gaussians)
    if args.disable_single_submap_passthrough:
        submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
        submap_cfg["render_single_submap_passthrough"] = False
    submap_cfg = base["PIPE"].setdefault("gaussian_submaps", {})
    submap_cfg["render_single_submap_passthrough_requires_single_logical_map"] = (
        args.require_single_logical_single_passthrough and
        not args.allow_multi_logical_single_passthrough
    )
    if args.enable_local_sdf_reference_render:
        submap_cfg["render_use_local_sdf_reference"] = True
    elif args.disable_local_sdf_reference_render:
        submap_cfg["render_use_local_sdf_reference"] = False
    if args.render_max_visible_gaussians is not None:
        submap_cfg["render_max_visible_gaussians"] = max(1, args.render_max_visible_gaussians)
    if args.render_selection_policy is not None:
        submap_cfg["render_selection_policy"] = args.render_selection_policy
    multi_cfg = base["PIPE"].setdefault("TSDF", {}).setdefault("multi", {})
    if args.tsdf_render_compositor is not None:
        multi_cfg["render_compositor"] = args.tsdf_render_compositor
    if args.tsdf_render_order_policy is not None:
        multi_cfg["render_order_policy"] = args.tsdf_render_order_policy
    if args.tsdf_max_render_maps is not None:
        multi_cfg["max_render_maps"] = max(1, args.tsdf_max_render_maps)
    if args.enable_tsdf_render_from_residency_pins:
        multi_cfg["render_from_residency_pins"] = True
    elif args.disable_tsdf_render_from_residency_pins:
        multi_cfg["render_from_residency_pins"] = False
    if args.enable_tsdf_render_pins_from_gaussian_selection:
        multi_cfg["render_pins_from_gaussian_selection"] = True
    elif args.disable_tsdf_render_pins_from_gaussian_selection:
        multi_cfg["render_pins_from_gaussian_selection"] = False
    if args.enable_tsdf_render_pin_primary_submap:
        multi_cfg["render_pin_primary_submap"] = True
    elif args.disable_tsdf_render_pin_primary_submap:
        multi_cfg["render_pin_primary_submap"] = False

    summary = []
    if args.selection_cap_sweep is not None:
        all_variants = selection_cap_variants(
            args.selection_cap_sweep,
            args.render_max_visible_gaussians,
            args.render_selection_policy,
        )
    elif args.lag_sweep:
        all_variants = lag_variants()
    elif args.owner_confidence_sweep:
        all_variants = owner_confidence_variants()
    elif args.boundary_sweep:
        all_variants = boundary_variants()
    elif args.overhead_benchmark:
        all_variants = overhead_variants()
    elif args.adaptive_sweep:
        all_variants = adaptive_variants()
    elif args.blend_sweep:
        all_variants = blend_variants()
    else:
        all_variants = variants()
    if args.only_hole_fill and not args.blend_sweep:
        all_variants = [variant for variant in all_variants if variant["render_policy"] == "sdf_guided_fill"]
    if args.only_suffix:
        all_variants = [variant for variant in all_variants if variant["suffix"] == args.only_suffix]
        if not all_variants:
            raise RuntimeError(f"No render-policy variant found for suffix: {args.only_suffix}")
    if args.limit > 0:
        all_variants = all_variants[:args.limit]

    log_dir = args.workspace / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for variant in all_variants:
        cfg = deepcopy(base)
        cfg["PIPE"]["eval_path"] = f"/val_{variant['suffix']}"
        submap_cfg = cfg["PIPE"].setdefault("gaussian_submaps", {})
        for key, value in variant.items():
            if key != "suffix":
                submap_cfg[key] = value
        cfg_path = CONFIG_DIR / f"{args.workspace.name}_{variant['suffix']}_eval.yaml"
        cfg_path.write_text(yaml.dump(cfg, sort_keys=False))

        eval_path = args.workspace / cfg["PIPE"]["eval_path"].lstrip("/")
        if eval_path.exists():
            shutil.rmtree(eval_path)

        proc = run_cmd([str(args.binary), str(cfg_path)])
        log_path = log_dir / f"{args.workspace.name}_{variant['suffix']}_eval.log"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        row = {
            "suffix": variant["suffix"],
            "status": "OK" if proc.returncode == 0 else "CRASH",
            "config": str(cfg_path),
            "log": str(log_path),
            "diag_pipe_bypass_libtorch_atexit": int(bool(cfg.get("PIPE", {}).get("bypass_libtorch_atexit", False))),
        }
        if proc.returncode == 0:
            metric_cmd = ["python", str(METRIC_SCRIPT), "--gt_path", str(args.gt_path), "--render_path", str(eval_path / "render")]
            if args.depth_mask:
                metric_cmd.append("--depth_mask")
            metric_proc = run_cmd(metric_cmd)
            metric_log = log_dir / f"{args.workspace.name}_{variant['suffix']}_metric.log"
            metric_log.write_text(metric_proc.stdout + "\n--- STDERR ---\n" + metric_proc.stderr)
            row["metric_log"] = str(metric_log)
            results_path = eval_path / "render" / "results.json"
            if results_path.exists():
                row.update(json.loads(results_path.read_text()))
            eval_sdf_results_path = eval_path / "raycast_color" / "results.json"
            if (eval_path / "raycast_color").exists():
                if not eval_sdf_results_path.exists():
                    sdf_metric_cmd = [
                        "python",
                        str(METRIC_SCRIPT),
                        "--gt_path",
                        str(args.gt_path),
                        "--render_path",
                        str(eval_path / "raycast_color"),
                    ]
                    if args.depth_mask:
                        sdf_metric_cmd.append("--depth_mask")
                    sdf_metric_proc = run_cmd(sdf_metric_cmd)
                    sdf_metric_log = log_dir / f"{args.workspace.name}_{variant['suffix']}_sdf_metric.log"
                    sdf_metric_log.write_text(sdf_metric_proc.stdout + "\n--- STDERR ---\n" + sdf_metric_proc.stderr)
                    row["sdf_metric_log"] = str(sdf_metric_log)
            sdf_results_path = eval_sdf_results_path
            if not sdf_results_path.exists():
                sdf_results_path = args.workspace / "val" / "raycast_color" / "results.json"
            if sdf_results_path.exists():
                sdf = json.loads(sdf_results_path.read_text())
                if "PSNR" in row:
                    row["delta_PSNR"] = row["PSNR"] - sdf["PSNR"]
                if "SSIM" in row:
                    row["delta_SSIM"] = row["SSIM"] - sdf["SSIM"]
                if "LPIPS" in row:
                    row["delta_LPIPS"] = row["LPIPS"] - sdf["LPIPS"]
            row.update(summarize_render_diagnostics(eval_path))
        summary.append(row)
        print(json.dumps(row, indent=2))

    summary_stem = "render_policy_sweep"
    if args.selection_cap_sweep is not None:
        cap_suffix = "_".join(str(value) for value in args.selection_cap_sweep)
        summary_stem = f"render_selection_cap_sweep_{cap_suffix}"
        if args.render_max_visible_gaussians is not None:
            summary_stem += f"_gcap{args.render_max_visible_gaussians}"
        if args.render_selection_policy is not None:
            summary_stem += f"_{args.render_selection_policy}"
        if args.tsdf_render_compositor is not None:
            summary_stem += f"_tsdf{args.tsdf_render_compositor}"
        if args.tsdf_render_order_policy is not None:
            summary_stem += f"_{args.tsdf_render_order_policy}"
        if args.enable_local_sdf_reference_render:
            summary_stem += "_localref"
        elif args.disable_local_sdf_reference_render:
            summary_stem += "_globalref"
        if args.enable_tsdf_render_from_residency_pins:
            summary_stem += "_frompins"
        elif args.disable_tsdf_render_from_residency_pins:
            summary_stem += "_inferred"
        if args.enable_tsdf_render_pins_from_gaussian_selection:
            summary_stem += "_gspins"
        elif args.disable_tsdf_render_pins_from_gaussian_selection:
            summary_stem += "_nogspins"
    if args.only_suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.only_suffix)
        summary_stem = f"{summary_stem}_{safe_suffix}"
    summary_path = args.workspace / f"{summary_stem}.csv"
    with summary_path.open("w", newline="") as f:
        fields = sorted({key for row in summary for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
