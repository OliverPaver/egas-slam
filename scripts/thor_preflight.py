#!/usr/bin/env python3
"""Validate a GPS-SLAM config before running it on Jetson Thor.

This is a conservative guard for embedded/long-run tests. It does not modify
configs or benchmark the system; it only reports settings that are likely to
cause memory pressure, unbounded output growth, or unfair/ambiguous runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from benchmark_protocol import PROTOCOL_NONE, PROTOCOLS, validate_protocol  # noqa: E402


HASH_PROFILE_SCENE_BYTES = {
    "compact_64k": 273_547_264,
    "medium_128k": 547_094_528,
    "legacy_large": 1_094_189_056,
}


def nested_get(root: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def resident_maps(value: Any) -> int:
    if value is None or str(value).strip().lower() == "auto":
        return 2
    try:
        return max(1, int(str(value)))
    except ValueError:
        return 2


def nvidia_smi_memory_mb() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return int(lines[0].strip())
    except ValueError:
        return None


def parse_meminfo_mb(text: str) -> dict[str, int]:
    """Parse kB-valued Linux meminfo fields into whole MiB."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.split()
        if not parts:
            continue
        try:
            amount = float(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else "bytes"
        if unit == "kb":
            amount /= 1024.0
        elif unit == "bytes":
            amount /= 1024.0**2
        elif unit != "mb":
            continue
        values[key] = int(amount)
    return values


def proc_meminfo_mb(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    try:
        return parse_meminfo_mb(path.read_text(errors="replace"))
    except OSError:
        return {}


def is_jetson_platform() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    for path in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            model = path.read_text(errors="ignore").replace("\x00", "").lower()
        except OSError:
            continue
        if "nvidia jetson" in model or "tegra" in model:
            return True
    return False


def gpu_memory_info() -> dict[str, Any]:
    """Report dedicated VRAM or Jetson's shared system-memory budget."""
    jetson = is_jetson_platform()
    nvidia_mb = nvidia_smi_memory_mb()
    meminfo = proc_meminfo_mb() if jetson else {}
    if nvidia_mb is not None:
        return {
            "total_mb": nvidia_mb,
            "available_mb": meminfo.get("MemAvailable") if jetson else None,
            "source": "nvidia-smi",
            "shared_with_system": jetson,
        }
    if jetson and "MemTotal" in meminfo:
        return {
            "total_mb": meminfo["MemTotal"],
            "available_mb": meminfo.get("MemAvailable"),
            "source": "/proc/meminfo:MemTotal",
            "shared_with_system": True,
        }
    return {
        "total_mb": None,
        "available_mb": None,
        "source": "unavailable",
        "shared_with_system": False,
    }


def gpu_memory_mb() -> int | None:
    """Backward-compatible total-memory accessor."""
    return gpu_memory_info()["total_mb"]


def existing_disk_path(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def paths_overlap(left: Path, right: Path) -> bool:
    try:
        left_resolved = left.resolve(strict=False)
        right_resolved = right.resolve(strict=False)
        return left_resolved == right_resolved or left_resolved in right_resolved.parents or right_resolved in left_resolved.parents
    except OSError:
        return False


CAPACITY_MIN_FRAMES_PER_SUBMAP = {
    # Conservative full-resolution defaults from observed GPS-SLAM submap growth.
    # The estimate is a guardrail, not a replacement for runtime diagnostics.
    "compact_64k": 45,
    "medium_128k": 90,
    "legacy_large": 180,
}


def estimate_multi_residency(
    cfg: dict[str, Any],
    submap_margin: int,
    capacity_min_frames_per_submap: int | None,
) -> dict[str, Any]:
    reader = nested_get(cfg, ("READER",), {})
    multi = nested_get(cfg, ("PIPE", "TSDF", "multi"), {})
    start = int(reader.get("start_frame", 0))
    end = int(reader.get("end_frame", start - 1))
    frame_step = max(1, int(reader.get("frame_step", 1)))
    frame_count = max(0, (end - start) // frame_step + 1) if end >= start else 0
    max_frames = max(1, int(multi.get("max_frames", frame_count or 1)))
    frame_limited_submaps = 1 if as_bool(multi.get("force_single_submap")) else (
        (frame_count + max_frames - 1) // max_frames + max(0, submap_margin)
    )
    profile = str(multi.get("hash_profile", "compact_64k"))
    boundary_policy = str(multi.get("boundary_policy", "")).lower()
    capacity_limited_submaps = 0
    if not as_bool(multi.get("force_single_submap")) and boundary_policy == "capacity":
        capacity_interval = capacity_min_frames_per_submap
        if capacity_interval is None or capacity_interval <= 0:
            capacity_interval = CAPACITY_MIN_FRAMES_PER_SUBMAP.get(profile, 90)
        capacity_limited_submaps = (frame_count + capacity_interval - 1) // capacity_interval + max(0, submap_margin)
    estimated_submaps = max(frame_limited_submaps, capacity_limited_submaps, 1)
    scene_bytes = HASH_PROFILE_SCENE_BYTES.get(profile, HASH_PROFILE_SCENE_BYTES["legacy_large"])
    resident = resident_maps(multi.get("resident_gpu_maps", "auto"))
    passive = max(0, estimated_submaps - resident)
    paging_storage = str(multi.get("paging_storage", "disk")).lower()
    cpu_cache_mb = max(0, int(multi.get("cpu_page_cache_limit_mb", 0)))
    return {
        "frame_count": frame_count,
        "estimated_submaps": estimated_submaps,
        "frame_limited_submaps": frame_limited_submaps,
        "capacity_limited_submaps": capacity_limited_submaps,
        "hash_profile": profile,
        "boundary_policy": boundary_policy,
        "scene_mb": scene_bytes / 1024**2,
        "resident_maps": resident,
        "passive_maps": passive,
        "paging_storage": paging_storage,
        "cpu_page_cache_limit_mb": cpu_cache_mb,
        "estimated_resident_scene_mb": resident * scene_bytes / 1024**2,
        "estimated_passive_scene_mb": passive * scene_bytes / 1024**2,
    }


def add_issue(issues: list[dict[str, str]], severity: str, key: str, message: str) -> None:
    issues.append({"severity": severity, "key": key, "message": message})


def validate(
    cfg: dict[str, Any],
    config_path: Path,
    submap_margin: int,
    capacity_min_frames_per_submap: int | None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    tsdf_engine = str(nested_get(cfg, ("PIPE", "TSDF", "engine"), "basic"))
    input_mode = str(nested_get(cfg, ("PIPE", "TSDF", "input_mode"), "preload"))
    input_prefetch = as_bool(nested_get(cfg, ("PIPE", "TSDF", "input_prefetch"), False))
    input_tensor_bridge = str(
        nested_get(cfg, ("PIPE", "TSDF", "input_tensor_bridge"), "direct")
    )
    input_tensor_bridge_validate = as_bool(
        nested_get(cfg, ("PIPE", "TSDF", "input_tensor_bridge_validate"), False)
    )
    preload_images = as_bool(nested_get(cfg, ("READER", "preload_images"), False))
    benchmark_protocol = str(
        nested_get(cfg, ("PIPE", "benchmark_protocol"), PROTOCOL_NONE)
    )
    benchmark_role = str(nested_get(cfg, ("PIPE", "benchmark_role"), ""))
    locked_benchmark = benchmark_protocol in PROTOCOLS and benchmark_protocol != PROTOCOL_NONE
    workspace_value = str(nested_get(cfg, ("workspace_dir",), "")).strip()
    workspace = Path(workspace_value).expanduser()

    if not workspace_value:
        add_issue(issues, "error", "workspace_dir", "workspace_dir must not be empty; startup clears it recursively.")
    else:
        resolved_workspace = workspace.resolve(strict=False)
        resolved_config = config_path.resolve(strict=False)
        resolved_cwd = Path.cwd().resolve(strict=False)
        resolved_home = Path.home().resolve(strict=False)
        if resolved_workspace == Path(resolved_workspace.anchor):
            add_issue(issues, "error", "workspace_dir", "Refusing filesystem root; startup clears workspace_dir recursively.")
        elif resolved_workspace == resolved_home or resolved_workspace in resolved_home.parents:
            add_issue(
                issues,
                "error",
                "workspace_dir",
                "Refusing the user home directory or one of its parents; startup clears workspace_dir recursively.",
            )
        if paths_overlap(resolved_workspace, resolved_cwd):
            add_issue(issues, "error", "workspace_dir", "workspace_dir overlaps the current/project directory and could delete source files.")
        if resolved_workspace == resolved_config.parent or resolved_workspace in resolved_config.parents:
            add_issue(issues, "error", "workspace_dir", "workspace_dir contains the active config and would delete it.")
        input_value = str(nested_get(cfg, ("READER", "input_dir"), "")).strip()
        if input_value and paths_overlap(resolved_workspace, Path(input_value).expanduser()):
            add_issue(issues, "error", "workspace_dir", "workspace_dir overlaps READER.input_dir and could delete dataset files.")
        if resolved_workspace.exists() and not resolved_workspace.is_dir():
            add_issue(issues, "error", "workspace_dir", "workspace_dir exists but is not a directory.")

    if locked_benchmark:
        try:
            validate_protocol(cfg, benchmark_protocol, benchmark_role, "unknown")
        except ValueError as exc:
            add_issue(issues, "error", "PIPE.benchmark_protocol", str(exc))
        if input_mode == "preload":
            add_issue(
                issues,
                "info",
                "PIPE.benchmark_protocol",
                "Locked paper-parity protocol intentionally preserves preloaded input; dataset loading is outside the timed frame loop.",
            )
        else:
            add_issue(
                issues,
                "info",
                "PIPE.benchmark_protocol",
                "Locked deployed protocol includes on-demand input and one-frame prefetch in the runtime path.",
            )
    elif input_mode != "on_demand":
        add_issue(issues, "error", "PIPE.TSDF.input_mode", "Thor long runs should use on_demand input.")
    if input_mode == "on_demand" and not input_prefetch:
        add_issue(issues, "warning", "PIPE.TSDF.input_prefetch", "Enable the persistent one-frame prefetch worker to hide decode latency.")
    if input_mode == "on_demand" and input_tensor_bridge != "direct":
        add_issue(
            issues,
            "warning",
            "PIPE.TSDF.input_tensor_bridge",
            "opencv_legacy adds avoidable full-frame conversions; use direct after parity validation.",
        )
    if input_tensor_bridge_validate:
        add_issue(
            issues,
            "warning",
            "PIPE.TSDF.input_tensor_bridge_validate",
            "Bridge validation intentionally executes both conversion paths; disable it for timing runs.",
        )
    schedule_mode = str(
        nested_get(cfg, ("PIPE", "gaussian_schedule", "mode"), "immediate")
    )
    schedule_iters = int(
        nested_get(cfg, ("PIPE", "gaussian_schedule", "iterations_per_frame"), 1)
    )
    local_opt_iters = int(nested_get(cfg, ("PIPE", "local_opt_iters"), 0))
    local_opt_interval = int(nested_get(cfg, ("PIPE", "local_opt_interval"), 1))
    if schedule_mode not in {"immediate", "sliced"}:
        add_issue(issues, "error", "PIPE.gaussian_schedule.mode", "Expected immediate or sliced.")
    if schedule_iters <= 0:
        add_issue(
            issues,
            "error",
            "PIPE.gaussian_schedule.iterations_per_frame",
            "Must be positive.",
        )
    if (
        schedule_mode == "sliced"
        and schedule_iters * max(1, local_opt_interval) < local_opt_iters
    ):
        add_issue(
            issues,
            "error",
            "PIPE.gaussian_schedule",
            "The sliced job cannot finish before the next local optimization trigger.",
        )
    if preload_images and not locked_benchmark:
        add_issue(issues, "error", "READER.preload_images", "Preloading images defeats the embedded memory target.")
    if as_bool(nested_get(cfg, ("PIPE", "save_after_train"), True)) and not locked_benchmark:
        add_issue(issues, "warning", "PIPE.save_after_train", "Saving full checkpoints can create large writes; disable for ATE/FPS smoke tests.")

    start_frame = int(nested_get(cfg, ("READER", "start_frame"), 0))
    end_frame = int(nested_get(cfg, ("READER", "end_frame"), start_frame - 1))
    frame_step = max(1, int(nested_get(cfg, ("READER", "frame_step"), 1)))
    frame_count = max(0, (end_frame - start_frame) // frame_step + 1) if end_frame >= start_frame else 0
    submap_stats_interval = max(
        0, int(nested_get(cfg, ("PIPE", "submap_stats_interval_frames"), 1))
    )
    if frame_count >= 500 and 0 < submap_stats_interval < 5:
        add_issue(
            issues,
            "warning",
            "PIPE.submap_stats_interval_frames",
            "Per-frame, per-submap CSV rows grow with map count; use 5-10 for long throughput runs.",
        )

    if workspace_value:
        usage = shutil.disk_usage(existing_disk_path(workspace))
        free_gb = usage.free / 1024**3
        if free_gb < 20.0:
            add_issue(issues, "warning", "workspace_dir", f"Only {free_gb:.1f} GB free near workspace_dir.")

    residency = {}
    if tsdf_engine == "multi":
        residency = estimate_multi_residency(cfg, submap_margin, capacity_min_frames_per_submap)
        multi = nested_get(cfg, ("PIPE", "TSDF", "multi"), {})
        if as_bool(multi.get("console_diagnostics"), False):
            add_issue(issues, "warning", "PIPE.TSDF.multi.console_diagnostics", "Verbose stdout can stall long embedded runs; use CSV diagnostics.")
        if not as_bool(multi.get("paging"), False):
            add_issue(issues, "error", "PIPE.TSDF.multi.paging", "MultiEngine Thor runs should bound resident SDF maps with paging.")
        if residency["hash_profile"] == "legacy_large":
            add_issue(issues, "warning", "PIPE.TSDF.multi.hash_profile", "legacy_large is a desktop/paper-style profile, not Thor stress-safe.")
        if residency["capacity_limited_submaps"] > residency["frame_limited_submaps"]:
            add_issue(
                issues,
                "warning",
                "PIPE.TSDF.multi.capacity_estimate",
                "Capacity-triggered boundaries may dominate submap count; check runtime submap_stats before long Thor runs.",
            )
        if residency["resident_maps"] < 2:
            add_issue(issues, "error", "PIPE.TSDF.multi.resident_gpu_maps", "Need at least primary + candidate resident maps.")
        if residency["paging_storage"] == "cpu" and residency["cpu_page_cache_limit_mb"] <= 0:
            add_issue(issues, "warning", "PIPE.TSDF.multi.cpu_page_cache_limit_mb", "CPU paging should have an explicit cache limit on shared-memory systems.")
        if (
            residency["paging_storage"] == "cpu"
            and residency["cpu_page_cache_limit_mb"] > 0
            and residency["estimated_passive_scene_mb"] > residency["cpu_page_cache_limit_mb"]
        ):
            add_issue(
                issues,
                "warning",
                "PIPE.TSDF.multi.cpu_page_cache_limit_mb",
                "Estimated passive TSDF pages exceed CPU cache; disk spill/page churn can cause multi-second spikes.",
            )
        if str(multi.get("primary_failure_handoff_raw_gate", "failed_only")) != "failed_only":
            add_issue(issues, "warning", "PIPE.TSDF.multi.primary_failure_handoff_raw_gate", "failed_or_weak can cut easy sequences early; use only for ablations.")

    if as_bool(nested_get(cfg, ("PIPE", "gaussian_submaps", "enable"), False)):
        ownership = str(nested_get(cfg, ("PIPE", "gaussian_submaps", "ownership"), "legacy"))
        frame = str(nested_get(cfg, ("PIPE", "gaussian_submaps", "coordinate_frame"), "global"))
        if tsdf_engine != "multi" or ownership != "tsdf" or frame != "local":
            add_issue(
                issues,
                "error",
                "PIPE.gaussian_submaps",
                "Gaussian submaps require TSDF engine=multi, ownership=tsdf, coordinate_frame=local.",
            )
        if as_bool(nested_get(cfg, ("PIPE", "gaussian_submaps", "debug"), False)):
            add_issue(
                issues,
                "warning",
                "PIPE.gaussian_submaps.debug",
                "Debug evaluation writes stage traces and synchronizes CUDA for timings; disable for pure throughput measurements.",
            )
        if as_bool(
            nested_get(cfg, ("PIPE", "gaussian_submaps", "gaussian_insert_diagnostics"), False)
        ):
            add_issue(
                issues,
                "warning",
                "PIPE.gaussian_submaps.gaussian_insert_diagnostics",
                "Insertion counters require GPU reductions and per-frame event rows; disable for pure throughput measurements.",
            )
        if as_bool(
            nested_get(cfg, ("PIPE", "gaussian_submaps", "render_overlap_diagnostics"), False)
        ):
            add_issue(
                issues,
                "warning",
                "PIPE.gaussian_submaps.render_overlap_diagnostics",
                "Overlap diagnostics add render reductions; enable them only while diagnosing quality/compositing.",
            )

    apply_mode = str(nested_get(cfg, ("PIPE", "pose_graph", "apply_mode"), "disabled"))
    final_candidate_scan = as_bool(
        nested_get(cfg, ("PIPE", "pose_graph", "final_candidate_scan"), False)
    )
    terminal_online_scan = as_bool(
        nested_get(cfg, ("PIPE", "pose_graph", "terminal_online_scan"), False)
    )
    if final_candidate_scan and terminal_online_scan:
        add_issue(
            issues,
            "warning",
            "PIPE.pose_graph.terminal_online_scan",
            "terminal_online_scan is inert while final_candidate_scan is enabled; select one terminal policy.",
        )
    if apply_mode == "loop_candidates" and not as_bool(nested_get(cfg, ("PIPE", "pose_graph", "apply_health_gate"), True), True):
        add_issue(issues, "warning", "PIPE.pose_graph.apply_health_gate", "Loop apply should keep the run-health gate enabled for Thor tests.")

    memory_info = gpu_memory_info()
    gpu_mb = memory_info["total_mb"]
    if memory_info["shared_with_system"]:
        add_issue(
            issues,
            "info",
            "memory_budget",
            "Jetson CPU and GPU share this system-memory budget; it is not dedicated VRAM.",
        )
    if gpu_mb is not None and residency:
        reserve_mb = int(nested_get(cfg, ("PIPE", "TSDF", "multi", "gpu_reserve_mb"), 3000))
        if residency["estimated_resident_scene_mb"] > max(0, gpu_mb - reserve_mb):
            memory_label = (
                "shared CPU/GPU memory"
                if memory_info["shared_with_system"]
                else "GPU memory"
            )
            add_issue(
                issues,
                "warning",
                "PIPE.TSDF.multi.resident_gpu_maps",
                f"Estimated resident TSDF scene memory exceeds {memory_label} after reserve.",
            )

    return {
        "config": str(config_path),
        "tsdf_engine": tsdf_engine,
        "input_mode": input_mode,
        "input_prefetch": input_prefetch,
        "input_tensor_bridge": input_tensor_bridge,
        "submap_stats_interval_frames": submap_stats_interval,
        "workspace_dir": str(workspace),
        "gpu_memory_mb": gpu_mb,
        "gpu_memory_available_mb": memory_info["available_mb"],
        "gpu_memory_source": memory_info["source"],
        "gpu_memory_shared_with_system": memory_info["shared_with_system"],
        "multi_residency_estimate": residency,
        "issues": issues,
        "ok": not any(issue["severity"] == "error" for issue in issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="GPS-SLAM YAML config to validate")
    parser.add_argument(
        "--submap-margin",
        type=int,
        default=4,
        help="Extra logical submaps added to the frame-count estimate for capacity/visibility boundaries.",
    )
    parser.add_argument(
        "--capacity-min-frames-per-submap",
        type=int,
        default=0,
        help=(
            "Override the conservative capacity-boundary estimate. "
            "0 uses the profile default; negative disables the capacity estimate."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit("Config did not parse as a YAML mapping")

    capacity_min_frames = None if args.capacity_min_frames_per_submap == 0 else args.capacity_min_frames_per_submap
    if args.capacity_min_frames_per_submap < 0:
        capacity_min_frames = 10**9
    report = validate(cfg, args.config, args.submap_margin, capacity_min_frames)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Thor preflight: {args.config}")
        print(f"  TSDF engine: {report['tsdf_engine']}")
        print(f"  Input mode: {report['input_mode']}")
        print(
            "  Input path: "
            f"prefetch={int(report['input_prefetch'])}, "
            f"bridge={report['input_tensor_bridge']}"
        )
        print(
            "  Submap stats interval: "
            f"{report['submap_stats_interval_frames']} frame(s)"
        )
        if report["gpu_memory_mb"] is not None:
            if report["gpu_memory_shared_with_system"]:
                available = report["gpu_memory_available_mb"]
                available_text = f", available={available} MB" if available is not None else ""
                print(
                    "  Shared CPU/GPU memory: "
                    f"total={report['gpu_memory_mb']} MB{available_text} "
                    f"(source={report['gpu_memory_source']})"
                )
            else:
                print(
                    f"  GPU memory: {report['gpu_memory_mb']} MB "
                    f"(source={report['gpu_memory_source']})"
                )
        if report["multi_residency_estimate"]:
            estimate = report["multi_residency_estimate"]
            print(
                "  MultiEngine estimate: "
                f"{estimate['estimated_submaps']} submaps, "
                f"{estimate['resident_maps']} resident, "
                f"{estimate['hash_profile']} ({estimate['scene_mb']:.1f} MB/map)"
            )
            if estimate.get("capacity_limited_submaps", 0) > estimate.get("frame_limited_submaps", 0):
                print(
                    "  Capacity estimate: "
                    f"{estimate['capacity_limited_submaps']} submaps "
                    f"vs {estimate['frame_limited_submaps']} from frame limits"
                )
        for issue in report["issues"]:
            print(f"  {issue['severity'].upper()}: {issue['key']}: {issue['message']}")
        print("  Result:", "OK" if report["ok"] else "BLOCKED")
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
