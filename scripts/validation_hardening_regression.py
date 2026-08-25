#!/usr/bin/env python3
"""Lightweight regression checks for validation-runner safety semantics."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import hybrid_validation  # noqa: E402
from run import tum_loop_validation  # noqa: E402
from scripts import summarize_tum_runs  # noqa: E402
from scripts import thor_preflight  # noqa: E402
from scripts import benchmark_protocol  # noqa: E402


def expect_value_error(callback, description: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {description}")


def main() -> int:
    for module in (hybrid_validation, tum_loop_validation):
        assert module.validate_path_token("thor_run-01.v2", "test") == "thor_run-01.v2"
        assert module.validate_path_token("", "test", allow_empty=True) == ""
        for unsafe in ("../escape", "nested/name", "nested\\name", "has space", ".hidden"):
            expect_value_error(
                lambda unsafe=unsafe, module=module: module.validate_path_token(unsafe, "test"),
                f"unsafe token {unsafe!r}",
            )

    hybrid_labels = {
        label
        for label, _overrides in hybrid_validation.experiment_matrix(10, True, "fast", "stress")
    }
    selected = hybrid_validation.parse_requested_labels(
        "basic_sdf, multi_hybrid_warm_fill", hybrid_labels
    )
    assert selected == {"basic_sdf", "multi_hybrid_warm_fill"}
    expect_value_error(
        lambda: hybrid_validation.parse_requested_labels("not_a_real_label", hybrid_labels),
        "unknown hybrid label",
    )
    expect_value_error(
        lambda: hybrid_validation.parse_requested_labels("basic_sdf,", hybrid_labels),
        "empty hybrid label",
    )

    assert hybrid_validation.validation_exit_failures([{"name": "ok", "status": "OK"}], {}) == ([], [])
    assert hybrid_validation.validation_exit_failures(
        [{"name": "generated", "status": "GENERATED"}], {}
    ) == ([], [])
    failed_runs, failed_gates = hybrid_validation.validation_exit_failures(
        [{"name": "crashed", "status": "CRASH", "quality_gate": "FAIL"}],
        {"hybrid_quality_gate": "FAIL"},
    )
    assert failed_runs == ["crashed"]
    assert failed_gates == ["crashed:quality_gate"]
    assert hybrid_validation.validation_exit_failures(
        [{"name": "ok", "status": "OK"}], {"multi_sdf_ate_gate": "FAIL"}
    ) == ([], ["multi_sdf_ate_gate"])

    tum_loop_validation.validate_known_labels(["fr1_desk"], set(tum_loop_validation.SEQUENCES), "test")
    tum_loop_validation.validate_known_labels(["multi_sdf"], tum_loop_validation.VARIANTS, "test")
    expect_value_error(
        lambda: tum_loop_validation.validate_known_labels(
            ["fr1_desk", "fr1_desk"], set(tum_loop_validation.SEQUENCES), "test"
        ),
        "duplicate TUM sequence",
    )
    expect_value_error(
        lambda: tum_loop_validation.validate_known_labels(
            ["../fr1_desk"], set(tum_loop_validation.SEQUENCES), "test"
        ),
        "unsafe TUM sequence",
    )
    robust_profile = tum_loop_validation.apply_loop_profile(
        {"PIPE": {"pose_graph": {}}}, "recommended_longspan_robust"
    )["PIPE"]["pose_graph"]
    assert robust_profile["final_candidate_scan"] is True
    assert robust_profile["terminal_online_scan"] is False
    assert robust_profile["apply_correction_scale"] == 0.75
    streaming_profile = tum_loop_validation.apply_loop_profile(
        {"PIPE": {"pose_graph": {}}}, "recommended_longspan_robust_streaming"
    )["PIPE"]["pose_graph"]
    assert streaming_profile["candidate_interval_frames"] == 60
    assert streaming_profile["max_scanned_pairs_per_check"] == 8
    assert streaming_profile["max_candidate_pairs_per_check"] == 1
    assert streaming_profile["span_reserve_pairs"] == 1
    assert streaming_profile["orb_feature_cache_limit"] == 512
    assert streaming_profile["run_on_submap_events"] is False
    assert streaming_profile["final_candidate_scan"] is False

    parsed = thor_preflight.parse_meminfo_mb(
        "MemTotal:       131072 kB\nMemAvailable:    65536 kB\n"
    )
    assert parsed == {"MemTotal": 128, "MemAvailable": 64}
    if thor_preflight.is_jetson_platform():
        memory = thor_preflight.gpu_memory_info()
        assert memory["shared_with_system"] is True
        assert memory["total_mb"] is not None

    home_parent_cfg = {
        # HOME is deliberately /tmp in the Thor container, whose parent is the
        # filesystem root and therefore triggers the earlier root-specific
        # guard. The home directory itself exercises the intended invariant in
        # both native and container environments.
        "workspace_dir": str(Path.home()),
        "READER": {
            "input_dir": "/tmp/gps_slam_validation_dataset",
            "start_frame": 0,
            "end_frame": 0,
            "frame_step": 1,
            "preload_images": False,
        },
        "PIPE": {
            "save_after_train": False,
            "submap_stats_interval_frames": 10,
            "TSDF": {
                "engine": "basic",
                "input_mode": "on_demand",
                "input_prefetch": True,
                "input_tensor_bridge": "direct",
            },
        },
    }
    report = thor_preflight.validate(
        home_parent_cfg,
        Path("/tmp/gps_slam_validation_config.yaml"),
        submap_margin=4,
        capacity_min_frames_per_submap=None,
    )
    assert any(
        issue["severity"] == "error"
        and issue["key"] == "workspace_dir"
        and "home directory or one of its parents" in issue["message"]
        for issue in report["issues"]
    )

    locked_cfg = benchmark_protocol.apply_protocol(
        {
            "workspace_dir": "/tmp/gps_slam_locked_workspace",
            "READER": {"input_dir": "/tmp/gps_slam_locked_dataset"},
            "PIPE": {"TSDF": {"multi": {}}, "gaussian_submaps": {}, "pose_graph": {}},
            "MODEL": {},
        },
        benchmark_protocol.REPLICA_PAPER_V1,
        "baseline",
        "legacy_large",
    )
    locked_report = thor_preflight.validate(
        locked_cfg,
        Path("/tmp/gps_slam_locked_config.yaml"),
        submap_margin=4,
        capacity_min_frames_per_submap=None,
    )
    assert not any(
        issue["severity"] == "error"
        and issue["key"] in {"PIPE.TSDF.input_mode", "READER.preload_images"}
        for issue in locked_report["issues"]
    )

    malformed_schedule = {
        "workspace_dir": "/tmp/gps_slam_schedule_workspace",
        "READER": {
            "input_dir": "/tmp/gps_slam_schedule_dataset",
            "start_frame": 0,
            "end_frame": 20,
            "frame_step": 1,
            "preload_images": False,
        },
        "PIPE": {
            "local_opt_iters": 20,
            "local_opt_interval": 10,
            "gaussian_schedule": {"mode": "sliced", "iterations_per_frame": 1},
            "TSDF": {"engine": "basic", "input_mode": "on_demand", "input_prefetch": True},
        },
    }
    malformed_report = thor_preflight.validate(
        malformed_schedule,
        Path("/tmp/gps_slam_schedule_config.yaml"),
        submap_margin=4,
        capacity_min_frames_per_submap=None,
    )
    assert any(
        issue["severity"] == "error" and issue["key"] == "PIPE.gaussian_schedule"
        for issue in malformed_report["issues"]
    )

    valid_rows = [{"run": "valid", "run_valid_for_acceptance": 1}]
    blocked_rows = [
        {"run": "blocked", "run_valid_for_acceptance": 0, "acceptance_blockers": "no_valid_ate"}
    ]
    assert summarize_tum_runs.acceptance_blocked_rows(valid_rows) == []
    assert summarize_tum_runs.acceptance_blocked_rows(blocked_rows) == blocked_rows

    print("validation hardening regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
