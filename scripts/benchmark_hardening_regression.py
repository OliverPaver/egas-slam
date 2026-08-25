#!/usr/bin/env python3
"""Fast regressions for locked benchmark protocols and timing accounting."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
from pathlib import Path

import yaml
import benchmark_protocol

from benchmark_protocol import (
    REPLICA_DEPLOYED_QUALITY_V1,
    REPLICA_PAPER_V1,
    TUM_DEPLOYED_V1,
    TUM_INTEGRATED_V1,
    TUM_TRACKING_V1,
    apply_protocol,
    dataset_fingerprint,
    protocol_release_status,
    replica_role_for_label,
    tum_role_for_variant,
    validate_protocol,
    write_manifest,
)
from summarize_tum_runs import add_terminal_inclusive_timing, parse_pose_graph, parse_timing


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_value_error(action, message: str) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError(message)


def base_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/release/replica/office0.yaml").read_text()
    )


def test_replica_roles_and_locks() -> None:
    require(protocol_release_status(REPLICA_PAPER_V1) == "publication_locked", "paper protocol status")
    require(
        protocol_release_status(REPLICA_DEPLOYED_QUALITY_V1) == "unpromoted_screen_only",
        "rejected deployment status",
    )
    require(replica_role_for_label("basic_unified_gaussian") == "baseline", "baseline label")
    require(replica_role_for_label("multi_hybrid_warm_fill") == "modified", "modified label")
    expect_value_error(lambda: replica_role_for_label("basic_sdf"), "unlocked Replica label accepted")

    baseline = apply_protocol(base_config(), REPLICA_PAPER_V1, "baseline", "legacy_large")
    require(baseline["READER"]["image_shape"] == [1200, 680], "Replica resolution not locked")
    require(baseline["READER"]["end_frame"] == 1999, "Replica frame range not locked")
    require(baseline["PIPE"]["TSDF"]["engine"] == "basic", "baseline engine changed")
    require(not baseline["PIPE"]["gaussian_submaps"]["enable"], "legacy Gaussian mode changed")
    require(not baseline["PIPE"]["gaussian_submaps"]["quality_finalize_enable"], "offline refinement enabled")
    expect_value_error(
        lambda: validate_protocol(baseline, REPLICA_PAPER_V1, "baseline", "medium_128k"),
        "reduced-capacity baseline binary accepted",
    )

    modified = apply_protocol(base_config(), REPLICA_PAPER_V1, "modified", "medium_128k")
    require(modified["PIPE"]["TSDF"]["engine"] == "multi", "modified engine not locked")
    require(modified["PIPE"]["gaussian_submaps"]["enable"], "Gaussian submaps not enabled")
    require(modified["PIPE"]["gaussian_submaps"]["ownership"] == "tsdf", "ownership not locked")
    require(not modified["PIPE"]["gaussian_submaps"]["debug"], "debug timing contamination")
    require(modified["PIPE"]["TSDF"]["multi"]["hash_profile"] == "medium_128k", "hash mismatch")
    require(modified["PIPE"]["weight_configs"]["ssim_weight"] == 0.0, "paper loss changed")
    require(modified["PIPE"]["gaussian_schedule"]["mode"] == "immediate", "paper scheduling changed")

    deployed = apply_protocol(base_config(), REPLICA_DEPLOYED_QUALITY_V1, "modified", "medium_128k")
    require(not deployed["READER"]["preload_images"], "deployed Replica preload enabled")
    require(deployed["PIPE"]["gaussian_schedule"] == {"mode": "sliced", "iterations_per_frame": 2}, "deployed Replica scheduler")
    require(deployed["PIPE"]["TSDF"]["multi"]["resident_gpu_maps"] == 8, "deployed Replica paging")
    require(not deployed["PIPE"]["gaussian_submaps"]["quality_finalize_enable"], "deployed Replica offline work")


def test_tum_roles_and_locks() -> None:
    require(tum_role_for_variant("basic_sdf", TUM_TRACKING_V1) == "baseline", "TUM baseline role")
    require(tum_role_for_variant("multi_sdf_orb_apply", TUM_TRACKING_V1) == "modified", "TUM method role")
    require(tum_role_for_variant("multi_hybrid_orb_apply", TUM_INTEGRATED_V1) == "integrated", "integrated role")

    modified = base_config()
    modified["PIPE"]["pose_graph"]["candidate_detection"] = True
    modified["PIPE"]["pose_graph"]["apply_mode"] = "loop_candidates"
    modified = apply_protocol(modified, TUM_TRACKING_V1, "modified", "medium_128k")
    require(modified["work_mode"] == "recon", "tracking benchmark creates Gaussians")
    require(not modified["PIPE"]["TSDF"]["use_gt_pose"], "TUM used GT poses")
    require(not modified["PIPE"]["gaussian_submaps"]["enable"], "TUM tracking used Gaussians")

    integrated = copy.deepcopy(modified)
    integrated = apply_protocol(integrated, TUM_INTEGRATED_V1, "integrated", "medium_128k")
    require(integrated["work_mode"] == "train", "integrated benchmark does not train")
    require(integrated["PIPE"]["gaussian_submaps"]["enable"], "integrated Gaussians disabled")
    require(not integrated["PIPE"]["gaussian_submaps"]["frozen_refinement_enable"], "offline refinement enabled")
    require(integrated["PIPE"]["gaussian_schedule"]["mode"] == "immediate", "parity scheduling changed")

    deployed = copy.deepcopy(integrated)
    deployed = apply_protocol(deployed, TUM_DEPLOYED_V1, "integrated", "medium_128k")
    require(deployed["PIPE"]["TSDF"]["tracking_profile"] == "robust_rpe", "deployed tracker")
    require(deployed["PIPE"]["TSDF"]["tracking_failure_mode"] == "contain_failed", "deployed failure policy")
    require(deployed["PIPE"]["pose_graph"]["candidate_interval_frames"] == 60, "deployed loop cadence")
    require(deployed["PIPE"]["pose_graph"]["max_scanned_pairs_per_check"] == 8, "deployed scan budget")
    require(not deployed["PIPE"]["pose_graph"]["final_candidate_scan"], "deployed terminal scan")
    require(deployed["PIPE"]["export_online_end_trajectory"], "online trajectory export")
    require(deployed["PIPE"]["TSDF"]["multi"]["async_page_out"], "deployed async page-out")
    require(deployed["PIPE"]["TSDF"]["multi"]["async_page_in"], "deployed async page-in")
    tampered = copy.deepcopy(deployed)
    tampered["PIPE"]["gaussian_schedule"]["iterations_per_frame"] = 1
    expect_value_error(
        lambda: validate_protocol(tampered, TUM_DEPLOYED_V1, "integrated", "medium_128k"),
        "tampered deployed scheduler accepted",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_terminal_inclusive_timing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        write_csv(
            run_dir / "timing.csv",
            [
                {"total_measured_ms": 10.0, "poseGraph_ms": 0.0},
                {"total_measured_ms": 20.0, "poseGraph_ms": 4.0},
            ],
        )
        write_csv(
            run_dir / "pose_graph_live/updates.csv",
            [
                {
                    "source": "online_interval",
                    "update_duration_ms": 4.0,
                    "candidate_scan_ms": 3.0,
                    "apply_ms": 1.0,
                    "loop_scanned_pairs": 2,
                    "loop_verified_pairs": 1,
                    "loop_accepted": 0,
                    "applied": 0,
                },
                {
                    "source": "live_end_of_sequence",
                    "update_duration_ms": 30.0,
                    "candidate_scan_ms": 20.0,
                    "apply_ms": 8.0,
                    "loop_scanned_pairs": 100,
                    "loop_verified_pairs": 3,
                    "loop_accepted": 1,
                    "applied": 1,
                },
            ],
        )
        row: dict[str, float | int | str] = {}
        row.update(parse_timing(run_dir))
        row.update(parse_pose_graph(run_dir))
        add_terminal_inclusive_timing(row)
        require(row["timing_frame_count"] == 2, "frame count missing")
        require(row["total_frame_ms_sum"] == 30.0, "frame time sum incorrect")
        require(row["pose_graph_update_total_ms"] == 34.0, "update diagnostics sum incorrect")
        require(row["terminal_pose_graph_ms"] == 30.0, "terminal work not isolated")
        require(row["end_to_end_sequence_ms"] == 60.0, "terminal work double-counted or omitted")
        require(abs(float(row["steady_state_fps"]) - 66.6666667) < 1e-5, "steady FPS")
        require(abs(float(row["end_to_end_fps"]) - 33.3333333) < 1e-5, "end-to-end FPS")
        require(
            abs(float(row["terminal_inclusive_pipeline_fps"]) - 33.3333333) < 1e-5,
            "terminal-inclusive fallback FPS",
        )


def test_dataset_fingerprints() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "a.bin").write_bytes(b"abcd")
        (root / "sub").mkdir()
        (root / "sub/b.bin").write_bytes(b"12")
        inventory_a = dataset_fingerprint(root, "inventory")
        inventory_b = dataset_fingerprint(root, "inventory")
        full_a = dataset_fingerprint(root, "full")
        require(inventory_a == inventory_b, "inventory fingerprint is nondeterministic")
        require(inventory_a["file_count"] == 2 and inventory_a["total_bytes"] == 6, "inventory counts")
        (root / "a.bin").write_bytes(b"wxyz")
        require(dataset_fingerprint(root, "inventory")["sha256"] == inventory_a["sha256"], "inventory hashes content")
        require(dataset_fingerprint(root, "full")["sha256"] != full_a["sha256"], "full hash missed content change")


def test_manifest_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        dataset.mkdir()
        (dataset / "frame.bin").write_bytes(b"frame")
        cfg = apply_protocol(base_config(), REPLICA_PAPER_V1, "baseline", "legacy_large")
        cfg_path = root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        output = root / "benchmark_manifest.json"
        original_freeze = benchmark_protocol.benchmark_freeze_provenance
        benchmark_protocol.benchmark_freeze_provenance = lambda: {
            "freeze_id": "regression-fixture",
            "manifest_sha256": "0" * 64,
            "functional_digest": "1" * 64,
            "verified_clean": True,
        }
        try:
            write_manifest(
                output,
                protocol=REPLICA_PAPER_V1,
                role="baseline",
                cfg=cfg,
                config_path=cfg_path,
                binary=None,
                dataset_root=dataset,
                dataset_hash_mode="inventory",
                extra={"state": "regression"},
            )
        finally:
            benchmark_protocol.benchmark_freeze_provenance = original_freeze
        manifest = json.loads(output.read_text())
        require(manifest["protocol"] == REPLICA_PAPER_V1, "manifest protocol")
        require(manifest["protocol_release_status"] == "publication_locked", "manifest protocol status")
        require(manifest["code_freeze"]["verified_clean"], "manifest freeze verification")
        require(manifest["dataset"]["file_count"] == 1, "manifest dataset")
        require(manifest["run"]["state"] == "regression", "manifest run metadata")
        require(bool(manifest["runtime"]["captured_utc"]), "manifest runtime provenance")


def test_tum_safety_uses_locked_configuration() -> None:
    source = (PROJECT_ROOT / "run/tum_loop_validation.py").read_text()
    protocol_call = source.rfind("                    apply_protocol(\n")
    safety_call = source.rfind("                enforce_multi_run_safety(cfg, seq, variant, args)\n")
    require(protocol_call >= 0, "TUM protocol application call missing")
    require(safety_call >= 0, "TUM resource-safety call missing")
    require(
        protocol_call < safety_call,
        "TUM resource safety ran before the locked protocol finalized memory settings",
    )


def main() -> None:
    tests = (
        test_replica_roles_and_locks,
        test_tum_roles_and_locks,
        test_terminal_inclusive_timing,
        test_dataset_fingerprints,
        test_manifest_round_trip,
        test_tum_safety_uses_locked_configuration,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"benchmark hardening regression passed ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    main()
