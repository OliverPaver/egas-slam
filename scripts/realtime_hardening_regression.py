#!/usr/bin/env python3
"""Fast regressions for FIFO trace analysis and Thor telemetry parsing."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from analyze_realtime_trace import (
    analyze,
    atomic_json,
    expected_frames_from_run_config,
    load_timing,
    write_trace,
)
from thor_telemetry import integer_deltas, parse_sample


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_value_error(action, message: str) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError(message)


def write_timing(path: Path, services: list[float], frames: list[int] | None = None) -> None:
    frame_ids = frames if frames is not None else list(range(len(services)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sequence_id", "frame_id", "trackingReady_ms", "algorithmicService_ms"),
        )
        writer.writeheader()
        for sequence_id, (frame_id, service) in enumerate(zip(frame_ids, services)):
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "trackingReady_ms": min(5.0, service),
                    "algorithmicService_ms": service,
                }
            )


def test_fixed_and_dataset_fifo() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_timing(root / "timing.csv", [20.0] * 4, [2, 3, 4, 5])
        metadata = {
            "frame_records": [
                {"rgb_timestamp": index / 30.0} for index in range(8)
            ]
        }
        (root / "metadata.json").write_text(json.dumps(metadata))
        trace, summary = analyze(root / "timing.csv", root / "metadata.json", 30.0)
        fixed = summary["arrival_models"]["fixed_rate"]
        dataset = summary["arrival_models"]["dataset_timestamps"]
        require(len(trace) == 8, "both arrival models were not emitted")
        require(fixed["processing_fps"] == 50.0, "processing throughput")
        require(fixed["sustained_30hz_gate"], "healthy fixed-rate stream failed")
        require(dataset["sustained_30hz_gate"], "healthy timestamp stream failed")
        require(fixed["full_latency_p50_ms"] == 20.0, "latency p50")
        write_trace(root / "realtime_trace.csv", trace)
        atomic_json(root / "realtime_summary.json", summary)
        require((root / "realtime_trace.csv").is_file(), "trace output missing")
        require((root / "realtime_summary.json").is_file(), "summary output missing")
        (root / "run.yaml").write_text(
            "READER:\n  start_frame: 2\n  end_frame: 5\n  frame_step: 1\n"
        )
        require(expected_frames_from_run_config(root) == 4, "config frame count")
        _, incomplete = analyze(root / "timing.csv", None, 30.0, 5)
        require(not incomplete["all_frames_processed"], "missing frame accepted")
        require(
            not incomplete["arrival_models"]["fixed_rate"]["sustained_30hz_gate"],
            "incomplete run passed sustained gate",
        )


def test_overload_and_recovery() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_timing(root / "timing.csv", [40.0] * 20)
        _, summary = analyze(root / "timing.csv", None, 30.0)
        fixed = summary["arrival_models"]["fixed_rate"]
        require(not fixed["sustained_30hz_gate"], "sustained overload passed")
        require(fixed["final_backlog_ms"] > 100.0, "overload backlog not accumulated")

        write_timing(root / "timing.csv", [20.0, 70.0, 10.0, 10.0, 10.0, 10.0])
        _, summary = analyze(root / "timing.csv", None, 30.0)
        fixed = summary["arrival_models"]["fixed_rate"]
        require(fixed["maximum_backlog_ms"] > 0.0, "latency spike not observed")
        require(fixed["final_backlog_ms"] == 0.0, "recoverable spike did not drain")


def test_malformed_timing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "timing.csv"
        path.write_text(
            "sequence_id,frame_id,trackingReady_ms,algorithmicService_ms\n"
            "0,1,5,10\n2,2,5,10\n"
        )
        expect_value_error(lambda: load_timing(path), "non-contiguous sequence accepted")
        path.write_text(
            "sequence_id,frame_id,trackingReady_ms,algorithmicService_ms\n"
            "0,1,11,10\n"
        )
        expect_value_error(lambda: load_timing(path), "pose time beyond service accepted")


def test_telemetry_parsing() -> None:
    sample = parse_sample(
        "RAM 100/1000MB cpu@41.5C gpu@44C VDD_GPU 7000mW/6500mW VIN 30000mW/29000mW",
        250.0,
    )
    require(sample["ram_used_mb"] == 100, "RAM parse")
    require(sample["temp_gpu_c"] == 44.0, "temperature parse")
    require(sample["power_vin_mw"] == 30000, "power parse")
    require(integer_deltas({"oc1": "2"}, {"oc1": "2"}) == {"oc1": 0}, "counter delta")


def main() -> None:
    tests = (
        test_fixed_and_dataset_fifo,
        test_overload_and_recovery,
        test_malformed_timing,
        test_telemetry_parsing,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"realtime hardening regression passed ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    main()
