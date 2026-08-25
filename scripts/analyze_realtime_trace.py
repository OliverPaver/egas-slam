#!/usr/bin/env python3
"""Analyze GPS-SLAM frame service times as deterministic FIFO sensor streams."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

import yaml


SERVICE_FIELDS = ("algorithmicService_ms", "total_measured_ms")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def finite_nonnegative(value: str, *, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {field}={value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"row {row_number}: {field} must be finite and non-negative")
    return parsed


def load_timing(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise FileNotFoundError(f"timing CSV not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"timing CSV has no data rows: {path}")

    service_field = next(
        (field for field in SERVICE_FIELDS if field in rows[0]),
        None,
    )
    if service_field is None:
        raise ValueError(
            f"timing CSV needs one of {', '.join(SERVICE_FIELDS)}"
        )

    parsed: list[dict[str, float | int]] = []
    seen_frames: set[int] = set()
    previous_sequence = -1
    for row_number, row in enumerate(rows, start=2):
        try:
            sequence_id = int(row.get("sequence_id", ""))
            frame_id = int(row.get("frame_id", ""))
        except ValueError as exc:
            raise ValueError(f"row {row_number}: invalid sequence_id/frame_id") from exc
        if sequence_id != previous_sequence + 1:
            raise ValueError(
                f"row {row_number}: non-contiguous sequence_id {sequence_id}; "
                f"expected {previous_sequence + 1}"
            )
        if frame_id in seen_frames:
            raise ValueError(f"row {row_number}: duplicate frame_id {frame_id}")
        seen_frames.add(frame_id)
        previous_sequence = sequence_id
        service_ms = finite_nonnegative(
            row.get(service_field, ""), field=service_field, row_number=row_number
        )
        pose_field = "trackingReady_ms" if "trackingReady_ms" in row else "fusion_ms"
        pose_ms = finite_nonnegative(
            row.get(pose_field, ""), field=pose_field, row_number=row_number
        )
        if pose_ms > service_ms + 1e-6:
            raise ValueError(
                f"row {row_number}: pose-ready time exceeds complete service time"
            )
        parsed.append(
            {
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "service_ms": service_ms,
                "pose_service_ms": pose_ms,
            }
        )
    return parsed


def normalized_arrivals(count: int, rate_hz: float) -> tuple[list[float], list[float]]:
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be finite and positive")
    period = 1000.0 / rate_hz
    return [index * period for index in range(count)], [period] * count


def timestamp_arrivals(
    metadata_path: Path,
    rows: list[dict[str, float | int]],
) -> tuple[list[float], list[float]]:
    metadata = json.loads(metadata_path.read_text())
    records = metadata.get("frame_records")
    if not isinstance(records, list):
        raise ValueError(
            "metadata frame_records must be a list"
        )
    frame_ids = [int(row["frame_id"]) for row in rows]
    if not frame_ids or min(frame_ids) < 0 or max(frame_ids) >= len(records):
        raise ValueError(
            f"timing frame_id range cannot be resolved against {len(records)} metadata records"
        )
    timestamps: list[float] = []
    for index, frame_id in enumerate(frame_ids):
        record = records[frame_id]
        if not isinstance(record, dict) or "rgb_timestamp" not in record:
            raise ValueError(f"metadata frame_records[{frame_id}] lacks rgb_timestamp")
        timestamp = float(record["rgb_timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError(f"metadata frame_records[{frame_id}] has non-finite timestamp")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError("metadata RGB timestamps must be strictly increasing")
        timestamps.append(timestamp)
    origin = timestamps[0]
    arrivals = [(value - origin) * 1000.0 for value in timestamps]
    intervals = [
        arrivals[index + 1] - arrivals[index]
        for index in range(len(arrivals) - 1)
    ]
    nominal = statistics.median(intervals) if intervals else 1000.0 / 30.0
    deadlines = intervals + [nominal]
    return arrivals, deadlines


def busy_period_recovery(backlogs: Iterable[float], finishes: list[float]) -> float:
    maximum = 0.0
    start_finish: float | None = None
    for backlog, finish in zip(backlogs, finishes):
        if backlog > 1e-9 and start_finish is None:
            start_finish = finish
        elif backlog <= 1e-9 and start_finish is not None:
            maximum = max(maximum, finish - start_finish)
            start_finish = None
    if start_finish is not None:
        maximum = max(maximum, finishes[-1] - start_finish)
    return maximum


def simulate(
    rows: list[dict[str, float | int]],
    arrivals: list[float],
    deadlines: list[float],
    mode: str,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | bool | str]]:
    if len(rows) != len(arrivals) or len(rows) != len(deadlines):
        raise ValueError("arrival/deadline lengths must match timing rows")
    finish_previous = 0.0
    trace: list[dict[str, float | int | str]] = []
    full_latencies: list[float] = []
    pose_latencies: list[float] = []
    services: list[float] = []
    backlog_ms_values: list[float] = []
    finishes: list[float] = []
    full_misses = 0
    pose_misses = 0

    for index, (row, arrival_ms, deadline_ms) in enumerate(
        zip(rows, arrivals, deadlines)
    ):
        service_ms = float(row["service_ms"])
        pose_service_ms = float(row["pose_service_ms"])
        start_ms = max(arrival_ms, finish_previous)
        pose_finish_ms = start_ms + pose_service_ms
        finish_ms = start_ms + service_ms
        full_latency_ms = finish_ms - arrival_ms
        pose_latency_ms = pose_finish_ms - arrival_ms
        next_deadline_ms = arrival_ms + deadline_ms
        backlog_ms = max(0.0, finish_ms - next_deadline_ms)
        queue_depth = max(0, bisect.bisect_right(arrivals, finish_ms) - index - 1)
        full_miss = full_latency_ms > deadline_ms + 1e-9
        pose_miss = pose_latency_ms > deadline_ms + 1e-9
        full_misses += int(full_miss)
        pose_misses += int(pose_miss)
        services.append(service_ms)
        full_latencies.append(full_latency_ms)
        pose_latencies.append(pose_latency_ms)
        backlog_ms_values.append(backlog_ms)
        finishes.append(finish_ms)
        trace.append(
            {
                "arrival_mode": mode,
                "sequence_id": int(row["sequence_id"]),
                "frame_id": int(row["frame_id"]),
                "arrival_ms": arrival_ms,
                "deadline_ms": deadline_ms,
                "start_ms": start_ms,
                "pose_finish_ms": pose_finish_ms,
                "finish_ms": finish_ms,
                "service_ms": service_ms,
                "pose_latency_ms": pose_latency_ms,
                "full_latency_ms": full_latency_ms,
                "backlog_ms": backlog_ms,
                "queue_depth_frames": queue_depth,
                "pose_deadline_miss": int(pose_miss),
                "full_deadline_miss": int(full_miss),
            }
        )
        finish_previous = finish_ms

    count = len(rows)
    nominal_period = statistics.median(deadlines)
    final_horizon = arrivals[-1] + deadlines[-1]
    final_backlog_ms = max(0.0, finishes[-1] - final_horizon)
    processing_fps = 1000.0 * count / sum(services) if sum(services) else 0.0
    summary: dict[str, float | int | bool | str] = {
        "arrival_mode": mode,
        "frame_count": count,
        "frame_continuity": True,
        "processing_fps": processing_fps,
        "nominal_rate_hz": 1000.0 / nominal_period,
        "service_p50_ms": percentile(services, 0.50),
        "service_p95_ms": percentile(services, 0.95),
        "service_p99_ms": percentile(services, 0.99),
        "service_max_ms": max(services),
        "pose_latency_p50_ms": percentile(pose_latencies, 0.50),
        "pose_latency_p95_ms": percentile(pose_latencies, 0.95),
        "pose_latency_p99_ms": percentile(pose_latencies, 0.99),
        "pose_latency_max_ms": max(pose_latencies),
        "full_latency_p50_ms": percentile(full_latencies, 0.50),
        "full_latency_p95_ms": percentile(full_latencies, 0.95),
        "full_latency_p99_ms": percentile(full_latencies, 0.99),
        "full_latency_max_ms": max(full_latencies),
        "pose_deadline_misses": pose_misses,
        "pose_deadline_miss_rate": pose_misses / count,
        "full_deadline_misses": full_misses,
        "full_deadline_miss_rate": full_misses / count,
        "maximum_backlog_ms": max(backlog_ms_values),
        "maximum_queue_depth_frames": max(
            int(row["queue_depth_frames"]) for row in trace
        ),
        "final_backlog_ms": final_backlog_ms,
        "maximum_recovery_ms": busy_period_recovery(backlog_ms_values, finishes),
    }
    summary["sustained_30hz_gate"] = bool(
        processing_fps >= 30.0
        and final_backlog_ms <= 1000.0 / 30.0 + 1e-9
        and float(summary["maximum_backlog_ms"]) <= 500.0 + 1e-9
        and float(summary["maximum_recovery_ms"]) <= 1000.0 + 1e-9
    )
    return trace, summary


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_trace(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze(
    timing_path: Path,
    metadata_path: Path | None,
    rate_hz: float,
    expected_frames: int | None = None,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    rows = load_timing(timing_path)
    trace: list[dict[str, float | int | str]] = []
    summaries: dict[str, object] = {}
    arrivals, deadlines = normalized_arrivals(len(rows), rate_hz)
    mode_trace, mode_summary = simulate(rows, arrivals, deadlines, f"fixed_{rate_hz:g}hz")
    trace.extend(mode_trace)
    summaries["fixed_rate"] = mode_summary
    if metadata_path is not None:
        arrivals, deadlines = timestamp_arrivals(metadata_path, rows)
        mode_trace, mode_summary = simulate(rows, arrivals, deadlines, "dataset_rgb_timestamps")
        trace.extend(mode_trace)
        summaries["dataset_timestamps"] = mode_summary
    expected = len(rows) if expected_frames is None else expected_frames
    if expected <= 0:
        raise ValueError("expected_frames must be positive")
    all_frames_processed = len(rows) == expected
    if not all_frames_processed:
        for values in summaries.values():
            if isinstance(values, dict):
                values["sustained_30hz_gate"] = False
    summary = {
        "schema_version": 1,
        "timing_path": str(timing_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path else None,
        "normalized_rate_hz": rate_hz,
        "expected_frame_count": expected,
        "processed_frame_count": len(rows),
        "all_frames_processed": all_frames_processed,
        "arrival_models": summaries,
    }
    return trace, summary


def expected_frames_from_run_config(run_dir: Path) -> int | None:
    configs = sorted(run_dir.glob("*.yaml"))
    if len(configs) != 1:
        return None
    payload = yaml.safe_load(configs[0].read_text())
    reader = payload.get("READER", {}) if isinstance(payload, dict) else {}
    try:
        start = int(reader["start_frame"])
        end = int(reader["end_frame"])
        step = int(reader.get("frame_step", 1))
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end < start or step <= 0:
        return None
    return (end - start) // step + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=0,
        help="Expected processed frames; 0 derives it from the run YAML when available",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    trace_path = args.trace_output or run_dir / "realtime_trace.csv"
    summary_path = args.summary_output or run_dir / "realtime_summary.json"
    expected_frames = args.expected_frames or expected_frames_from_run_config(run_dir)
    trace, summary = analyze(
        run_dir / "timing.csv",
        args.metadata,
        args.rate_hz,
        expected_frames,
    )
    write_trace(trace_path, trace)
    atomic_json(summary_path, summary)
    fixed = summary["arrival_models"]["fixed_rate"]
    print(
        "Realtime trace: "
        f"fps={fixed['processing_fps']:.2f} "
        f"max_backlog={fixed['maximum_backlog_ms']:.2f} ms "
        f"final_backlog={fixed['final_backlog_ms']:.2f} ms "
        f"gate={'PASS' if fixed['sustained_30hz_gate'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
