#!/usr/bin/env python3
"""Run a command while recording Thor power, thermal, clock, and throttle state."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


OC_COUNTER_GLOB = "/sys/devices/platform/nvtherm-oc-event/hwmon/hwmon*/oc*_event_cnt"
OC_ENABLE_GLOB = "/sys/devices/platform/nvtherm-oc-event/hwmon/hwmon*/oc*_throt_en"
FAN_GLOBS = (
    "/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1",
    "/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1_enable",
)
THERMAL_GLOB = "/sys/devices/virtual/thermal/thermal_zone*"
COOLING_GLOB = "/sys/devices/virtual/thermal/cooling_device*"
TEMP_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@([0-9.]+)C\b")
POWER_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+([0-9]+)mW/([0-9]+)mW\b")
RAM_RE = re.compile(r"\bRAM\s+([0-9]+)/([0-9]+)MB\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip().rstrip("\x00")
    except OSError:
        return ""


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip()


def clock_state() -> dict[str, object]:
    groups: list[dict[str, str]] = []
    for cpu_root in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq")):
        groups.append(
            {
                "name": cpu_root.parent.name,
                "kind": "cpu",
                "governor": read_text(cpu_root / "scaling_governor"),
                "current_hz": read_text(cpu_root / "scaling_cur_freq"),
                "min_hz": read_text(cpu_root / "scaling_min_freq"),
                "max_hz": read_text(cpu_root / "scaling_max_freq"),
            }
        )
    for gpu_root in sorted(Path("/sys/class/devfreq").glob("gpu-*")):
        groups.append(
            {
                "name": gpu_root.name,
                "kind": "gpu",
                "governor": read_text(gpu_root / "governor"),
                "current_hz": read_text(gpu_root / "cur_freq"),
                "min_hz": read_text(gpu_root / "min_freq"),
                "max_hz": read_text(gpu_root / "max_freq"),
            }
        )
    constrained = [group for group in groups if group["min_hz"] and group["max_hz"]]
    return {
        "groups": groups,
        "all_reported_clocks_locked": bool(constrained)
        and all(group["min_hz"] == group["max_hz"] for group in constrained),
    }


def sysfs_values(patterns: tuple[str, ...] | list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pattern in patterns:
        for item in sorted(glob.glob(pattern)):
            path = Path(item)
            values[str(path)] = read_text(path)
    return values


def thermal_values() -> dict[str, object]:
    values: dict[str, object] = {}
    for item in sorted(glob.glob(THERMAL_GLOB)):
        root = Path(item)
        name = read_text(root / "type") or root.name
        raw = read_text(root / "temp")
        try:
            values[name] = float(raw) / 1000.0
        except ValueError:
            values[name] = raw
    return values


def cooling_states(*, sample_keys: bool = False) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for item in sorted(glob.glob(COOLING_GLOB)):
        root = Path(item)
        name = read_text(root / "type") or root.name
        raw = read_text(root / "cur_state")
        key = (
            "cooling_"
            + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            + "_state"
            if sample_keys
            else f"{root}:{name}"
        )
        try:
            values[key] = int(raw)
        except ValueError:
            values[key] = raw
    return values


def snapshot() -> dict[str, object]:
    return {
        "captured_utc": utc_now(),
        "nvpmodel": command_output(["nvpmodel", "-q"]),
        "jetson_clocks": command_output(["jetson_clocks", "--show"]),
        "clock_state": clock_state(),
        "over_current_event_counters": sysfs_values([OC_COUNTER_GLOB]),
        "over_current_throttle_enable": sysfs_values([OC_ENABLE_GLOB]),
        "fan": sysfs_values(list(FAN_GLOBS)),
        "thermal_c": thermal_values(),
        "thermal_cooling_state": cooling_states(),
    }


def integer_deltas(before: dict[str, str], after: dict[str, str]) -> dict[str, int | None]:
    deltas: dict[str, int | None] = {}
    for path in sorted(set(before) | set(after)):
        try:
            deltas[path] = int(after[path]) - int(before[path])
        except (KeyError, TypeError, ValueError):
            deltas[path] = None
    return deltas


def parse_sample(line: str, elapsed_ms: float) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_elapsed_ms": round(elapsed_ms, 3),
        "raw": line.rstrip(),
    }
    ram = RAM_RE.search(line)
    if ram:
        row["ram_used_mb"] = int(ram.group(1))
        row["ram_total_mb"] = int(ram.group(2))
    for name, value in TEMP_RE.findall(line):
        row[f"temp_{name.lower()}_c"] = float(value)
    for name, current, average in POWER_RE.findall(line):
        key = name.lower()
        row[f"power_{key}_mw"] = int(current)
        row[f"power_{key}_avg_mw"] = int(average)
    return row


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_samples(path: Path, samples: list[dict[str, object]]) -> None:
    fields = ["sample_elapsed_ms", "ram_used_mb", "ram_total_mb"]
    dynamic = sorted({key for sample in samples for key in sample if key not in fields and key != "raw"})
    fields.extend(dynamic)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    temporary.replace(path)


def aggregate_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"sample_count": len(samples)}
    numeric_keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if key not in {"sample_elapsed_ms", "raw"} and isinstance(value, (int, float))
        }
    )
    for key in numeric_keys:
        values = [float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_max"] = max(values)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="return exit 3 when performance state or throttle counters invalidate the run",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.interval_ms < 100:
        parser.error("--interval-ms must be at least 100")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    samples: list[dict[str, object]] = []
    raw_path = output_dir / "tegrastats.log"
    start = time.monotonic()
    tegrastats = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def collect() -> None:
        assert tegrastats.stdout is not None
        with raw_path.open("w", encoding="utf-8") as raw:
            for line in tegrastats.stdout:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                raw.write(f"{elapsed_ms:.3f},{line}")
                raw.flush()
                sample = parse_sample(line, elapsed_ms)
                sample.update(cooling_states(sample_keys=True))
                samples.append(sample)

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    command_start_utc = utc_now()
    try:
        child = subprocess.Popen(command)
        child_returncode = child.wait()
    except BaseException:
        if "child" in locals() and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        raise
    finally:
        if tegrastats.poll() is None:
            tegrastats.send_signal(signal.SIGTERM)
        try:
            tegrastats.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegrastats.kill()
            tegrastats.wait()
        collector.join(timeout=5)

    after = snapshot()
    deltas = integer_deltas(
        before["over_current_event_counters"],  # type: ignore[arg-type]
        after["over_current_event_counters"],  # type: ignore[arg-type]
    )
    nvpmodel_ok = "MAXN" in str(before["nvpmodel"]).upper() and "MAXN" in str(after["nvpmodel"]).upper()
    clocks_ok = bool(before["clock_state"]["all_reported_clocks_locked"]) and bool(  # type: ignore[index]
        after["clock_state"]["all_reported_clocks_locked"]  # type: ignore[index]
    )
    fan_before = before["fan"]
    fan_after = after["fan"]
    fan_pwm_values = [
        int(value)
        for path, value in {**fan_before, **fan_after}.items()  # type: ignore[union-attr]
        if path.endswith("/pwm1") and str(value).isdigit()
    ]
    fan_max_ok = bool(fan_pwm_values) and min(fan_pwm_values) >= 255
    throttle_clean = bool(deltas) and all(value == 0 for value in deltas.values())
    thermal_state_values: list[int] = []
    for state in (before["thermal_cooling_state"], after["thermal_cooling_state"]):
        thermal_state_values.extend(
            int(value)
            for path, value in state.items()  # type: ignore[union-attr]
            if "pwm-fan" not in path and isinstance(value, int)
        )
    thermal_sample_values = [
        int(value)
        for sample in samples
        for key, value in sample.items()
        if key.startswith("cooling_")
        and not key.startswith("cooling_pwm_fan_")
        and isinstance(value, int)
    ]
    thermal_clean = bool(thermal_state_values) and all(
        value == 0 for value in thermal_state_values + thermal_sample_values
    )
    valid = (
        nvpmodel_ok
        and clocks_ok
        and fan_max_ok
        and throttle_clean
        and thermal_clean
        and child_returncode == 0
    )
    summary = {
        "schema_version": 1,
        "command": command,
        "command_start_utc": command_start_utc,
        "command_end_utc": utc_now(),
        "command_elapsed_ms": (time.monotonic() - start) * 1000.0,
        "command_returncode": child_returncode,
        "before": before,
        "after": after,
        "over_current_event_counter_deltas": deltas,
        "nvpmodel_maxn": nvpmodel_ok,
        "clocks_locked": clocks_ok,
        "fan_max": fan_max_ok,
        "throttle_counters_unchanged": throttle_clean,
        "thermal_throttle_clean": thermal_clean,
        "run_valid_for_headline": valid,
        "tegrastats": aggregate_samples(samples),
    }
    write_samples(output_dir / "tegrastats.csv", samples)
    atomic_json(output_dir / "thor_telemetry_summary.json", summary)
    print(
        "Thor telemetry: "
        f"samples={len(samples)} maxn={nvpmodel_ok} clocks={clocks_ok} "
        f"fan_max={fan_max_ok} throttle_clean={throttle_clean} thermal_clean={thermal_clean} "
        f"valid={'yes' if valid else 'no'}"
    )
    if child_returncode != 0:
        return child_returncode
    if args.require_clean and not valid:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
