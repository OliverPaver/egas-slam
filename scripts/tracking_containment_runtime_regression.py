#!/usr/bin/env python3
"""Validate a short run made with ``tracker_config: type=forcefail``."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"missing regression artifact: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_run(run_dir: Path, mode: str) -> None:
    tracking = rows(run_dir / "tracking.csv")
    if not tracking:
        raise AssertionError(f"no tracking rows in {run_dir}")

    robust = mode in {"contain_failed", "stop_integration"}
    expected_policy = 1 if robust else 2
    expected_flag = "1" if robust else "0"
    for index, row in enumerate(tracking):
        if row.get("raw_primary_tracking_result") != "0":
            raise AssertionError(f"row {index}: forcefail did not report raw FAILED")
        if row.get("policy_primary_tracking_result") != str(expected_policy):
            raise AssertionError(f"row {index}: unexpected policy result")
        if row.get("containment_pose_rollback") != expected_flag:
            raise AssertionError(f"row {index}: unexpected rollback flag")
        if row.get("containment_fusion_suppressed") != expected_flag:
            raise AssertionError(f"row {index}: unexpected fusion suppression flag")

    backend = tracking[0].get("backend")
    diagnostics_path = run_dir / "basic_tracking_diagnostics.csv"
    if backend == "basic" and diagnostics_path.exists():
        prepared = [row for row in rows(diagnostics_path) if row.get("stage") == "post_prepare"]
        if len(prepared) != len(tracking):
            raise AssertionError("basic diagnostics do not cover every forcefail frame")
        expected_fusion = "0" if robust else "1"
        if any(row.get("did_fusion") != expected_fusion for row in prepared):
            raise AssertionError("BasicEngine fusion behavior does not match the configured mode")

    if backend == "multi":
        stats = rows(run_dir / "submap_stats.csv")
        block_counts = [int(row["blocks"]) for row in stats if int(row["blocks"]) >= 0]
        if robust and any(block_counts):
            raise AssertionError("MultiEngine allocated TSDF blocks during contained failures")
        if not robust and (not block_counts or max(block_counts) <= 0):
            raise AssertionError("legacy baseline no longer fuses forcefail frames")

    print(f"runtime containment regression passed: {run_dir} ({mode}, {backend})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["legacy_ignore", "contain_failed", "stop_integration"], required=True
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        check_run(run_dir, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
