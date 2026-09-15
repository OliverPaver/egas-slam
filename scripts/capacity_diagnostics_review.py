#!/usr/bin/env python3
"""Read native allocation diagnostics; gate capacity independently of ATE scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--min-free-fraction", type=float, default=0.10)
    parser.add_argument("--max-failed-fraction", type=float, default=0.05)
    parser.add_argument("--max-failed-streak", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_frames < 1 or args.max_failed_streak < 0 or not 0 <= args.min_free_fraction < 1 or not 0 <= args.max_failed_fraction < 1:
        parser.error("Invalid gate threshold")
    rows = list(csv.DictReader(args.csv.open()))
    reasons = []
    if len(rows) != args.expected_frames:
        reasons.append("incomplete_frame_count")
    if [int(r["frame"]) for r in rows] != list(range(len(rows))):
        reasons.append("non_contiguous_frames")
    metrics = {}
    if rows:
        if len({(r["blocks_capacity"], r["hash_buckets"], r["excess_capacity"]) for r in rows}) != 1:
            reasons.append("capacity_changed_within_run")
        min_blocks = min(int(r["blocks_free"]) / int(r["blocks_capacity"]) for r in rows)
        min_excess = min(int(r["excess_free"]) / int(r["excess_capacity"]) for r in rows)
        failed_voxel = sum(int(r["failed_voxel_requests"]) for r in rows)
        failed_excess = sum(int(r["failed_excess_requests"]) for r in rows)
        failure_count = sum(int(r["raw_result"]) == 0 for r in rows)
        longest = streak = 0
        for row in rows:
            if int(row["raw_result"]) not in (0, 1, 2) or int(row["policy_result"]) not in (0, 1, 2):
                reasons.append("unknown_tracking_result")
            streak = streak + 1 if int(row["raw_result"]) == 0 else 0
            longest = max(longest, streak)
            if int(row["blocks_free"]) + int(row["blocks_used"]) != int(row["blocks_capacity"]):
                reasons.append("inconsistent_capacity_counts")
            if not 0 <= int(row["blocks_free"]) <= int(row["blocks_capacity"]) or not 0 <= int(row["excess_free"]) <= int(row["excess_capacity"]):
                reasons.append("out_of_bounds_free_counter")
        if min_blocks < args.min_free_fraction or min_excess < args.min_free_fraction:
            reasons.append("capacity_headroom_below_threshold")
        if failed_voxel or failed_excess:
            reasons.append("allocation_requests_rejected")
        if any(int(r["counters_available"]) != 1 for r in rows):
            reasons.append("allocation_counters_unavailable")
        if any(int(r["pose_finite"]) != 1 for r in rows):
            reasons.append("nonfinite_pose")
        if failure_count / len(rows) > args.max_failed_fraction or longest > args.max_failed_streak:
            reasons.append("tracking_health_floor_failed")
        metrics = {"frames": len(rows), "min_free_block_fraction": min_blocks, "min_free_excess_fraction": min_excess,
                   "failed_voxel_requests": failed_voxel, "failed_excess_requests": failed_excess,
                   "raw_failed_frames": failure_count, "raw_failed_fraction": failure_count / len(rows),
                   "raw_poor_frames": sum(int(r["raw_result"]) == 1 for r in rows),
                   "longest_raw_failed_streak": longest}
    report = {"status": "pass" if not reasons else "fail", "selection_rule": "capacity_headroom_not_ate",
              "scope": "This CSV gate alone does not certify full baseline fidelity or hardware conditions",
              "source_csv": str(args.csv.resolve()), "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              "thresholds": {"minimum_pool_and_excess_free_fraction": args.min_free_fraction,
                             "maximum_raw_failed_fraction": args.max_failed_fraction,
                             "maximum_consecutive_raw_failed_frames": args.max_failed_streak},
              "metrics": metrics, "failure_reasons": sorted(set(reasons))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
