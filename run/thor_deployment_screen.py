#!/usr/bin/env python3
"""Run the three fixed, one-pass TUM deployment screening configurations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tum_loop_validation import SEQUENCES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREEN_CONFIGS = (
    (
        "unsliced_reference",
        "recommended_longspan_robust",
        "thor_cpu_bounded",
        "immediate",
    ),
    (
        "config_only_streaming",
        "recommended_longspan_robust_streaming",
        "thor_streaming_bounded",
        "immediate",
    ),
    (
        "combined_sliced_streaming",
        "recommended_longspan_robust_streaming",
        "thor_streaming_bounded",
        "sliced",
    ),
)


def prepared_sequences(tum_root: Path) -> list[str]:
    return [
        alias
        for alias, directory in SEQUENCES.items()
        if (tum_root / directory / "metadata.json").is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "configs/release/replica/office0.yaml")
    parser.add_argument("--sequences", nargs="+", choices=sorted(SEQUENCES), default=None)
    parser.add_argument(
        "--screens",
        nargs="+",
        choices=[entry[0] for entry in SCREEN_CONFIGS],
        default=None,
        help="run only the named screening configurations (default: all three)",
    )
    parser.add_argument("--hash-profile", choices=["compact_64k", "medium_128k", "legacy_large"], default="medium_128k")
    parser.add_argument("--run", action="store_true", help="execute runs; otherwise only generate and preflight configs")
    parser.add_argument("--quiet-run-logs", action="store_true")
    args = parser.parse_args()

    sequences = args.sequences or prepared_sequences(args.tum_root)
    if not sequences:
        parser.error("no prepared TUM sequences were found")
    missing = [
        alias
        for alias in sequences
        if not (args.tum_root / SEQUENCES[alias] / "metadata.json").is_file()
    ]
    if missing:
        parser.error("missing prepared sequence(s): " + ", ".join(missing))

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_screens = set(args.screens or (entry[0] for entry in SCREEN_CONFIGS))
    for name, loop_profile, residency_profile, schedule in SCREEN_CONFIGS:
        if name not in selected_screens:
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run/tum_loop_validation.py"),
            "--tum-root",
            str(args.tum_root),
            "--template",
            str(args.template),
            "--config-dir",
            str(args.output_root / "configs" / name),
            "--output-root",
            str(args.output_root / name),
            "--trainer",
            str(args.trainer),
            "--sequences",
            *sequences,
            "--variants",
            "multi_hybrid_orb_apply",
            "--full-sequence",
            "--loop-profile",
            loop_profile,
            "--tracking-profile",
            "robust_rpe",
            "--tracking-failure-mode",
            "contain_failed",
            "--hash-profile",
            args.hash_profile,
            "--tsdf-lifecycle-profile",
            "paper_parity",
            "--tsdf-residency-profile",
            residency_profile,
            "--tsdf-input-mode",
            "on_demand",
            "--tsdf-input-prefetch",
            "on",
            "--submap-stats-interval-frames",
            "0",
            "--gaussian-schedule",
            schedule,
            "--gaussian-opt-iters-per-frame",
            "2",
            "--export-online-end-trajectory",
            "true",
            "--analyze-realtime",
            "--thor-preflight",
            "--no-save-after-train",
            "--tag",
            f"screen_{name}",
            "--summary-output",
            str(args.output_root / f"{name}_summary.csv"),
        ]
        if args.quiet_run_logs:
            command.append("--quiet-run-logs")
        if args.run:
            command.append("--run")
        print("[SCREEN]", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
