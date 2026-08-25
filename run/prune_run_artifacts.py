#!/usr/bin/env python3
"""Archive run evidence and prune bulky generated artifacts.

This is meant for external experiment roots such as TUM validation folders on
large drives. It is intentionally narrow: by default it only considers
``tsdf_pages`` directories and only deletes them when ``--execute`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_NAMES = {
    "time_log.txt",
    "results.json",
    "pose_eval.txt",
    "submap_events.jsonl",
    "frame_assignments.csv",
    "submap_stats.csv",
    "submap_transitions.csv",
    "timing.csv",
    "tracking.csv",
    "paging.csv",
    "render_selection.csv",
    "submap_render_debug.csv",
    "loop_candidates.csv",
    "pose_graph.csv",
    "pose_graph_summary.json",
}

EVIDENCE_SUFFIXES = {".yaml", ".yml", ".json", ".jsonl", ".csv", ".txt", ".log"}
SAFE_ARTIFACT_NAMES = {
    "before_opt",
    "gs_model",
    "raycasted",
    "tensorboard",
    "tsdf_engine",
    "tsdf_pages",
}


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.lstat().st_size
        except FileNotFoundError:
            pass
    return total


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_evidence_file(source: Path, archive_root: Path, rel: Path) -> dict:
    dst = archive_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    return {
        "source": str(source),
        "archive": str(dst),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(dst),
    }


def archive_run_evidence(run_dir: Path, archive_run_dir: Path) -> list[dict]:
    copied: list[dict] = []
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    for item in run_dir.rglob("*"):
        if not item.is_file():
            continue
        if any(part in SAFE_ARTIFACT_NAMES for part in item.relative_to(run_dir).parts):
            continue
        keep = item.name in EVIDENCE_NAMES or item.suffix in EVIDENCE_SUFFIXES
        if not keep:
            continue
        copied.append(copy_evidence_file(item, archive_run_dir, item.relative_to(run_dir)))
    return copied


def archive_parent_summaries(run_root: Path, archive_root: Path) -> list[dict]:
    copied: list[dict] = []
    summary_archive = archive_root / "_parent_summaries"
    for item in sorted(run_root.glob("*summary*.csv")):
        if item.is_file():
            copied.append(copy_evidence_file(item, summary_archive, item.name))
    return copied


def iter_run_dirs(run_roots: list[Path], patterns: list[str]) -> list[Path]:
    runs: list[Path] = []
    for root in run_roots:
        if not root.exists():
            raise FileNotFoundError(f"Run root does not exist: {root}")
        candidates = [root] if (root / "time_log.txt").exists() else sorted(p for p in root.iterdir() if p.is_dir())
        for candidate in candidates:
            if patterns and not any(fnmatch.fnmatch(candidate.name, pattern) for pattern in patterns):
                continue
            runs.append(candidate)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", type=Path, required=True, help="Run directory or parent containing run directories")
    parser.add_argument("--pattern", action="append", default=[], help="Shell-style run-name pattern. Repeatable. Defaults to all runs.")
    parser.add_argument("--artifact", action="append", choices=sorted(SAFE_ARTIFACT_NAMES), default=["tsdf_pages"])
    parser.add_argument("--archive-dir", type=Path, default=None, help="Evidence archive directory")
    parser.add_argument("--execute", action="store_true", help="Actually delete artifact directories after archiving evidence")
    parser.add_argument("--min-artifact-mb", type=float, default=1.0, help="Ignore artifacts smaller than this size")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = args.archive_dir or (PROJECT_ROOT / "output" / "audit_archive" / f"external_prune_{stamp}")
    archive_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    evidence_manifest: list[dict] = []
    reclaimable = 0
    min_artifact_bytes = int(max(0.0, args.min_artifact_mb) * 1024 * 1024)

    for run_root in args.run_root:
        evidence_manifest.extend(archive_parent_summaries(run_root, archive_root / run_root.name))

    for run_dir in iter_run_dirs(args.run_root, args.pattern):
        copied = archive_run_evidence(run_dir, archive_root / run_dir.parent.name / run_dir.name)
        evidence_manifest.extend(copied)
        for artifact_name in args.artifact:
            artifact_path = run_dir / artifact_name
            if not artifact_path.exists():
                continue
            if artifact_path.is_symlink() or not artifact_path.is_dir():
                manifest_rows.append({
                    "run": str(run_dir),
                    "artifact": str(artifact_path),
                    "size_bytes": 0,
                    "size_gb": 0.0,
                    "action": "skip",
                    "reason": "artifact is not a real directory",
                    "evidence_files": len(copied),
                })
                continue
            size_bytes = dir_size(artifact_path)
            if size_bytes < min_artifact_bytes:
                action = "skip"
                reason = "below minimum artifact size"
            else:
                action = "delete" if args.execute else "dry_run_delete"
                reason = "archived evidence and pruned bulky runtime artifact"
                reclaimable += size_bytes
            manifest_rows.append({
                "run": str(run_dir),
                "artifact": str(artifact_path),
                "size_bytes": size_bytes,
                "size_gb": round(size_bytes / 1024**3, 3),
                "action": action,
                "reason": reason,
                "evidence_files": len(copied),
            })
            if args.execute and action == "delete":
                shutil.rmtree(artifact_path)

    (archive_root / "prune_manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    (archive_root / "evidence_manifest.json").write_text(json.dumps(evidence_manifest, indent=2) + "\n")
    with (archive_root / "prune_manifest.csv").open("w", newline="") as f:
        fieldnames = ["run", "artifact", "size_bytes", "size_gb", "action", "reason", "evidence_files"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Archive: {archive_root}")
    print(f"Mode: {'execute' if args.execute else 'dry-run'}")
    print(f"Runs inspected: {len(set(row['run'] for row in manifest_rows))}")
    print(f"Artifact rows: {len(manifest_rows)}")
    print(f"Reclaimable/deleted by selected policy: {reclaimable / 1024**3:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
