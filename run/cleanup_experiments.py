#!/usr/bin/env python3
"""
Archive experiment evidence and remove bulky obsolete outputs.

The script is intentionally conservative: it copies metrics/config/log files into
output/audit_archive before deleting raw render/model/TSDF artifacts.
"""
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiments"

EVIDENCE_NAMES = {
    "time_log.txt",
    "results.json",
    "pose_eval.txt",
    "submap_events.jsonl",
    "frame_assignments.csv",
    "submap_stats.csv",
    "timing.csv",
    "paging.csv",
    "render_selection.csv",
    "submap_render_debug.csv",
    "summary.csv",
}
EVIDENCE_SUFFIXES = {".yaml", ".yml", ".json", ".jsonl", ".csv", ".txt", ".log"}

GOLDEN_EXPERIMENTS = {
    "exp_19_baseline_smoke_300f",
    "exp_50_multi_single_smoke_50f",
    "exp_54_multi_nongt_recon_120f",
    "exp_55_multi_hybrid_paging_65f",
    "exp_60_multi_disk_stream_tracking_180f",
    "exp_62_multi_paging_loop_prefetch_180f",
    "exp_65_multi_auto_residency_2f",
}

OBSOLETE_PREFIXES = tuple(f"exp_{idx:02d}_" for idx in range(1, 13))
ARCHIVED_SWEEP_ROOTS = {
    "hybrid_validation",
    "hybrid_training_sweep",
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


def archive_evidence(exp_dir: Path, archive_exp_dir: Path) -> list[dict]:
    archive_exp_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    config_path = CONFIG_ROOT / f"{exp_dir.name}.yaml"
    if config_path.exists():
        dst = archive_exp_dir / "config.yaml"
        shutil.copy2(config_path, dst)
        copied.append({"source": str(config_path), "archive": str(dst), "sha256": sha256_file(dst)})

    root_log = PROJECT_ROOT / f"output_{exp_dir.name}.log"
    if root_log.exists():
        dst = archive_exp_dir / root_log.name
        shutil.copy2(root_log, dst)
        copied.append({"source": str(root_log), "archive": str(dst), "sha256": sha256_file(dst)})

    for item in exp_dir.rglob("*"):
        if not item.is_file():
            continue
        keep = item.name in EVIDENCE_NAMES or item.suffix in EVIDENCE_SUFFIXES
        if not keep:
            continue
        rel = item.relative_to(exp_dir)
        dst = archive_exp_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)
        copied.append({"source": str(item), "archive": str(dst), "sha256": sha256_file(dst)})
    return copied


def classify(exp_dir: Path, size_bytes: int) -> tuple[str, str]:
    name = exp_dir.name
    if name in GOLDEN_EXPERIMENTS:
        return "keep", "golden run"
    if exp_dir.parent.name in ARCHIVED_SWEEP_ROOTS:
        if size_bytes >= 64 * 1024**2:
            return "delete", "archived old hybrid sweep raw artifacts"
        return "keep", "small hybrid diagnostic run"
    if name.startswith(OBSOLETE_PREFIXES):
        return "delete", "obsolete early config sweep"
    if size_bytes >= 2 * 1024**3:
        return "delete", "large superseded raw artifact"
    if "before_sequential_fix" in name:
        return "delete", "pre-fix comparison archived as evidence only"
    return "keep", "small or current diagnostic run"


def iter_experiment_dirs() -> list[Path]:
    roots = [
        OUTPUT_ROOT / "experiments",
        OUTPUT_ROOT / "experiments_clean",
        OUTPUT_ROOT / "hybrid_validation",
        OUTPUT_ROOT / "hybrid_training_sweep",
    ]
    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        dirs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="delete/prune after archiving")
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--min-free-gb", type=float, default=60.0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = args.archive_dir or (OUTPUT_ROOT / "audit_archive" / stamp)
    archive_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_reclaimable = 0
    for exp_dir in iter_experiment_dirs():
        size_bytes = dir_size(exp_dir)
        action, reason = classify(exp_dir, size_bytes)
        if action == "delete":
            total_reclaimable += size_bytes
        copied = archive_evidence(exp_dir, archive_root / exp_dir.parent.name / exp_dir.name)
        row = {
            "experiment": str(exp_dir),
            "size_bytes": size_bytes,
            "size_gb": round(size_bytes / 1024**3, 3),
            "action": action,
            "reason": reason,
            "evidence_files": len(copied),
        }
        manifest_rows.append(row)
        if args.execute and action == "delete":
            shutil.rmtree(exp_dir)

    manifest_json = archive_root / "cleanup_manifest.json"
    manifest_json.write_text(json.dumps(manifest_rows, indent=2) + "\n")
    manifest_csv = archive_root / "cleanup_manifest.csv"
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "size_bytes", "size_gb", "action", "reason", "evidence_files"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    print(f"Archive: {archive_root}")
    print(f"Mode: {'execute' if args.execute else 'dry-run'}")
    print(f"Reclaimable by policy: {total_reclaimable / 1024**3:.1f} GB")
    print(f"Current free space: {free_gb:.1f} GB")
    if args.execute:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
        print(f"Free space after cleanup: {free_gb:.1f} GB")
        if free_gb < args.min_free_gb:
            print(f"[WARN] free space remains below requested {args.min_free_gb:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
