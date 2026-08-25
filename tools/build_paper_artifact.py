#!/usr/bin/env python3
"""Build or verify the compact, checksum-addressed R2 paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXCLUDED_PARTS = {"gs_model", "tsdf_engine", "meshes", "models", "checkpoints"}
INDIVIDUAL_POSE = re.compile(r"^(?:frame|pose)\d+.*\.txt$")
FRAME_NUMBER = re.compile(r"(\d+)")
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
METADATA_FILES = (
    "BENCHMARK_FREEZE.json",
    "BENCHMARK_FREEZE_2026-08-19.md",
    "BENCHMARK_PROTOCOL.md",
    "PAPER_READINESS_LOCK_2026-08-25.md",
    "THOR_DEPLOYMENT_SCREEN_2026-08-19.md",
    "THOR_HEADLINE_BENCHMARK_2026-08-19.md",
    "CITATION.cff",
    "PROVENANCE.md",
    "REPRODUCIBILITY.md",
    "RELEASE_EQUIVALENCE.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/paper_soak_r2_20260825.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stdout
            + completed.stderr
        )
    return completed.stdout.strip()


def selected_text_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if INDIVIDUAL_POSE.match(path.name):
            continue
        selected.append(path)
    return sorted(selected, key=lambda value: value.relative_to(root).as_posix())


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)


def pose_directories(root: Path) -> list[Path]:
    directories: list[Path] = []
    for directory in root.rglob("*"):
        if not directory.is_dir() or "pose" not in directory.name.lower():
            continue
        if any(part in EXCLUDED_PARTS for part in directory.relative_to(root).parts):
            continue
        if any(INDIVIDUAL_POSE.match(path.name) for path in directory.iterdir() if path.is_file()):
            directories.append(directory)
    return sorted(directories, key=lambda value: value.relative_to(root).as_posix())


def write_lossless_trajectory(source_dir: Path, destination: Path) -> int:
    paths = sorted(
        (path for path in source_dir.iterdir() if path.is_file() and INDIVIDUAL_POSE.match(path.name)),
        key=lambda path: path.name,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for path in paths:
            raw = path.read_bytes()
            record = {
                "file": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "matrix_text": raw.decode("utf-8"),
            }
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    return len(paths)


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_NUMBER.search(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if not paths or count <= 0:
        return []
    if len(paths) <= count:
        return paths
    indices = sorted({round(index * (len(paths) - 1) / (count - 1)) for index in range(count)})
    return [paths[index] for index in indices]


def representative_composites(headline: Path, per_run: int) -> list[Path]:
    directories: list[Path] = []
    directories.extend(headline.glob("replica_baseline/*/val/comp"))
    directories.extend(headline.glob("replica_modified/*/val_final_map*/comp"))
    selected: list[Path] = []
    for directory in sorted(directories, key=lambda value: value.as_posix()):
        images = sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=frame_sort_key,
        )
        selected.extend(evenly_spaced(images, per_run))
    return selected


def write_manifest(artifact_root: Path) -> tuple[int, int]:
    paths = sorted(
        (
            path
            for path in artifact_root.rglob("*")
            if path.is_file() and path.name != "MANIFEST.sha256"
        ),
        key=lambda value: value.relative_to(artifact_root).as_posix(),
    )
    total_bytes = sum(path.stat().st_size for path in paths)
    manifest = artifact_root / "MANIFEST.sha256"
    with manifest.open("w", encoding="utf-8") as handle:
        for path in paths:
            relative = path.relative_to(artifact_root).as_posix()
            handle.write(f"{sha256_file(path)}  {relative}\n")
    return len(paths), total_bytes


def verify_directory(artifact_root: Path) -> tuple[int, int]:
    manifest = artifact_root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"missing {manifest}")
    count = 0
    total_bytes = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"malformed manifest line: {line!r}")
        path = artifact_root / relative
        resolved = path.resolve()
        try:
            resolved.relative_to(artifact_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"manifest path escapes artifact: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"manifest file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {relative}")
        count += 1
        total_bytes += path.stat().st_size
    return count, total_bytes


def secret_scan(artifact_root: Path) -> None:
    findings: list[str] = []
    for path in artifact_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | {".md", ".cff"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(artifact_root)}")
    if findings:
        raise RuntimeError("artifact secret scan failed:\n- " + "\n- ".join(findings))


def validate_archive_members(archive: Path) -> None:
    """Reject absolute, parent-traversing, or multi-root archive members."""
    listing = run(["tar", "-I", "zstd", "-tf", str(archive)])
    members = [line for line in listing.splitlines() if line]
    if not members:
        raise RuntimeError("archive contains no members")
    roots: set[str] = set()
    for member in members:
        path = Path(member)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive member: {member}")
        if not path.parts or path.parts[0] in {"", "."}:
            raise RuntimeError(f"malformed archive member: {member}")
        roots.add(path.parts[0])
    if len(roots) != 1:
        raise RuntimeError("archive must contain exactly one top-level directory")


def build(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    headline = run_root / args.headline
    soak = run_root / args.soak
    for source in (headline, soak):
        if not source.is_dir():
            raise FileNotFoundError(f"required run root is missing: {source}")
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("building the release archive requires GNU tar and zstd")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / args.name
    archive = output_dir / f"{args.name}.tar.zst"
    archive_checksum = output_dir / f"{archive.name}.sha256"
    existing = [path for path in (artifact_root, archive, archive_checksum) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite release output: " + ", ".join(map(str, existing)))
    artifact_root.mkdir()

    copied_text = 0
    consolidated_trajectories = 0
    consolidated_pose_files = 0
    for source_root in (headline, soak):
        destination_root = artifact_root / "runs" / source_root.name
        for source in selected_text_files(source_root):
            copy_file(source, destination_root / source.relative_to(source_root))
            copied_text += 1
        for pose_dir in pose_directories(source_root):
            destination = (
                artifact_root
                / "trajectories"
                / source_root.name
                / pose_dir.relative_to(source_root)
                / "trajectory.jsonl"
            )
            consolidated_pose_files += write_lossless_trajectory(pose_dir, destination)
            consolidated_trajectories += 1

    representative_count = 0
    for source in representative_composites(headline, args.representative_per_run):
        destination = artifact_root / "representative_renders" / source.relative_to(headline)
        copy_file(source, destination)
        representative_count += 1

    for relative in METADATA_FILES:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required release metadata is missing: {relative}")
        copy_file(source, artifact_root / "release" / relative)

    git_commit = run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    git_describe = run(["git", "describe", "--tags", "--always", "--dirty"], PROJECT_ROOT)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_name": args.name,
        "source_roots": [headline.name, soak.name],
        "git_commit": git_commit,
        "git_describe": git_describe,
        "copied_text_files": copied_text,
        "consolidated_trajectory_directories": consolidated_trajectories,
        "consolidated_pose_files": consolidated_pose_files,
        "representative_render_files": representative_count,
        "exclusions": [
            "datasets and metric weights",
            "compiled binaries and caches",
            "Gaussian/TSDF models and checkpoints",
            "meshes and point clouds",
            "all-frame renders/raycasts/composites",
            "individual pose files after lossless JSONL consolidation",
        ],
    }
    (artifact_root / "ARTIFACT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_root / "README.md").write_text(
        "# GPS-SLAM R2 paper artifact\n\n"
        "This compact artifact preserves the exact configurations, manifests, summaries, "
        "per-frame traces, trajectories, submap/pose-graph records, metric coverage, logs, "
        "environment/telemetry evidence, and selected comparison renders used by the R2 paper "
        "evaluation. It contains no datasets, weights, executables, models, meshes, TSDF state, "
        "or full render sequences.\n\n"
        "Each `trajectories/**/trajectory.jsonl` record contains the original pose filename, "
        "SHA-256 digest, and exact matrix text. Verify every retained file with "
        "`sha256sum -c MANIFEST.sha256`.\n",
        encoding="utf-8",
    )

    secret_scan(artifact_root)
    file_count, total_bytes = write_manifest(artifact_root)
    verified_count, verified_bytes = verify_directory(artifact_root)
    if (file_count, total_bytes) != (verified_count, verified_bytes):
        raise RuntimeError("artifact verification count mismatch")

    run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--pax-option=delete=atime,delete=ctime",
            "-I",
            "zstd -19 -T0",
            "-cf",
            str(archive),
            artifact_root.name,
        ],
        output_dir,
    )
    archive_checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    print(
        f"artifact built: files={file_count} uncompressed={total_bytes / 1024**2:.1f} MiB "
        f"archive={archive.stat().st_size / 1024**2:.1f} MiB path={archive}"
    )


def verify(args: argparse.Namespace) -> None:
    target = args.target.resolve()
    if target.is_dir():
        count, total_bytes = verify_directory(target)
        print(f"artifact verified: files={count} bytes={total_bytes} path={target}")
        return
    if not target.is_file() or not target.name.endswith(".tar.zst"):
        raise FileNotFoundError(f"expected artifact directory or .tar.zst archive: {target}")
    validate_archive_members(target)
    with tempfile.TemporaryDirectory(prefix="gps_slam_artifact_verify_") as temporary:
        destination = Path(temporary)
        run(["tar", "--no-same-owner", "-I", "zstd", "-xf", str(target), "-C", str(destination)])
        roots = [path for path in destination.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("archive must contain exactly one top-level directory")
        count, total_bytes = verify_directory(roots[0])
    print(f"archive verified in second extraction: files={count} bytes={total_bytes} path={target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--run-root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--headline", default="headline_r2_2026-08-24")
    build_parser.add_argument("--soak", default="paper_soak_r2_20260825")
    build_parser.add_argument("--name", default="gps-slam-paper-artifact-v1.0.0")
    build_parser.add_argument("--representative-per-run", type=int, default=3)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("target", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            if args.representative_per_run < 1:
                raise ValueError("--representative-per-run must be positive")
            build(args)
        else:
            verify(args)
    except Exception as exc:
        print(f"artifact error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
