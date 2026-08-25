#!/usr/bin/env python3
"""Create or verify the immutable source/binary envelope for paper benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "BENCHMARK_FREEZE.json"
SCHEMA_VERSION = 1

ROOT_FILES = (
    "BENCHMARK_PROTOCOL.md",
    "CMakeLists.txt",
    "VERSION",
    "build_third_libs.sh",
    "download_replica.sh",
    "json.hpp",
    "remote_viewer.cpp",
    "slam_trainer.cpp",
)

SOURCE_ROOTS = (
    "InfiniTAM",
    "ThirdLibs/fetchcontent/cxxopts",
    "ThirdLibs/fetchcontent/glm",
    "ThirdLibs/fetchcontent/nanoflann",
    "ThirdLibs/fetchcontent/nlohmann_json",
    "ThirdLibs/indicators",
    "ThirdLibs/install",
    "ThirdLibs/tensorboard_logger",
    "ThirdLibs/tinyply",
    "ThirdLibs/yaml-cpp",
    "configs/release",
    "configs/viewer",
    "containers",
    "gsplat",
    "include",
    "run",
    "scripts",
    "slam",
    "src",
)

IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".swp", ".tmp"}
BINARY_PATHS = ("build_legacy/slam_trainer", "build_thor/slam_trainer")
CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CUDA_ARCHITECTURES",
    "CMAKE_CUDA_COMPILER",
    "GPS_SLAM_BUILD_LEGACY_UI",
    "GPS_SLAM_BUILD_REMOTE_VIEWER",
    "GPS_SLAM_CUDA_FAST_MATH",
    "GPS_SLAM_ENABLE_TENSORBOARD",
    "GPS_SLAM_TSDF_HASH_PROFILE",
)


class FreezeVerificationError(RuntimeError):
    """Raised when the working tree no longer matches the benchmark freeze."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_inventory_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in IGNORED_PARTS for part in path.parts):
        return False
    return path.suffix not in IGNORED_SUFFIXES


def discover_functional_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    paths: set[Path] = set()
    for relative in ROOT_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required freeze input is missing: {relative}")
        paths.add(path)
    for relative in SOURCE_ROOTS:
        root = project_root / relative
        if not root.is_dir():
            raise FileNotFoundError(f"required freeze source root is missing: {relative}")
        paths.update(path for path in root.rglob("*") if _is_inventory_file(path))
    return sorted(paths, key=lambda path: path.relative_to(project_root).as_posix())


def file_records(paths: Iterable[Path], project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        records.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def records_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in CACHE_KEYS:
            values[key] = value
    return values


def binary_records(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    records = []
    for relative in BINARY_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen binary is missing: {relative}")
        build_dir = path.parent
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "cmake_cache": parse_cmake_cache(build_dir / "CMakeCache.txt"),
            }
        )
    return records


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout + completed.stderr).strip()


def container_image_id() -> str:
    configured = os.environ.get("GPS_SLAM_CONTAINER_IMAGE_ID", "").strip()
    if configured:
        return configured
    image = os.environ.get("GPS_SLAM_CONTAINER_IMAGE", "gps-slam-thor:pytorch-26.07-jp7.1")
    return _command_output(["docker", "image", "inspect", "--format", "{{.Id}}", image])


def create_manifest(freeze_id: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if not freeze_id or any(character.isspace() for character in freeze_id):
        raise ValueError("freeze ID must be a non-empty token without whitespace")
    records = file_records(discover_functional_files(project_root), project_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "scope": "comparison-only",
            "default_gaussian_schedule": "immediate",
            "paper_protocols": [
                "gps_slam_replica_paper_v1",
                "gps_slam_tum_tracking_v1",
                "gps_slam_tum_integrated_v1",
            ],
            "unpromoted_protocols": [
                "gps_slam_replica_deployed_quality_v1",
                "gps_slam_tum_deployed_v1",
            ],
        },
        "platform": {
            "machine": platform.machine(),
            "container_image_id": container_image_id(),
        },
        "functional_digest": records_digest(records),
        "functional_file_count": len(records),
        "files": records,
        "binaries": binary_records(project_root),
    }


def compare_records(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    expected_by_path = {str(record["path"]): record for record in expected}
    actual_by_path = {str(record["path"]): record for record in actual}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    added = sorted(set(actual_by_path) - set(expected_by_path))
    modified = sorted(
        path
        for path in set(expected_by_path) & set(actual_by_path)
        if expected_by_path[path].get("size") != actual_by_path[path].get("size")
        or expected_by_path[path].get("sha256") != actual_by_path[path].get("sha256")
    )
    return {"missing": missing, "added": added, "modified": modified}


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    project_root: Path = PROJECT_ROOT,
    *,
    require_container: bool = True,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FreezeVerificationError(f"benchmark freeze manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise FreezeVerificationError("unsupported benchmark freeze schema")

    actual_files = file_records(discover_functional_files(project_root), project_root)
    source_changes = compare_records(payload.get("files", []), actual_files)
    expected_binaries = payload.get("binaries", [])
    try:
        actual_binaries = binary_records(project_root)
        binary_changes = compare_records(expected_binaries, actual_binaries)
        expected_binary_by_path = {record["path"]: record for record in expected_binaries}
        actual_binary_by_path = {record["path"]: record for record in actual_binaries}
        cache_changes = sorted(
            path
            for path in set(expected_binary_by_path) & set(actual_binary_by_path)
            if expected_binary_by_path[path].get("cmake_cache")
            != actual_binary_by_path[path].get("cmake_cache")
        )
    except FileNotFoundError as exc:
        actual_binaries = []
        binary_changes = {"missing": [str(exc)], "added": [], "modified": []}
        cache_changes = []

    expected_container = str(payload.get("platform", {}).get("container_image_id", ""))
    actual_container = container_image_id()
    container_observable = bool(actual_container)
    container_match = bool(expected_container) and actual_container == expected_container
    machine_match = payload.get("platform", {}).get("machine") == platform.machine()
    source_clean = not any(source_changes.values())
    binaries_clean = not any(binary_changes.values()) and not cache_changes
    clean = bool(
        source_clean
        and binaries_clean
        and machine_match
        and (container_match if require_container else (container_match or not container_observable))
    )
    return {
        "freeze_id": payload.get("freeze_id", ""),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "functional_digest": payload.get("functional_digest", ""),
        "functional_file_count": len(actual_files),
        "source_clean": source_clean,
        "source_changes": source_changes,
        "binaries_clean": binaries_clean,
        "binary_changes": binary_changes,
        "cmake_cache_changes": cache_changes,
        "machine_match": machine_match,
        "expected_container_image_id": expected_container,
        "actual_container_image_id": actual_container,
        "container_observable": container_observable,
        "container_match": container_match,
        "clean": clean,
    }


def require_clean_freeze(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    report = verify_manifest(manifest_path)
    if not report["clean"]:
        raise FreezeVerificationError(
            "benchmark code freeze verification failed:\n" + json.dumps(report, indent=2, sort_keys=True)
        )
    return report


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="write a new freeze manifest")
    create.add_argument("--freeze-id", required=True)
    create.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify = subparsers.add_parser("verify", help="verify sources, binaries, build caches, and container")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--allow-unobservable-container", action="store_true")
    verify.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "create":
        payload = create_manifest(args.freeze_id)
        atomic_json(args.manifest, payload)
        print(
            f"benchmark freeze created: id={payload['freeze_id']} "
            f"files={payload['functional_file_count']} digest={payload['functional_digest']}"
        )
        return 0

    report = verify_manifest(
        args.manifest,
        require_container=not args.allow_unobservable_container,
    )
    if args.output:
        atomic_json(args.output, report)
    print(
        f"benchmark freeze: id={report['freeze_id']} "
        f"source={'clean' if report['source_clean'] else 'CHANGED'} "
        f"binaries={'clean' if report['binaries_clean'] else 'CHANGED'} "
        f"container={'match' if report['container_match'] else 'MISMATCH'}"
    )
    if not report["clean"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
