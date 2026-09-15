#!/usr/bin/env python3
"""Certify/recheck a capacity-independent rasterizer archive from recorded Ninja dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

CAPACITY_MACROS = re.compile(rb"\b(?:GPS_SLAM_(?:COMPACT_HASH|MEDIUM_HASH|CAPACITY_CONTROL|COMPILED_HASH_PROFILE)|SDF_(?:LOCAL_BLOCK_NUM|BUCKET_NUM|HASH_MASK|EXCESS_LIST_SIZE|BLOCK_SIZE3?))\b")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(args.verify.read_text())
        if manifest["status"] != "pass" or manifest["capacity_references"]:
            raise SystemExit("Archive dependency independence was not established")
        if not args.archive or not args.sha256 or sha(args.archive) != args.sha256 or args.sha256 != manifest["archive_sha256"]:
            raise SystemExit("Archive SHA256 mismatch")
        if os.environ.get("GPS_SLAM_CONTAINER_IMAGE_ID") != manifest["container_image_id"]:
            raise SystemExit("Archive container identity mismatch")
        for expect in args.expect:
            key, value = expect.split("=", 1)
            if str(manifest["build_cache"].get(key)) != value:
                raise SystemExit("Archive compiler/flag mismatch: " + key + " source=" +
                                 repr(manifest["build_cache"].get(key)) + " requested=" + repr(value))
        for row in manifest["dependencies"]:
            if sha(Path(row["path"])) != row["sha256"]:
                raise SystemExit("Archive dependency changed: " + row["path"])
        print("Verified capacity-independent archive: " + args.sha256)
        return
    if not args.build_dir or not args.output:
        parser.error("Inventory requires --build-dir and --output")
    build = args.build_dir.resolve()
    commands = [r for r in json.loads((build / "compile_commands.json").read_text()) if "/rasterizer.dir/" in r["output"]]
    if not commands:
        raise SystemExit("No recorded rasterizer translation units")
    dep_output = subprocess.check_output(["ninja", "-C", str(build), "-t", "deps", *[r["output"] for r in commands]], text=True)
    dependencies = set()
    for line in dep_output.splitlines():
        if line.startswith("    "):
            path = Path(line.strip())
            dependencies.add(path.resolve() if path.is_absolute() else (build / path).resolve())
        elif line and "#deps" in line and "VALID" not in line:
            raise SystemExit("Stale Ninja dependency information: " + line)
    if not dependencies:
        raise SystemExit("No dependency inventory")
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^/#][^:=]*):[^=]+=(.*)$", line)
        if match:
            cache[match[1]] = match[2]
    # Pin compiler and build-system inputs as well as preprocessor dependencies.
    for name in ("CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER"):
        dependencies.add(Path(cache[name]).resolve())
    dependencies.add(Path(cache["Torch_DIR"]) / "TorchConfig.cmake")
    records, capacity_references = [], []
    for path in sorted(dependencies):
        data = path.read_bytes()
        references = sorted(set(token.decode() for token in CAPACITY_MACROS.findall(data)))
        if references:
            capacity_references.append({"path": str(path), "macros": references})
        records.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    archive = build / "librasterizer.a"
    record = {"status": "pass" if not capacity_references else "fail", "archive": str(archive),
              "archive_sha256": sha(archive), "container_image_id": os.environ.get("GPS_SLAM_CONTAINER_IMAGE_ID"),
              "commands": commands, "build_cache": cache, "dependencies": records,
              "capacity_references": capacity_references,
              "scope": "Ninja-recorded transitive dependencies contain no capacity macros; exact source/compiler/container identity pinned"}
    if not record["container_image_id"]:
        raise SystemExit("Missing pinned container identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "translation_units": len(commands), "dependencies": len(records),
                      "archive_sha256": record["archive_sha256"], "capacity_references": capacity_references}))
    if record["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
