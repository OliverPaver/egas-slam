#!/usr/bin/env python3
"""Check working-tree hygiene and source identity in the working tree or a Git ref."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024
ALLOWED_EXECUTABLES = {"egas", "scripts/desktop_run.py"}
PROTECTED_PATHS = (
    "slam",
    "src",
    "include",
    "gsplat",
    "InfiniTAM",
    "slam_trainer.cpp",
    "CMakeLists.txt",
    "configs/release",
)
REQUIRED_FILES = (
    ".gitleaks.toml",
    "README.md",
    "LICENSE.txt",
    "CITATION.cff",
    "PROVENANCE.md",
    "REPRODUCIBILITY.md",
    "RELEASE_EQUIVALENCE.md",
    "THIRD_PARTY_NOTICES.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "SOURCE_MANIFEST.json",
)
FORBIDDEN_PREFIXES = (
    "ThirdLibs/install/",
    "build/",
    "data/",
    "datasets/",
    "output/",
    ".runs/",
    ".cache/",
    "checkpoints/",
    "models/",
    "renders/",
    "telemetry/",
    "benchmark_repair_",
    "thesis_results_",
)
FORBIDDEN_SUFFIXES = (
    ".o",
    ".obj",
    ".so",
    ".a",
    ".dll",
    ".exe",
    ".pt",
    ".pth",
    ".ckpt",
    ".tar",
    ".tar.gz",
    ".tar.zst",
    ".zip",
    ".ply",
)
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cff",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".tpp",
    ".tcu",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
ABSOLUTE_USER_PATH = re.compile(r"(?:/home/[A-Za-z0-9._-]+/|[A-Za-z]:\\\\Users\\\\[^\\\\\s]+)")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class CheckFailure(RuntimeError):
    pass


def git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode:
        raise CheckFailure(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout


def candidate_paths() -> list[Path]:
    output = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted((ROOT / value for value in output.split("\0") if value), key=lambda p: p.as_posix())


def is_vendored(relative: str) -> bool:
    return relative.startswith(("ThirdLibs/", "InfiniTAM/"))


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def check_markdown_links(path: Path, text: str, errors: list[str]) -> None:
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            errors.append(f"documentation link escapes repository: {path.relative_to(ROOT)} -> {target}")
            continue
        if not resolved.exists():
            errors.append(f"broken documentation link: {path.relative_to(ROOT)} -> {target}")


def hygiene() -> None:
    errors: list[str] = []
    paths = candidate_paths()
    relatives = {path.relative_to(ROOT).as_posix() for path in paths}

    for required in REQUIRED_FILES:
        if required not in relatives:
            errors.append(f"missing release file: {required}")

    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            errors.append(f"source release must not contain symbolic links: {relative}")
            continue
        if not path.is_file():
            continue
        first_part = relative.split("/", 1)[0]
        generated_build_dir = "/" in relative and first_part.startswith(("build_", "cmake-build-"))
        if generated_build_dir or any(
            relative == prefix.rstrip("/") or relative.startswith(prefix)
            for prefix in FORBIDDEN_PREFIXES
        ):
            errors.append(f"forbidden generated path: {relative}")
        if relative.lower().endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden binary/archive: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 50 MiB: {relative} ({path.stat().st_size} bytes)")

        text = read_text(path)
        if text is None or is_vendored(relative):
            continue
        if ABSOLUTE_USER_PATH.search(text):
            errors.append(f"absolute user path: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label}: {relative}")
        if path.suffix.lower() == ".md":
            check_markdown_links(path, text, errors)

    modes = git("ls-files", "--stage").splitlines()
    for line in modes:
        metadata, relative = line.split("\t", 1)
        mode = metadata.split()[0]
        if mode == "100755" and not relative.endswith(".sh") and relative not in ALLOWED_EXECUTABLES:
            errors.append(f"unexpected executable mode: {relative}")

    if errors:
        raise CheckFailure("release hygiene failed:\n- " + "\n- ".join(sorted(set(errors))))
    print(f"working-tree release hygiene passed ({len(paths)} files checked)")


def tree_digest(ref: str) -> str:
    listing = git("ls-tree", "-r", ref, "--", *PROTECTED_PATHS)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def algorithm_equivalence(tag: str) -> None:
    git("rev-parse", "--verify", f"{tag}^{{commit}}")
    changes = git("diff", "--name-status", tag, "--", *PROTECTED_PATHS)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *PROTECTED_PATHS)
    if changes.strip() or untracked.strip():
        detail = "\n".join(value for value in (changes.strip(), untracked.strip()) if value)
        raise CheckFailure(f"algorithm tree differs from {tag}:\n{detail}")
    print(f"algorithm equivalence passed: tag={tag} digest={tree_digest(tag)}")


def git_blob_identities(object_ids: set[str]) -> dict[str, tuple[int, str]]:
    """Hash raw Git blobs in one batch, without checkout filters or per-file Git calls."""
    requested = sorted(object_ids)
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        input="".join(object_id + "\n" for object_id in requested).encode("ascii"),
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise CheckFailure(completed.stderr.decode("utf-8", errors="replace").strip())
    output = completed.stdout
    identities = {}
    offset = 0
    for object_id in requested:
        header_end = output.find(b"\n", offset)
        header = output[offset:header_end].split()
        if header_end < 0 or len(header) != 3 or header[:2] != [object_id.encode("ascii"), b"blob"]:
            raise CheckFailure(f"cannot read Git blob: {object_id}")
        size = int(header[2])
        start = header_end + 1
        end = start + size
        if end >= len(output) or output[end:end + 1] != b"\n":
            raise CheckFailure(f"incomplete Git blob: {object_id}")
        digest = hashlib.sha256(memoryview(output)[start:end]).hexdigest()
        identities[object_id] = (size, digest)
        offset = end + 1
    if offset != len(output):
        raise CheckFailure("unexpected output after Git blob batch")
    return identities


def source_manifest_at_ref(records: dict, relative_manifest: str, ref: str) -> None:
    """Verify a manifest against the complete tracked file set of one commit."""
    commit = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").strip()
    entries = {}
    for entry in git("ls-tree", "-r", "-z", "--full-tree", commit).split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, kind, object_id = metadata.split()
        entries[name] = (mode, kind, object_id)
    manifest_entry = entries.pop(relative_manifest, None)
    if manifest_entry is None or manifest_entry[:2] != ("100644", "blob"):
        raise CheckFailure(f"source manifest is not a regular tracked file at {commit}: {relative_manifest}")
    if set(records) != set(entries):
        raise CheckFailure(f"source-manifest file-set mismatch at {commit}: "
                           f"missing={sorted(set(records) - set(entries))}; "
                           f"added={sorted(set(entries) - set(records))}")
    unsafe = [name for name, (mode, kind, _) in entries.items()
              if mode not in {"100644", "100755"} or kind != "blob"]
    if unsafe:
        raise CheckFailure("source-manifest verification failed:\n- "
                           + "\n- ".join(name + ": not a regular Git blob" for name in unsafe))
    identities = git_blob_identities({entry[2] for entry in entries.values()})
    errors = []
    for name, record in records.items():
        size, digest = identities[entries[name][2]]
        if size != record["bytes"] or digest != record["sha256"]:
            errors.append(name + ": contents changed")
    if errors:
        raise CheckFailure(f"source-manifest verification failed at {commit}:\n- " + "\n- ".join(errors))
    print(f"Git source identity passed: ref={commit}; {relative_manifest}; {len(records)} files")


def source_manifest(path: str, ref: str | None = None) -> None:
    """Verify every shipped file except the manifest's unavoidable self entry."""
    manifest_path = (ROOT / path).resolve()
    relative_manifest = manifest_path.relative_to(ROOT).as_posix()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise CheckFailure("unsupported source-manifest schema")
    records = manifest["files"]
    if ref is not None:
        source_manifest_at_ref(records, relative_manifest, ref)
        return
    actual = {p.relative_to(ROOT).as_posix(): p for p in candidate_paths()
              if p.is_file() and p != manifest_path}
    if set(records) != set(actual):
        raise CheckFailure("source-manifest file-set mismatch: "
                           f"missing={sorted(set(records) - set(actual))}; "
                           f"added={sorted(set(actual) - set(records))}")
    errors = []
    for name, record in records.items():
        p = actual[name]
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            errors.append(name + ": unsafe source path")
            continue
        data = p.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            errors.append(name + ": contents changed")
    if errors:
        raise CheckFailure("source-manifest verification failed:\n- " + "\n- ".join(errors))
    print(f"working-tree source identity passed: {relative_manifest}; {len(records)} files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm-equivalence", metavar="TAG")
    parser.add_argument("--source-manifest", metavar="JSON")
    parser.add_argument(
        "--source-manifest-ref", metavar="REF",
        help="verify --source-manifest against this Git commit; hygiene still checks the working tree",
    )
    parser.add_argument("--skip-hygiene", action="store_true")
    args = parser.parse_args()
    if args.source_manifest_ref is not None and not args.source_manifest:
        parser.error("--source-manifest-ref requires --source-manifest")
    try:
        if not args.skip_hygiene:
            hygiene()
        if args.algorithm_equivalence:
            algorithm_equivalence(args.algorithm_equivalence)
        if args.source_manifest:
            source_manifest(args.source_manifest, args.source_manifest_ref)
    except CheckFailure as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
