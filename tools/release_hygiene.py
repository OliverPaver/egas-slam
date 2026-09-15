#!/usr/bin/env python3
"""Check source-only release hygiene and the complete source-file manifest."""

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
        if mode == "100755" and not relative.endswith(".sh"):
            errors.append(f"unexpected executable mode: {relative}")

    if errors:
        raise CheckFailure("release hygiene failed:\n- " + "\n- ".join(sorted(set(errors))))
    print(f"release hygiene passed ({len(paths)} files checked)")


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


def source_manifest(path: str) -> None:
    """Verify every shipped file except the manifest's unavoidable self entry."""
    manifest_path = (ROOT / path).resolve()
    relative_manifest = manifest_path.relative_to(ROOT).as_posix()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise CheckFailure("unsupported source-manifest schema")
    records = manifest["files"]
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
    print(f"complete source identity passed: {relative_manifest}; {len(records)} files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm-equivalence", metavar="TAG")
    parser.add_argument("--source-manifest", metavar="JSON")
    parser.add_argument("--skip-hygiene", action="store_true")
    args = parser.parse_args()
    try:
        if not args.skip_hygiene:
            hygiene()
        if args.algorithm_equivalence:
            algorithm_equivalence(args.algorithm_equivalence)
        if args.source_manifest:
            source_manifest(args.source_manifest)
    except CheckFailure as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
