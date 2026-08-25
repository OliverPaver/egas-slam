"""Shared validation and output helpers for evaluation scripts.

This module deliberately has no Torch, Open3D, or image-library imports so the
pairing and failure contracts can be tested on a build host before the runtime
metric dependencies are installed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def frame_id(name: str | Path) -> int | None:
    """Return the explicit ``frameNNN`` ID, or the first numeric fallback."""
    filename = Path(name).name
    match = re.search(r"frame(\d+)", filename, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def collect_indexed_images(
    directory: str | Path,
    *,
    required_name_token: str | None = None,
) -> dict[int, Path]:
    """Collect image files by frame ID, rejecting ambiguous inputs.

    Generated comparison montages remain excluded for compatibility. A duplicate
    numeric ID is an error rather than silently replacing an earlier image.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    images: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if "_comp" in lower_name or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if required_name_token is not None and required_name_token.lower() not in lower_name:
            continue
        idx = frame_id(path)
        if idx is None:
            raise ValueError(f"Image filename has no numeric frame ID: {path}")
        previous = images.get(idx)
        if previous is not None:
            raise ValueError(
                f"Duplicate frame ID {idx} in {directory}: "
                f"{previous.name!r} and {path.name!r}"
            )
        images[idx] = path
    return images


def require_primary_coverage(
    primary_ids: Iterable[int],
    matched_ids: Iterable[int],
    *,
    primary_label: str,
    allow_partial: bool,
) -> tuple[int, float]:
    """Require every primary item to have references unless explicitly relaxed."""
    primary = set(primary_ids)
    matched = set(matched_ids)
    missing = sorted(primary - matched)
    coverage = len(matched) / len(primary) if primary else 0.0
    if not primary:
        raise RuntimeError(f"No {primary_label} files found")
    if missing and not allow_partial:
        preview = ", ".join(str(idx) for idx in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"{len(missing)} of {len(primary)} {primary_label} frames lack all "
            f"required references (IDs: {preview}{suffix})"
        )
    return len(missing), coverage


def resolve_torch_device(torch_module: Any, requested: str):
    """Resolve ``auto``, CPU, or an explicit CUDA device without false fallback."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda:0" if torch_module.cuda.is_available() else "cpu"
    elif normalized == "cuda":
        normalized = "cuda:0"
    elif normalized != "cpu" and not re.fullmatch(r"cuda:\d+", normalized):
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:N")

    if normalized.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {normalized}")
        device_index = int(normalized.split(":", 1)[1])
        device_count = int(torch_module.cuda.device_count())
        if device_index >= device_count:
            raise RuntimeError(
                f"CUDA device {device_index} requested, but only {device_count} device(s) are visible"
            )
    device = torch_module.device(normalized)
    if normalized.startswith("cuda"):
        torch_module.cuda.set_device(device)
    return device


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically replace a JSON artifact so readers never observe partial data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_status_error(path: str | Path, exc: BaseException, **context: Any) -> None:
    payload = {
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        **context,
    }
    atomic_write_json(path, payload)
