#!/usr/bin/env python3
"""Fail-fast ABI, CUDA-architecture, linkage, and Torch checks for a Thor build."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path


def run(*args: str, timeout: int = 30) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def cache_value(cache_text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:[^=]+=(.*)$", cache_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build_thor"))
    parser.add_argument("--hash-profile", default="medium_128k")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    binary = build_dir / "slam_trainer"
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("host architecture", platform.machine() == "aarch64", platform.machine())
    check("trainer exists", binary.is_file() and binary.stat().st_mode & 0o111 != 0, str(binary))

    if binary.is_file():
        code, output = run("file", str(binary))
        check("binary ABI", code == 0 and "aarch64" in output.lower(), output)

        code, output = run("ldd", str(binary))
        links_ok = code == 0 and "not found" not in output.lower() and "x86_64" not in output.lower()
        check("runtime linkage", links_ok, output)

        code, output = run("readelf", "-d", str(binary))
        protobuf_needed = re.search(r"NEEDED.*libprotobuf", output) is not None
        check(
            "no direct system protobuf dependency",
            code == 0 and not protobuf_needed,
            f"readelf={code}, protobuf_needed={protobuf_needed}",
        )

        cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        code, output = run(cuobjdump, "--list-elf", str(binary))
        check("embedded SM 110 code", code == 0 and "sm_110" in output, output)

    cache_path = build_dir / "CMakeCache.txt"
    if cache_path.is_file():
        cache_text = cache_path.read_text(errors="replace")
        architecture = cache_value(cache_text, "CMAKE_CUDA_ARCHITECTURES")
        profile = cache_value(cache_text, "GPS_SLAM_TSDF_HASH_PROFILE")
        legacy_ui = cache_value(cache_text, "GPS_SLAM_BUILD_LEGACY_UI")
        fast_math = cache_value(cache_text, "GPS_SLAM_CUDA_FAST_MATH")
        tensorboard = cache_value(cache_text, "GPS_SLAM_ENABLE_TENSORBOARD")
        check("CMake CUDA architecture", architecture == "110", str(architecture))
        check("TSDF hash profile", profile == args.hash_profile, str(profile))
        check("legacy UI disabled", legacy_ui == "OFF", str(legacy_ui))
        check("CUDA fast math disabled", fast_math == "OFF", str(fast_math))
        check("TensorBoard/protobuf isolation", tensorboard == "OFF", str(tensorboard))
    else:
        check("CMake cache", False, f"missing {cache_path}")

    commands_path = build_dir / "compile_commands.json"
    if commands_path.is_file():
        commands_text = commands_path.read_text(errors="replace")
        has_110 = re.search(r"(?:compute|sm)_110", commands_text) is not None
        has_removed_arch = re.search(r"(?:compute|sm)_70\b", commands_text) is not None
        check("compile architecture flags", has_110 and not has_removed_arch, f"sm110={has_110}, sm70={has_removed_arch}")
    else:
        check("compile commands", False, f"missing {commands_path}")

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        capability = torch.cuda.get_device_capability(0) if cuda_ok else None
        check(
            "Torch CUDA runtime",
            cuda_ok and capability == (11, 0),
            f"torch={torch.__version__}, CUDA={torch.version.cuda}, capability={capability}",
        )
    except Exception as exc:
        check("Torch CUDA runtime", False, repr(exc))

    report = {
        "build_dir": str(build_dir),
        "binary": str(binary),
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for item in checks:
            status = "PASS" if item["ok"] else "FAIL"
            detail = str(item["detail"])
            if "\n" in detail:
                detail = detail.splitlines()[0] + " ..."
            print(f"[{status}] {item['name']}: {detail}")
        print("Result:", "READY" if report["ok"] else "BLOCKED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
