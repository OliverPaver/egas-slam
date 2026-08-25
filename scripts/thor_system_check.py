#!/usr/bin/env python3
"""Report whether the current environment can build and run GPS-SLAM on Thor."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def command(*args: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return result.returncode, result.stdout.strip()


def result(ok: bool, label: str, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")


def find_nvcc() -> Path | None:
    configured = os.environ.get("CUDACXX")
    candidates = [configured, shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate).resolve()
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside-container", action="store_true")
    args = parser.parse_args()

    failures = 0
    machine = platform.machine()
    ok = machine == "aarch64"
    result(ok, "architecture", machine)
    failures += not ok

    memory_gib = 0.0
    try:
        memory_kib = int(Path("/proc/meminfo").read_text().split("MemTotal:", 1)[1].split()[0])
        memory_gib = memory_kib / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    ok = memory_gib >= 100.0
    result(ok, "unified memory", f"{memory_gib:.1f} GiB detected")
    failures += not ok

    nvcc = find_nvcc()
    ok = nvcc is not None
    result(ok, "CUDA compiler", str(nvcc) if nvcc else "nvcc not found")
    failures += not ok
    if nvcc:
        code, output = command(str(nvcc), "--list-gpu-arch")
        architectures = set(output.splitlines()) if code == 0 else set()
        ok = "compute_110" in architectures
        result(ok, "Thor CUDA target", "compute_110 available" if ok else output)
        failures += not ok

    code, output = command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    ok = code == 0 and "Thor" in output
    result(ok, "GPU runtime", output or "nvidia-smi failed")
    failures += not ok

    if args.inside_container:
        ucc = Path("/opt/hpcx/ucc/lib/libucc.so.1")
        code, output = command("ldd", str(ucc))
        libucs_match = re.search(r"libucs\.so(?:\.\d+)*\s+=>\s+(\S+)", output)
        libucs_path = libucs_match.group(1) if libucs_match else "not resolved"
        libucs_resolved = (
            str(Path(libucs_path).resolve(strict=False))
            if Path(libucs_path).is_absolute()
            else libucs_path
        )
        torch_cpu = Path(
            "/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so"
        )
        mpi_code, mpi_output = command("ldd", str(torch_cpu))
        libmpi_match = re.search(r"libmpi\.so(?:\.\d+)*\s+=>\s+(\S+)", mpi_output)
        libmpi_path = libmpi_match.group(1) if libmpi_match else "not resolved"
        libmpi_resolved = (
            str(Path(libmpi_path).resolve(strict=False))
            if Path(libmpi_path).is_absolute()
            else libmpi_path
        )
        hpcx_ok = (
            ucc.is_file()
            and code == 0
            and libucs_resolved.startswith("/opt/hpcx/")
            and "not found" not in output
            and torch_cpu.is_file()
            and mpi_code == 0
            and libmpi_resolved.startswith("/opt/hpcx/")
            and "not found" not in mpi_output
        )
        result(
            hpcx_ok,
            "HPC-X UCC/UCX/OpenMPI linkage",
            f"libucs={libucs_resolved}, libmpi={libmpi_resolved}",
        )
        failures += not hpcx_ok

    try:
        import torch

        torch_ok = bool(torch.cuda.is_available())
        capability = torch.cuda.get_device_capability(0) if torch_ok else None
        probe_sum = None
        if torch_ok:
            # Force allocation, kernel launch, synchronization, and a device to
            # host result transfer. Device enumeration alone does not prove
            # that CUDA minor-version compatibility can execute this image.
            probe = torch.arange(1024, device="cuda", dtype=torch.float32)
            probe_sum = float(probe.sum().item())
            torch.cuda.synchronize()
        torch_ok = torch_ok and capability == (11, 0) and probe_sum == 523776.0
        detail = (
            f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
            f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}, "
            f"capability={capability}, probe_sum={probe_sum}, "
            f"cmake={torch.utils.cmake_prefix_path}"
        )
    except Exception as exc:  # Import failures must remain actionable.
        torch_ok = False
        detail = repr(exc)
    result(torch_ok, "CUDA-enabled Torch", detail)
    failures += not torch_ok

    if args.inside_container:
        imex_ok = os.environ.get("NVIDIA_IMEX_CHANNELS") == "0"
        result(
            imex_ok,
            "NVIDIA 26.07 NVSHMEM workaround",
            f"NVIDIA_IMEX_CHANNELS={os.environ.get('NVIDIA_IMEX_CHANNELS', '<unset>')}",
        )
        failures += not imex_ok

        try:
            import scipy
            import torchvision
            import trimesh

            metric_ok = getattr(trimesh, "__version__", "") == "4.12.2"
            metric_detail = (
                f"torchvision={torchvision.__version__}, scipy={scipy.__version__}, "
                f"trimesh={trimesh.__version__}"
            )
        except Exception as exc:
            metric_ok = False
            metric_detail = repr(exc)
        result(metric_ok, "evaluation dependencies", metric_detail)
        failures += not metric_ok

        torch_home = Path(os.environ.get("TORCH_HOME", ""))
        checkpoints = torch_home / "hub" / "checkpoints"
        expected_weights = {
            checkpoints / "vgg.pth": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
            checkpoints / "vgg16-397923af.pth": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
        }
        weights_ok = all(
            path.is_file() and sha256(path) == expected
            for path, expected in expected_weights.items()
        )
        result(
            weights_ok,
            "offline metric weights",
            ", ".join(str(path) for path in expected_weights),
        )
        failures += not weights_ok

    code, output = command("findmnt", "-n", "-o", "FSTYPE", "-T", ".")
    fs_ok = code == 0 and output.strip() not in {"exfat", "vfat", "fuseblk"}
    result(fs_ok, "build filesystem", output or "unknown")
    failures += not fs_ok

    free_gib = shutil.disk_usage(Path.cwd()).free / 1024**3
    disk_ok = free_gib >= 30.0
    result(disk_ok, "free build space", f"{free_gib:.1f} GiB")
    failures += not disk_ok

    cmake = shutil.which("cmake")
    code, output = command(cmake or "cmake", "--version")
    match = re.search(r"cmake version (\d+)\.(\d+)", output)
    cmake_ok = code == 0 and match is not None and tuple(map(int, match.groups())) >= (3, 21)
    result(cmake_ok, "CMake", output.splitlines()[0] if output else "missing")
    failures += not cmake_ok

    if not args.inside_container:
        release = Path("/etc/nv_tegra_release")
        detail = release.read_text().splitlines()[0] if release.exists() else "not found"
        l4t_ok = "# R38" in detail or "# R39" in detail
        result(l4t_ok, "Jetson Linux BSP", detail)
        failures += not l4t_ok
        if not torch_ok:
            print("[INFO] Host Torch is optional when using scripts/thor_container.sh.")

    print(f"Result: {'READY' if failures == 0 else 'BLOCKED'} ({failures} failed checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
