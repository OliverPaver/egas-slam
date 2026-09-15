#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${GPS_SLAM_THOR_BUILD_DIR:-${PROJECT_ROOT}/build_thor}"
BUILD_JOBS="${GPS_SLAM_BUILD_JOBS:-$(nproc)}"
HASH_PROFILE="${GPS_SLAM_TSDF_HASH_PROFILE:-medium_128k}"
BUILD_REMOTE_VIEWER="${GPS_SLAM_BUILD_REMOTE_VIEWER:-0}"
BASIC_INSTRUMENTATION="${GPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION:-OFF}"

case "${BASIC_INSTRUMENTATION}" in
    ON|OFF) ;;
    *) echo "GPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION must be ON or OFF" >&2; exit 1 ;;
esac

for command_name in cmake python3; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done

NVCC="${CUDACXX:-}"
if [[ -z "${NVCC}" ]] || [[ ! -x "${NVCC}" ]]; then
    NVCC=$(command -v nvcc 2>/dev/null || true)
fi
if [[ -z "${NVCC}" ]] && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC=/usr/local/cuda/bin/nvcc
fi
if [[ -z "${NVCC}" ]] || [[ ! -x "${NVCC}" ]]; then
    echo "Missing nvcc. Install the JetPack CUDA development tools or set CUDACXX." >&2
    exit 1
fi
NVCC=$(readlink -f "${NVCC}")
CUDA_ROOT=$(cd "$(dirname "${NVCC}")/.." && pwd)
export CUDACXX="${NVCC}"
export PATH="$(dirname "${NVCC}"):${PATH}"

if [[ $(uname -m) != "aarch64" ]]; then
    echo "Warning: thor_build.sh is intended for native aarch64/Jetson builds." >&2
fi

if ! "${NVCC}" --list-gpu-arch 2>/dev/null | grep -q '^compute_110$'; then
    echo "The installed CUDA compiler does not advertise compute_110 support." >&2
    echo "Install the Thor JetPack CUDA development package before building." >&2
    exit 1
fi

case "${HASH_PROFILE}" in
    legacy_large|medium_128k|compact_64k|legacy_pool_512k|legacy_pool_1m|legacy_256k_hash4m|legacy_pool_2m|legacy_256k_hash8m) ;;
    *)
        echo "Unsupported GPS_SLAM_TSDF_HASH_PROFILE=${HASH_PROFILE}" >&2
        exit 1
        ;;
esac

if ! [[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPS_SLAM_BUILD_JOBS must be a positive integer, got: ${BUILD_JOBS}" >&2
    exit 1
fi

case "${BUILD_REMOTE_VIEWER}" in
    0|1) ;;
    *)
        echo "GPS_SLAM_BUILD_REMOTE_VIEWER must be 0 or 1, got: ${BUILD_REMOTE_VIEWER}" >&2
        exit 1
        ;;
esac

case "${GPS_SLAM_BUILD_DEPS:-auto}" in
    auto|0|1) ;;
    *)
        echo "GPS_SLAM_BUILD_DEPS must be auto, 0, or 1, got: ${GPS_SLAM_BUILD_DEPS}" >&2
        exit 1
        ;;
esac

TORCH_PREFIX="${GPS_SLAM_TORCH_PREFIX:-}"
if [[ -z "${TORCH_PREFIX}" ]]; then
    TORCH_PREFIX=$(python3 - <<'PY'
try:
    import torch
    print(torch.utils.cmake_prefix_path)
except Exception:
    pass
PY
)
fi
if [[ -z "${TORCH_PREFIX}" ]] || \
   [[ ! -f "${TORCH_PREFIX}/Torch/TorchConfig.cmake" && \
      ! -f "${TORCH_PREFIX}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    echo "Could not locate an AArch64 Torch CMake package." >&2
    echo "Install a JetPack-compatible NVIDIA PyTorch package/container or set GPS_SLAM_TORCH_PREFIX." >&2
    exit 1
fi

if ! TORCH_REPORT=$(python3 - <<'PY' 2>&1
import sys
import os
import torch

capability = "not checked (build-only mode)"
device = "not initialized"
if os.environ.get("GPS_SLAM_BUILD_SKIP_GPU_CHECK", "0") != "1":
    if not torch.cuda.is_available():
        raise SystemExit("Torch cannot access CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability != (11, 0):
        raise SystemExit(f"Expected Thor capability (11, 0), got {capability}")
    device = torch.cuda.get_device_name(0)
abi_fn = getattr(torch, "compiled_with_cxx11_abi", None)
abi = abi_fn() if abi_fn is not None else getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown")
print(
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"device={device} capability={capability} cxx11_abi={abi}"
)
PY
); then
    echo "The selected Torch installation is not usable on Thor: ${TORCH_REPORT}" >&2
    exit 1
fi

# Resolve the framework first so a missing Torch install fails before the
# comparatively expensive native third-party build begins.
DEPS_MARKER="${PROJECT_ROOT}/ThirdLibs/install/.gps_slam_native_deps_complete"
DEPS_READY=0
DEPS_COMPILER=$(${CXX:-c++} -dumpmachine 2>/dev/null || echo unknown)
DEPS_REQUIRED_CONFIGS=(
    "${PROJECT_ROOT}/ThirdLibs/install/lib/cmake/indicators/indicatorsConfig.cmake"
    "${PROJECT_ROOT}/ThirdLibs/install/cmake/tensorboard_loggerConfig.cmake"
    "${PROJECT_ROOT}/ThirdLibs/install/lib/cmake/tinyply/tinyplyConfig.cmake"
    "${PROJECT_ROOT}/ThirdLibs/install/lib/cmake/yaml-cpp/yaml-cpp-config.cmake"
)
DEPS_CONFIGS_READY=1
for config_path in "${DEPS_REQUIRED_CONFIGS[@]}"; do
    if [[ ! -f "${config_path}" ]]; then
        DEPS_CONFIGS_READY=0
        break
    fi
done
if [[ -f "${DEPS_MARKER}" ]] && \
   grep -qx 'schema=3' "${DEPS_MARKER}" && \
   grep -qx "architecture=$(uname -m)" "${DEPS_MARKER}" && \
   grep -qx "compiler=${DEPS_COMPILER}" "${DEPS_MARKER}" && \
   [[ "${DEPS_CONFIGS_READY}" == "1" ]]; then
    DEPS_READY=1
fi
if [[ "${GPS_SLAM_BUILD_DEPS:-auto}" == "1" ]] || \
   { [[ "${GPS_SLAM_BUILD_DEPS:-auto}" == "auto" ]] && [[ "${DEPS_READY}" == "0" ]]; }; then
    GPS_SLAM_BUILD_JOBS="${BUILD_JOBS}" bash "${PROJECT_ROOT}/build_third_libs.sh"
elif [[ "${GPS_SLAM_BUILD_DEPS:-auto}" == "0" && "${DEPS_READY}" == "0" ]]; then
    echo "Native ThirdLibs are incomplete for $(uname -m); remove GPS_SLAM_BUILD_DEPS=0 or build them first." >&2
    exit 1
fi

CMAKE_GENERATOR="${GPS_SLAM_CMAKE_GENERATOR:-}"
if [[ -z "${CMAKE_GENERATOR}" ]]; then
    if command -v ninja >/dev/null 2>&1; then
        CMAKE_GENERATOR=Ninja
    else
        CMAKE_GENERATOR="Unix Makefiles"
    fi
fi

echo "[thor] CUDA compiler: ${NVCC}"
echo "[thor] CUDA root:     ${CUDA_ROOT}"
echo "[thor] Torch prefix:  ${TORCH_PREFIX}"
echo "[thor] Torch runtime: ${TORCH_REPORT}"
echo "[thor] Generator:     ${CMAKE_GENERATOR}"
echo "[thor] Hash profile:  ${HASH_PROFILE}"
echo "[thor] Remote viewer: ${BUILD_REMOTE_VIEWER}"

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_CUDA_ARCHITECTURES=110 \
    -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DGPS_SLAM_BUILD_LEGACY_UI=OFF \
    -DGPS_SLAM_BUILD_REMOTE_VIEWER="$([[ "${BUILD_REMOTE_VIEWER}" == "1" ]] && echo ON || echo OFF)" \
    -DGPS_SLAM_ENABLE_TENSORBOARD=OFF \
    -DGPS_SLAM_CUDA_FAST_MATH=OFF \
    -DGPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION="${BASIC_INSTRUMENTATION}" \
    -DGPS_SLAM_TSDF_HASH_PROFILE="${HASH_PROFILE}" \
    -DCMAKE_PREFIX_PATH="${TORCH_PREFIX};${PROJECT_ROOT}/ThirdLibs/install" \
    -DGPS_SLAM_RASTERIZER_ARCHIVE="${GPS_SLAM_RASTERIZER_ARCHIVE:-}" \
    -DGPS_SLAM_RASTERIZER_ARCHIVE_SHA256="${GPS_SLAM_RASTERIZER_ARCHIVE_SHA256:-}" \
    -DGPS_SLAM_RASTERIZER_ARCHIVE_MANIFEST="${GPS_SLAM_RASTERIZER_ARCHIVE_MANIFEST:-}"

BUILD_TARGETS=(slam_trainer)
if [[ "${BUILD_REMOTE_VIEWER}" == "1" ]]; then
    BUILD_TARGETS+=(remote_viewer)
fi
cmake --build "${BUILD_DIR}" --target "${BUILD_TARGETS[@]}" --parallel "${BUILD_JOBS}"
POST_BUILD_ARGS=()
if [[ "${GPS_SLAM_BUILD_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    POST_BUILD_ARGS+=(--skip-cuda-runtime)
fi
python3 "${PROJECT_ROOT}/scripts/thor_post_build_check.py" \
    --build-dir "${BUILD_DIR}" \
    --hash-profile "${HASH_PROFILE}" "${POST_BUILD_ARGS[@]}"

echo "[thor] build complete"
echo "[thor] trainer: ${BUILD_DIR}/slam_trainer"
if [[ "${BUILD_REMOTE_VIEWER}" == "1" ]]; then
    echo "[thor] viewer:  ${BUILD_DIR}/remote_viewer"
fi
