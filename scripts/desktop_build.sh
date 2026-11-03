#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${PROJECT_ROOT}/scripts/desktop_env.sh"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:${PATH}
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
BUILD_DIR="${EGAS_BUILD_DIR:-${PROJECT_ROOT}/build_desktop}"
BUILD_JOBS="${EGAS_BUILD_JOBS:-4}"
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo 'EGAS_BUILD_JOBS must be positive' >&2; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo 'Use thor_build.sh on AArch64' >&2; exit 1; }
TORCH_PREFIX=$("${EGAS_PYTHON}" -c 'import torch; print(torch.utils.cmake_prefix_path)')
"${EGAS_PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available(), 'Torch cannot access CUDA'
assert torch.cuda.get_device_capability() == (8, 6), 'This desktop profile targets RTX 3070 / SM 86'
print(f'Torch {torch.__version__}, CUDA {torch.version.cuda}, ABI={torch._C._GLIBCXX_USE_CXX11_ABI}')
print(torch.cuda.get_device_name())
PY
MARKER="${PROJECT_ROOT}/ThirdLibs/install/.gps_slam_native_deps_complete"
if [[ ! -f "${MARKER}" ]] || ! grep -qx architecture=x86_64 "${MARKER}"; then
    GPS_SLAM_BUILD_JOBS="${BUILD_JOBS}" bash "${PROJECT_ROOT}/build_third_libs.sh"
fi
/usr/bin/cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DTorch_DIR="${TORCH_PREFIX}/Torch" \
    -DCMAKE_PREFIX_PATH="${TORCH_PREFIX};${PROJECT_ROOT}/ThirdLibs/install" \
    -DGPS_SLAM_TSDF_HASH_PROFILE=medium_128k \
    -DGPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION=OFF \
    -DGPS_SLAM_BUILD_LEGACY_UI=OFF -DGPS_SLAM_BUILD_REMOTE_VIEWER=OFF \
    -DGPS_SLAM_ENABLE_TENSORBOARD=OFF -DGPS_SLAM_CUDA_FAST_MATH=OFF \
    -DGPS_SLAM_PREFER_HOST_CXX_RUNTIME=ON
/usr/bin/cmake --build "${BUILD_DIR}" --target slam_trainer desktop_filesystem_regression --parallel "${BUILD_JOBS}"
"${BUILD_DIR}/desktop_filesystem_regression"
printf 'Desktop trainer: %s/slam_trainer\n' "${BUILD_DIR}"
