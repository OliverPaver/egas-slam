#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
THIRDLIB_PATH="${PROJECT_ROOT}/ThirdLibs"
INSTALL_PATH="${THIRDLIB_PATH}/install"
BUILD_ROOT="${GPS_SLAM_THIRDLIB_BUILD_ROOT:-}"
BUILD_JOBS="${GPS_SLAM_BUILD_JOBS:-$(nproc)}"
COMPLETION_MARKER="${INSTALL_PATH}/.gps_slam_native_deps_complete"

if ! [[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPS_SLAM_BUILD_JOBS must be a positive integer, got: ${BUILD_JOBS}" >&2
    exit 1
fi

if [[ ! -d "${THIRDLIB_PATH}" ]]; then
    echo "Missing ${THIRDLIB_PATH}" >&2
    exit 1
fi

CMAKE_GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR_ARGS=(-G Ninja)
    CMAKE_GENERATOR_KEY=ninja
else
    CMAKE_GENERATOR_KEY=makefiles
fi
if [[ -z "${BUILD_ROOT}" ]]; then
    BUILD_ROOT="${PROJECT_ROOT}/build_thirdlibs/$(uname -m)-${CMAKE_GENERATOR_KEY}"
fi
mkdir -p "${INSTALL_PATH}" "${BUILD_ROOT}"
rm -f "${COMPLETION_MARKER}"

build_and_install()
{
    local name="$1"
    shift
    local source_dir="${THIRDLIB_PATH}/${name}"
    if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
        echo "Missing third-party source: ${source_dir}" >&2
        exit 1
    fi
    local build_dir="${BUILD_ROOT}/${name}"
    echo "[thirdlibs] configuring ${name}"
    cmake -S "${source_dir}" -B "${build_dir}" "${CMAKE_GENERATOR_ARGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_PREFIX_PATH="${INSTALL_PATH}" \
        "$@"
    cmake --build "${build_dir}" --parallel "${BUILD_JOBS}"
    cmake --install "${build_dir}"
}

build_and_install indicators \
    -DINDICATORS_BUILD_TESTS=OFF \
    -DINDICATORS_SAMPLES=OFF \
    -DINDICATORS_DEMO=OFF
build_and_install tensorboard_logger -DBUILD_TEST=OFF
build_and_install tinyply -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
build_and_install yaml-cpp \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF

{
    echo "schema=3"
    echo "architecture=$(uname -m)"
    echo "compiler=$(${CXX:-c++} -dumpmachine 2>/dev/null || echo unknown)"
} > "${COMPLETION_MARKER}"

echo "[thirdlibs] installed native dependencies to ${INSTALL_PATH}"
