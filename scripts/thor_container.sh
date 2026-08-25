#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINER_DIR="${PROJECT_ROOT}/containers/thor"
IMAGE="${GPS_SLAM_THOR_IMAGE:-gps-slam-thor:pytorch-26.07-jp7.1}"
ENGINE="${GPS_SLAM_CONTAINER_ENGINE:-docker}"
USE_SUDO="${GPS_SLAM_DOCKER_SUDO:-0}"
HOST_DATA_ROOT="${GPS_SLAM_DATA_ROOT:-${PROJECT_ROOT}/data}"
HOST_WEIGHT_ROOT="${GPS_SLAM_WEIGHT_ROOT:-${PROJECT_ROOT}/.cache/torch_hub}"
HOST_RUN_ROOT="${GPS_SLAM_RUN_ROOT:-${PROJECT_ROOT}/.runs}"
HOST_LEGACY_TRANSFER_ROOT="${GPS_SLAM_LEGACY_TRANSFER_ROOT:-}"
ACTION="${1:-help}"

usage()
{
    cat <<'EOF'
Usage: scripts/thor_container.sh <command> [arguments]

Commands:
  image        Build the JetPack 7.1-compatible development image.
  check        Verify CUDA, Torch, and SM 11.0 access inside the image.
  build        Build GPS-SLAM and native ThirdLibs inside the image.
  baseline-build
               Build the faithful legacy_large GPS-SLAM baseline in build_legacy/.
  regressions  Run the lightweight policy, metric, and pose-graph regressions.
  source-check Verify the algorithm tree against the exact imported R2 tag.
  shell        Open an interactive development shell.
  run CMD...   Run an arbitrary command in the development image.
  telemetry-run CMD...
               Run a container command while recording host tegrastats,
               clocks, fan, temperatures, and throttle-counter deltas.

Set GPS_SLAM_DOCKER_SUDO=1 to use an interactive one-off `sudo docker`
workflow instead of granting the login user root-equivalent docker-group access.

Host paths (defaults shown):
  GPS_SLAM_DATA_ROOT=<repository>/data
  GPS_SLAM_WEIGHT_ROOT=<repository>/.cache/torch_hub
  GPS_SLAM_RUN_ROOT=<repository>/.runs
  GPS_SLAM_LEGACY_TRANSFER_ROOT=  (optional; mounted at /workspace/transfer)
EOF
}

if [[ "${ACTION}" == "help" || "${ACTION}" == "--help" || "${ACTION}" == "-h" ]]; then
    usage
    exit 0
fi

if ! command -v "${ENGINE}" >/dev/null 2>&1; then
    echo "Missing container engine: ${ENGINE}" >&2
    exit 1
fi

case "${USE_SUDO}" in
    0)
        ENGINE_CMD=("${ENGINE}")
        ;;
    1)
        if ! command -v sudo >/dev/null 2>&1; then
            echo "GPS_SLAM_DOCKER_SUDO=1 was requested, but sudo is unavailable." >&2
            exit 1
        fi
        sudo -v
        ENGINE_CMD=(sudo "${ENGINE}")
        ;;
    *)
        echo "GPS_SLAM_DOCKER_SUDO must be 0 or 1, got: ${USE_SUDO}" >&2
        exit 1
        ;;
esac

if ! "${ENGINE_CMD[@]}" info >/dev/null 2>&1; then
    echo "Cannot access the ${ENGINE} daemon." >&2
    echo "Use GPS_SLAM_DOCKER_SUDO=1 for a one-off sudo workflow, or deliberately configure daemon access." >&2
    echo "Membership in the docker group grants root-equivalent host privileges." >&2
    exit 1
fi

if [[ "${ACTION}" == "image" ]]; then
    exec "${ENGINE_CMD[@]}" build --pull -t "${IMAGE}" "${CONTAINER_DIR}"
fi

mkdir -p "${HOST_DATA_ROOT}" "${HOST_WEIGHT_ROOT}" "${HOST_RUN_ROOT}"
HOST_DATA_ROOT=$(cd "${HOST_DATA_ROOT}" && pwd)
HOST_WEIGHT_ROOT=$(cd "${HOST_WEIGHT_ROOT}" && pwd)
HOST_RUN_ROOT=$(cd "${HOST_RUN_ROOT}" && pwd)
if [[ -n "${HOST_LEGACY_TRANSFER_ROOT}" ]]; then
    if [[ ! -d "${HOST_LEGACY_TRANSFER_ROOT}" ]]; then
        echo "GPS_SLAM_LEGACY_TRANSFER_ROOT is not a directory: ${HOST_LEGACY_TRANSFER_ROOT}" >&2
        exit 1
    fi
    HOST_LEGACY_TRANSFER_ROOT=$(cd "${HOST_LEGACY_TRANSFER_ROOT}" && pwd)
fi

CONTAINER_HASH_PROFILE="${GPS_SLAM_TSDF_HASH_PROFILE:-medium_128k}"
CONTAINER_BUILD_DIR="${GPS_SLAM_THOR_BUILD_DIR:-/workspace/project/build_thor}"
if [[ "${ACTION}" == "baseline-build" ]]; then
    CONTAINER_HASH_PROFILE="legacy_large"
    CONTAINER_BUILD_DIR="/workspace/project/build_legacy"
fi
IMAGE_ID=$("${ENGINE_CMD[@]}" image inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null || true)
HOST_NVPMODEL_STATE="unavailable"
if command -v nvpmodel >/dev/null 2>&1; then
    HOST_NVPMODEL_STATE=$(nvpmodel -q 2>&1 || true)
fi

run_args=(
    run --rm
    --runtime=nvidia
    --gpus=all
    --ipc=host
    --network=host
    --ulimit memlock=-1
    --user "$(id -u):$(id -g)"
    --env HOME=/tmp
    --env NVIDIA_IMEX_CHANNELS=0
    --env TORCH_HOME=/tmp/gps_slam_torch_cache
    --env GPS_SLAM_BUILD_JOBS="${GPS_SLAM_BUILD_JOBS:-6}"
    --env GPS_SLAM_BUILD_REMOTE_VIEWER="${GPS_SLAM_BUILD_REMOTE_VIEWER:-0}"
    --env GPS_SLAM_TSDF_HASH_PROFILE="${CONTAINER_HASH_PROFILE}"
    --env GPS_SLAM_THOR_BUILD_DIR="${CONTAINER_BUILD_DIR}"
    --env GPS_SLAM_CONTAINER_IMAGE="${IMAGE}"
    --env GPS_SLAM_CONTAINER_IMAGE_ID="${IMAGE_ID}"
    --env GPS_SLAM_HOST_NVPMODEL_STATE="${HOST_NVPMODEL_STATE}"
    --env GPS_SLAM_DATA_ROOT=/workspace/data
    --env GPS_SLAM_RUN_ROOT=/workspace/runs
    --env GPS_SLAM_BENCHMARK_FREEZE="${GPS_SLAM_BENCHMARK_FREEZE:-/workspace/project/BENCHMARK_FREEZE.json}"
    --volume "${PROJECT_ROOT}:/workspace/project"
    --volume "${HOST_DATA_ROOT}:/workspace/data:ro"
    --volume "${HOST_DATA_ROOT}:/workspace/project/data:ro"
    --volume "${HOST_WEIGHT_ROOT}:/tmp/gps_slam_torch_cache/hub:ro"
    --volume "${HOST_RUN_ROOT}:/workspace/runs"
    --volume "${HOST_RUN_ROOT}:/workspace/project/output"
    --workdir /workspace/project
)
if [[ -n "${HOST_LEGACY_TRANSFER_ROOT}" ]]; then
    run_args+=(--volume "${HOST_LEGACY_TRANSFER_ROOT}:/workspace/transfer")
fi
container_exec=(/usr/local/bin/gps-slam-hpcx-exec)

for group_name in video render; do
    group_id=$(getent group "${group_name}" | cut -d: -f3 || true)
    if [[ -n "${group_id}" ]]; then
        run_args+=(--group-add "${group_id}")
    fi
done

case "${ACTION}" in
    check)
        exec "${ENGINE_CMD[@]}" "${run_args[@]}" "${IMAGE}" "${container_exec[@]}" \
            python3 scripts/thor_system_check.py --inside-container
        ;;
    build|baseline-build)
        exec "${ENGINE_CMD[@]}" "${run_args[@]}" "${IMAGE}" "${container_exec[@]}" \
            bash scripts/thor_build.sh
        ;;
    regressions)
        exec "${ENGINE_CMD[@]}" "${run_args[@]}" "${IMAGE}" "${container_exec[@]}" bash -lc \
            'python3 scripts/tracking_failure_policy_regression.py && python3 scripts/validation_hardening_regression.py && python3 scripts/metrics_hardening_regression.py && python3 scripts/benchmark_freeze_regression.py && python3 scripts/benchmark_hardening_regression.py && python3 scripts/realtime_hardening_regression.py && python3 scripts/promotion_hardening_regression.py && python3 scripts/pose_graph_solver_regression.py --build-dir build_thor'
        ;;
    source-check)
        exec "${ENGINE_CMD[@]}" "${run_args[@]}" "${IMAGE}" "${container_exec[@]}" \
            python3 tools/release_hygiene.py --algorithm-equivalence thor-paper-20260819-r2-source
        ;;
    shell)
        exec "${ENGINE_CMD[@]}" "${run_args[@]}" --interactive --tty "${IMAGE}" "${container_exec[@]}" bash
        ;;
    run)
        shift
        if [[ $# -eq 0 ]]; then
            echo "scripts/thor_container.sh run requires a command" >&2
            exit 1
        fi
        exec "${ENGINE_CMD[@]}" "${run_args[@]}" "${IMAGE}" "${container_exec[@]}" "$@"
        ;;
    telemetry-run)
        shift
        if [[ $# -eq 0 ]]; then
            echo "scripts/thor_container.sh telemetry-run requires a command" >&2
            exit 1
        fi
        telemetry_stamp=$(date -u +%Y%m%dT%H%M%SZ)
        telemetry_dir="${GPS_SLAM_TELEMETRY_DIR:-${HOST_RUN_ROOT}/telemetry/${telemetry_stamp}}"
        telemetry_args=(
            python3 "${PROJECT_ROOT}/scripts/thor_telemetry.py"
            --output-dir "${telemetry_dir}"
            --interval-ms "${GPS_SLAM_TELEMETRY_INTERVAL_MS:-500}"
        )
        if [[ "${GPS_SLAM_REQUIRE_CLEAN_TELEMETRY:-1}" == "1" ]]; then
            telemetry_args+=(--require-clean)
        fi
        exec "${telemetry_args[@]}" -- \
            "${ENGINE_CMD[@]}" "${run_args[@]}" "${IMAGE}" "${container_exec[@]}" "$@"
        ;;
    *)
        echo "Unknown command: ${ACTION}" >&2
        usage >&2
        exit 1
        ;;
esac
