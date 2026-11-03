#!/usr/bin/env bash
# Source after activating the Python environment used to build and run EGAS-SLAM.
export EGAS_PYTHON="${EGAS_PYTHON:-$(command -v python3)}"
if [[ -z "${EGAS_PYTHON}" ]] || ! command -v "${EGAS_PYTHON}" >/dev/null 2>&1; then
    echo 'Activate a Python environment or set EGAS_PYTHON to its interpreter.' >&2
    return 1
fi
export EGAS_WORKSPACE_ROOT="${EGAS_WORKSPACE_ROOT:-${HOME}/egas_workspace}"
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GPS_SLAM_DATA_ROOT="${GPS_SLAM_DATA_ROOT:-${EGAS_WORKSPACE_ROOT}/data}"
export GPS_SLAM_RUN_ROOT="${GPS_SLAM_RUN_ROOT:-${EGAS_WORKSPACE_ROOT}/runs}"
