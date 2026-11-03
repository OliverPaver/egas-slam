#!/usr/bin/env bash
# Source this file for the local x86-64 development environment.
export EGAS_PYTHON="${EGAS_PYTHON:-/home/oliver/miniconda3/bin/python}"
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GPS_SLAM_DATA_ROOT="${GPS_SLAM_DATA_ROOT:-/home/oliver/egas_workspace/data}"
export GPS_SLAM_RUN_ROOT="${GPS_SLAM_RUN_ROOT:-/home/oliver/egas_workspace/runs}"
