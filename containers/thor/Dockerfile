ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Build and runtime dependencies not supplied by the NVIDIA PyTorch image.
# CUDA, cuDNN, and LibTorch deliberately come from the JetPack-compatible
# framework image rather than Ubuntu or PyPI packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        pkg-config \
        python3-joblib \
        python3-pil \
        python3-scipy \
        python3-tqdm \
        python3-yaml \
        libboost-system-dev \
        libeigen3-dev \
        libopencv-dev \
        libprotobuf-dev \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-evaluation.txt /tmp/requirements-evaluation.txt
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /tmp/requirements-evaluation.txt \
    && rm /tmp/requirements-evaluation.txt

ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --chmod=0755 hpcx_exec.sh /usr/local/bin/gps-slam-hpcx-exec

# Catch communication-stack ABI skew while building the image. The helper
# configures OpenMPI, UCC, and UCX as one versioned HPC-X stack; selecting only
# UCX causes libtorch_cpu to resolve incompatible MPI symbols.
RUN gps-slam-hpcx-exec bash -c \
        "ldd /opt/hpcx/ucc/lib/libucc.so.1 \
         && ldd /opt/hpcx/ucc/lib/libucc.so.1 \
            | grep -E 'libucs\\.so[^ ]* => /opt/hpcx/.*ucx/lib/' \
         && ldd /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so \
         && ldd /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so \
            | grep -E 'libmpi\\.so[^ ]* => /opt/hpcx/.*ompi/lib/' \
         && python3 -c \"import torch; print('torch import:', torch.__version__)\""
