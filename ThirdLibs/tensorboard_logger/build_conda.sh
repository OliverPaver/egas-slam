#!/bin/bash
# Build tensorboard_logger using conda protobuf

# Clean previous build
rm -rf build
mkdir -p build
cd build

# Configure with conda paths
CONDA_PREFIX=$HOME/miniconda3
cmake .. \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DProtobuf_INCLUDE_DIR="${CONDA_PREFIX}/include" \
    -DProtobuf_LIBRARY="${CONDA_PREFIX}/lib/libprotobuf.so" \
    -DProtobuf_PROTOC_EXECUTABLE="${CONDA_PREFIX}/bin/protoc" \
    -DCMAKE_INSTALL_PREFIX=/home/oliver/GPS_SLAM/GPS-SLAM/ThirdLibs/install

# Build
make -j16

# Install
make install

echo "tensorboard_logger built and installed successfully!"
