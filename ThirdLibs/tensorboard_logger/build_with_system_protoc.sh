#!/bin/bash
# Build tensorboard_logger using system protoc (3.21.12)

cd /home/oliver/GPS_SLAM/GPS-SLAM/ThirdLibs/tensorboard_logger

# Clean previous build
rm -rf build
mkdir -p build
cd build

# Use system protoc and library
# protoc 3.21.12 should match libprotobuf 3.21.12
cmake .. \
    -DProtobuf_INCLUDE_DIR="/usr/include" \
    -DProtobuf_LIBRARY="/usr/lib/x86_64-linux-gnu/libprotobuf.so" \
    -DProtobuf_PROTOC_EXECUTABLE="/usr/bin/protoc" \
    -DCMAKE_INSTALL_PREFIX=/home/oliver/GPS_SLAM/GPS-SLAM/ThirdLibs/install

# Build
make -j16

# Install
sudo make install

echo "tensorboard_logger built successfully with system protoc!"
