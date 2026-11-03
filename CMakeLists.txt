cmake_minimum_required(VERSION 3.21)
project(opensplat LANGUAGES CXX)

set(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF CACHE BOOL "Build for maximum CUDA device compatibility")
option(GPS_SLAM_BUILD_LEGACY_UI "Build the unused InfiniTAM GLUT UI implementation" OFF)
option(GPS_SLAM_BUILD_REMOTE_VIEWER "Build the TCP remote viewer" ON)
option(GPS_SLAM_CUDA_FAST_MATH "Use CUDA fast math (changes numerical results)" OFF)
option(GPS_SLAM_PREFER_HOST_CXX_RUNTIME "Resolve the host C++ runtime before LibTorch's exported filesystem symbols" OFF)
set(GPS_SLAM_HOST_CXX_RUNTIME_FIRST "")
if(GPS_SLAM_PREFER_HOST_CXX_RUNTIME)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "GPS_SLAM_PREFER_HOST_CXX_RUNTIME currently supports GNU/Linux only")
    endif()
    # Some desktop LibTorch wheels export an incompatible remove_all() whose
    # weak recursive-iterator dependency is absent from the host libstdc++.
    # Keep the host runtime first even with the linker's --as-needed default.
    set(GPS_SLAM_HOST_CXX_RUNTIME_FIRST "-Wl,--no-as-needed" stdc++ "-Wl,--as-needed")
endif()
include(cmake/TSDFCapacityProfiles.cmake)
option(GPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION "Compile added BasicEngine allocation and CSV diagnostics" ON)
if(NOT GPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION)
    add_compile_definitions(GPS_SLAM_DISABLE_BASIC_CAPACITY_INSTRUMENTATION)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(GPS_SLAM_LOCAL_PREFIXES "")
if(EXISTS "${PROJECT_SOURCE_DIR}/ThirdLibs/libtorch/share/cmake/Torch")
    list(APPEND GPS_SLAM_LOCAL_PREFIXES "${PROJECT_SOURCE_DIR}/ThirdLibs/libtorch")
endif()
if(EXISTS "${PROJECT_SOURCE_DIR}/ThirdLibs/install")
    list(APPEND GPS_SLAM_LOCAL_PREFIXES "${PROJECT_SOURCE_DIR}/ThirdLibs/install")
endif()
# Keep command-line/toolchain prefixes available for native AArch64 builds.
# Local bundled dependencies take precedence when they exist on desktop.
list(PREPEND CMAKE_PREFIX_PATH ${GPS_SLAM_LOCAL_PREFIXES})

# Read version
file(READ "VERSION" APP_VERSION)
string(STRIP "${APP_VERSION}" APP_VERSION)

# Read git commit
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" 
                OUTPUT_VARIABLE GIT_REV 
                ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if (NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git commit ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()

message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# Don't complain about the override from NANOFLANN_BUILD_EXAMPLES
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Use time-of-extraction for FetchContent'ed files modification time
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)
set(GPS_SLAM_FETCHCONTENT_ROOT
    "${PROJECT_SOURCE_DIR}/ThirdLibs/fetchcontent"
    CACHE PATH "Optional offline source root for FetchContent dependencies")
foreach(GPS_SLAM_FETCHCONTENT_DEP nlohmann_json nanoflann cxxopts glm)
    string(TOUPPER "${GPS_SLAM_FETCHCONTENT_DEP}" GPS_SLAM_FETCHCONTENT_DEP_UPPER)
    if(EXISTS "${GPS_SLAM_FETCHCONTENT_ROOT}/${GPS_SLAM_FETCHCONTENT_DEP}/CMakeLists.txt")
        set("FETCHCONTENT_SOURCE_DIR_${GPS_SLAM_FETCHCONTENT_DEP_UPPER}"
            "${GPS_SLAM_FETCHCONTENT_ROOT}/${GPS_SLAM_FETCHCONTENT_DEP}")
    else()
        message(FATAL_ERROR
            "Missing vendored FetchContent source ${GPS_SLAM_FETCHCONTENT_ROOT}/${GPS_SLAM_FETCHCONTENT_DEP}. "
            "Restore the transfer bundle instead of downloading an unpinned replacement.")
    endif()
endforeach()
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
)
FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
)
FetchContent_MakeAvailable(nlohmann_json nanoflann cxxopts)
FetchContent_Declare(glm
    URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
)
FetchContent_MakeAvailable(glm)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
# Test before enable_language(CUDA): CMake otherwise fills this variable from
# a compiler default, making an accidental non-Thor build look explicit.
if(NOT OPENSPLAT_MAX_CUDA_COMPATIBILITY AND
   (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL ""))
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES must be explicit (110 for Jetson AGX Thor). "
        "Compiler defaults can produce binaries that configure successfully but cannot run on the target GPU.")
endif()
enable_language(CUDA)
if(OPENSPLAT_MAX_CUDA_COMPATIBILITY)
    execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-arch
        OUTPUT_VARIABLE LIST_GPU_ARCH
        ERROR_QUIET)
    if(NOT LIST_GPU_ARCH)
        message(FATAL_ERROR
            "Cannot compile for maximum CUDA compatibility: nvcc does not support --list-gpu-arch")
    endif()

    # This mode is intentionally separate from Thor's explicit SM 110 build.
    string(REPLACE "\r" "" LIST_GPU_ARCH "${LIST_GPU_ARCH}")
    string(REPLACE "\n" ";" LIST_GPU_ARCH "${LIST_GPU_ARCH}")
    execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-code
        OUTPUT_VARIABLE LIST_GPU_CODE
        ERROR_QUIET)
    string(REPLACE "\r" "" LIST_GPU_CODE "${LIST_GPU_CODE}")
    string(REPLACE "\n" ";" LIST_GPU_CODE "${LIST_GPU_CODE}")

    set(CMAKE_CUDA_ARCHITECTURES "")
    list(GET LIST_GPU_CODE 0 TARGET_GPU_CODE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${TARGET_GPU_CODE}")
    set(IDX 0)
    foreach(GPU_ARCH ${LIST_GPU_ARCH})
        string(REGEX MATCH "compute_([0-9]+)" GPU_ARCH_VERSION "${GPU_ARCH}")
        list(APPEND CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
        list(GET LIST_GPU_CODE ${IDX} GPU_CODE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=${GPU_ARCH},code=${GPU_CODE}")
        math(EXPR IDX "${IDX}+1")
    endforeach()
    message(STATUS "CUDA compatibility flags: ${CMAKE_CUDA_FLAGS}")
else()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "^([0-9]+)([0-9])$" "\\1.\\2")
string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
message(STATUS "Torch CUDA architecture list: ${TORCH_CUDA_ARCH_LIST}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

##################### InfiniTAM ############################
add_subdirectory(InfiniTAM)


##################### LIBTORCH and OpenCV ############################
find_package(Torch REQUIRED)
separate_arguments(GPS_SLAM_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d features2d)

##################### My LIBS ############################
# yaml-cpp
find_package(yaml-cpp CONFIG REQUIRED)

# TensorBoard is useful for desktop experiments, but NVIDIA's PyTorch container
# bundles private protobuf headers which are not ABI-compatible with the system
# protobuf used by tensorboard_logger. Keep the optional logger out of embedded
# builds so a single executable never mixes both protobuf implementations.
option(GPS_SLAM_ENABLE_TENSORBOARD "Write TensorBoard scalar event files" ON)
if(GPS_SLAM_ENABLE_TENSORBOARD)
    find_package(Protobuf REQUIRED)
    set(TENSORBOARD_LOGGER_DIR "${CMAKE_SOURCE_DIR}/ThirdLibs/install")
    find_package(tensorboard_logger CONFIG REQUIRED HINTS ${TENSORBOARD_LOGGER_DIR})
    set(GPS_SLAM_PROTOBUF_ROOT "" CACHE PATH "Optional system protobuf installation prefix")
    # Torch can register a dummy protobuf::libprotobuf target and can place its
    # own private static protobuf ahead of the platform library.
    find_library(GPS_SLAM_PROTOBUF_LIBRARY
        NAMES protobuf
        HINTS ${GPS_SLAM_PROTOBUF_ROOT}
        PATHS
            "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
            "/usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
            /usr/lib /usr/local/lib
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )
    if(NOT GPS_SLAM_PROTOBUF_LIBRARY)
        message(FATAL_ERROR
            "Could not find the platform libprotobuf. Install libprotobuf-dev or set "
            "GPS_SLAM_PROTOBUF_ROOT to its installation prefix.")
    endif()
    message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}")
    message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
    message(STATUS "GPS_SLAM_PROTOBUF_LIBRARY: ${GPS_SLAM_PROTOBUF_LIBRARY}")
    message(STATUS "Protobuf_PROTOC_EXECUTABLE: ${Protobuf_PROTOC_EXECUTABLE}")
endif()

# eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# indicators
find_package(indicators REQUIRED)

# tinyply
find_package(tinyply REQUIRED)
message(STATUS "tinyply_FOUND: ${tinyply_FOUND}")
message(STATUS "tinyply_INCLUDE_DIRS: ${tinyply_INCLUDE_DIR}")
message(STATUS "tinyply_LIBRARIES: ${tinyply_LIBRARY_DIR}")

option(GPS_SLAM_USE_NVML "Use NVML for GPU memory diagnostics when available" ON)
set(GPS_SLAM_NVML_ROOT "" CACHE PATH "Optional NVML installation prefix")
set(GPS_SLAM_NVML_TARGET "")
if(GPS_SLAM_USE_NVML AND TARGET CUDA::nvml)
    set(GPS_SLAM_NVML_TARGET CUDA::nvml)
    set(GPS_SLAM_HAS_NVML ON)
    message(STATUS "NVML enabled through CUDA::nvml")
elseif(GPS_SLAM_USE_NVML)
    find_path(NVML_INCLUDE_DIR
        NAMES nvml.h
        HINTS ${GPS_SLAM_NVML_ROOT} ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_TARGET_DIR}
        PATH_SUFFIXES include
    )
    find_library(NVML_LIBRARY
        NAMES nvidia-ml libnvidia-ml.so.1
        HINTS
            ${GPS_SLAM_NVML_ROOT}
            "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia"
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_TARGET_DIR}
        PATH_SUFFIXES lib lib64 lib/stubs lib/aarch64-linux-gnu lib/x86_64-linux-gnu
    )
endif()
set(GPS_SLAM_NVML_INCLUDE_DIRS "")
if(GPS_SLAM_NVML_TARGET)
    # Imported targets carry their own include path.
elseif(GPS_SLAM_USE_NVML AND NVML_INCLUDE_DIR AND NVML_LIBRARY)
    message(STATUS "NVML enabled: ${NVML_LIBRARY}")
    set(GPS_SLAM_NVML_TARGET ${NVML_LIBRARY})
    list(APPEND GPS_SLAM_NVML_INCLUDE_DIRS ${NVML_INCLUDE_DIR})
    set(GPS_SLAM_HAS_NVML ON)
else()
    message(WARNING "NVML disabled or not found; GPU memory diagnostics will use cudaMemGetInfo")
endif()

##############################################################
#####################   gsplat rasterizer
set(RASTERIZER_LIBS rasterizer)
file(GLOB RASTERIZER_SRC_FILES ${PROJECT_SOURCE_DIR}/gsplat/rasterizer/*.cu ${PROJECT_SOURCE_DIR}/gsplat/rasterizer/*.cuh)

set(GPS_SLAM_RASTERIZER_ARCHIVE "" CACHE FILEPATH "Optional SHA-pinned capacity-independent rasterizer archive")
set(GPS_SLAM_RASTERIZER_ARCHIVE_SHA256 "" CACHE STRING "Required SHA256 for the reused archive")
set(GPS_SLAM_RASTERIZER_ARCHIVE_MANIFEST "" CACHE FILEPATH "Required transitive dependency inventory")
if(GPS_SLAM_RASTERIZER_ARCHIVE)
    if(NOT GPS_SLAM_RASTERIZER_ARCHIVE_SHA256 OR NOT GPS_SLAM_RASTERIZER_ARCHIVE_MANIFEST)
        message(FATAL_ERROR "Rasterizer reuse requires both a SHA256 and dependency inventory")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    get_property(GPS_SLAM_BASE_CXX_FLAGS CACHE CMAKE_CXX_FLAGS PROPERTY VALUE)
    get_property(GPS_SLAM_BASE_CUDA_FLAGS CACHE CMAKE_CUDA_FLAGS PROPERTY VALUE)
    # Torch emits its own -gencode flags and can set the local architecture
    # variable to OFF; compare the explicitly selected cache value instead.
    get_property(GPS_SLAM_SELECTED_ARCHITECTURES CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/capacity_rasterizer_inventory.py"
        --verify "${GPS_SLAM_RASTERIZER_ARCHIVE_MANIFEST}"
        --archive "${GPS_SLAM_RASTERIZER_ARCHIVE}" --sha256 "${GPS_SLAM_RASTERIZER_ARCHIVE_SHA256}"
        --expect "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        --expect "CMAKE_CUDA_ARCHITECTURES=${GPS_SLAM_SELECTED_ARCHITECTURES}"
        --expect "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        --expect "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        --expect "CMAKE_CXX_FLAGS=${GPS_SLAM_BASE_CXX_FLAGS}"
        --expect "CMAKE_CUDA_FLAGS=${GPS_SLAM_BASE_CUDA_FLAGS}"
        --expect "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
        --expect "CMAKE_CUDA_FLAGS_RELEASE=${CMAKE_CUDA_FLAGS_RELEASE}"
        --expect "GPS_SLAM_CUDA_FAST_MATH=${GPS_SLAM_CUDA_FAST_MATH}"
        RESULT_VARIABLE GPS_SLAM_ARCHIVE_CHECK)
    if(NOT GPS_SLAM_ARCHIVE_CHECK EQUAL 0)
        message(FATAL_ERROR "Rasterizer archive source/compiler/container verification failed")
    endif()
    add_library(rasterizer STATIC IMPORTED)
    set_target_properties(rasterizer PROPERTIES IMPORTED_LOCATION "${GPS_SLAM_RASTERIZER_ARCHIVE}")
    target_link_libraries(rasterizer INTERFACE CUDA::cudart glm::glm-header-only)
else()
    if(GPS_SLAM_RASTERIZER_ARCHIVE_SHA256 OR GPS_SLAM_RASTERIZER_ARCHIVE_MANIFEST)
        message(FATAL_ERROR "Rasterizer reuse metadata supplied without an archive")
    endif()
    add_library(rasterizer ${RASTERIZER_SRC_FILES})
    target_link_libraries(rasterizer PUBLIC CUDA::cudart)
    target_include_directories(rasterizer PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
    target_link_libraries(rasterizer PUBLIC glm::glm-header-only)
    target_compile_options(rasterizer PRIVATE
        ${GPS_SLAM_TORCH_CXX_FLAGS}
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${GPS_SLAM_CUDA_FAST_MATH}>>:--use_fast_math>)
    set_target_properties(rasterizer PROPERTIES LINKER_LANGUAGE CXX)
endif()


##############################################################
#####################   gsplat c++ binding
file(GLOB GSPLAT_SRC_FILES ${PROJECT_SOURCE_DIR}/gsplat/*.cpp)
add_library(gsplat ${GSPLAT_SRC_FILES})
install(TARGETS gsplat DESTINATION bin)
set_property(TARGET gsplat PROPERTY CXX_STANDARD 17)
target_include_directories(gsplat PRIVATE
    ${PROJECT_SOURCE_DIR}/gsplat/rasterizer
)
target_link_libraries(gsplat PUBLIC ${STDPPFS_LIBRARY} ${RASTERIZER_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(gsplat PUBLIC
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)
if (NOT WIN32)
    target_link_libraries(gsplat PUBLIC pthread)
endif()
target_compile_definitions(gsplat PRIVATE USE_CUDA)
target_compile_options(gsplat PRIVATE ${GPS_SLAM_TORCH_CXX_FLAGS})


file(GLOB OURS_SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB SLAM_SRC_FILES ${PROJECT_SOURCE_DIR}/slam/*.cpp ${PROJECT_SOURCE_DIR}/slam/TsdfFusion/*.cpp)
if(NOT GPS_SLAM_BUILD_LEGACY_UI)
    list(FILTER SLAM_SRC_FILES EXCLUDE REGEX "/UIEngine\\.cpp$")
else()
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

##############################################################
#####################   shared GPS-SLAM implementation
add_library(gps_slam_core STATIC ${OURS_SRC_FILES} ${SLAM_SRC_FILES})
target_link_libraries(gps_slam_core PUBLIC
        gsplat
        Eigen3::Eigen
        indicators::indicators
        tinyply
        yaml-cpp::yaml-cpp
        InputSource 
        ITMLib 
        MiniSlamGraphLib 
        ORUtils 
        FernRelocLib 
        CUDA::cudart
        ${GPS_SLAM_NVML_TARGET}
)
if(GPS_SLAM_ENABLE_TENSORBOARD)
    target_link_libraries(gps_slam_core PUBLIC
        tensorboard_logger::tensorboard_logger
        ${GPS_SLAM_PROTOBUF_LIBRARY}
    )
    # PUBLIC is required because pipeline.h changes layout with this feature.
    target_compile_definitions(gps_slam_core PUBLIC GPS_SLAM_ENABLE_TENSORBOARD=1)
endif()
if(GPS_SLAM_BUILD_LEGACY_UI)
    target_link_libraries(gps_slam_core PUBLIC OpenGL::GL GLUT::GLUT)
endif()
target_include_directories(gps_slam_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/slam
    ${PROJECT_SOURCE_DIR}/gsplat
    ${PROJECT_SOURCE_DIR}/InfiniTAM
    ${GPS_SLAM_NVML_INCLUDE_DIRS}
)
target_compile_options(gps_slam_core PRIVATE ${GPS_SLAM_TORCH_CXX_FLAGS})
if(GPS_SLAM_HAS_NVML)
    target_compile_definitions(gps_slam_core PRIVATE GPS_SLAM_HAS_NVML=1)
endif()

#####################   slam trainer
add_executable(slam_trainer slam_trainer.cpp)
target_link_libraries(slam_trainer PRIVATE ${GPS_SLAM_HOST_CXX_RUNTIME_FIRST} gps_slam_core)
target_compile_options(slam_trainer PRIVATE ${GPS_SLAM_TORCH_CXX_FLAGS})

if(GPS_SLAM_PREFER_HOST_CXX_RUNTIME)
    add_executable(desktop_filesystem_regression scripts/desktop_filesystem_regression.cpp)
    target_link_libraries(desktop_filesystem_regression PRIVATE
        ${GPS_SLAM_HOST_CXX_RUNTIME_FIRST} gps_slam_core ${CMAKE_DL_LIBS})
endif()

#####################   optional TCP remote viewer
if(GPS_SLAM_BUILD_REMOTE_VIEWER)
    find_package(Boost REQUIRED COMPONENTS system)
    add_executable(remote_viewer remote_viewer.cpp)
    target_link_libraries(remote_viewer PRIVATE ${GPS_SLAM_HOST_CXX_RUNTIME_FIRST} gps_slam_core Boost::system)
    target_compile_options(remote_viewer PRIVATE ${GPS_SLAM_TORCH_CXX_FLAGS})
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB OPENCV_DLL "${OPENCV_DIR}/x64/vc16/bin/opencv_world490.dll")
    set(DLLS_TO_COPY ${TORCH_DLLS} ${OPENCV_DLL})
    add_custom_command(TARGET opensplat
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLLS_TO_COPY}
        $<TARGET_FILE_DIR:opensplat>)
endif (MSVC)

add_compile_definitions(GLOG_USE_GLOG_EXPORT)
