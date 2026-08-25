#################
# UseCUDA.cmake #
#################

enable_language(CUDA)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA Toolkit found")
else()
  message(WARNING "CUDA Toolkit not found!")
endif()

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDAToolkit_FOUND})

IF(WITH_CUDA)
  message(STATUS "Building with CUDA support")
  # Honour the architecture selected by the parent project/toolchain.  The
  # previous unconditional assignment silently replaced Thor's SM 11.0 with
  # 70;75;80; CUDA 13 no longer accepts the Volta (70) target.
  IF(NOT CMAKE_CUDA_ARCHITECTURES)
    MESSAGE(FATAL_ERROR "InfiniTAM requires an explicit CMAKE_CUDA_ARCHITECTURES value")
  ENDIF()
  message(STATUS "InfiniTAM CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  # Auto-detect the CUDA compute capability.
  # SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  # IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  #   INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
  # ENDIF()

  # # Set the compute capability flags.
  # FOREACH(compute_capability ${CUDA_COMPUTE_CAPABILITY})
  #   LIST(APPEND CUDA_NVCC_FLAGS --generate-code arch=compute_${compute_capability},code=sm_${compute_capability})
  # ENDFOREACH()
  # message(STATUS "CUDA compute capability: ${CUDA_COMPUTE_CAPABILITY}")

  message(STATUS "CUDA fast math: ${GPS_SLAM_CUDA_FAST_MATH}")
ELSE()
  message(STATUS "Building without CUDA support")
  ADD_DEFINITIONS(-DCOMPILE_WITHOUT_CUDA)
ENDIF()
