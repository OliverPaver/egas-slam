##########################
# SetCUDALibTarget.cmake #
##########################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

# message(STATUS "sources ${sources}")
ADD_LIBRARY(${targetname} STATIC ${sources} ${headers} ${templates})
target_link_libraries(${targetname} PUBLIC CUDA::cudart)
target_include_directories(${targetname} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(${targetname} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${GPS_SLAM_CUDA_FAST_MATH}>>:--use_fast_math>
)
