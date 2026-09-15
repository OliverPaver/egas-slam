// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <cooperative_groups.h>

// All callers enter with the complete first warp (local ids 0..31).
// At each stage only the lower lanes write; their source lanes do not write
// during that stage. Barriers separate stages under independent scheduling.
// Lane zero retains the original +32,+16,+8,+4,+2,+1 addition tree.
template<class T>
inline __device__ void warpReduce(volatile T* sdata, int tid)
{
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncwarp(0xffffffffu);
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncwarp(0xffffffffu);
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncwarp(0xffffffffu);
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncwarp(0xffffffffu);
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncwarp(0xffffffffu);
    if (tid < 1) sdata[tid] += sdata[tid + 1];
    __syncwarp(0xffffffffu);
}

inline __device__ void warpReduce3(volatile float* sdata, int tid)
{
    if (tid < 32)
    {
        sdata[3 * tid + 0] += sdata[3 * (tid + 32) + 0];
        sdata[3 * tid + 1] += sdata[3 * (tid + 32) + 1];
        sdata[3 * tid + 2] += sdata[3 * (tid + 32) + 2];
    }
    __syncwarp(0xffffffffu);
    if (tid < 16)
    {
        sdata[3 * tid + 0] += sdata[3 * (tid + 16) + 0];
        sdata[3 * tid + 1] += sdata[3 * (tid + 16) + 1];
        sdata[3 * tid + 2] += sdata[3 * (tid + 16) + 2];
    }
    __syncwarp(0xffffffffu);
    if (tid < 8)
    {
        sdata[3 * tid + 0] += sdata[3 * (tid + 8) + 0];
        sdata[3 * tid + 1] += sdata[3 * (tid + 8) + 1];
        sdata[3 * tid + 2] += sdata[3 * (tid + 8) + 2];
    }
    __syncwarp(0xffffffffu);
    if (tid < 4)
    {
        sdata[3 * tid + 0] += sdata[3 * (tid + 4) + 0];
        sdata[3 * tid + 1] += sdata[3 * (tid + 4) + 1];
        sdata[3 * tid + 2] += sdata[3 * (tid + 4) + 2];
    }
    __syncwarp(0xffffffffu);
    if (tid < 2)
    {
        sdata[3 * tid + 0] += sdata[3 * (tid + 2) + 0];
        sdata[3 * tid + 1] += sdata[3 * (tid + 2) + 1];
        sdata[3 * tid + 2] += sdata[3 * (tid + 2) + 2];
    }
    __syncwarp(0xffffffffu);
    if (tid < 1)
    {
        sdata[3 * tid + 0] += sdata[3 * (tid + 1) + 0];
        sdata[3 * tid + 1] += sdata[3 * (tid + 1) + 1];
        sdata[3 * tid + 2] += sdata[3 * (tid + 1) + 2];
    }
    __syncwarp(0xffffffffu);
}

template <typename T>
__device__ int computePrefixSum_device(uint element, T *sum, int localSize, int localId)
{
    // Several callers return out-of-range threads before entering this scan.
    // A block-wide scan then reads unwritten shared slots in partial blocks.
    // Compact the currently participating lanes and reserve one contiguous
    // range per group; global ordering was already unspecified across blocks.
    (void)localSize;
    (void)localId;
    const auto group = cooperative_groups::coalesced_threads();
    const unsigned int rank = group.thread_rank();
    unsigned int inclusive = element;
    for (unsigned int offset = 1; offset < group.size(); offset <<= 1)
    {
        const unsigned int preceding = group.shfl_up(inclusive, offset);
        if (rank >= offset)
            inclusive += preceding;
    }
    const unsigned int total = group.shfl(inclusive, group.size() - 1);
    T base = 0;
    if (rank == 0 && total > 0)
        base = atomicAdd(sum, static_cast<T>(total));
    base = group.shfl(base, 0);
    return element == 0 ? -1 : static_cast<int>(base + inclusive - element);
}

__device__ static inline void atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

__device__ static inline void atomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

template<typename T>
__global__ void memsetKernel_device(T *devPtr, const T val, size_t nwords)
{
	size_t offset = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= nwords) return;
	devPtr[offset] = val;
}

template<typename T>
__global__ void memsetKernelLarge_device(T *devPtr, const T val, size_t nwords)
{
	size_t offset = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	if (offset >= nwords) return;
	devPtr[offset] = val;
}

template<typename T>
inline void memsetKernel(T *devPtr, const T val, size_t nwords)
{
	dim3 blockSize(256);
	dim3 gridSize((int)ceil((float)nwords / (float)blockSize.x));
	if (gridSize.x <= 65535) {
		memsetKernel_device<T> <<<gridSize,blockSize>>>(devPtr, val, nwords);
		ORcudaKernelCheck;
	} else {
		gridSize.x = (int)ceil(sqrt((float)gridSize.x));
		gridSize.y = (int)ceil((float)nwords / (float)(blockSize.x * gridSize.x));
		memsetKernelLarge_device<T> <<<gridSize,blockSize>>>(devPtr, val, nwords);
		ORcudaKernelCheck;
	}
}

template<typename T>
__global__ void fillArrayKernel_device(T *devPtr, size_t nwords)
{
	size_t offset = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= nwords) return;
	devPtr[offset] = offset;
}

template<typename T>
inline void fillArrayKernel(T *devPtr, size_t nwords)
{
	dim3 blockSize(256);
	dim3 gridSize((int)ceil((float)nwords / (float)blockSize.x));
	fillArrayKernel_device<T> <<<gridSize,blockSize>>>(devPtr, nwords);
	ORcudaKernelCheck;
}

