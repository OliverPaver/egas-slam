#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#define ORcudaKernelCheck do { if (cudaGetLastError() != cudaSuccess) std::abort(); } while(0)
#include "ITMCUDAUtils.h"

__global__ void checkPrefix(unsigned int *total, int *offsets, int count)
{
    const int id = threadIdx.x;
    if (id >= count) return; // Same partial block contract as the raycast caller.
    offsets[id] = computePrefixSum_device<unsigned int>(1, total, 256, id);
}
int main()
{
    unsigned int *total;
    int *offsets;
    cudaMalloc(&total, sizeof(unsigned int));
    cudaMalloc(&offsets, 256 * sizeof(int));
    int failures = 0;
    for (int count : {1, 31, 32, 63, 129, 255, 256})
    {
        cudaMemset(total, 0, sizeof(unsigned int));
        cudaMemset(offsets, 0xff, 256 * sizeof(int));
        checkPrefix<<<1, 256>>>(total, offsets, count);
        if (cudaDeviceSynchronize() != cudaSuccess) return 2;
        unsigned int sum;
        std::vector<int> positions(count);
        cudaMemcpy(&sum, total, sizeof(sum), cudaMemcpyDeviceToHost);
        cudaMemcpy(positions.data(), offsets, count * sizeof(int), cudaMemcpyDeviceToHost);
        std::sort(positions.begin(), positions.end());
        bool pass = sum == count;
        for (int i = 0; i < count; ++i) pass = pass && positions[i] == i;
        std::printf("count=%d total=%u first=%d last=%d passed=%d\n", count, sum, positions.front(), positions.back(), pass);
        failures += !pass;
    }
    cudaFree(offsets);
    cudaFree(total);
    return failures ? 1 : 0;
}
