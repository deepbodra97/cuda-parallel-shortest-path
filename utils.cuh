#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cassert>

/*  Wrapper to provide error checking for CUDA API calls */

inline
cudaError_t cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__
void warmpupGpu() {
    __shared__ int s_tid;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        s_tid = tid;
    }
    __syncthreads();
    tid = s_tid;
}

#endif /*CUDA_CHECK_CUH*/