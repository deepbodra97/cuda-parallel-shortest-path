#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include "cuda_runtime.h"

#include <cstdio>
#include <cassert>

/*  Wrapper to provide error checking for CUDA API calls */

inline
cudaError_t cudaCheck(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

#endif /*CUDA_CHECK_CUH*/