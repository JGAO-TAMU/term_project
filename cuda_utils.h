#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

inline void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", call, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#endif
