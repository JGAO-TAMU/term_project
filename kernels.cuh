#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "sim_types.h"

__global__ void simulate_agents_kernel(float* pos_x,
                                       float* pos_y,
                                       float* energy,
                                       float* size,
                                       bool* alive,
                                       int* species,
                                       float* dir_x,
                                       float* dir_y,
                                       const float* food_x,
                                       const float* food_y,
                                       int* food_active,
                                       int agent_count,
                                       int food_count,
                                       int step,
                                       SimParams params);

#endif
