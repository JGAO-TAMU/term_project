#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "sim_types.h"

__global__ void simulate_agents_kernel(float* pos_x,
                                       float* pos_y,
                                       float* energy,
                                       const float* death_energy,
                                       float* size,
                                       int* spawn_cooldown,
                                       bool* alive,
                                       int* species,
                                       float* dir_x,
                                       float* dir_y,
                                       const float* food_x,
                                       const float* food_y,
                                       int* food_active,
                                       const int* agent_cell_head,
                                       const int* agent_cell_next,
                                       const int* food_cell_head,
                                       const int* food_cell_next,
                                       int agent_count,
                                       int food_count,
                                       int tick,
                                       SimParams params);

#endif
