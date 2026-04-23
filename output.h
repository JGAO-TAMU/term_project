#ifndef OUTPUT_H
#define OUTPUT_H

#include <cstdio>

void write_state(FILE* out,
                 const char* label,
                 const float* pos_x,
                 const float* pos_y,
                 const float* energy,
                 const float* size,
                 const bool* alive,
                 const int* species,
                 int agent_count,
                 const float* food_x,
                 const float* food_y,
                 const int* food_active,
                 int food_count);

void stream_tick_state(int tick,
                       float day,
                       const float* pos_x,
                       const float* pos_y,
                       const float* energy,
                       const float* size,
                       const bool* alive,
                       const bool* exists,
                       const int* species,
                       int agent_count,
                       const float* food_x,
                       const float* food_y,
                       const int* food_active,
                       int food_count);

#endif
