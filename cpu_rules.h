#ifndef CPU_RULES_H
#define CPU_RULES_H

#include "sim_types.h"

int count_alive_species(const bool* alive, const int* species, int agent_count, int target_species);
int count_active_food(const int* food_active, int food_count);

void clear_newly_dead_agents(const bool* was_alive,
                             bool* alive,
                             bool* exists,
                             const float* pos_x,
                             const float* pos_y,
                             float* food_x,
                             float* food_y,
                             int* food_active,
                             int agent_count,
                             int food_count);

int respawn_food(float* food_x,
                 float* food_y,
                 int* food_active,
                 int food_count,
                 const SimParams& params);

int process_predation(float* pos_x,
                      float* pos_y,
                      float* energy,
                      const float* size,
                      bool* alive,
                      bool* exists,
                      const int* species,
                      const float* cover_x,
                      const float* cover_y,
                      const float* cover_intensity,
                      int agent_count,
                      int cover_count,
                      const SimParams& params);

int reproduce_bluegill(float* pos_x,
                       float* pos_y,
                       float* energy,
                       float* death_energy,
                       float* size,
                       int* spawn_cooldown,
                       bool* alive,
                       bool* exists,
                       int* species,
                       float* dir_x,
                       float* dir_y,
                       bool* reproduced,
                       int tick,
                       int agent_count,
                       const SimParams& params);

int reproduce_minnow(float* pos_x,
                     float* pos_y,
                     float* energy,
                     float* death_energy,
                     float* size,
                     int* spawn_cooldown,
                     bool* alive,
                     bool* exists,
                     int* species,
                     float* dir_x,
                     float* dir_y,
                     bool* reproduced,
                     int tick,
                     int agent_count,
                     const SimParams& params);

int reproduce_bass(float* pos_x,
                   float* pos_y,
                   float* energy,
                   float* death_energy,
                   float* size,
                   int* spawn_cooldown,
                   bool* alive,
                   bool* exists,
                   int* species,
                   float* dir_x,
                   float* dir_y,
                   bool* reproduced,
                   int tick,
                   int agent_count,
                   const SimParams& params);

#endif
