#include "cpu_rules.h"

#include <cstring>
#include <cstdlib>

#include "sim_math.h"

int count_alive_species(const bool* alive, const int* species, int agent_count, int target_species) {
    int count = 0;
    for (int i = 0; i < agent_count; ++i) {
        if (alive[i] && species[i] == target_species) {
            ++count;
        }
    }
    return count;
}

int count_active_food(const int* food_active, int food_count) {
    int count = 0;
    for (int i = 0; i < food_count; ++i) {
        if (food_active[i] != 0) {
            ++count;
        }
    }
    return count;
}

static int find_empty_agent_slot(const bool* alive, int agent_count) {
    for (int i = 0; i < agent_count; ++i) {
        if (!alive[i]) {
            return i;
        }
    }
    return -1;
}

static int find_empty_food_slot(const int* food_active, int food_count) {
    for (int i = 0; i < food_count; ++i) {
        if (food_active[i] == 0) {
            return i;
        }
    }
    return -1;
}

static void activate_food(float* food_x,
                          float* food_y,
                          int* food_active,
                          int food_idx,
                          float x,
                          float y) {
    food_x[food_idx] = clamp01_host(x);
    food_y[food_idx] = clamp01_host(y);
    food_active[food_idx] = 1;
}

void clear_newly_dead_agents(const bool* was_alive,
                             bool* alive,
                             bool* exists,
                             const float* pos_x,
                             const float* pos_y,
                             float* food_x,
                             float* food_y,
                             int* food_active,
                             int agent_count,
                             int food_count) {
    for (int i = 0; i < agent_count; ++i) {
        if (was_alive[i] && !alive[i]) {
            int food_idx = find_empty_food_slot(food_active, food_count);
            if (food_idx >= 0) {
                activate_food(food_x, food_y, food_active, food_idx, pos_x[i], pos_y[i]);
            }
            exists[i] = false;
        }
    }
}

int respawn_food(float* food_x,
                 float* food_y,
                 int* food_active,
                 int food_count,
                 const SimParams& params) {
    int spawned = 0;
    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        if (food_active[food_idx] != 0) {
            continue;
        }

        int roll = std::rand() % 100;
        if (roll >= params.food_respawn_chance_percent) {
            continue;
        }

        activate_food(food_x,
                      food_y,
                      food_active,
                      food_idx,
                      random_unit_float(),
                      random_unit_float());
        ++spawned;
    }

    return spawned;
}

int process_predation(float* pos_x,
                      float* pos_y,
                      float* energy,
                      const float* size,
                      bool* alive,
                      bool* exists,
                      const int* species,
                      int agent_count,
                      const SimParams& params) {
    const float predation_radius_sq = params.predation_radius * params.predation_radius;
    int kills = 0;

    for (int bass_idx = 0; bass_idx < agent_count; ++bass_idx) {
        if (!alive[bass_idx] || species[bass_idx] != SPECIES_BASS) {
            continue;
        }
        if (energy[bass_idx] >= params.bass_max_energy) {
            energy[bass_idx] = params.bass_max_energy;
            continue;
        }

        for (int prey_idx = 0; prey_idx < agent_count; ++prey_idx) {
            if (!alive[prey_idx] ||
                (species[prey_idx] != SPECIES_BLUEGILL &&
                 species[prey_idx] != SPECIES_MINNOW)) {
                continue;
            }

            float diff_x = pos_x[prey_idx] - pos_x[bass_idx];
            float diff_y = pos_y[prey_idx] - pos_y[bass_idx];
            float distance_sq = diff_x * diff_x + diff_y * diff_y;
            if (distance_sq < predation_radius_sq) {
                int success_chance = species[prey_idx] == SPECIES_MINNOW ?
                                     params.minnow_predation_success_percent :
                                     params.bluegill_predation_success_percent;
                int roll = std::rand() % 100;
                if (roll >= success_chance) {
                    break;
                }

                alive[prey_idx] = false;
                exists[prey_idx] = false;
                energy[prey_idx] = 0.0f;
                energy[bass_idx] += params.bass_energy_gain * size[prey_idx];
                if (energy[bass_idx] > params.bass_max_energy) {
                    energy[bass_idx] = params.bass_max_energy;
                }
                ++kills;
                break;
            }
        }
    }

    return kills;
}

static int reproduce_species(float* pos_x,
                             float* pos_y,
                             float* energy,
                             float* size,
                             bool* alive,
                             bool* exists,
                             int* species,
                             float* dir_x,
                             float* dir_y,
                             bool* reproduced,
                             int agent_count,
                             int target_species,
                             float reproduction_energy,
                             float reproduction_cost,
                             float child_energy,
                             float child_size,
                             int max_children,
                             const SimParams& params) {
    const float mate_radius_sq = params.mate_radius * params.mate_radius;
    int births = 0;

    std::memset(reproduced, 0, static_cast<size_t>(agent_count) * sizeof(bool));

    for (int parent_a = 0; parent_a < agent_count; ++parent_a) {
        if (!alive[parent_a] ||
            species[parent_a] != target_species ||
            reproduced[parent_a] ||
            energy[parent_a] < reproduction_energy ||
            energy[parent_a] - reproduction_cost <= 0.0f) {
            continue;
        }

        for (int parent_b = parent_a + 1; parent_b < agent_count; ++parent_b) {
            if (!alive[parent_b] ||
                species[parent_b] != target_species ||
                reproduced[parent_b] ||
                energy[parent_b] < reproduction_energy ||
                energy[parent_b] - reproduction_cost <= 0.0f) {
                continue;
            }

            float diff_x = pos_x[parent_b] - pos_x[parent_a];
            float diff_y = pos_y[parent_b] - pos_y[parent_a];
            float distance_sq = diff_x * diff_x + diff_y * diff_y;
            if (distance_sq > mate_radius_sq) {
                continue;
            }

            int roll = std::rand() % 100;
            if (roll >= params.reproduction_chance_percent) {
                continue;
            }

            if (find_empty_agent_slot(alive, agent_count) < 0) {
                return births;
            }

            energy[parent_a] -= reproduction_cost;
            energy[parent_b] -= reproduction_cost;

            reproduced[parent_a] = true;
            reproduced[parent_b] = true;

            int child_count = 1 + (std::rand() % max_children);
            for (int child_number = 0; child_number < child_count; ++child_number) {
                int child_idx = find_empty_agent_slot(alive, agent_count);
                if (child_idx < 0) {
                    return births;
                }

                float child_jitter_x = 0.02f * (2.0f * random_unit_float() - 1.0f);
                float child_jitter_y = 0.02f * (2.0f * random_unit_float() - 1.0f);
                pos_x[child_idx] = clamp01_host(0.5f * (pos_x[parent_a] + pos_x[parent_b]) + child_jitter_x);
                pos_y[child_idx] = clamp01_host(0.5f * (pos_y[parent_a] + pos_y[parent_b]) + child_jitter_y);
                energy[child_idx] = child_energy;
                size[child_idx] = child_size;
                alive[child_idx] = true;
                exists[child_idx] = true;
                species[child_idx] = target_species;
                dir_x[child_idx] = 2.0f * random_unit_float() - 1.0f;
                dir_y[child_idx] = 2.0f * random_unit_float() - 1.0f;
                normalize_direction(&dir_x[child_idx], &dir_y[child_idx]);

                reproduced[child_idx] = true;
                ++births;
            }

            break;
        }
    }

    return births;
}

int reproduce_bluegill(float* pos_x,
                       float* pos_y,
                       float* energy,
                       float* size,
                       bool* alive,
                       bool* exists,
                       int* species,
                       float* dir_x,
                       float* dir_y,
                       bool* reproduced,
                       int agent_count,
                       const SimParams& params) {
    return reproduce_species(pos_x,
                             pos_y,
                             energy,
                             size,
                             alive,
                             exists,
                             species,
                             dir_x,
                             dir_y,
                             reproduced,
                             agent_count,
                             SPECIES_BLUEGILL,
                             params.reproduction_energy,
                             params.reproduction_cost,
                             params.child_energy,
                             params.bluegill_min_size,
                             params.max_children_per_birth,
                             params);
}

int reproduce_minnow(float* pos_x,
                     float* pos_y,
                     float* energy,
                     float* size,
                     bool* alive,
                     bool* exists,
                     int* species,
                     float* dir_x,
                     float* dir_y,
                     bool* reproduced,
                     int agent_count,
                     const SimParams& params) {
    return reproduce_species(pos_x,
                             pos_y,
                             energy,
                             size,
                             alive,
                             exists,
                             species,
                             dir_x,
                             dir_y,
                             reproduced,
                             agent_count,
                             SPECIES_MINNOW,
                             params.minnow_reproduction_energy,
                             params.minnow_reproduction_cost,
                             params.minnow_child_energy,
                             params.minnow_min_size,
                             params.minnow_max_children_per_birth,
                             params);
}

int reproduce_bass(float* pos_x,
                   float* pos_y,
                   float* energy,
                   float* size,
                   bool* alive,
                   bool* exists,
                   int* species,
                   float* dir_x,
                   float* dir_y,
                   bool* reproduced,
                   int agent_count,
                   const SimParams& params) {
    return reproduce_species(pos_x,
                             pos_y,
                             energy,
                             size,
                             alive,
                             exists,
                             species,
                             dir_x,
                             dir_y,
                             reproduced,
                             agent_count,
                             SPECIES_BASS,
                             params.bass_reproduction_energy,
                             params.bass_reproduction_cost,
                             params.bass_child_energy,
                             params.bass_min_size,
                             params.bass_max_children_per_birth,
                             params);
}
