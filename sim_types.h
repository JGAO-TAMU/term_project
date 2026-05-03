#ifndef SIM_TYPES_H
#define SIM_TYPES_H

#include <cstddef>

static const int SPECIES_BLUEGILL = 0;
static const int SPECIES_BASS = 1;
static const int SPECIES_MINNOW = 2;

static const int PATTERN_GRID = 0;
static const int PATTERN_RANDOM = 1;
static const int PATTERN_CLUSTER = 2;
static const int PATTERN_RING = 3;
static const int PATTERN_DENSE_CIRCLE = 4;
static const int PATTERN_DENSE_RECT = 5;

struct SimParams {
    int tick_count;
    int ticks_per_day;
    int reproduction_chance_percent;
    int food_respawn_chance_percent;
    int predation_success_percent;
    int bluegill_predation_success_percent;
    int minnow_predation_success_percent;
    int max_children_per_birth;
    int bass_max_children_per_birth;
    int minnow_max_children_per_birth;
    int food_pattern;
    int cover_pattern;
    int spatial_grid_dim;
    unsigned int seed;
    float bluegill_energy_gain;
    float bluegill_energy_drain;
    float bluegill_max_energy;
    float bluegill_soft_cap;
    float bluegill_refuge_scale;
    float minnow_energy_gain;
    float minnow_energy_drain;
    float minnow_max_energy;
    float minnow_soft_cap;
    float minnow_refuge_scale;
    float bass_energy_drain;
    float bass_energy_gain;
    float bass_max_energy;
    float bass_soft_cap;
    float death_energy_variation;
    int bass_starvation_resistant_percent;
    float bass_starvation_reserve;
    float detection_radius;
    float predation_radius;
    float move_step;
    float decay_factor;
    float noise_scale;
    float reproduction_energy;
    float reproduction_cost;
    float reproduction_cooldown_days;
    float mate_radius;
    float bass_mate_radius;
    float child_energy;
    float minnow_reproduction_energy;
    float minnow_reproduction_cost;
    float minnow_child_energy;
    float minnow_reproduction_cooldown_days;
    float bass_reproduction_energy;
    float bass_reproduction_cost;
    float bass_child_energy;
    float bass_reproduction_cooldown_days;
    float size_growth_threshold;
    float size_shrink_threshold;
    float size_growth_rate;
    float size_shrink_rate;
    float min_size;
    float max_size;
    float bluegill_min_size;
    float bluegill_max_size;
    float minnow_min_size;
    float minnow_max_size;
    float bass_min_size;
    float bass_max_size;
    float food_energy_gain;
    float food_eat_radius;
    float cover_radius;
    float cover_protection_scale;
    float predation_half_sat;
    float min_refuge_multiplier;
    float max_prey_per_bass_per_day;
};

struct HostBuffers {
    float* pos_x;
    float* pos_y;
    float* energy;
    float* death_energy;
    float* size;
    int* spawn_cooldown;
    bool* alive;
    bool* was_alive;
    bool* exists;
    bool* reproduced;
    int* species;
    float* dir_x;
    float* dir_y;
    float* food_x;
    float* food_y;
    int* food_active;
    int* agent_cell_head;
    int* agent_cell_next;
    int* food_cell_head;
    int* food_cell_next;
    float* cover_x;
    float* cover_y;
    float* cover_intensity;
};

struct DeviceBuffers {
    float* pos_x;
    float* pos_y;
    float* energy;
    float* death_energy;
    float* size;
    int* spawn_cooldown;
    bool* alive;
    int* species;
    float* dir_x;
    float* dir_y;
    float* food_x;
    float* food_y;
    int* food_active;
    int* agent_cell_head;
    int* agent_cell_next;
    int* food_cell_head;
    int* food_cell_next;
};

#endif
