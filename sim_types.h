#ifndef SIM_TYPES_H
#define SIM_TYPES_H

#include <cstddef>

static const int SPECIES_BLUEGILL = 0;
static const int SPECIES_BASS = 1;
static const int SPECIES_MINNOW = 2;

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
    unsigned int seed;
    float bluegill_energy_gain;
    float bluegill_energy_drain;
    float bluegill_max_energy;
    float minnow_energy_gain;
    float minnow_energy_drain;
    float minnow_max_energy;
    float bass_energy_drain;
    float bass_energy_gain;
    float bass_max_energy;
    float detection_radius;
    float predation_radius;
    float move_step;
    float decay_factor;
    float noise_scale;
    float reproduction_energy;
    float reproduction_cost;
    float reproduction_cooldown_days;
    float mate_radius;
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
};

struct HostBuffers {
    float* pos_x;
    float* pos_y;
    float* energy;
    float* size;
    bool* alive;
    bool* was_alive;
    bool* exists;
    bool* reproduced;
    int* reproduction_cooldown;
    int* species;
    float* dir_x;
    float* dir_y;
    float* food_x;
    float* food_y;
    int* food_active;
};

struct DeviceBuffers {
    float* pos_x;
    float* pos_y;
    float* energy;
    float* size;
    bool* alive;
    int* reproduction_cooldown;
    int* species;
    float* dir_x;
    float* dir_y;
    float* food_x;
    float* food_y;
    int* food_active;
};

#endif
