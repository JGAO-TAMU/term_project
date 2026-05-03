#include "cpu_rules.h"

#include <cstring>
#include <cstdlib>
#include <vector>

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

static int find_empty_agent_slot_from(const bool* alive, int agent_count, int* cursor) {
    int start = cursor != nullptr ? *cursor : 0;
    if (start < 0) {
        start = 0;
    }

    for (int i = start; i < agent_count; ++i) {
        if (!alive[i]) {
            if (cursor != nullptr) {
                *cursor = i + 1;
            }
            return i;
        }
    }

    for (int i = 0; i < start && i < agent_count; ++i) {
        if (!alive[i]) {
            if (cursor != nullptr) {
                *cursor = i + 1;
            }
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

static int cooldown_ticks_for_days_host(float cooldown_days, const SimParams& params) {
    if (cooldown_days <= 0.0f || params.ticks_per_day <= 0) {
        return 0;
    }

    int ticks = static_cast<int>(cooldown_days * static_cast<float>(params.ticks_per_day) + 0.5f);
    if (ticks < 1) {
        ticks = 1;
    }
    return ticks;
}

static float crowding_multiplier(int population, float soft_cap) {
    if (soft_cap <= 0.0f) {
        return 1.0f;
    }

    float crowding = 1.0f - static_cast<float>(population) / soft_cap;
    if (crowding < 0.0f) {
        return 0.0f;
    }
    if (crowding > 1.0f) {
        return 1.0f;
    }
    return crowding;
}

static float random_death_energy_threshold(int target_species, const SimParams& params) {
    if (target_species == SPECIES_BASS &&
        params.bass_starvation_resistant_percent > 0 &&
        (std::rand() % 100) < params.bass_starvation_resistant_percent) {
        float reserve_fraction = 0.25f + 0.75f * random_unit_float();
        return -params.bass_starvation_reserve * reserve_fraction;
    }

    return params.death_energy_variation * random_unit_float();
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

struct CpuSpatialGrid {
    int grid_dim;
    std::vector<int> head;
    std::vector<int> next;
};

static int clamp_grid_coord_host(int coord, int grid_dim) {
    if (coord < 0) {
        return 0;
    }
    if (coord >= grid_dim) {
        return grid_dim - 1;
    }
    return coord;
}

static int spatial_cell_for_point_host(float x, float y, int grid_dim) {
    int cell_x = clamp_grid_coord_host(static_cast<int>(x * static_cast<float>(grid_dim)), grid_dim);
    int cell_y = clamp_grid_coord_host(static_cast<int>(y * static_cast<float>(grid_dim)), grid_dim);
    return cell_y * grid_dim + cell_x;
}

static int cell_min_for_radius_host(float value, float radius, int grid_dim) {
    return clamp_grid_coord_host(static_cast<int>((value - radius) * static_cast<float>(grid_dim)), grid_dim);
}

static int cell_max_for_radius_host(float value, float radius, int grid_dim) {
    return clamp_grid_coord_host(static_cast<int>((value + radius) * static_cast<float>(grid_dim)), grid_dim);
}

static int normalized_grid_dim(const SimParams& params) {
    if (params.spatial_grid_dim < 1) {
        return 1;
    }
    return params.spatial_grid_dim;
}

static CpuSpatialGrid build_cover_grid(const float* cover_x,
                                       const float* cover_y,
                                       int cover_count,
                                       const SimParams& params) {
    CpuSpatialGrid grid;
    grid.grid_dim = normalized_grid_dim(params);
    int cell_count = grid.grid_dim * grid.grid_dim;
    grid.head.assign(cell_count, -1);
    grid.next.assign(cover_count, -1);

    for (int cover_idx = 0; cover_idx < cover_count; ++cover_idx) {
        int cell = spatial_cell_for_point_host(cover_x[cover_idx],
                                               cover_y[cover_idx],
                                               grid.grid_dim);
        grid.next[cover_idx] = grid.head[cell];
        grid.head[cell] = cover_idx;
    }
    return grid;
}

static CpuSpatialGrid build_eligible_agent_grid(const float* pos_x,
                                                const float* pos_y,
                                                const float* energy,
                                                const int* spawn_cooldown,
                                                const bool* alive,
                                                const int* species,
                                                const bool* reproduced,
                                                int agent_count,
                                                int target_species,
                                                float reproduction_energy,
                                                float reproduction_cost,
                                                const SimParams& params) {
    CpuSpatialGrid grid;
    grid.grid_dim = normalized_grid_dim(params);
    int cell_count = grid.grid_dim * grid.grid_dim;
    grid.head.assign(cell_count, -1);
    grid.next.assign(agent_count, -1);

    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (!alive[agent_idx] ||
            species[agent_idx] != target_species ||
            reproduced[agent_idx] ||
            spawn_cooldown[agent_idx] > 0 ||
            energy[agent_idx] < reproduction_energy ||
            energy[agent_idx] - reproduction_cost <= 0.0f) {
            continue;
        }

        int cell = spatial_cell_for_point_host(pos_x[agent_idx],
                                               pos_y[agent_idx],
                                               grid.grid_dim);
        grid.next[agent_idx] = grid.head[cell];
        grid.head[cell] = agent_idx;
    }
    return grid;
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

static float local_cover_intensity(const float* cover_x,
                                   const float* cover_y,
                                   const float* cover_intensity,
                                   const CpuSpatialGrid& cover_grid,
                                   float x,
                                   float y,
                                   float cover_radius) {
    if (cover_grid.next.empty() || cover_radius <= 0.0f) {
        return 0.0f;
    }

    float cover_radius_sq = cover_radius * cover_radius;
    float total_intensity = 0.0f;
    int min_cell_x = cell_min_for_radius_host(x, cover_radius, cover_grid.grid_dim);
    int max_cell_x = cell_max_for_radius_host(x, cover_radius, cover_grid.grid_dim);
    int min_cell_y = cell_min_for_radius_host(y, cover_radius, cover_grid.grid_dim);
    int max_cell_y = cell_max_for_radius_host(y, cover_radius, cover_grid.grid_dim);

    for (int cell_y = min_cell_y; cell_y <= max_cell_y; ++cell_y) {
        for (int cell_x = min_cell_x; cell_x <= max_cell_x; ++cell_x) {
            int cell = cell_y * cover_grid.grid_dim + cell_x;
            for (int cover_idx = cover_grid.head[cell];
                 cover_idx >= 0;
                 cover_idx = cover_grid.next[cover_idx]) {
                float diff_x = cover_x[cover_idx] - x;
                float diff_y = cover_y[cover_idx] - y;
                float distance_sq = diff_x * diff_x + diff_y * diff_y;
                if (distance_sq > cover_radius_sq) {
                    continue;
                }

                float falloff = 1.0f - distance_sq / cover_radius_sq;
                total_intensity += cover_intensity[cover_idx] * falloff;
            }
        }
    }

    if (total_intensity > 1.0f) {
        total_intensity = 1.0f;
    }
    return total_intensity;
}

static float average_cover_for_species(const float* pos_x,
                                       const float* pos_y,
                                       const bool* alive,
                                       const int* species,
                                       int agent_count,
                                       int target_species,
                                       const float* cover_x,
                                       const float* cover_y,
                                       const float* cover_intensity,
                                       const CpuSpatialGrid& cover_grid,
                                       const SimParams& params) {
    float total_cover = 0.0f;
    int covered_count = 0;

    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (!alive[agent_idx] || species[agent_idx] != target_species) {
            continue;
        }

        total_cover += local_cover_intensity(cover_x,
                                             cover_y,
                                             cover_intensity,
                                             cover_grid,
                                             pos_x[agent_idx],
                                             pos_y[agent_idx],
                                             params.cover_radius);
        ++covered_count;
    }

    if (covered_count <= 0) {
        return 0.0f;
    }

    return total_cover / static_cast<float>(covered_count);
}

static float refuge_multiplier(float average_cover,
                               float refuge_scale,
                               const SimParams& params) {
    float refuge = 1.0f - params.cover_protection_scale * refuge_scale * average_cover;
    if (refuge < params.min_refuge_multiplier) {
        refuge = params.min_refuge_multiplier;
    }
    if (refuge > 1.0f) {
        refuge = 1.0f;
    }
    return refuge;
}

static float prey_availability(int prey_count,
                               int base_predation_percent,
                               float average_cover,
                               float refuge_scale,
                               const SimParams& params) {
    if (prey_count <= 0 || base_predation_percent <= 0) {
        return 0.0f;
    }

    float vulnerability = static_cast<float>(base_predation_percent) / 100.0f;
    float refuge = refuge_multiplier(average_cover, refuge_scale, params);
    return static_cast<float>(prey_count) * vulnerability * refuge * refuge;
}

static int stochastic_round(float value) {
    if (value <= 0.0f) {
        return 0;
    }

    int rounded = static_cast<int>(value);
    if (random_unit_float() < value - static_cast<float>(rounded)) {
        ++rounded;
    }
    return rounded;
}

static int remove_prey_kills(float* energy,
                             const float* size,
                             bool* alive,
                             bool* exists,
                             const int* species,
                             int agent_count,
                             int target_species,
                             int kill_count,
                             float* total_prey_size) {
    int remaining_candidates = count_alive_species(alive, species, agent_count, target_species);
    int remaining_kills = kill_count;
    int kills = 0;
    float consumed_size = 0.0f;

    for (int agent_idx = 0; agent_idx < agent_count && remaining_kills > 0; ++agent_idx) {
        if (!alive[agent_idx] || species[agent_idx] != target_species) {
            continue;
        }

        int roll = std::rand() % remaining_candidates;
        if (roll < remaining_kills) {
            alive[agent_idx] = false;
            exists[agent_idx] = false;
            energy[agent_idx] = 0.0f;
            consumed_size += size[agent_idx];
            --remaining_kills;
            ++kills;
        }
        --remaining_candidates;
    }

    if (total_prey_size != nullptr) {
        *total_prey_size = consumed_size;
    }
    return kills;
}

static void distribute_predation_energy(float* energy,
                                        const bool* alive,
                                        const int* species,
                                        int agent_count,
                                        float total_gain,
                                        const SimParams& params) {
    if (total_gain <= 0.0f) {
        return;
    }

    int hungry_bass = 0;
    float total_room = 0.0f;
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx] &&
            species[agent_idx] == SPECIES_BASS &&
            energy[agent_idx] < params.bass_max_energy) {
            ++hungry_bass;
            total_room += params.bass_max_energy - energy[agent_idx];
        }
    }

    if (hungry_bass <= 0) {
        return;
    }

    if (total_gain >= total_room) {
        for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
            if (!alive[agent_idx] || species[agent_idx] != SPECIES_BASS) {
                continue;
            }
            if (energy[agent_idx] < params.bass_max_energy) {
                energy[agent_idx] = params.bass_max_energy;
            }
        }
        return;
    }

    float weight_sum = 0.0f;
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (!alive[agent_idx] ||
            species[agent_idx] != SPECIES_BASS ||
            energy[agent_idx] >= params.bass_max_energy) {
            continue;
        }

        float room = params.bass_max_energy - energy[agent_idx];
        unsigned int hash = static_cast<unsigned int>(agent_idx * 1103515245u + 12345u);
        float noise = static_cast<float>(hash % 1024u) / 1024.0f;
        float weight = room * (0.25f + noise);
        weight_sum += weight;
    }

    if (weight_sum <= 0.0f) {
        return;
    }

    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (!alive[agent_idx] ||
            species[agent_idx] != SPECIES_BASS ||
            energy[agent_idx] >= params.bass_max_energy) {
            continue;
        }

        float room = params.bass_max_energy - energy[agent_idx];
        unsigned int hash = static_cast<unsigned int>(agent_idx * 1103515245u + 12345u);
        float noise = static_cast<float>(hash % 1024u) / 1024.0f;
        float weight = room * (0.25f + noise);
        energy[agent_idx] += total_gain * (weight / weight_sum);
        if (energy[agent_idx] > params.bass_max_energy) {
            energy[agent_idx] = params.bass_max_energy;
        }
    }
}

static int count_hungry_bass(const float* energy,
                             const bool* alive,
                             const int* species,
                             int agent_count,
                             const SimParams& params) {
    int hungry_bass = 0;
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx] &&
            species[agent_idx] == SPECIES_BASS &&
            energy[agent_idx] < params.bass_max_energy) {
            ++hungry_bass;
        }
    }
    return hungry_bass;
}

static float total_hungry_bass_energy_room(const float* energy,
                                           const bool* alive,
                                           const int* species,
                                           int agent_count,
                                           const SimParams& params) {
    float total_room = 0.0f;
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (!alive[agent_idx] || species[agent_idx] != SPECIES_BASS) {
            continue;
        }

        float room = params.bass_max_energy - energy[agent_idx];
        if (room > 0.0f) {
            total_room += room;
        }
    }
    return total_room;
}

static float total_prey_size_for_species(const float* size,
                                         const bool* alive,
                                         const int* species,
                                         int agent_count,
                                         int target_species) {
    float total_size = 0.0f;
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx] && species[agent_idx] == target_species) {
            total_size += size[agent_idx];
        }
    }
    return total_size;
}

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
                      const SimParams& params) {
    int predator_count = count_hungry_bass(energy, alive, species, agent_count, params);
    if (predator_count <= 0) {
        return 0;
    }

    int total_kills = 0;
    float total_consumed_size = 0.0f;
    CpuSpatialGrid cover_grid = build_cover_grid(cover_x, cover_y, cover_count, params);

    int bluegill_count = count_alive_species(alive, species, agent_count, SPECIES_BLUEGILL);
    float bluegill_total_size =
        total_prey_size_for_species(size, alive, species, agent_count, SPECIES_BLUEGILL);
    float bluegill_cover = average_cover_for_species(pos_x,
                                                     pos_y,
                                                     alive,
                                                     species,
                                                     agent_count,
                                                     SPECIES_BLUEGILL,
                                                     cover_x,
                                                     cover_y,
                                                     cover_intensity,
                                                     cover_grid,
                                                     params);
    float bluegill_available = prey_availability(bluegill_count,
                                                 params.bluegill_predation_success_percent,
                                                 bluegill_cover,
                                                 params.bluegill_refuge_scale,
                                                 params);

    int minnow_count = count_alive_species(alive, species, agent_count, SPECIES_MINNOW);
    float minnow_total_size =
        total_prey_size_for_species(size, alive, species, agent_count, SPECIES_MINNOW);
    float minnow_cover = average_cover_for_species(pos_x,
                                                   pos_y,
                                                   alive,
                                                   species,
                                                   agent_count,
                                                   SPECIES_MINNOW,
                                                   cover_x,
                                                   cover_y,
                                                   cover_intensity,
                                                   cover_grid,
                                                   params);
    float minnow_available = prey_availability(minnow_count,
                                               params.minnow_predation_success_percent,
                                               minnow_cover,
                                               params.minnow_refuge_scale,
                                               params);

    float total_available = bluegill_available + minnow_available;
    if (total_available <= 0.0f) {
        return 0;
    }

    int total_prey_count = bluegill_count + minnow_count;
    float total_prey_size_available = bluegill_total_size + minnow_total_size;
    if (total_prey_count <= 0 || total_prey_size_available <= 0.0f) {
        return 0;
    }

    float daily_appetite_limit =
        params.max_prey_per_bass_per_day * static_cast<float>(predator_count);
    if (params.bass_energy_gain > 0.0f) {
        float average_prey_size =
            total_prey_size_available / static_cast<float>(total_prey_count);
        float energy_limited_kills =
            total_hungry_bass_energy_room(energy, alive, species, agent_count, params) /
            (params.bass_energy_gain * average_prey_size);
        if (energy_limited_kills < daily_appetite_limit) {
            daily_appetite_limit = energy_limited_kills;
        }
    }
    if (daily_appetite_limit <= 0.0f) {
        return 0;
    }

    float half_sat = params.predation_half_sat > 0.0f ? params.predation_half_sat : 1.0f;
    float encounter_factor = total_available / (total_available + half_sat);
    float expected_tick_kills =
        (daily_appetite_limit * encounter_factor) /
        static_cast<float>(params.ticks_per_day > 0 ? params.ticks_per_day : 1);
    int total_tick_kills = stochastic_round(expected_tick_kills);
    if (total_tick_kills <= 0) {
        return 0;
    }
    if (total_tick_kills > total_prey_count) {
        total_tick_kills = total_prey_count;
    }

    int bluegill_kills = stochastic_round(
        static_cast<float>(total_tick_kills) * bluegill_available / total_available);
    if (bluegill_kills > bluegill_count) {
        bluegill_kills = bluegill_count;
    }
    int minnow_kills = total_tick_kills - bluegill_kills;
    if (minnow_kills > minnow_count) {
        minnow_kills = minnow_count;
    }
    if (bluegill_kills + minnow_kills < total_tick_kills) {
        int remaining = total_tick_kills - (bluegill_kills + minnow_kills);
        if (bluegill_count - bluegill_kills >= minnow_count - minnow_kills) {
            bluegill_kills += remaining;
            if (bluegill_kills > bluegill_count) {
                bluegill_kills = bluegill_count;
            }
        } else {
            minnow_kills += remaining;
            if (minnow_kills > minnow_count) {
                minnow_kills = minnow_count;
            }
        }
    }

    float bluegill_size_eaten = 0.0f;
    total_kills += remove_prey_kills(energy,
                                     size,
                                     alive,
                                     exists,
                                     species,
                                     agent_count,
                                     SPECIES_BLUEGILL,
                                     bluegill_kills,
                                     &bluegill_size_eaten);
    total_consumed_size += bluegill_size_eaten;

    float minnow_size_eaten = 0.0f;
    total_kills += remove_prey_kills(energy,
                                     size,
                                     alive,
                                     exists,
                                     species,
                                     agent_count,
                                     SPECIES_MINNOW,
                                     minnow_kills,
                                     &minnow_size_eaten);
    total_consumed_size += minnow_size_eaten;

    distribute_predation_energy(energy,
                                alive,
                                species,
                                agent_count,
                                params.bass_energy_gain * total_consumed_size,
                                params);
    return total_kills;
}

static int reproduce_species(float* pos_x,
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
                             int target_species,
                             float mate_radius,
                             float reproduction_energy,
                             float reproduction_cost,
                             float reproduction_cooldown_days,
                             float child_energy,
                             float child_size,
                             int max_children,
                             float soft_cap,
                             const SimParams& params) {
    const float mate_radius_sq = mate_radius * mate_radius;
    int births = 0;
    int cooldown_ticks = cooldown_ticks_for_days_host(reproduction_cooldown_days, params);
    int empty_slot_cursor = 0;

    int current_population = count_alive_species(alive, species, agent_count, target_species);
    float pulse_crowding = crowding_multiplier(current_population, soft_cap);
    int effective_reproduction_chance =
        static_cast<int>(params.reproduction_chance_percent * pulse_crowding + 0.5f);
    if (pulse_crowding <= 0.0f || effective_reproduction_chance <= 0) {
        return 0;
    }

    std::memset(reproduced, 0, static_cast<size_t>(agent_count) * sizeof(bool));
    CpuSpatialGrid mate_grid = build_eligible_agent_grid(pos_x,
                                                         pos_y,
                                                         energy,
                                                         spawn_cooldown,
                                                         alive,
                                                         species,
                                                         reproduced,
                                                         agent_count,
                                                         target_species,
                                                         reproduction_energy,
                                                         reproduction_cost,
                                                         params);

    for (int parent_a = 0; parent_a < agent_count; ++parent_a) {
        if (!alive[parent_a] ||
            species[parent_a] != target_species ||
            reproduced[parent_a] ||
            spawn_cooldown[parent_a] > 0 ||
            energy[parent_a] < reproduction_energy ||
            energy[parent_a] - reproduction_cost <= 0.0f) {
            continue;
        }

        int min_cell_x = cell_min_for_radius_host(pos_x[parent_a], mate_radius, mate_grid.grid_dim);
        int max_cell_x = cell_max_for_radius_host(pos_x[parent_a], mate_radius, mate_grid.grid_dim);
        int min_cell_y = cell_min_for_radius_host(pos_y[parent_a], mate_radius, mate_grid.grid_dim);
        int max_cell_y = cell_max_for_radius_host(pos_y[parent_a], mate_radius, mate_grid.grid_dim);
        bool paired = false;

        for (int cell_y = min_cell_y; cell_y <= max_cell_y && !paired; ++cell_y) {
            for (int cell_x = min_cell_x; cell_x <= max_cell_x && !paired; ++cell_x) {
                int cell = cell_y * mate_grid.grid_dim + cell_x;
                for (int parent_b = mate_grid.head[cell];
                     parent_b >= 0;
                     parent_b = mate_grid.next[parent_b]) {
                    if (parent_b <= parent_a ||
                        !alive[parent_b] ||
                        species[parent_b] != target_species ||
                        reproduced[parent_b] ||
                        spawn_cooldown[parent_b] > 0 ||
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
                    if (roll >= effective_reproduction_chance) {
                        continue;
                    }

                    if (find_empty_agent_slot_from(alive, agent_count, &empty_slot_cursor) < 0) {
                        return births;
                    }
                    --empty_slot_cursor;

                    energy[parent_a] -= reproduction_cost;
                    energy[parent_b] -= reproduction_cost;

                    reproduced[parent_a] = true;
                    reproduced[parent_b] = true;
                    spawn_cooldown[parent_a] = cooldown_ticks;
                    spawn_cooldown[parent_b] = cooldown_ticks;

                    int child_count = 1 + (std::rand() % max_children);
                    for (int child_number = 0; child_number < child_count; ++child_number) {
                        float juvenile_crowding = crowding_multiplier(current_population + births, soft_cap);
                        if (juvenile_crowding <= 0.0f) {
                            break;
                        }
                        if (random_unit_float() > juvenile_crowding) {
                            continue;
                        }

                        int child_idx = find_empty_agent_slot_from(alive, agent_count, &empty_slot_cursor);
                        if (child_idx < 0) {
                            return births;
                        }

                        float child_jitter_x = 0.02f * (2.0f * random_unit_float() - 1.0f);
                        float child_jitter_y = 0.02f * (2.0f * random_unit_float() - 1.0f);
                        pos_x[child_idx] = clamp01_host(0.5f * (pos_x[parent_a] + pos_x[parent_b]) + child_jitter_x);
                        pos_y[child_idx] = clamp01_host(0.5f * (pos_y[parent_a] + pos_y[parent_b]) + child_jitter_y);
                        energy[child_idx] = child_energy;
                        death_energy[child_idx] = random_death_energy_threshold(target_species, params);
                        size[child_idx] = child_size;
                        spawn_cooldown[child_idx] = cooldown_ticks;
                        alive[child_idx] = true;
                        exists[child_idx] = true;
                        species[child_idx] = target_species;
                        dir_x[child_idx] = 2.0f * random_unit_float() - 1.0f;
                        dir_y[child_idx] = 2.0f * random_unit_float() - 1.0f;
                        normalize_direction(&dir_x[child_idx], &dir_y[child_idx]);

                        reproduced[child_idx] = true;
                        ++births;
                    }

                    paired = true;
                    break;
                }
            }
        }
    }

    return births;
}

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
                       const SimParams& params) {
    return reproduce_species(pos_x,
                             pos_y,
                             energy,
                             death_energy,
                             size,
                             spawn_cooldown,
                             alive,
                             exists,
                             species,
                             dir_x,
                             dir_y,
                             reproduced,
                             tick,
                             agent_count,
                             SPECIES_BLUEGILL,
                             params.mate_radius,
                             params.reproduction_energy,
                             params.reproduction_cost,
                             params.reproduction_cooldown_days,
                             params.child_energy,
                             params.bluegill_min_size,
                             params.max_children_per_birth,
                             params.bluegill_soft_cap,
                             params);
}

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
                     const SimParams& params) {
    return reproduce_species(pos_x,
                             pos_y,
                             energy,
                             death_energy,
                             size,
                             spawn_cooldown,
                             alive,
                             exists,
                             species,
                             dir_x,
                             dir_y,
                             reproduced,
                             tick,
                             agent_count,
                             SPECIES_MINNOW,
                             params.mate_radius,
                             params.minnow_reproduction_energy,
                             params.minnow_reproduction_cost,
                             params.minnow_reproduction_cooldown_days,
                             params.minnow_child_energy,
                             params.minnow_min_size,
                             params.minnow_max_children_per_birth,
                             params.minnow_soft_cap,
                             params);
}

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
                   const SimParams& params) {
    return reproduce_species(pos_x,
                             pos_y,
                             energy,
                             death_energy,
                             size,
                             spawn_cooldown,
                             alive,
                             exists,
                             species,
                             dir_x,
                             dir_y,
                             reproduced,
                             tick,
                             agent_count,
                             SPECIES_BASS,
                             params.bass_mate_radius,
                             params.bass_reproduction_energy,
                             params.bass_reproduction_cost,
                             params.bass_reproduction_cooldown_days,
                             params.bass_child_energy,
                             params.bass_min_size,
                             params.bass_max_children_per_birth,
                             params.bass_soft_cap,
                             params);
}
