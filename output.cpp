#include "output.h"

#include <cstdio>

#include "cpu_rules.h"
#include "sim_math.h"
#include "sim_types.h"

static int position_to_grid_index(float value, int grid_size) {
    int index = static_cast<int>(value * static_cast<float>(grid_size - 1) + 0.5f);
    if (index < 0) {
        return 0;
    }
    if (index >= grid_size) {
        return grid_size - 1;
    }
    return index;
}

static void render_grid(const float* pos_x,
                        const float* pos_y,
                        const bool* alive,
                        const int* species,
                        int agent_count,
                        const float* food_x,
                        const float* food_y,
                        const int* food_active,
                        int food_count) {
    const int grid_width = 20;
    const int grid_height = 20;
    char grid[grid_height][grid_width];

    for (int row = 0; row < grid_height; ++row) {
        for (int col = 0; col < grid_width; ++col) {
            grid[row][col] = '.';
        }
    }

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        if (food_active[food_idx] != 0) {
            int col = position_to_grid_index(food_x[food_idx], grid_width);
            int row = grid_height - 1 - position_to_grid_index(food_y[food_idx], grid_height);
            grid[row][col] = 'f';
        }
    }

    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx]) {
            int col = position_to_grid_index(pos_x[agent_idx], grid_width);
            int row = grid_height - 1 - position_to_grid_index(pos_y[agent_idx], grid_height);
            if (species[agent_idx] == SPECIES_BASS) {
                grid[row][col] = 'B';
            } else if (species[agent_idx] == SPECIES_MINNOW) {
                grid[row][col] = 'm';
            } else {
                grid[row][col] = 'g';
            }
        }
    }

    for (int row = 0; row < grid_height; ++row) {
        for (int col = 0; col < grid_width; ++col) {
            std::printf("%c", grid[row][col]);
        }
        std::printf("\n");
    }
}

void print_state(const char* label,
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
                 int food_count) {
    std::printf("%s\n", label);
    render_grid(pos_x, pos_y, alive, species, agent_count, food_x, food_y, food_active, food_count);
    std::printf("Bluegill: %d | Minnow: %d | Bass: %d | Food: %d/%d | Capacity: %d\n",
                count_alive_species(alive, species, agent_count, SPECIES_BLUEGILL),
                count_alive_species(alive, species, agent_count, SPECIES_MINNOW),
                count_alive_species(alive, species, agent_count, SPECIES_BASS),
                count_active_food(food_active, food_count),
                food_count,
                agent_count);
    std::printf("Energies:");
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx]) {
            std::printf(" %s%d=%.2f(size %.2f)",
                        species_name(species[agent_idx]),
                        agent_idx,
                        energy[agent_idx],
                        size[agent_idx]);
        }
    }
    std::printf("\n\n");
}

void stream_step_state(int step,
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
                       int food_count) {
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (exists[agent_idx]) {
            std::printf("%d,agent,%s,%d,%.6f,%.6f,%.6f,%d,%.6f\n",
                        step,
                        species_name(species[agent_idx]),
                        agent_idx,
                        pos_x[agent_idx],
                        pos_y[agent_idx],
                        energy[agent_idx],
                        alive[agent_idx] ? 1 : 0,
                        size[agent_idx]);
        }
    }

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        std::printf("%d,food,food,%d,%.6f,%.6f,0.000000,%d,0.000000\n",
                    step,
                    food_idx,
                    food_x[food_idx],
                    food_y[food_idx],
                    food_active[food_idx] != 0 ? 1 : 0);
    }

    std::fflush(stdout);
}
