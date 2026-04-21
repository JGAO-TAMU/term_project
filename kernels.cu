#include "kernels.cuh"

#include <cuda_runtime.h>

__device__ unsigned int hash_u32(unsigned int value) {
    value ^= value >> 16;
    value *= 0xdeadbeefU;
    value ^= value >> 15;
    value *= 0xbeefbeefU;
    value ^= value >> 16;
    return value;
}

__device__ float hash_to_unit_float(unsigned int value) {
    const unsigned int masked = hash_u32(value) & 0x00FFFFFFU;
    return static_cast<float>(masked) / static_cast<float>(0x01000000U);
}

__device__ float clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

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
                                       SimParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= agent_count || !alive[idx]) {
        return;
    }

    float x = pos_x[idx];
    float y = pos_y[idx];
    float e = energy[idx];
    float s = size[idx];
    int my_species = species[idx];
    bool is_prey = my_species == SPECIES_BLUEGILL || my_species == SPECIES_MINNOW;
    float species_min_size = params.bluegill_min_size;
    float species_max_size = params.bluegill_max_size;

    if (my_species == SPECIES_MINNOW) {
        species_min_size = params.minnow_min_size;
        species_max_size = params.minnow_max_size;
    } else if (my_species == SPECIES_BASS) {
        species_min_size = params.bass_min_size;
        species_max_size = params.bass_max_size;
    }

    if (my_species == SPECIES_MINNOW) {
        e += params.minnow_energy_gain;
        e -= params.minnow_energy_drain;
        if (e > params.minnow_max_energy) {
            e = params.minnow_max_energy;
        }
    } else if (my_species == SPECIES_BLUEGILL) {
        e += params.bluegill_energy_gain;
        e -= params.bluegill_energy_drain;
        if (e > params.bluegill_max_energy) {
            e = params.bluegill_max_energy;
        }
    } else {
        e -= params.bass_energy_drain;
        if (e > params.bass_max_energy) {
            e = params.bass_max_energy;
        }
    }

    if (e <= 0.0f) {
        energy[idx] = 0.0f;
        alive[idx] = false;
        return;
    }

    const float detection_radius_sq = params.detection_radius * params.detection_radius;
    int target_agent_idx = -1;
    int target_food_idx = -1;
    float target_distance_sq = detection_radius_sq;
    float my_reproduction_energy = params.reproduction_energy;
    float my_reproduction_cost = params.reproduction_cost;
    if (my_species == SPECIES_BASS) {
        my_reproduction_energy = params.bass_reproduction_energy;
        my_reproduction_cost = params.bass_reproduction_cost;
    } else if (my_species == SPECIES_MINNOW) {
        my_reproduction_energy = params.minnow_reproduction_energy;
        my_reproduction_cost = params.minnow_reproduction_cost;
    }
    bool can_mate = e >= my_reproduction_energy &&
                    e - my_reproduction_cost > 0.0f;

    if (can_mate) {
        for (int other_idx = 0; other_idx < agent_count; ++other_idx) {
            if (other_idx != idx &&
                alive[other_idx] &&
                species[other_idx] == my_species &&
                energy[other_idx] >= my_reproduction_energy &&
                energy[other_idx] - my_reproduction_cost > 0.0f) {
                float diff_x = pos_x[other_idx] - x;
                float diff_y = pos_y[other_idx] - y;
                float distance_sq = diff_x * diff_x + diff_y * diff_y;

                if (distance_sq <= target_distance_sq) {
                    target_agent_idx = other_idx;
                    target_distance_sq = distance_sq;
                }
            }
        }
    }

    if (target_agent_idx < 0) {
        if (my_species == SPECIES_BASS && e < params.bass_max_energy) {
            for (int other_idx = 0; other_idx < agent_count; ++other_idx) {
                if (alive[other_idx] &&
                    (species[other_idx] == SPECIES_BLUEGILL ||
                     species[other_idx] == SPECIES_MINNOW)) {
                    float diff_x = pos_x[other_idx] - x;
                    float diff_y = pos_y[other_idx] - y;
                    float distance_sq = diff_x * diff_x + diff_y * diff_y;

                    if (distance_sq <= target_distance_sq) {
                        target_agent_idx = other_idx;
                        target_distance_sq = distance_sq;
                    }
                }
            }
        } else if (is_prey) {
            for (int food_idx = 0; food_idx < food_count; ++food_idx) {
                if (food_active[food_idx] == 0) {
                    continue;
                }

                float diff_x = food_x[food_idx] - x;
                float diff_y = food_y[food_idx] - y;
                float distance_sq = diff_x * diff_x + diff_y * diff_y;

                if (distance_sq <= target_distance_sq) {
                    target_food_idx = food_idx;
                    target_distance_sq = distance_sq;
                }
            }
        }
    }

    float dx = 0.0f;
    float dy = 0.0f;
    bool is_wandering = false;

    if (target_agent_idx >= 0 || target_food_idx >= 0) {
        float target_x = target_agent_idx >= 0 ? pos_x[target_agent_idx] : food_x[target_food_idx];
        float target_y = target_agent_idx >= 0 ? pos_y[target_agent_idx] : food_y[target_food_idx];
        float target_dx = target_x - x;
        float target_dy = target_y - y;
        float target_distance = sqrtf(target_dx * target_dx + target_dy * target_dy);

        if (target_distance > 0.0f) {
            dx = params.move_step * (target_dx / target_distance);
            dy = params.move_step * (target_dy / target_distance);
        }
    } else {
        is_wandering = true;
        const unsigned int base = static_cast<unsigned int>(step * 131 + idx * 977);
        float wander_dir_x = dir_x[idx] * params.decay_factor;
        float wander_dir_y = dir_y[idx] * params.decay_factor;
        float noise_x = params.noise_scale * (2.0f * hash_to_unit_float(base + 17U) - 1.0f);
        float noise_y = params.noise_scale * (2.0f * hash_to_unit_float(base + 53U) - 1.0f);

        wander_dir_x += noise_x;
        wander_dir_y += noise_y;

        float wander_length = sqrtf(wander_dir_x * wander_dir_x + wander_dir_y * wander_dir_y);
        if (wander_length > 0.0f) {
            wander_dir_x /= wander_length;
            wander_dir_y /= wander_length;
        } else {
            wander_dir_x = 1.0f;
            wander_dir_y = 0.0f;
        }

        dir_x[idx] = wander_dir_x;
        dir_y[idx] = wander_dir_y;
        dx = params.move_step * wander_dir_x;
        dy = params.move_step * wander_dir_y;
    }

    float next_x = x + dx;
    float next_y = y + dy;
    bool hit_x_border = next_x < 0.0f || next_x > 1.0f;
    bool hit_y_border = next_y < 0.0f || next_y > 1.0f;

    x = clamp01(next_x);
    y = clamp01(next_y);

    if (is_wandering && (hit_x_border || hit_y_border)) {
        float bounced_dir_x = dir_x[idx];
        float bounced_dir_y = dir_y[idx];

        if (hit_x_border) {
            bounced_dir_x = -bounced_dir_x;
        }
        if (hit_y_border) {
            bounced_dir_y = -bounced_dir_y;
        }

        float bounced_length = sqrtf(bounced_dir_x * bounced_dir_x + bounced_dir_y * bounced_dir_y);
        if (bounced_length > 0.0f) {
            dir_x[idx] = bounced_dir_x / bounced_length;
            dir_y[idx] = bounced_dir_y / bounced_length;
        }
    }

    pos_x[idx] = x;
    pos_y[idx] = y;

    if (is_prey && target_food_idx >= 0) {
        float food_diff_x = food_x[target_food_idx] - x;
        float food_diff_y = food_y[target_food_idx] - y;
        float eat_radius_sq = params.food_eat_radius * params.food_eat_radius;
        float food_distance_sq = food_diff_x * food_diff_x + food_diff_y * food_diff_y;

        if (food_distance_sq <= eat_radius_sq &&
            atomicCAS(&food_active[target_food_idx], 1, 0) == 1) {
            e += params.food_energy_gain;
            float prey_max_energy = my_species == SPECIES_MINNOW ?
                                    params.minnow_max_energy :
                                    params.bluegill_max_energy;
            if (e > prey_max_energy) {
                e = prey_max_energy;
            }
        }
    }

    if (e >= params.size_growth_threshold) {
        s += params.size_growth_rate * (e - params.size_growth_threshold);
        if (s > species_max_size) {
            s = species_max_size;
        }
    } else if (e <= params.size_shrink_threshold) {
        s -= params.size_shrink_rate * (params.size_shrink_threshold - e);
        if (s < species_min_size) {
            s = species_min_size;
        }
    }

    size[idx] = s;
    energy[idx] = e;
}
