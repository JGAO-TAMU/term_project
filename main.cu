#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

static void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", call, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

static float random_unit_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

static float clamp01_host(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static void normalize_direction(float* x, float* y) {
    float length = std::sqrt((*x) * (*x) + (*y) * (*y));
    if (length > 0.0f) {
        *x /= length;
        *y /= length;
    } else {
        *x = 1.0f;
        *y = 0.0f;
    }
}

struct SimParams {
    int step_count;
    int food_respawn_chance_percent;
    int reproduction_chance_percent;
    unsigned int seed;
    float energy_drain;
    float food_energy_gain;
    float detection_radius;
    float eat_radius;
    float move_step;
    float decay_factor;
    float noise_scale;
    float reproduction_energy;
    float reproduction_cost;
    float mate_radius;
    float child_energy;
};

static void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  --stream\n");
    std::printf("  --steps N\n");
    std::printf("  --agent-capacity N\n");
    std::printf("  --max-agents N\n");
    std::printf("  --seed N\n");
    std::printf("  --energy-drain X\n");
    std::printf("  --food-energy X\n");
    std::printf("  --detection-radius X\n");
    std::printf("  --eat-radius X\n");
    std::printf("  --move-step X\n");
    std::printf("  --decay X\n");
    std::printf("  --noise X\n");
    std::printf("  --respawn-chance N\n");
    std::printf("  --reproduce-energy X\n");
    std::printf("  --reproduce-cost X\n");
    std::printf("  --mate-radius X\n");
    std::printf("  --child-energy X\n");
    std::printf("  --reproduce-chance N\n");
}

static bool read_arg_value(int argc, char** argv, int* index, const char** value) {
    if (*index + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", argv[*index]);
        return false;
    }

    ++(*index);
    *value = argv[*index];
    return true;
}

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
                                       bool* alive,
                                       float* dir_x,
                                       float* dir_y,
                                       const float* food_x,
                                       const float* food_y,
                                       bool* food_eaten,
                                       int agent_count,
                                       int food_count,
                                       int step,
                                       SimParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= agent_count) {
        return;
    }
    if (!alive[idx]) {
        return;
    }

    float x = pos_x[idx];
    float y = pos_y[idx];
    float e = energy[idx] - params.energy_drain;
    if (e <= 0.0f) {
        energy[idx] = 0.0f;
        alive[idx] = false;
        return;
    }

    const float detection_radius_sq = params.detection_radius * params.detection_radius;
    const float eat_radius_sq = params.eat_radius * params.eat_radius;

    int closest_mate_idx = -1;
    float closest_mate_distance_sq = detection_radius_sq;
    bool can_mate = e >= params.reproduction_energy &&
                    e - params.reproduction_cost > 0.0f;

    if (can_mate) {
        for (int mate_idx = 0; mate_idx < agent_count; ++mate_idx) {
            if (mate_idx != idx &&
                alive[mate_idx] &&
                energy[mate_idx] >= params.reproduction_energy &&
                energy[mate_idx] - params.reproduction_cost > 0.0f) {
                float diff_x = pos_x[mate_idx] - x;
                float diff_y = pos_y[mate_idx] - y;
                float distance_sq = diff_x * diff_x + diff_y * diff_y;

                if (distance_sq <= closest_mate_distance_sq) {
                    closest_mate_distance_sq = distance_sq;
                    closest_mate_idx = mate_idx;
                }
            }
        }
    }

    int closest_food_idx = -1;
    float closest_food_distance_sq = detection_radius_sq;

    if (closest_mate_idx < 0) {
        for (int food_idx = 0; food_idx < food_count; ++food_idx) {
            if (!food_eaten[food_idx]) {
                float diff_x = food_x[food_idx] - x;
                float diff_y = food_y[food_idx] - y;
                float distance_sq = diff_x * diff_x + diff_y * diff_y;

                if (distance_sq <= closest_food_distance_sq) {
                    closest_food_distance_sq = distance_sq;
                    closest_food_idx = food_idx;
                }
            }
        }
    }

    float dx = 0.0f;
    float dy = 0.0f;
    bool is_wandering = false;

    if (closest_mate_idx >= 0) {
        float target_dx = pos_x[closest_mate_idx] - x;
        float target_dy = pos_y[closest_mate_idx] - y;
        float target_distance = sqrtf(target_dx * target_dx + target_dy * target_dy);

        if (target_distance > 0.0f) {
            dx = params.move_step * (target_dx / target_distance);
            dy = params.move_step * (target_dy / target_distance);
        }
    } else if (closest_food_idx >= 0) {
        float target_dx = food_x[closest_food_idx] - x;
        float target_dy = food_y[closest_food_idx] - y;
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
    bool hit_x_border = false;
    bool hit_y_border = false;

    if (next_x < 0.0f || next_x > 1.0f) {
        hit_x_border = true;
    }
    if (next_y < 0.0f || next_y > 1.0f) {
        hit_y_border = true;
    }

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

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        if (!food_eaten[food_idx]) {
            float diff_x = x - food_x[food_idx];
            float diff_y = y - food_y[food_idx];
            float distance_sq = diff_x * diff_x + diff_y * diff_y;

            if (distance_sq < eat_radius_sq) {
                food_eaten[food_idx] = true;
                e += params.food_energy_gain;
            }
        }
    }

    pos_x[idx] = x;
    pos_y[idx] = y;
    energy[idx] = e;
}

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

static int count_alive_agents(const bool* alive, int agent_count);

static void render_grid(const float* pos_x,
                        const float* pos_y,
                        const bool* alive,
                        const float* food_x,
                        const float* food_y,
                        const bool* food_eaten,
                        int agent_count,
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
        if (!food_eaten[food_idx]) {
            int col = position_to_grid_index(food_x[food_idx], grid_width);
            int row = grid_height - 1 - position_to_grid_index(food_y[food_idx], grid_height);
            grid[row][col] = 'F';
        }
    }

    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx]) {
            int col = position_to_grid_index(pos_x[agent_idx], grid_width);
            int row = grid_height - 1 - position_to_grid_index(pos_y[agent_idx], grid_height);
            grid[row][col] = 'A';
        }
    }

    for (int row = 0; row < grid_height; ++row) {
        for (int col = 0; col < grid_width; ++col) {
            std::printf("%c", grid[row][col]);
        }
        std::printf("\n");
    }
}

static void print_step_state(int step,
                             const float* pos_x,
                             const float* pos_y,
                             const float* energy,
                             const bool* alive,
                             const float* food_x,
                             const float* food_y,
                             const bool* food_eaten,
                             int agent_count,
                             int food_count) {
    std::printf("Step %d\n", step + 1);
    render_grid(pos_x, pos_y, alive, food_x, food_y, food_eaten, agent_count, food_count);
    std::printf("Alive agents: %d / %d capacity\n", count_alive_agents(alive, agent_count), agent_count);
    std::printf("Energies:");
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx]) {
            std::printf(" A%d=%.2f", agent_idx, energy[agent_idx]);
        }
    }
    std::printf("\n\n");
}

static void print_initial_state(const float* pos_x,
                                const float* pos_y,
                                const float* energy,
                                const bool* alive,
                                const float* food_x,
                                const float* food_y,
                                const bool* food_eaten,
                                int agent_count,
                                int food_count) {
    std::printf("Initial state\n");
    render_grid(pos_x, pos_y, alive, food_x, food_y, food_eaten, agent_count, food_count);
    std::printf("Alive agents: %d / %d capacity\n", count_alive_agents(alive, agent_count), agent_count);
    std::printf("Energies:");
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (alive[agent_idx]) {
            std::printf(" A%d=%.2f", agent_idx, energy[agent_idx]);
        }
    }
    std::printf("\n\n");
}

static void stream_step_state(int step,
                              const float* pos_x,
                              const float* pos_y,
                              const float* energy,
                              const bool* alive,
                              const bool* exists,
                              const float* food_x,
                              const float* food_y,
                              const bool* food_eaten,
                              int agent_count,
                              int food_count) {
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (exists[agent_idx]) {
            std::printf("%d,agent,%d,%.6f,%.6f,%.6f,%d\n",
                        step,
                        agent_idx,
                        pos_x[agent_idx],
                        pos_y[agent_idx],
                        energy[agent_idx],
                        alive[agent_idx] ? 1 : 0);
        }
    }

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        std::printf("%d,food,%d,%.6f,%.6f,0,%d\n",
                    step,
                    food_idx,
                    food_x[food_idx],
                    food_y[food_idx],
                    food_eaten[food_idx] ? 0 : 1);
    }

    std::fflush(stdout);
}

static int count_alive_agents(const bool* alive, int agent_count) {
    int alive_count = 0;
    for (int i = 0; i < agent_count; ++i) {
        if (alive[i]) {
            ++alive_count;
        }
    }
    return alive_count;
}

static int find_empty_agent_slot(const bool* alive, int agent_count) {
    for (int i = 0; i < agent_count; ++i) {
        if (!alive[i]) {
            return i;
        }
    }
    return -1;
}

static int find_inactive_food_slot(const bool* food_eaten, int food_count) {
    for (int i = 0; i < food_count; ++i) {
        if (food_eaten[i]) {
            return i;
        }
    }
    return -1;
}

static void place_food_particle(float x,
                                float y,
                                float* food_x,
                                float* food_y,
                                bool* food_eaten,
                                int food_count,
                                int replacement_hint) {
    int food_idx = find_inactive_food_slot(food_eaten, food_count);
    if (food_idx < 0) {
        food_idx = replacement_hint % food_count;
    }

    food_x[food_idx] = x;
    food_y[food_idx] = y;
    food_eaten[food_idx] = false;
}

static int convert_dead_agents_to_food(const bool* was_alive,
                                       float* pos_x,
                                       float* pos_y,
                                       bool* alive,
                                       bool* exists,
                                       float* food_x,
                                       float* food_y,
                                       bool* food_eaten,
                                       int agent_count,
                                       int food_count) {
    int converted_count = 0;
    for (int i = 0; i < agent_count; ++i) {
        if (was_alive[i] && !alive[i]) {
            place_food_particle(pos_x[i],
                                pos_y[i],
                                food_x,
                                food_y,
                                food_eaten,
                                food_count,
                                converted_count);
            exists[i] = false;
            ++converted_count;
        }
    }
    return converted_count;
}

static int reproduce_agents(float* pos_x,
                            float* pos_y,
                            float* energy,
                            bool* alive,
                            bool* exists,
                            float* dir_x,
                            float* dir_y,
                            bool* reproduced,
                            int agent_count,
                            const SimParams& params) {
    const float mate_radius_sq = params.mate_radius * params.mate_radius;
    int births = 0;

    std::memset(reproduced, 0, static_cast<size_t>(agent_count) * sizeof(bool));

    for (int parent_a = 0; parent_a < agent_count; ++parent_a) {
        if (!alive[parent_a] || reproduced[parent_a]) {
            continue;
        }
        if (energy[parent_a] < params.reproduction_energy ||
            energy[parent_a] - params.reproduction_cost <= 0.0f) {
            continue;
        }

        for (int parent_b = parent_a + 1; parent_b < agent_count; ++parent_b) {
            if (!alive[parent_b] || reproduced[parent_b]) {
                continue;
            }
            if (energy[parent_b] < params.reproduction_energy ||
                energy[parent_b] - params.reproduction_cost <= 0.0f) {
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

            int child_idx = find_empty_agent_slot(alive, agent_count);
            if (child_idx < 0) {
                return births;
            }

            energy[parent_a] -= params.reproduction_cost;
            energy[parent_b] -= params.reproduction_cost;

            float child_jitter_x = 0.02f * (2.0f * random_unit_float() - 1.0f);
            float child_jitter_y = 0.02f * (2.0f * random_unit_float() - 1.0f);
            pos_x[child_idx] = clamp01_host(0.5f * (pos_x[parent_a] + pos_x[parent_b]) + child_jitter_x);
            pos_y[child_idx] = clamp01_host(0.5f * (pos_y[parent_a] + pos_y[parent_b]) + child_jitter_y);
            energy[child_idx] = params.child_energy;
            alive[child_idx] = true;
            exists[child_idx] = true;
            dir_x[child_idx] = 2.0f * random_unit_float() - 1.0f;
            dir_y[child_idx] = 2.0f * random_unit_float() - 1.0f;
            normalize_direction(&dir_x[child_idx], &dir_y[child_idx]);

            reproduced[parent_a] = true;
            reproduced[parent_b] = true;
            reproduced[child_idx] = true;
            ++births;
            break;
        }
    }

    return births;
}

int main(int argc, char** argv) {
    const int initial_agent_count = 10;
    int agent_count = 50;
    const int food_count = 24;
    bool stream_mode = false;
    SimParams params;
    params.step_count = 50;
    params.food_respawn_chance_percent = 30;
    params.reproduction_chance_percent = 100;
    params.seed = 12345U;
    params.energy_drain = 0.05f;
    params.food_energy_gain = 0.25f;
    params.detection_radius = 0.15f;
    params.eat_radius = 0.02f;
    params.move_step = 0.03f;
    params.decay_factor = 0.95f;
    params.noise_scale = 0.20f;
    params.reproduction_energy = 1.25f;
    params.reproduction_cost = 0.40f;
    params.mate_radius = 0.06f;
    params.child_energy = 0.75f;

    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        const char* value = nullptr;

        if (std::strcmp(argv[arg_idx], "--stream") == 0) {
            stream_mode = true;
        } else if (std::strcmp(argv[arg_idx], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (std::strcmp(argv[arg_idx], "--steps") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.step_count = std::atoi(value);
        } else if (std::strcmp(argv[arg_idx], "--agent-capacity") == 0 ||
                   std::strcmp(argv[arg_idx], "--max-agents") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            agent_count = std::atoi(value);
        } else if (std::strcmp(argv[arg_idx], "--seed") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.seed = static_cast<unsigned int>(std::strtoul(value, nullptr, 10));
        } else if (std::strcmp(argv[arg_idx], "--energy-drain") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.energy_drain = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--food-energy") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.food_energy_gain = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--detection-radius") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.detection_radius = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--eat-radius") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.eat_radius = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--move-step") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.move_step = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--decay") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.decay_factor = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--noise") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.noise_scale = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--respawn-chance") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.food_respawn_chance_percent = std::atoi(value);
        } else if (std::strcmp(argv[arg_idx], "--reproduce-energy") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.reproduction_energy = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--reproduce-cost") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.reproduction_cost = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--mate-radius") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.mate_radius = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--child-energy") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.child_energy = static_cast<float>(std::atof(value));
        } else if (std::strcmp(argv[arg_idx], "--reproduce-chance") == 0) {
            if (!read_arg_value(argc, argv, &arg_idx, &value)) {
                return EXIT_FAILURE;
            }
            params.reproduction_chance_percent = std::atoi(value);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[arg_idx]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (params.step_count < 0) {
        params.step_count = 0;
    }
    if (agent_count < initial_agent_count) {
        agent_count = initial_agent_count;
    }
    if (params.food_respawn_chance_percent < 0) {
        params.food_respawn_chance_percent = 0;
    }
    if (params.food_respawn_chance_percent > 100) {
        params.food_respawn_chance_percent = 100;
    }
    if (params.reproduction_chance_percent < 0) {
        params.reproduction_chance_percent = 0;
    }
    if (params.reproduction_chance_percent > 100) {
        params.reproduction_chance_percent = 100;
    }

    const size_t agent_bytes = static_cast<size_t>(agent_count) * sizeof(float);
    const size_t agent_flag_bytes = static_cast<size_t>(agent_count) * sizeof(bool);
    const size_t food_bytes = static_cast<size_t>(food_count) * sizeof(float);
    const size_t food_flag_bytes = static_cast<size_t>(food_count) * sizeof(bool);

    float* host_pos_x = static_cast<float*>(std::malloc(agent_bytes));
    float* host_pos_y = static_cast<float*>(std::malloc(agent_bytes));
    float* host_energy = static_cast<float*>(std::malloc(agent_bytes));
    bool* host_alive = static_cast<bool*>(std::malloc(agent_flag_bytes));
    bool* host_was_alive = static_cast<bool*>(std::malloc(agent_flag_bytes));
    bool* host_exists = static_cast<bool*>(std::malloc(agent_flag_bytes));
    bool* host_reproduced = static_cast<bool*>(std::malloc(agent_flag_bytes));
    float* host_dir_x = static_cast<float*>(std::malloc(agent_bytes));
    float* host_dir_y = static_cast<float*>(std::malloc(agent_bytes));
    float* host_food_x = static_cast<float*>(std::malloc(food_bytes));
    float* host_food_y = static_cast<float*>(std::malloc(food_bytes));
    bool* host_food_eaten = static_cast<bool*>(std::malloc(food_flag_bytes));

    if (host_pos_x == nullptr || host_pos_y == nullptr || host_energy == nullptr ||
        host_alive == nullptr || host_was_alive == nullptr ||
        host_exists == nullptr || host_reproduced == nullptr ||
        host_dir_x == nullptr || host_dir_y == nullptr ||
        host_food_x == nullptr || host_food_y == nullptr || host_food_eaten == nullptr) {
        std::fprintf(stderr, "Host memory allocation failed.\n");
        std::free(host_pos_x);
        std::free(host_pos_y);
        std::free(host_energy);
        std::free(host_alive);
        std::free(host_was_alive);
        std::free(host_exists);
        std::free(host_reproduced);
        std::free(host_dir_x);
        std::free(host_dir_y);
        std::free(host_food_x);
        std::free(host_food_y);
        std::free(host_food_eaten);
        return EXIT_FAILURE;
    }

    const float initial_agent_x[initial_agent_count] = {
        0.38f, 0.42f, 0.58f, 0.62f, 0.50f,
        0.34f, 0.40f, 0.50f, 0.60f, 0.66f
    };
    const float initial_agent_y[initial_agent_count] = {
        0.68f, 0.68f, 0.68f, 0.68f, 0.53f,
        0.43f, 0.37f, 0.33f, 0.37f, 0.43f
    };
    const float initial_energy[initial_agent_count] = {
        1.00f, 1.00f, 1.00f, 1.00f, 1.00f,
        1.00f, 1.00f, 1.00f, 1.00f, 1.00f
    };
    const float face_center_x = 0.50f;
    const float face_center_y = 0.55f;
    const float face_radius = 0.34f;
    const float two_pi = 6.28318530718f;

    std::srand(params.seed);

    for (int i = 0; i < agent_count; ++i) {
        host_pos_x[i] = 0.0f;
        host_pos_y[i] = 0.0f;
        host_energy[i] = 0.0f;
        host_alive[i] = false;
        host_was_alive[i] = false;
        host_exists[i] = false;
        host_dir_x[i] = 2.0f * random_unit_float() - 1.0f;
        host_dir_y[i] = 2.0f * random_unit_float() - 1.0f;
        normalize_direction(&host_dir_x[i], &host_dir_y[i]);
    }

    for (int i = 0; i < initial_agent_count; ++i) {
        host_pos_x[i] = initial_agent_x[i];
        host_pos_y[i] = initial_agent_y[i];
        host_energy[i] = initial_energy[i];
        host_alive[i] = true;
        host_exists[i] = true;
    }

    for (int i = 0; i < food_count; ++i) {
        float angle = (two_pi * static_cast<float>(i)) / static_cast<float>(food_count);
        host_food_x[i] = face_center_x + face_radius * std::cos(angle);
        host_food_y[i] = face_center_y + face_radius * std::sin(angle);
        host_food_eaten[i] = false;
    }

    float* device_pos_x = nullptr;
    float* device_pos_y = nullptr;
    float* device_energy = nullptr;
    bool* device_alive = nullptr;
    float* device_dir_x = nullptr;
    float* device_dir_y = nullptr;
    float* device_food_x = nullptr;
    float* device_food_y = nullptr;
    bool* device_food_eaten = nullptr;

    check_cuda(cudaMalloc(&device_pos_x, agent_bytes), "cudaMalloc(device_pos_x)");
    check_cuda(cudaMalloc(&device_pos_y, agent_bytes), "cudaMalloc(device_pos_y)");
    check_cuda(cudaMalloc(&device_energy, agent_bytes), "cudaMalloc(device_energy)");
    check_cuda(cudaMalloc(&device_alive, agent_flag_bytes), "cudaMalloc(device_alive)");
    check_cuda(cudaMalloc(&device_dir_x, agent_bytes), "cudaMalloc(device_dir_x)");
    check_cuda(cudaMalloc(&device_dir_y, agent_bytes), "cudaMalloc(device_dir_y)");
    check_cuda(cudaMalloc(&device_food_x, food_bytes), "cudaMalloc(device_food_x)");
    check_cuda(cudaMalloc(&device_food_y, food_bytes), "cudaMalloc(device_food_y)");
    check_cuda(cudaMalloc(&device_food_eaten, food_flag_bytes), "cudaMalloc(device_food_eaten)");

    check_cuda(cudaMemcpy(device_pos_x, host_pos_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_pos_x, host_pos_x)");
    check_cuda(cudaMemcpy(device_pos_y, host_pos_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_pos_y, host_pos_y)");
    check_cuda(cudaMemcpy(device_energy, host_energy, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_energy, host_energy)");
    check_cuda(cudaMemcpy(device_alive,
                          host_alive,
                          agent_flag_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(device_alive, host_alive)");
    check_cuda(cudaMemcpy(device_dir_x, host_dir_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_dir_x, host_dir_x)");
    check_cuda(cudaMemcpy(device_dir_y, host_dir_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_dir_y, host_dir_y)");
    check_cuda(cudaMemcpy(device_food_x, host_food_x, food_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_food_x, host_food_x)");
    check_cuda(cudaMemcpy(device_food_y, host_food_y, food_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_food_y, host_food_y)");
    check_cuda(cudaMemcpy(device_food_eaten, host_food_eaten, food_flag_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device_food_eaten, host_food_eaten)");

    const int threads_per_block = 32;
    const int blocks = (agent_count + threads_per_block - 1) / threads_per_block;

    if (!stream_mode) {
        std::printf("Alive agents: %d / %d capacity\n", count_alive_agents(host_alive, agent_count), agent_count);
        std::printf("Total food: %d\n\n", food_count);
    } else {
        std::printf("step,type,id,x,y,energy,active\n");
        std::fflush(stdout);
    }

    if (!stream_mode) {
        print_initial_state(host_pos_x,
                            host_pos_y,
                            host_energy,
                            host_alive,
                            host_food_x,
                            host_food_y,
                            host_food_eaten,
                            agent_count,
                            food_count);
    } else {
        stream_step_state(0,
                          host_pos_x,
                          host_pos_y,
                          host_energy,
                          host_alive,
                          host_exists,
                          host_food_x,
                          host_food_y,
                          host_food_eaten,
                          agent_count,
                          food_count);
    }

    for (int step = 0; step < params.step_count; ++step) {
        std::memcpy(host_was_alive, host_alive, agent_flag_bytes);

        simulate_agents_kernel<<<blocks, threads_per_block>>>(device_pos_x,
                                                              device_pos_y,
                                                              device_energy,
                                                              device_alive,
                                                              device_dir_x,
                                                              device_dir_y,
                                                              device_food_x,
                                                              device_food_y,
                                                              device_food_eaten,
                                                              agent_count,
                                                              food_count,
                                                              step,
                                                              params);
        check_cuda(cudaGetLastError(), "simulate_agents_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda(cudaMemcpy(host_pos_x, device_pos_x, agent_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host_pos_x, device_pos_x)");
        check_cuda(cudaMemcpy(host_pos_y, device_pos_y, agent_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host_pos_y, device_pos_y)");
        check_cuda(cudaMemcpy(host_energy, device_energy, agent_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host_energy, device_energy)");
        check_cuda(cudaMemcpy(host_alive,
                              device_alive,
                              agent_flag_bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host_alive, device_alive)");
        check_cuda(cudaMemcpy(host_food_eaten, device_food_eaten, food_flag_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host_food_eaten, device_food_eaten)");

        convert_dead_agents_to_food(host_was_alive,
                                    host_pos_x,
                                    host_pos_y,
                                    host_alive,
                                    host_exists,
                                    host_food_x,
                                    host_food_y,
                                    host_food_eaten,
                                    agent_count,
                                    food_count);

        for (int food_idx = 0; food_idx < food_count; ++food_idx) {
            if (host_food_eaten[food_idx]) {
                int roll = std::rand() % 100;
                if (roll < params.food_respawn_chance_percent) {
                    host_food_x[food_idx] = random_unit_float();
                    host_food_y[food_idx] = random_unit_float();
                    host_food_eaten[food_idx] = false;
                }
            }
        }

        reproduce_agents(host_pos_x,
                         host_pos_y,
                         host_energy,
                         host_alive,
                         host_exists,
                         host_dir_x,
                         host_dir_y,
                         host_reproduced,
                         agent_count,
                         params);

        check_cuda(cudaMemcpy(device_pos_x, host_pos_x, agent_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_pos_x, host_pos_x)");
        check_cuda(cudaMemcpy(device_pos_y, host_pos_y, agent_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_pos_y, host_pos_y)");
        check_cuda(cudaMemcpy(device_energy, host_energy, agent_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_energy, host_energy)");
        check_cuda(cudaMemcpy(device_alive, host_alive, agent_flag_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_alive, host_alive)");
        check_cuda(cudaMemcpy(device_dir_x, host_dir_x, agent_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_dir_x, host_dir_x)");
        check_cuda(cudaMemcpy(device_dir_y, host_dir_y, agent_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_dir_y, host_dir_y)");
        check_cuda(cudaMemcpy(device_food_x, host_food_x, food_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_food_x, host_food_x)");
        check_cuda(cudaMemcpy(device_food_y, host_food_y, food_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_food_y, host_food_y)");
        check_cuda(cudaMemcpy(device_food_eaten, host_food_eaten, food_flag_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device_food_eaten, host_food_eaten)");

        if (!stream_mode) {
            print_step_state(step,
                             host_pos_x,
                             host_pos_y,
                             host_energy,
                             host_alive,
                             host_food_x,
                             host_food_y,
                             host_food_eaten,
                             agent_count,
                             food_count);
        } else {
            stream_step_state(step + 1,
                              host_pos_x,
                              host_pos_y,
                              host_energy,
                              host_alive,
                              host_exists,
                              host_food_x,
                              host_food_y,
                              host_food_eaten,
                              agent_count,
                              food_count);
        }
    }

    check_cuda(cudaFree(device_pos_x), "cudaFree(device_pos_x)");
    check_cuda(cudaFree(device_pos_y), "cudaFree(device_pos_y)");
    check_cuda(cudaFree(device_energy), "cudaFree(device_energy)");
    check_cuda(cudaFree(device_alive), "cudaFree(device_alive)");
    check_cuda(cudaFree(device_dir_x), "cudaFree(device_dir_x)");
    check_cuda(cudaFree(device_dir_y), "cudaFree(device_dir_y)");
    check_cuda(cudaFree(device_food_x), "cudaFree(device_food_x)");
    check_cuda(cudaFree(device_food_y), "cudaFree(device_food_y)");
    check_cuda(cudaFree(device_food_eaten), "cudaFree(device_food_eaten)");

    std::free(host_pos_x);
    std::free(host_pos_y);
    std::free(host_energy);
    std::free(host_alive);
    std::free(host_was_alive);
    std::free(host_exists);
    std::free(host_reproduced);
    std::free(host_dir_x);
    std::free(host_dir_y);
    std::free(host_food_x);
    std::free(host_food_y);
    std::free(host_food_eaten);

    return EXIT_SUCCESS;
}
