#include "simulation_app.h"

#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "cpu_rules.h"
#include "cuda_utils.h"
#include "kernels.cuh"
#include "output.h"
#include "sim_math.h"

static void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  --stream\n");
    std::printf("  --stats\n");
    std::printf("  --ticks N\n");
    std::printf("  --ticks-per-day N\n");
    std::printf("  --agent-capacity N\n");
    std::printf("  --bluegill N\n");
    std::printf("  --minnow N\n");
    std::printf("  --bass N\n");
    std::printf("  --food N\n");
    std::printf("  --food-pattern grid|random|cluster|ring|dense-circle|dense-rect\n");
    std::printf("  --cover N\n");
    std::printf("  --cover-pattern grid|random|cluster|ring|dense-circle|dense-rect\n");
    std::printf("  --spatial-grid N\n");
    std::printf("  --seed N\n");
    std::printf("  --bluegill-gain X\n");
    std::printf("  --bluegill-drain X\n");
    std::printf("  --bluegill-max-energy X\n");
    std::printf("  --bluegill-soft-cap X\n");
    std::printf("  --bluegill-refuge-scale X\n");
    std::printf("  --minnow-gain X\n");
    std::printf("  --minnow-drain X\n");
    std::printf("  --minnow-max-energy X\n");
    std::printf("  --minnow-soft-cap X\n");
    std::printf("  --minnow-refuge-scale X\n");
    std::printf("  --bass-drain X\n");
    std::printf("  --bass-gain X\n");
    std::printf("  --bass-max-energy X\n");
    std::printf("  --bass-soft-cap X\n");
    std::printf("  --death-energy-variation X\n");
    std::printf("  --bass-starvation-resistant N\n");
    std::printf("  --bass-starvation-reserve X\n");
    std::printf("  --detection-radius X\n");
    std::printf("  --predation-radius X\n");
    std::printf("  --move-step X\n");
    std::printf("  --decay X\n");
    std::printf("  --noise X\n");
    std::printf("  --reproduce-energy X\n");
    std::printf("  --reproduce-cost X\n");
    std::printf("  --reproduce-cooldown-days X\n");
    std::printf("  --mate-radius X\n");
    std::printf("  --bass-mate-radius X\n");
    std::printf("  --child-energy X\n");
    std::printf("  --reproduce-chance N\n");
    std::printf("  --max-children N\n");
    std::printf("  --bass-reproduce-energy X\n");
    std::printf("  --bass-reproduce-cost X\n");
    std::printf("  --bass-reproduce-cooldown-days X\n");
    std::printf("  --bass-child-energy X\n");
    std::printf("  --bass-max-children N\n");
    std::printf("  --minnow-reproduce-energy X\n");
    std::printf("  --minnow-reproduce-cost X\n");
    std::printf("  --minnow-reproduce-cooldown-days X\n");
    std::printf("  --minnow-child-energy X\n");
    std::printf("  --minnow-max-children N\n");
    std::printf("  --size-growth-threshold X\n");
    std::printf("  --size-shrink-threshold X\n");
    std::printf("  --size-growth-rate X\n");
    std::printf("  --size-shrink-rate X\n");
    std::printf("  --min-size X\n");
    std::printf("  --max-size X\n");
    std::printf("  --bluegill-min-size X\n");
    std::printf("  --bluegill-max-size X\n");
    std::printf("  --minnow-min-size X\n");
    std::printf("  --minnow-max-size X\n");
    std::printf("  --bass-min-size X\n");
    std::printf("  --bass-max-size X\n");
    std::printf("  --predation-success N\n");
    std::printf("  --bluegill-predation-success N\n");
    std::printf("  --minnow-predation-success N\n");
    std::printf("  --predation-half-sat X\n");
    std::printf("  --max-prey-per-bass-per-day X\n");
    std::printf("  --food-gain X\n");
    std::printf("  --food-eat-radius X\n");
    std::printf("  --food-respawn-chance N\n");
    std::printf("  --cover-radius X\n");
    std::printf("  --cover-protection X\n");
    std::printf("  --min-refuge-multiplier X\n");
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

static int parse_pattern_name(const char* value, int fallback) {
    if (std::strcmp(value, "grid") == 0) {
        return PATTERN_GRID;
    }
    if (std::strcmp(value, "random") == 0) {
        return PATTERN_RANDOM;
    }
    if (std::strcmp(value, "cluster") == 0 ||
        std::strcmp(value, "clustered") == 0) {
        return PATTERN_CLUSTER;
    }
    if (std::strcmp(value, "ring") == 0) {
        return PATTERN_RING;
    }
    if (std::strcmp(value, "dense-circle") == 0 ||
        std::strcmp(value, "dense_circle") == 0 ||
        std::strcmp(value, "circle") == 0) {
        return PATTERN_DENSE_CIRCLE;
    }
    if (std::strcmp(value, "dense-rect") == 0 ||
        std::strcmp(value, "dense_rect") == 0 ||
        std::strcmp(value, "rectangle") == 0 ||
        std::strcmp(value, "rect") == 0) {
        return PATTERN_DENSE_RECT;
    }
    return fallback;
}

static const char* pattern_name(int pattern) {
    if (pattern == PATTERN_RANDOM) {
        return "random";
    }
    if (pattern == PATTERN_CLUSTER) {
        return "cluster";
    }
    if (pattern == PATTERN_RING) {
        return "ring";
    }
    if (pattern == PATTERN_DENSE_CIRCLE) {
        return "dense-circle";
    }
    if (pattern == PATTERN_DENSE_RECT) {
        return "dense-rect";
    }
    return "grid";
}

static float tick_to_day(int tick, const SimParams& params) {
    if (params.ticks_per_day <= 0) {
        return 0.0f;
    }

    return static_cast<float>(tick) / static_cast<float>(params.ticks_per_day);
}

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

static const char* kReportFilename = "sim_report.txt";

static float initial_death_energy_threshold(int species, const SimParams& params) {
    if (species == SPECIES_BASS &&
        params.bass_starvation_resistant_percent > 0 &&
        (std::rand() % 100) < params.bass_starvation_resistant_percent) {
        float reserve_fraction = 0.25f + 0.75f * random_unit_float();
        return -params.bass_starvation_reserve * reserve_fraction;
    }

    return params.death_energy_variation * random_unit_float();
}

static void grid_point(int index, int count, float* x, float* y) {
    int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(count))));
    if (columns < 1) {
        columns = 1;
    }
    int rows = (count + columns - 1) / columns;
    if (rows < 1) {
        rows = 1;
    }

    int column = index % columns;
    int row = index / columns;
    *x = (static_cast<float>(column) + 0.5f) / static_cast<float>(columns);
    *y = (static_cast<float>(row) + 0.5f) / static_cast<float>(rows);
}

static void ring_point(int index, int count, float base_radius, float jitter, float* x, float* y) {
    float angle = 6.28318530718f * static_cast<float>(index) /
                  static_cast<float>(count > 0 ? count : 1);
    float radius = base_radius + jitter * (2.0f * random_unit_float() - 1.0f);
    *x = clamp01_host(0.50f + radius * std::cos(angle));
    *y = clamp01_host(0.50f + radius * std::sin(angle));
}

static void cluster_point(int index, int count, int points_per_patch, float spread, float* x, float* y) {
    int patch_count = count / points_per_patch;
    if (patch_count < 1) {
        patch_count = 1;
    }

    int patch_idx = index % patch_count;
    float patch_angle = 6.28318530718f * static_cast<float>(patch_idx) /
                        static_cast<float>(patch_count);
    float patch_radius = 0.16f + 0.30f * random_unit_float();
    float center_x = 0.50f + patch_radius * std::cos(patch_angle);
    float center_y = 0.50f + patch_radius * std::sin(patch_angle);
    float local_radius = spread * random_unit_float();
    float local_angle = 6.28318530718f * random_unit_float();

    *x = clamp01_host(center_x + local_radius * std::cos(local_angle));
    *y = clamp01_host(center_y + local_radius * std::sin(local_angle));
}

static void dense_circle_point(int index, int count, float radius, float* x, float* y) {
    int side = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(count) * 4.0f / 3.14159265359f)));
    if (side < 1) {
        side = 1;
    }

    int candidates_seen = 0;
    for (int row = 0; row < side; ++row) {
        for (int column = 0; column < side; ++column) {
            float candidate_x = (static_cast<float>(column) + 0.5f) / static_cast<float>(side);
            float candidate_y = (static_cast<float>(row) + 0.5f) / static_cast<float>(side);
            float dx = candidate_x - 0.5f;
            float dy = candidate_y - 0.5f;
            if (dx * dx + dy * dy > 0.25f) {
                continue;
            }

            if (candidates_seen == index) {
                *x = clamp01_host(0.5f + radius * 2.0f * dx);
                *y = clamp01_host(0.5f + radius * 2.0f * dy);
                return;
            }
            ++candidates_seen;
        }
    }

    float angle = 2.39996322973f * static_cast<float>(index);
    float local_radius = radius * std::sqrt((static_cast<float>(index) + 0.5f) /
                                            static_cast<float>(count > 0 ? count : 1));
    *x = clamp01_host(0.50f + local_radius * std::cos(angle));
    *y = clamp01_host(0.50f + local_radius * std::sin(angle));
}

static void dense_rect_point(int index, int count, float min_x, float max_x, float min_y, float max_y, float* x, float* y) {
    int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(count) * 2.0f)));
    if (columns < 1) {
        columns = 1;
    }
    int rows = (count + columns - 1) / columns;
    if (rows < 1) {
        rows = 1;
    }

    int column = index % columns;
    int row = index / columns;
    float u = (static_cast<float>(column) + 0.5f) / static_cast<float>(columns);
    float v = (static_cast<float>(row) + 0.5f) / static_cast<float>(rows);
    *x = clamp01_host(min_x + (max_x - min_x) * u);
    *y = clamp01_host(min_y + (max_y - min_y) * v);
}

SimulationApp::SimulationApp() {
    host.pos_x = nullptr;
    host.pos_y = nullptr;
    host.energy = nullptr;
    host.death_energy = nullptr;
    host.size = nullptr;
    host.spawn_cooldown = nullptr;
    host.alive = nullptr;
    host.was_alive = nullptr;
    host.exists = nullptr;
    host.reproduced = nullptr;
    host.species = nullptr;
    host.dir_x = nullptr;
    host.dir_y = nullptr;
    host.food_x = nullptr;
    host.food_y = nullptr;
    host.food_active = nullptr;
    host.agent_cell_head = nullptr;
    host.agent_cell_next = nullptr;
    host.food_cell_head = nullptr;
    host.food_cell_next = nullptr;
    host.cover_x = nullptr;
    host.cover_y = nullptr;
    host.cover_intensity = nullptr;

    device.pos_x = nullptr;
    device.pos_y = nullptr;
    device.energy = nullptr;
    device.death_energy = nullptr;
    device.size = nullptr;
    device.spawn_cooldown = nullptr;
    device.alive = nullptr;
    device.species = nullptr;
    device.dir_x = nullptr;
    device.dir_y = nullptr;
    device.food_x = nullptr;
    device.food_y = nullptr;
    device.food_active = nullptr;
    device.agent_cell_head = nullptr;
    device.agent_cell_next = nullptr;
    device.food_cell_head = nullptr;
    device.food_cell_next = nullptr;

    total_kernel_ms = 0.0f;
    max_kernel_ms = 0.0f;
    kernel_launch_count = 0;

    apply_defaults();
}

SimulationApp::~SimulationApp() {
    free_device();
    free_host();
}

void SimulationApp::apply_defaults() {
    initial_bluegill_count = 30;
    initial_minnow_count = 50;
    initial_bass_count = 1;
    agent_count = 320;
    food_count = 260;
    cover_count = 180;
    stream_mode = false;
    stats_mode = false;
    agent_bytes = 0;
    species_bytes = 0;
    agent_flag_bytes = 0;
    food_bytes = 0;
    food_flag_bytes = 0;
    cover_bytes = 0;
    agent_grid_head_bytes = 0;
    food_grid_head_bytes = 0;

    params.tick_count = 50;
    params.ticks_per_day = 96;
    params.reproduction_chance_percent = 34;
    params.food_respawn_chance_percent = 20;
    params.predation_success_percent = 70;
    params.bluegill_predation_success_percent = 16;
    params.minnow_predation_success_percent = 32;
    params.max_children_per_birth = 5;
    params.bass_max_children_per_birth = 5;
    params.minnow_max_children_per_birth = 5;
    params.food_pattern = PATTERN_GRID;
    params.cover_pattern = PATTERN_CLUSTER;
    params.spatial_grid_dim = 48;
    params.seed = 12345U;
    params.bluegill_energy_gain = 0.00f;
    params.bluegill_energy_drain = 0.00060f;
    params.bluegill_max_energy = 2.50f;
    params.bluegill_soft_cap = 0.0f;
    params.bluegill_refuge_scale = 1.20f;
    params.minnow_energy_gain = 0.00f;
    params.minnow_energy_drain = 0.00080f;
    params.minnow_max_energy = 1.70f;
    params.minnow_soft_cap = 0.0f;
    params.minnow_refuge_scale = 0.90f;
    params.bass_energy_drain = 0.0038f;
    params.bass_energy_gain = 6.00f;
    params.bass_max_energy = 4.00f;
    params.bass_soft_cap = 0.0f;
    params.death_energy_variation = 0.02f;
    params.bass_starvation_resistant_percent = 8;
    params.bass_starvation_reserve = 0.45f;
    params.detection_radius = 0.35f;
    params.predation_radius = 0.025f;
    params.move_step = 0.025f;
    params.decay_factor = 0.95f;
    params.noise_scale = 0.20f;
    params.reproduction_energy = 1.35f;
    params.reproduction_cost = 0.85f;
    params.reproduction_cooldown_days = 6.0f;
    params.mate_radius = 0.055f;
    params.bass_mate_radius = 0.14f;
    params.child_energy = 0.28f;
    params.minnow_reproduction_energy = 1.35f;
    params.minnow_reproduction_cost = 0.80f;
    params.minnow_child_energy = 0.18f;
    params.minnow_reproduction_cooldown_days = 5.0f;
    params.bass_reproduction_energy = 1.80f;
    params.bass_reproduction_cost = 1.20f;
    params.bass_child_energy = 0.45f;
    params.bass_reproduction_cooldown_days = 9.0f;
    params.size_growth_threshold = 1.30f;
    params.size_shrink_threshold = 0.70f;
    params.size_growth_rate = 0.015f;
    params.size_shrink_rate = 0.020f;
    params.min_size = 0.50f;
    params.max_size = 2.00f;
    params.bluegill_min_size = 0.10f;
    params.bluegill_max_size = 0.50f;
    params.minnow_min_size = 0.05f;
    params.minnow_max_size = 0.10f;
    params.bass_min_size = 0.80f;
    params.bass_max_size = 10.00f;
    params.food_energy_gain = 0.12f;
    params.food_eat_radius = 0.025f;
    params.cover_radius = 0.070f;
    params.cover_protection_scale = 0.82f;
    params.predation_half_sat = 120.0f;
    params.min_refuge_multiplier = 0.15f;
    params.max_prey_per_bass_per_day = 3.0f;
}

bool SimulationApp::parse_option(int argc, char** argv, int* arg_idx) {
    const char* value = nullptr;

    if (std::strcmp(argv[*arg_idx], "--stream") == 0) {
        stream_mode = true;
    } else if (std::strcmp(argv[*arg_idx], "--stats") == 0) {
        stats_mode = true;
    } else if (std::strcmp(argv[*arg_idx], "--help") == 0) {
        print_usage(argv[0]);
        std::exit(EXIT_SUCCESS);
    } else if (std::strcmp(argv[*arg_idx], "--ticks") == 0 ||
               std::strcmp(argv[*arg_idx], "--steps") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.tick_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--ticks-per-day") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.ticks_per_day = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--agent-capacity") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        agent_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--bluegill") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        initial_bluegill_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--minnow") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        initial_minnow_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--bass") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        initial_bass_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--food") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        food_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--food-pattern") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.food_pattern = parse_pattern_name(value, params.food_pattern);
    } else if (std::strcmp(argv[*arg_idx], "--cover") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        cover_count = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--cover-pattern") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.cover_pattern = parse_pattern_name(value, params.cover_pattern);
    } else if (std::strcmp(argv[*arg_idx], "--spatial-grid") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.spatial_grid_dim = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--seed") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.seed = static_cast<unsigned int>(std::strtoul(value, nullptr, 10));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-gain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_energy_gain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-drain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_energy_drain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-max-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_max_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-soft-cap") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_soft_cap = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-refuge-scale") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_refuge_scale = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-gain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_energy_gain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-drain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_energy_drain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-max-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_max_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-soft-cap") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_soft_cap = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-refuge-scale") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_refuge_scale = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-drain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_energy_drain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-gain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_energy_gain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-max-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_max_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-soft-cap") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_soft_cap = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--death-energy-variation") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.death_energy_variation = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-starvation-resistant") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_starvation_resistant_percent = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--bass-starvation-reserve") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_starvation_reserve = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--detection-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.detection_radius = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--predation-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.predation_radius = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--move-step") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.move_step = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--decay") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.decay_factor = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--noise") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.noise_scale = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--reproduce-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.reproduction_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--reproduce-cost") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.reproduction_cost = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--reproduce-cooldown-days") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.reproduction_cooldown_days = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--mate-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.mate_radius = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-mate-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_mate_radius = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--child-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.child_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--reproduce-chance") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.reproduction_chance_percent = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--max-children") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.max_children_per_birth = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--bass-reproduce-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_reproduction_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-reproduce-cost") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_reproduction_cost = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-reproduce-cooldown-days") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_reproduction_cooldown_days = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-child-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_child_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-max-children") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_max_children_per_birth = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--minnow-reproduce-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_reproduction_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-reproduce-cost") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_reproduction_cost = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-reproduce-cooldown-days") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_reproduction_cooldown_days = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-child-energy") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_child_energy = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-max-children") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_max_children_per_birth = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--size-growth-threshold") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.size_growth_threshold = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--size-shrink-threshold") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.size_shrink_threshold = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--size-growth-rate") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.size_growth_rate = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--size-shrink-rate") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.size_shrink_rate = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--min-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.min_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--max-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.max_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-min-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_min_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-max-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_max_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-min-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_min_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--minnow-max-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_max_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-min-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_min_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--bass-max-size") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bass_max_size = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--predation-success") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.predation_success_percent = std::atoi(value);
        params.bluegill_predation_success_percent = params.predation_success_percent;
        params.minnow_predation_success_percent = params.predation_success_percent;
    } else if (std::strcmp(argv[*arg_idx], "--bluegill-predation-success") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.bluegill_predation_success_percent = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--minnow-predation-success") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.minnow_predation_success_percent = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--predation-half-sat") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.predation_half_sat = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--max-prey-per-bass-per-day") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.max_prey_per_bass_per_day = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--food-gain") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.food_energy_gain = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--food-eat-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.food_eat_radius = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--food-respawn-chance") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.food_respawn_chance_percent = std::atoi(value);
    } else if (std::strcmp(argv[*arg_idx], "--cover-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.cover_radius = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--cover-protection") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.cover_protection_scale = static_cast<float>(std::atof(value));
    } else if (std::strcmp(argv[*arg_idx], "--min-refuge-multiplier") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.min_refuge_multiplier = static_cast<float>(std::atof(value));
    } else {
        std::fprintf(stderr, "Unknown option: %s\n", argv[*arg_idx]);
        print_usage(argv[0]);
        return false;
    }

    return true;
}

bool SimulationApp::configure(int argc, char** argv) {
    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (!parse_option(argc, argv, &arg_idx)) {
            return false;
        }
    }

    normalize_config();
    if (stats_mode) {
        stream_mode = false;
    }
    return true;
}

void SimulationApp::normalize_config() {
    if (params.tick_count < 0) {
        params.tick_count = 0;
    }
    if (params.ticks_per_day < 1) {
        params.ticks_per_day = 1;
    }
    if (initial_bluegill_count < 0) {
        initial_bluegill_count = 0;
    }
    if (initial_minnow_count < 0) {
        initial_minnow_count = 0;
    }
    if (initial_bass_count < 0) {
        initial_bass_count = 0;
    }
    if (food_count < 0) {
        food_count = 0;
    }
    if (cover_count < 0) {
        cover_count = 0;
    }
    int initial_agent_count = initial_bluegill_count + initial_minnow_count + initial_bass_count;
    if (agent_count < initial_agent_count) {
        agent_count = initial_agent_count;
    }
    if (params.reproduction_chance_percent < 0) {
        params.reproduction_chance_percent = 0;
    }
    if (params.reproduction_chance_percent > 100) {
        params.reproduction_chance_percent = 100;
    }
    if (params.food_respawn_chance_percent < 0) {
        params.food_respawn_chance_percent = 0;
    }
    if (params.food_respawn_chance_percent > 100) {
        params.food_respawn_chance_percent = 100;
    }
    if (params.predation_success_percent < 0) {
        params.predation_success_percent = 0;
    }
    if (params.predation_success_percent > 100) {
        params.predation_success_percent = 100;
    }
    if (params.bluegill_predation_success_percent < 0) {
        params.bluegill_predation_success_percent = 0;
    }
    if (params.bluegill_predation_success_percent > 100) {
        params.bluegill_predation_success_percent = 100;
    }
    if (params.minnow_predation_success_percent < 0) {
        params.minnow_predation_success_percent = 0;
    }
    if (params.minnow_predation_success_percent > 100) {
        params.minnow_predation_success_percent = 100;
    }
    if (params.max_children_per_birth < 1) {
        params.max_children_per_birth = 1;
    }
    if (params.bass_max_children_per_birth < 1) {
        params.bass_max_children_per_birth = 1;
    }
    if (params.minnow_max_children_per_birth < 1) {
        params.minnow_max_children_per_birth = 1;
    }
    if (params.bluegill_max_energy <= 0.0f) {
        params.bluegill_max_energy = 0.01f;
    }
    if (params.bluegill_soft_cap < 0.0f) {
        params.bluegill_soft_cap = 0.0f;
    }
    if (params.bluegill_refuge_scale < 0.0f) {
        params.bluegill_refuge_scale = 0.0f;
    }
    if (params.minnow_max_energy <= 0.0f) {
        params.minnow_max_energy = 0.01f;
    }
    if (params.minnow_soft_cap < 0.0f) {
        params.minnow_soft_cap = 0.0f;
    }
    if (params.minnow_refuge_scale < 0.0f) {
        params.minnow_refuge_scale = 0.0f;
    }
    if (params.bass_max_energy <= 0.0f) {
        params.bass_max_energy = 0.01f;
    }
    if (params.bass_soft_cap < 0.0f) {
        params.bass_soft_cap = 0.0f;
    }
    if (params.death_energy_variation < 0.0f) {
        params.death_energy_variation = 0.0f;
    }
    if (params.bass_starvation_resistant_percent < 0) {
        params.bass_starvation_resistant_percent = 0;
    }
    if (params.bass_starvation_resistant_percent > 100) {
        params.bass_starvation_resistant_percent = 100;
    }
    if (params.bass_starvation_reserve < 0.0f) {
        params.bass_starvation_reserve = 0.0f;
    }
    if (params.minnow_reproduction_energy <= 0.0f) {
        params.minnow_reproduction_energy = 0.01f;
    }
    if (params.bass_reproduction_energy <= 0.0f) {
        params.bass_reproduction_energy = 0.01f;
    }
    if (params.reproduction_cooldown_days < 0.0f) {
        params.reproduction_cooldown_days = 0.0f;
    }
    if (params.minnow_reproduction_cooldown_days < 0.0f) {
        params.minnow_reproduction_cooldown_days = 0.0f;
    }
    if (params.bass_reproduction_cooldown_days < 0.0f) {
        params.bass_reproduction_cooldown_days = 0.0f;
    }
    if (params.bass_mate_radius < 0.0f) {
        params.bass_mate_radius = 0.0f;
    }
    if (params.min_size <= 0.0f) {
        params.min_size = 0.01f;
    }
    if (params.max_size < params.min_size) {
        params.max_size = params.min_size;
    }
    if (params.bluegill_min_size <= 0.0f) {
        params.bluegill_min_size = 0.01f;
    }
    if (params.bluegill_max_size < params.bluegill_min_size) {
        params.bluegill_max_size = params.bluegill_min_size;
    }
    if (params.minnow_min_size <= 0.0f) {
        params.minnow_min_size = 0.01f;
    }
    if (params.minnow_max_size < params.minnow_min_size) {
        params.minnow_max_size = params.minnow_min_size;
    }
    if (params.bass_min_size <= 0.0f) {
        params.bass_min_size = 0.01f;
    }
    if (params.bass_max_size < params.bass_min_size) {
        params.bass_max_size = params.bass_min_size;
    }
    if (params.size_growth_rate < 0.0f) {
        params.size_growth_rate = 0.0f;
    }
    if (params.size_shrink_rate < 0.0f) {
        params.size_shrink_rate = 0.0f;
    }
    if (params.food_eat_radius < 0.0f) {
        params.food_eat_radius = 0.0f;
    }
    if (params.cover_radius < 0.0f) {
        params.cover_radius = 0.0f;
    }
    if (params.predation_half_sat <= 0.0f) {
        params.predation_half_sat = 1.0f;
    }
    if (params.max_prey_per_bass_per_day < 0.0f) {
        params.max_prey_per_bass_per_day = 0.0f;
    }
    if (params.cover_protection_scale < 0.0f) {
        params.cover_protection_scale = 0.0f;
    }
    if (params.cover_protection_scale > 1.0f) {
        params.cover_protection_scale = 1.0f;
    }
    if (params.min_refuge_multiplier < 0.0f) {
        params.min_refuge_multiplier = 0.0f;
    }
    if (params.min_refuge_multiplier > 1.0f) {
        params.min_refuge_multiplier = 1.0f;
    }
    if (params.spatial_grid_dim < 1) {
        params.spatial_grid_dim = 1;
    }
    if (params.spatial_grid_dim > 512) {
        params.spatial_grid_dim = 512;
    }
}

void SimulationApp::allocate_host() {
    agent_bytes = static_cast<size_t>(agent_count) * sizeof(float);
    species_bytes = static_cast<size_t>(agent_count) * sizeof(int);
    agent_flag_bytes = static_cast<size_t>(agent_count) * sizeof(bool);
    food_bytes = static_cast<size_t>(food_count) * sizeof(float);
    food_flag_bytes = static_cast<size_t>(food_count) * sizeof(int);
    cover_bytes = static_cast<size_t>(cover_count) * sizeof(float);
    food_grid_head_bytes = static_cast<size_t>(params.spatial_grid_dim) *
                           static_cast<size_t>(params.spatial_grid_dim) *
                           sizeof(int);
    agent_grid_head_bytes = food_grid_head_bytes * 3;

    host.pos_x = static_cast<float*>(std::malloc(agent_bytes));
    host.pos_y = static_cast<float*>(std::malloc(agent_bytes));
    host.energy = static_cast<float*>(std::malloc(agent_bytes));
    host.death_energy = static_cast<float*>(std::malloc(agent_bytes));
    host.size = static_cast<float*>(std::malloc(agent_bytes));
    host.spawn_cooldown = static_cast<int*>(std::malloc(species_bytes));
    host.alive = static_cast<bool*>(std::malloc(agent_flag_bytes));
    host.was_alive = static_cast<bool*>(std::malloc(agent_flag_bytes));
    host.exists = static_cast<bool*>(std::malloc(agent_flag_bytes));
    host.reproduced = static_cast<bool*>(std::malloc(agent_flag_bytes));
    host.species = static_cast<int*>(std::malloc(species_bytes));
    host.dir_x = static_cast<float*>(std::malloc(agent_bytes));
    host.dir_y = static_cast<float*>(std::malloc(agent_bytes));
    host.food_x = static_cast<float*>(std::malloc(food_bytes));
    host.food_y = static_cast<float*>(std::malloc(food_bytes));
    host.food_active = static_cast<int*>(std::malloc(food_flag_bytes));
    host.agent_cell_head = static_cast<int*>(std::malloc(agent_grid_head_bytes));
    host.agent_cell_next = static_cast<int*>(std::malloc(species_bytes));
    host.food_cell_head = static_cast<int*>(std::malloc(food_grid_head_bytes));
    host.food_cell_next = static_cast<int*>(std::malloc(food_flag_bytes));
    host.cover_x = static_cast<float*>(std::malloc(cover_bytes));
    host.cover_y = static_cast<float*>(std::malloc(cover_bytes));
    host.cover_intensity = static_cast<float*>(std::malloc(cover_bytes));

    if (host.pos_x == nullptr || host.pos_y == nullptr ||
        host.energy == nullptr || host.death_energy == nullptr || host.size == nullptr ||
        host.spawn_cooldown == nullptr ||
        host.alive == nullptr || host.was_alive == nullptr ||
        host.exists == nullptr || host.reproduced == nullptr ||
        host.species == nullptr ||
        host.dir_x == nullptr || host.dir_y == nullptr ||
        host.agent_cell_head == nullptr ||
        host.agent_cell_next == nullptr ||
        host.food_cell_head == nullptr ||
        (food_count > 0 && (host.food_x == nullptr ||
                            host.food_y == nullptr ||
                            host.food_active == nullptr ||
                            host.food_cell_next == nullptr)) ||
        (cover_count > 0 && (host.cover_x == nullptr ||
                             host.cover_y == nullptr ||
                             host.cover_intensity == nullptr))) {
        std::fprintf(stderr, "Host memory allocation failed.\n");
        std::exit(EXIT_FAILURE);
    }
}

void SimulationApp::initialize_host() {
    std::srand(params.seed);
    float bluegill_start_size = 0.5f * (params.bluegill_min_size + params.bluegill_max_size);
    float minnow_start_size = 0.5f * (params.minnow_min_size + params.minnow_max_size);
    float bass_start_size = params.bass_min_size +
                            0.05f * (params.bass_max_size - params.bass_min_size);

    for (int i = 0; i < agent_count; ++i) {
        host.pos_x[i] = 0.0f;
        host.pos_y[i] = 0.0f;
        host.energy[i] = 0.0f;
        host.death_energy[i] = 0.0f;
        host.size[i] = params.bluegill_min_size;
        host.spawn_cooldown[i] = 0;
        host.alive[i] = false;
        host.was_alive[i] = false;
        host.exists[i] = false;
        host.species[i] = SPECIES_BLUEGILL;
        host.dir_x[i] = 2.0f * random_unit_float() - 1.0f;
        host.dir_y[i] = 2.0f * random_unit_float() - 1.0f;
        normalize_direction(&host.dir_x[i], &host.dir_y[i]);
    }

    for (int i = 0; i < initial_bluegill_count; ++i) {
        float angle = 6.28318530718f * static_cast<float>(i) /
                      static_cast<float>(initial_bluegill_count > 0 ? initial_bluegill_count : 1);
        host.pos_x[i] = 0.50f + 0.18f * std::cos(angle);
        host.pos_y[i] = 0.50f + 0.18f * std::sin(angle);
        host.energy[i] = 1.00f;
        host.death_energy[i] = initial_death_energy_threshold(SPECIES_BLUEGILL, params);
        host.size[i] = bluegill_start_size;
        host.alive[i] = true;
        host.exists[i] = true;
        host.species[i] = SPECIES_BLUEGILL;
    }

    int bluegill_offset = initial_bluegill_count;
    for (int i = 0; i < initial_minnow_count; ++i) {
        int idx = bluegill_offset + i;
        float angle = 6.28318530718f * static_cast<float>(i) /
                      static_cast<float>(initial_minnow_count > 0 ? initial_minnow_count : 1);
        host.pos_x[idx] = 0.50f + 0.10f * std::cos(angle + 0.30f);
        host.pos_y[idx] = 0.50f + 0.10f * std::sin(angle + 0.30f);
        host.energy[idx] = 1.05f;
        host.death_energy[idx] = initial_death_energy_threshold(SPECIES_MINNOW, params);
        host.size[idx] = minnow_start_size;
        host.alive[idx] = true;
        host.exists[idx] = true;
        host.species[idx] = SPECIES_MINNOW;
    }

    int bass_offset = initial_bluegill_count + initial_minnow_count;
    for (int i = 0; i < initial_bass_count; ++i) {
        int idx = bass_offset + i;
        float angle = 6.28318530718f * static_cast<float>(i) /
                      static_cast<float>(initial_bass_count > 0 ? initial_bass_count : 1);
        host.pos_x[idx] = 0.50f + 0.32f * std::cos(angle);
        host.pos_y[idx] = 0.50f + 0.32f * std::sin(angle);
        host.energy[idx] = 1.25f;
        host.death_energy[idx] = initial_death_energy_threshold(SPECIES_BASS, params);
        host.size[idx] = bass_start_size;
        host.alive[idx] = true;
        host.exists[idx] = true;
        host.species[idx] = SPECIES_BASS;
    }

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        if (params.food_pattern == PATTERN_RANDOM) {
            host.food_x[food_idx] = random_unit_float();
            host.food_y[food_idx] = random_unit_float();
        } else if (params.food_pattern == PATTERN_CLUSTER) {
            cluster_point(food_idx, food_count, 24, 0.10f, &host.food_x[food_idx], &host.food_y[food_idx]);
        } else if (params.food_pattern == PATTERN_RING) {
            ring_point(food_idx, food_count, 0.32f, 0.08f, &host.food_x[food_idx], &host.food_y[food_idx]);
        } else if (params.food_pattern == PATTERN_DENSE_CIRCLE) {
            dense_circle_point(food_idx, food_count, 0.44f, &host.food_x[food_idx], &host.food_y[food_idx]);
        } else if (params.food_pattern == PATTERN_DENSE_RECT) {
            dense_rect_point(food_idx, food_count, 0.02f, 0.50f, 0.02f, 0.98f, &host.food_x[food_idx], &host.food_y[food_idx]);
        } else {
            grid_point(food_idx, food_count, &host.food_x[food_idx], &host.food_y[food_idx]);
        }
        host.food_active[food_idx] = 1;
    }

    for (int cover_idx = 0; cover_idx < cover_count; ++cover_idx) {
        if (params.cover_pattern == PATTERN_RANDOM) {
            host.cover_x[cover_idx] = random_unit_float();
            host.cover_y[cover_idx] = random_unit_float();
            host.cover_intensity[cover_idx] = 0.50f + 0.50f * random_unit_float();
        } else if (params.cover_pattern == PATTERN_GRID) {
            grid_point(cover_idx, cover_count, &host.cover_x[cover_idx], &host.cover_y[cover_idx]);
            host.cover_intensity[cover_idx] = 0.45f + 0.35f * random_unit_float();
        } else if (params.cover_pattern == PATTERN_RING) {
            ring_point(cover_idx, cover_count, 0.34f, 0.07f, &host.cover_x[cover_idx], &host.cover_y[cover_idx]);
            host.cover_intensity[cover_idx] = 0.68f + 0.32f * random_unit_float();
        } else if (params.cover_pattern == PATTERN_DENSE_CIRCLE) {
            dense_circle_point(cover_idx, cover_count, 0.42f, &host.cover_x[cover_idx], &host.cover_y[cover_idx]);
            host.cover_intensity[cover_idx] = 0.70f + 0.30f * random_unit_float();
        } else if (params.cover_pattern == PATTERN_DENSE_RECT) {
            dense_rect_point(cover_idx, cover_count, 0.02f, 0.50f, 0.02f, 0.98f, &host.cover_x[cover_idx], &host.cover_y[cover_idx]);
            host.cover_intensity[cover_idx] = 0.72f + 0.28f * random_unit_float();
        } else {
            cluster_point(cover_idx, cover_count, 18, 0.08f, &host.cover_x[cover_idx], &host.cover_y[cover_idx]);
            host.cover_intensity[cover_idx] = 0.72f + 0.28f * random_unit_float();
        }
    }
}

void SimulationApp::allocate_device() {
    check_cuda(cudaMalloc(&device.pos_x, agent_bytes), "cudaMalloc(device.pos_x)");
    check_cuda(cudaMalloc(&device.pos_y, agent_bytes), "cudaMalloc(device.pos_y)");
    check_cuda(cudaMalloc(&device.energy, agent_bytes), "cudaMalloc(device.energy)");
    check_cuda(cudaMalloc(&device.death_energy, agent_bytes), "cudaMalloc(device.death_energy)");
    check_cuda(cudaMalloc(&device.size, agent_bytes), "cudaMalloc(device.size)");
    check_cuda(cudaMalloc(&device.spawn_cooldown, species_bytes), "cudaMalloc(device.spawn_cooldown)");
    check_cuda(cudaMalloc(&device.alive, agent_flag_bytes), "cudaMalloc(device.alive)");
    check_cuda(cudaMalloc(&device.species, species_bytes), "cudaMalloc(device.species)");
    check_cuda(cudaMalloc(&device.dir_x, agent_bytes), "cudaMalloc(device.dir_x)");
    check_cuda(cudaMalloc(&device.dir_y, agent_bytes), "cudaMalloc(device.dir_y)");
    check_cuda(cudaMalloc(&device.agent_cell_head, agent_grid_head_bytes),
               "cudaMalloc(device.agent_cell_head)");
    check_cuda(cudaMalloc(&device.agent_cell_next, species_bytes),
               "cudaMalloc(device.agent_cell_next)");
    check_cuda(cudaMalloc(&device.food_cell_head, food_grid_head_bytes),
               "cudaMalloc(device.food_cell_head)");
    if (food_count > 0) {
        check_cuda(cudaMalloc(&device.food_x, food_bytes), "cudaMalloc(device.food_x)");
        check_cuda(cudaMalloc(&device.food_y, food_bytes), "cudaMalloc(device.food_y)");
        check_cuda(cudaMalloc(&device.food_active, food_flag_bytes), "cudaMalloc(device.food_active)");
        check_cuda(cudaMalloc(&device.food_cell_next, food_flag_bytes),
                   "cudaMalloc(device.food_cell_next)");
    }
}

void SimulationApp::copy_all_host_to_device() {
    check_cuda(cudaMemcpy(device.pos_x, host.pos_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.pos_x, host.pos_x)");
    check_cuda(cudaMemcpy(device.pos_y, host.pos_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.pos_y, host.pos_y)");
    check_cuda(cudaMemcpy(device.energy, host.energy, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.energy, host.energy)");
    check_cuda(cudaMemcpy(device.death_energy, host.death_energy, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.death_energy, host.death_energy)");
    check_cuda(cudaMemcpy(device.size, host.size, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.size, host.size)");
    check_cuda(cudaMemcpy(device.spawn_cooldown, host.spawn_cooldown, species_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.spawn_cooldown, host.spawn_cooldown)");
    check_cuda(cudaMemcpy(device.alive, host.alive, agent_flag_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.alive, host.alive)");
    check_cuda(cudaMemcpy(device.species, host.species, species_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.species, host.species)");
    check_cuda(cudaMemcpy(device.dir_x, host.dir_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.dir_x, host.dir_x)");
    check_cuda(cudaMemcpy(device.dir_y, host.dir_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.dir_y, host.dir_y)");
    if (food_count > 0) {
        check_cuda(cudaMemcpy(device.food_x, host.food_x, food_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_x, host.food_x)");
        check_cuda(cudaMemcpy(device.food_y, host.food_y, food_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_y, host.food_y)");
        check_cuda(cudaMemcpy(device.food_active, host.food_active, food_flag_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_active, host.food_active)");
    }
}

void SimulationApp::build_spatial_grids() {
    int cell_count = params.spatial_grid_dim * params.spatial_grid_dim;
    for (int cell = 0; cell < cell_count * 3; ++cell) {
        host.agent_cell_head[cell] = -1;
    }
    for (int cell = 0; cell < cell_count; ++cell) {
        host.food_cell_head[cell] = -1;
    }

    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        host.agent_cell_next[agent_idx] = -1;
        if (!host.alive[agent_idx]) {
            continue;
        }

        int agent_species = host.species[agent_idx];
        if (agent_species < SPECIES_BLUEGILL || agent_species > SPECIES_MINNOW) {
            continue;
        }

        int cell = spatial_cell_for_point_host(host.pos_x[agent_idx],
                                               host.pos_y[agent_idx],
                                               params.spatial_grid_dim);
        int species_cell = agent_species * cell_count + cell;
        host.agent_cell_next[agent_idx] = host.agent_cell_head[species_cell];
        host.agent_cell_head[species_cell] = agent_idx;
    }

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        host.food_cell_next[food_idx] = -1;
        if (host.food_active[food_idx] == 0) {
            continue;
        }

        int cell = spatial_cell_for_point_host(host.food_x[food_idx],
                                               host.food_y[food_idx],
                                               params.spatial_grid_dim);
        host.food_cell_next[food_idx] = host.food_cell_head[cell];
        host.food_cell_head[cell] = food_idx;
    }
}

void SimulationApp::copy_spatial_grids_to_device() {
    check_cuda(cudaMemcpy(device.agent_cell_head,
                          host.agent_cell_head,
                          agent_grid_head_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(device.agent_cell_head, host.agent_cell_head)");
    check_cuda(cudaMemcpy(device.agent_cell_next,
                          host.agent_cell_next,
                          species_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(device.agent_cell_next, host.agent_cell_next)");
    check_cuda(cudaMemcpy(device.food_cell_head,
                          host.food_cell_head,
                          food_grid_head_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(device.food_cell_head, host.food_cell_head)");
    if (food_count > 0) {
        check_cuda(cudaMemcpy(device.food_cell_next,
                              host.food_cell_next,
                              food_flag_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_cell_next, host.food_cell_next)");
    }
}

void SimulationApp::copy_kernel_results_to_host() {
    check_cuda(cudaMemcpy(host.pos_x, device.pos_x, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.pos_x, device.pos_x)");
    check_cuda(cudaMemcpy(host.pos_y, device.pos_y, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.pos_y, device.pos_y)");
    check_cuda(cudaMemcpy(host.energy, device.energy, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.energy, device.energy)");
    check_cuda(cudaMemcpy(host.death_energy, device.death_energy, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.death_energy, device.death_energy)");
    check_cuda(cudaMemcpy(host.size, device.size, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.size, device.size)");
    check_cuda(cudaMemcpy(host.spawn_cooldown, device.spawn_cooldown, species_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.spawn_cooldown, device.spawn_cooldown)");
    check_cuda(cudaMemcpy(host.alive, device.alive, agent_flag_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.alive, device.alive)");
    check_cuda(cudaMemcpy(host.species, device.species, species_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.species, device.species)");
    check_cuda(cudaMemcpy(host.dir_x, device.dir_x, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.dir_x, device.dir_x)");
    check_cuda(cudaMemcpy(host.dir_y, device.dir_y, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.dir_y, device.dir_y)");
    if (food_count > 0) {
        check_cuda(cudaMemcpy(host.food_x, device.food_x, food_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host.food_x, device.food_x)");
        check_cuda(cudaMemcpy(host.food_y, device.food_y, food_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host.food_y, device.food_y)");
        check_cuda(cudaMemcpy(host.food_active, device.food_active, food_flag_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(host.food_active, device.food_active)");
    }
}

void SimulationApp::copy_cpu_results_to_device() {
    check_cuda(cudaMemcpy(device.pos_x, host.pos_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.pos_x, host.pos_x)");
    check_cuda(cudaMemcpy(device.pos_y, host.pos_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.pos_y, host.pos_y)");
    check_cuda(cudaMemcpy(device.energy, host.energy, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.energy, host.energy)");
    check_cuda(cudaMemcpy(device.death_energy, host.death_energy, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.death_energy, host.death_energy)");
    check_cuda(cudaMemcpy(device.size, host.size, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.size, host.size)");
    check_cuda(cudaMemcpy(device.spawn_cooldown, host.spawn_cooldown, species_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.spawn_cooldown, host.spawn_cooldown)");
    check_cuda(cudaMemcpy(device.alive, host.alive, agent_flag_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.alive, host.alive)");
    check_cuda(cudaMemcpy(device.species, host.species, species_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.species, host.species)");
    check_cuda(cudaMemcpy(device.dir_x, host.dir_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.dir_x, host.dir_x)");
    check_cuda(cudaMemcpy(device.dir_y, host.dir_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.dir_y, host.dir_y)");
    if (food_count > 0) {
        check_cuda(cudaMemcpy(device.food_x, host.food_x, food_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_x, host.food_x)");
        check_cuda(cudaMemcpy(device.food_y, host.food_y, food_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_y, host.food_y)");
        check_cuda(cudaMemcpy(device.food_active, host.food_active, food_flag_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(device.food_active, host.food_active)");
    }
}

void SimulationApp::initialize_report() const {
    std::ofstream report(kReportFilename, std::ios::trunc);
    if (!report) {
        std::fprintf(stderr, "Warning: failed to initialize %s\n", kReportFilename);
        return;
    }

    report << "Detailed Simulation Output\n";
    report << "========================\n";
    report << "stream_mode=" << (stream_mode ? 1 : 0) << "\n\n";
}

void SimulationApp::append_report_state(const char* label) const {
    FILE* report = std::fopen(kReportFilename, "a");
    if (report == nullptr) {
        std::fprintf(stderr, "Warning: failed to append to %s\n", kReportFilename);
        return;
    }

    write_state(report,
                label,
                host.pos_x,
                host.pos_y,
                host.energy,
                host.size,
                host.alive,
                host.species,
                agent_count,
                host.food_x,
                host.food_y,
                host.food_active,
                food_count,
                host.cover_x,
                host.cover_y,
                host.cover_intensity,
                cover_count);
    std::fclose(report);
}

void SimulationApp::print_initial_output() {
    if (stats_mode) {
        std::printf("day,bluegill,minnow,bass,food,births,deaths,predation\n");
        stream_daily_stats(0.0f,
                           count_alive_species(host.alive, host.species, agent_count, SPECIES_BLUEGILL),
                           count_alive_species(host.alive, host.species, agent_count, SPECIES_MINNOW),
                           count_alive_species(host.alive, host.species, agent_count, SPECIES_BASS),
                           count_active_food(host.food_active, food_count),
                           0,
                           0,
                           0);
    } else if (stream_mode) {
        std::printf("tick,day,type,species,id,x,y,energy,active,size,intensity\n");
        std::fflush(stdout);
        stream_tick_state(0,
                          0.0f,
                          host.pos_x,
                          host.pos_y,
                          host.energy,
                          host.size,
                          host.alive,
                          host.exists,
                          host.species,
                          agent_count,
                          host.food_x,
                          host.food_y,
                          host.food_active,
                          food_count,
                          host.cover_x,
                          host.cover_y,
                          host.cover_intensity,
                          cover_count);
    } else {
        append_report_state("Initial state");
    }
}

void SimulationApp::run_tick(int tick,
                             int blocks,
                             int threads_per_block,
                             int* births,
                             int* deaths,
                             int* predation_count) {
    std::memcpy(host.was_alive, host.alive, agent_flag_bytes);
    build_spatial_grids();
    copy_spatial_grids_to_device();

    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate(kernel_start)");
    check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate(kernel_stop)");
    check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord(kernel_start)");

    simulate_agents_kernel<<<blocks, threads_per_block>>>(device.pos_x,
                                                          device.pos_y,
                                                          device.energy,
                                                          device.death_energy,
                                                          device.size,
                                                          device.spawn_cooldown,
                                                          device.alive,
                                                          device.species,
                                                          device.dir_x,
                                                          device.dir_y,
                                                          device.food_x,
                                                          device.food_y,
                                                          device.food_active,
                                                          device.agent_cell_head,
                                                          device.agent_cell_next,
                                                          device.food_cell_head,
                                                          device.food_cell_next,
                                                          agent_count,
                                                          food_count,
                                                          tick,
                                                          params);
    check_cuda(cudaEventRecord(kernel_stop), "cudaEventRecord(kernel_stop)");
    check_cuda(cudaGetLastError(), "simulate_agents_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize(kernel_stop)");

    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop),
               "cudaEventElapsedTime(kernel)");
    total_kernel_ms += kernel_ms;
    if (kernel_ms > max_kernel_ms) {
        max_kernel_ms = kernel_ms;
    }
    ++kernel_launch_count;

    check_cuda(cudaEventDestroy(kernel_start), "cudaEventDestroy(kernel_start)");
    check_cuda(cudaEventDestroy(kernel_stop), "cudaEventDestroy(kernel_stop)");

    copy_kernel_results_to_host();

    int tick_deaths = 0;
    for (int agent_idx = 0; agent_idx < agent_count; ++agent_idx) {
        if (host.was_alive[agent_idx] && !host.alive[agent_idx]) {
            ++tick_deaths;
        }
    }

    clear_newly_dead_agents(host.was_alive,
                            host.alive,
                            host.exists,
                            host.pos_x,
                            host.pos_y,
                            host.food_x,
                            host.food_y,
                            host.food_active,
                            agent_count,
                            food_count);
    int tick_predation = process_predation(host.pos_x,
                                           host.pos_y,
                                           host.energy,
                                           host.size,
                                           host.alive,
                                           host.exists,
                                           host.species,
                                           host.cover_x,
                                           host.cover_y,
                                           host.cover_intensity,
                                           agent_count,
                                           cover_count,
                                           params);
    int tick_births = 0;
    tick_births += reproduce_bass(host.pos_x,
                                  host.pos_y,
                                  host.energy,
                                  host.death_energy,
                                  host.size,
                                  host.spawn_cooldown,
                                  host.alive,
                                  host.exists,
                                  host.species,
                                  host.dir_x,
                                  host.dir_y,
                                  host.reproduced,
                                  tick,
                                  agent_count,
                                  params);
    tick_births += reproduce_bluegill(host.pos_x,
                                      host.pos_y,
                                      host.energy,
                                      host.death_energy,
                                      host.size,
                                      host.spawn_cooldown,
                                      host.alive,
                                      host.exists,
                                      host.species,
                                      host.dir_x,
                                      host.dir_y,
                                      host.reproduced,
                                      tick,
                                      agent_count,
                                      params);
    tick_births += reproduce_minnow(host.pos_x,
                                    host.pos_y,
                                    host.energy,
                                    host.death_energy,
                                    host.size,
                                    host.spawn_cooldown,
                                    host.alive,
                                    host.exists,
                                    host.species,
                                    host.dir_x,
                                    host.dir_y,
                                    host.reproduced,
                                    tick,
                                    agent_count,
                                    params);
    respawn_food(host.food_x,
                 host.food_y,
                 host.food_active,
                 food_count,
                 params);

    if (births != nullptr) {
        *births = tick_births;
    }
    if (deaths != nullptr) {
        *deaths = tick_deaths;
    }
    if (predation_count != nullptr) {
        *predation_count = tick_predation;
    }

    copy_cpu_results_to_device();

    if (stats_mode) {
        return;
    }

    if (stream_mode) {
        stream_tick_state(tick + 1,
                          tick_to_day(tick + 1, params),
                          host.pos_x,
                          host.pos_y,
                          host.energy,
                          host.size,
                          host.alive,
                          host.exists,
                          host.species,
                          agent_count,
                          host.food_x,
                          host.food_y,
                          host.food_active,
                          food_count,
                          host.cover_x,
                          host.cover_y,
                          host.cover_intensity,
                          cover_count);
    } else {
        char label[64];
        std::snprintf(label,
                      sizeof(label),
                      "Tick %d | Day %.2f",
                      tick + 1,
                      tick_to_day(tick + 1, params));
        append_report_state(label);
    }
}

void SimulationApp::run() {
    allocate_host();
    initialize_host();
    allocate_device();
    copy_all_host_to_device();

    const int threads_per_block = 32;
    const int blocks = (agent_count + threads_per_block - 1) / threads_per_block;

    total_kernel_ms = 0.0f;
    max_kernel_ms = 0.0f;
    kernel_launch_count = 0;

    initialize_report();
    auto program_start = std::chrono::steady_clock::now();

    print_initial_output();

    int day_births = 0;
    int day_deaths = 0;
    int day_predation = 0;

    for (int tick = 0; tick < params.tick_count; ++tick) {
        int tick_births = 0;
        int tick_deaths = 0;
        int tick_predation = 0;
        run_tick(tick,
                 blocks,
                 threads_per_block,
                 &tick_births,
                 &tick_deaths,
                 &tick_predation);
        day_births += tick_births;
        day_deaths += tick_deaths;
        day_predation += tick_predation;

        bool day_boundary = ((tick + 1) % params.ticks_per_day) == 0;
        bool final_tick = (tick + 1) == params.tick_count;
        if (stats_mode && (day_boundary || final_tick)) {
            stream_daily_stats(tick_to_day(tick + 1, params),
                               count_alive_species(host.alive, host.species, agent_count, SPECIES_BLUEGILL),
                               count_alive_species(host.alive, host.species, agent_count, SPECIES_MINNOW),
                               count_alive_species(host.alive, host.species, agent_count, SPECIES_BASS),
                               count_active_food(host.food_active, food_count),
                               day_births,
                               day_deaths,
                               day_predation);
            day_births = 0;
            day_deaths = 0;
            day_predation = 0;
        }
    }

    auto program_end = std::chrono::steady_clock::now();
    double program_ms = std::chrono::duration<double, std::milli>(program_end - program_start).count();

    print_run_summary(program_ms);
    write_report(program_ms);
}

void SimulationApp::print_run_summary(double program_ms) const {
    int final_bluegill = count_alive_species(host.alive, host.species, agent_count, SPECIES_BLUEGILL);
    int final_minnow = count_alive_species(host.alive, host.species, agent_count, SPECIES_MINNOW);
    int final_bass = count_alive_species(host.alive, host.species, agent_count, SPECIES_BASS);
    int active_food = count_active_food(host.food_active, food_count);
    float average_kernel_ms = kernel_launch_count > 0 ?
                              total_kernel_ms / static_cast<float>(kernel_launch_count) :
                              0.0f;

    FILE* out = (stream_mode || stats_mode) ? stderr : stdout;
    std::fprintf(out, "\nRun Summary\n");
    std::fprintf(out, "Program time: %.3f ms\n", program_ms);
    std::fprintf(out, "Kernel total: %.3f ms | avg: %.3f ms | max: %.3f ms | launches: %d\n",
                 total_kernel_ms,
                 average_kernel_ms,
                 max_kernel_ms,
                 kernel_launch_count);
    std::fprintf(out, "Final day: %.2f | Final tick: %d\n",
                 tick_to_day(params.tick_count, params),
                 params.tick_count);
    std::fprintf(out, "Final counts | Bluegill: %d | Minnow: %d | Bass: %d | Food: %d/%d | Cover: %d | Alive total: %d\n",
                 final_bluegill,
                 final_minnow,
                 final_bass,
                 active_food,
                 food_count,
                 cover_count,
                 final_bluegill + final_minnow + final_bass);
    std::fflush(out);
}

void SimulationApp::write_report(double program_ms) const {
    std::ofstream report(kReportFilename, std::ios::app);
    if (!report) {
        std::fprintf(stderr, "Warning: failed to write %s\n", kReportFilename);
        return;
    }

    int final_bluegill = count_alive_species(host.alive, host.species, agent_count, SPECIES_BLUEGILL);
    int final_minnow = count_alive_species(host.alive, host.species, agent_count, SPECIES_MINNOW);
    int final_bass = count_alive_species(host.alive, host.species, agent_count, SPECIES_BASS);
    int active_food = count_active_food(host.food_active, food_count);
    float average_kernel_ms = kernel_launch_count > 0 ?
                              total_kernel_ms / static_cast<float>(kernel_launch_count) :
                              0.0f;

    report << "Run Summary\n";
    report << "===========\n";
    report << "Program time: " << program_ms << " ms\n";
    report << "Kernel total: " << total_kernel_ms
           << " ms | avg: " << average_kernel_ms
           << " ms | max: " << max_kernel_ms
           << " ms | launches: " << kernel_launch_count << "\n";
    report << "Final day: " << tick_to_day(params.tick_count, params)
           << " | Final tick: " << params.tick_count << "\n";
    report << "Final counts | Bluegill: " << final_bluegill
           << " | Minnow: " << final_minnow
           << " | Bass: " << final_bass
           << " | Food: " << active_food << "/" << food_count
           << " | Cover: " << cover_count
           << " | Alive total: " << (final_bluegill + final_minnow + final_bass) << "\n\n";

    report << "Simulation Report\n";
    report << "=================\n";
    report << "timing.program_ms=" << program_ms << "\n";
    report << "timing.kernel_total_ms=" << total_kernel_ms << "\n";
    report << "timing.kernel_avg_ms=" << average_kernel_ms << "\n";
    report << "timing.kernel_max_ms=" << max_kernel_ms << "\n";
    report << "timing.kernel_launches=" << kernel_launch_count << "\n";
    report << "sim.ticks=" << params.tick_count << "\n";
    report << "sim.ticks_per_day=" << params.ticks_per_day << "\n";
    report << "sim.final_day=" << tick_to_day(params.tick_count, params) << "\n";
    report << "sim.stream_mode=" << (stream_mode ? 1 : 0) << "\n";
    report << "sim.agent_capacity=" << agent_count << "\n";
    report << "sim.initial_bluegill=" << initial_bluegill_count << "\n";
    report << "sim.initial_minnow=" << initial_minnow_count << "\n";
    report << "sim.initial_bass=" << initial_bass_count << "\n";
    report << "sim.final_bluegill=" << final_bluegill << "\n";
    report << "sim.final_minnow=" << final_minnow << "\n";
    report << "sim.final_bass=" << final_bass << "\n";
    report << "sim.active_food=" << active_food << "\n";
    report << "sim.total_food=" << food_count << "\n";
    report << "sim.total_cover=" << cover_count << "\n";
    report << "debug.detection_radius=" << params.detection_radius << "\n";
    report << "debug.spatial_grid_dim=" << params.spatial_grid_dim << "\n";
    report << "debug.predation_radius=" << params.predation_radius << "\n";
    report << "debug.move_step=" << params.move_step << "\n";
    report << "debug.food_respawn_chance_percent=" << params.food_respawn_chance_percent << "\n";
    report << "debug.food_energy_gain=" << params.food_energy_gain << "\n";
    report << "debug.food_pattern=" << pattern_name(params.food_pattern) << "\n";
    report << "debug.cover_radius=" << params.cover_radius << "\n";
    report << "debug.cover_protection_scale=" << params.cover_protection_scale << "\n";
    report << "debug.cover_pattern=" << pattern_name(params.cover_pattern) << "\n";
    report << "debug.predation_half_sat=" << params.predation_half_sat << "\n";
    report << "debug.min_refuge_multiplier=" << params.min_refuge_multiplier << "\n";
    report << "debug.max_prey_per_bass_per_day=" << params.max_prey_per_bass_per_day << "\n";
    report << "debug.bluegill_predation_success_percent=" << params.bluegill_predation_success_percent << "\n";
    report << "debug.minnow_predation_success_percent=" << params.minnow_predation_success_percent << "\n";
    report << "debug.bluegill_soft_cap=" << params.bluegill_soft_cap << "\n";
    report << "debug.minnow_soft_cap=" << params.minnow_soft_cap << "\n";
    report << "debug.bass_soft_cap=" << params.bass_soft_cap << "\n";
    report << "debug.bluegill_refuge_scale=" << params.bluegill_refuge_scale << "\n";
    report << "debug.minnow_refuge_scale=" << params.minnow_refuge_scale << "\n";
    report << "debug.death_energy_variation=" << params.death_energy_variation << "\n";
    report << "debug.bass_starvation_resistant_percent="
           << params.bass_starvation_resistant_percent << "\n";
    report << "debug.bass_starvation_reserve=" << params.bass_starvation_reserve << "\n";
    report << "debug.mate_radius=" << params.mate_radius << "\n";
    report << "debug.bass_mate_radius=" << params.bass_mate_radius << "\n";
    report << "debug.bluegill_reproduction_pulse_days=" << params.reproduction_cooldown_days << "\n";
    report << "debug.minnow_reproduction_pulse_days=" << params.minnow_reproduction_cooldown_days << "\n";
    report << "debug.bass_reproduction_pulse_days=" << params.bass_reproduction_cooldown_days << "\n";
}

void SimulationApp::free_host() {
    std::free(host.pos_x);
    std::free(host.pos_y);
    std::free(host.energy);
    std::free(host.death_energy);
    std::free(host.size);
    std::free(host.spawn_cooldown);
    std::free(host.alive);
    std::free(host.was_alive);
    std::free(host.exists);
    std::free(host.reproduced);
    std::free(host.species);
    std::free(host.dir_x);
    std::free(host.dir_y);
    std::free(host.food_x);
    std::free(host.food_y);
    std::free(host.food_active);
    std::free(host.agent_cell_head);
    std::free(host.agent_cell_next);
    std::free(host.food_cell_head);
    std::free(host.food_cell_next);
    std::free(host.cover_x);
    std::free(host.cover_y);
    std::free(host.cover_intensity);

    host.pos_x = nullptr;
    host.pos_y = nullptr;
    host.energy = nullptr;
    host.death_energy = nullptr;
    host.size = nullptr;
    host.spawn_cooldown = nullptr;
    host.alive = nullptr;
    host.was_alive = nullptr;
    host.exists = nullptr;
    host.reproduced = nullptr;
    host.species = nullptr;
    host.dir_x = nullptr;
    host.dir_y = nullptr;
    host.food_x = nullptr;
    host.food_y = nullptr;
    host.food_active = nullptr;
    host.agent_cell_head = nullptr;
    host.agent_cell_next = nullptr;
    host.food_cell_head = nullptr;
    host.food_cell_next = nullptr;
    host.cover_x = nullptr;
    host.cover_y = nullptr;
    host.cover_intensity = nullptr;
}

void SimulationApp::free_device() {
    if (device.pos_x != nullptr) {
        check_cuda(cudaFree(device.pos_x), "cudaFree(device.pos_x)");
    }
    if (device.pos_y != nullptr) {
        check_cuda(cudaFree(device.pos_y), "cudaFree(device.pos_y)");
    }
    if (device.energy != nullptr) {
        check_cuda(cudaFree(device.energy), "cudaFree(device.energy)");
    }
    if (device.death_energy != nullptr) {
        check_cuda(cudaFree(device.death_energy), "cudaFree(device.death_energy)");
    }
    if (device.size != nullptr) {
        check_cuda(cudaFree(device.size), "cudaFree(device.size)");
    }
    if (device.spawn_cooldown != nullptr) {
        check_cuda(cudaFree(device.spawn_cooldown), "cudaFree(device.spawn_cooldown)");
    }
    if (device.alive != nullptr) {
        check_cuda(cudaFree(device.alive), "cudaFree(device.alive)");
    }
    if (device.species != nullptr) {
        check_cuda(cudaFree(device.species), "cudaFree(device.species)");
    }
    if (device.dir_x != nullptr) {
        check_cuda(cudaFree(device.dir_x), "cudaFree(device.dir_x)");
    }
    if (device.dir_y != nullptr) {
        check_cuda(cudaFree(device.dir_y), "cudaFree(device.dir_y)");
    }
    if (device.food_x != nullptr) {
        check_cuda(cudaFree(device.food_x), "cudaFree(device.food_x)");
    }
    if (device.food_y != nullptr) {
        check_cuda(cudaFree(device.food_y), "cudaFree(device.food_y)");
    }
    if (device.food_active != nullptr) {
        check_cuda(cudaFree(device.food_active), "cudaFree(device.food_active)");
    }
    if (device.agent_cell_head != nullptr) {
        check_cuda(cudaFree(device.agent_cell_head), "cudaFree(device.agent_cell_head)");
    }
    if (device.agent_cell_next != nullptr) {
        check_cuda(cudaFree(device.agent_cell_next), "cudaFree(device.agent_cell_next)");
    }
    if (device.food_cell_head != nullptr) {
        check_cuda(cudaFree(device.food_cell_head), "cudaFree(device.food_cell_head)");
    }
    if (device.food_cell_next != nullptr) {
        check_cuda(cudaFree(device.food_cell_next), "cudaFree(device.food_cell_next)");
    }

    device.pos_x = nullptr;
    device.pos_y = nullptr;
    device.energy = nullptr;
    device.death_energy = nullptr;
    device.size = nullptr;
    device.spawn_cooldown = nullptr;
    device.alive = nullptr;
    device.species = nullptr;
    device.dir_x = nullptr;
    device.dir_y = nullptr;
    device.food_x = nullptr;
    device.food_y = nullptr;
    device.food_active = nullptr;
    device.agent_cell_head = nullptr;
    device.agent_cell_next = nullptr;
    device.food_cell_head = nullptr;
    device.food_cell_next = nullptr;
}
