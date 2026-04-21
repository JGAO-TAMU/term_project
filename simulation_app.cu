#include "simulation_app.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cpu_rules.h"
#include "cuda_utils.h"
#include "kernels.cuh"
#include "output.h"
#include "sim_math.h"

static void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  --stream\n");
    std::printf("  --steps N\n");
    std::printf("  --agent-capacity N\n");
    std::printf("  --bluegill N\n");
    std::printf("  --minnow N\n");
    std::printf("  --bass N\n");
    std::printf("  --food N\n");
    std::printf("  --seed N\n");
    std::printf("  --bluegill-gain X\n");
    std::printf("  --bluegill-drain X\n");
    std::printf("  --bluegill-max-energy X\n");
    std::printf("  --minnow-gain X\n");
    std::printf("  --minnow-drain X\n");
    std::printf("  --minnow-max-energy X\n");
    std::printf("  --bass-drain X\n");
    std::printf("  --bass-gain X\n");
    std::printf("  --bass-max-energy X\n");
    std::printf("  --detection-radius X\n");
    std::printf("  --predation-radius X\n");
    std::printf("  --move-step X\n");
    std::printf("  --decay X\n");
    std::printf("  --noise X\n");
    std::printf("  --reproduce-energy X\n");
    std::printf("  --reproduce-cost X\n");
    std::printf("  --mate-radius X\n");
    std::printf("  --child-energy X\n");
    std::printf("  --reproduce-chance N\n");
    std::printf("  --max-children N\n");
    std::printf("  --bass-reproduce-energy X\n");
    std::printf("  --bass-reproduce-cost X\n");
    std::printf("  --bass-child-energy X\n");
    std::printf("  --bass-max-children N\n");
    std::printf("  --minnow-reproduce-energy X\n");
    std::printf("  --minnow-reproduce-cost X\n");
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
    std::printf("  --food-gain X\n");
    std::printf("  --food-eat-radius X\n");
    std::printf("  --food-respawn-chance N\n");
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

SimulationApp::SimulationApp() {
    host.pos_x = nullptr;
    host.pos_y = nullptr;
    host.energy = nullptr;
    host.size = nullptr;
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

    device.pos_x = nullptr;
    device.pos_y = nullptr;
    device.energy = nullptr;
    device.size = nullptr;
    device.alive = nullptr;
    device.species = nullptr;
    device.dir_x = nullptr;
    device.dir_y = nullptr;
    device.food_x = nullptr;
    device.food_y = nullptr;
    device.food_active = nullptr;

    apply_defaults();
}

SimulationApp::~SimulationApp() {
    free_device();
    free_host();
}

void SimulationApp::apply_defaults() {
    initial_bluegill_count = 10;
    initial_minnow_count = 20;
    initial_bass_count = 6;
    agent_count = 120;
    food_count = 80;
    stream_mode = false;
    agent_bytes = 0;
    species_bytes = 0;
    agent_flag_bytes = 0;
    food_bytes = 0;
    food_flag_bytes = 0;

    params.step_count = 50;
    params.reproduction_chance_percent = 60;
    params.food_respawn_chance_percent = 5;
    params.predation_success_percent = 70;
    params.bluegill_predation_success_percent = 45;
    params.minnow_predation_success_percent = 70;
    params.max_children_per_birth = 5;
    params.bass_max_children_per_birth = 3;
    params.minnow_max_children_per_birth = 8;
    params.seed = 12345U;
    params.bluegill_energy_gain = 0.00f;
    params.bluegill_energy_drain = 0.012f;
    params.bluegill_max_energy = 2.00f;
    params.minnow_energy_gain = 0.00f;
    params.minnow_energy_drain = 0.006f;
    params.minnow_max_energy = 1.80f;
    params.bass_energy_drain = 0.0125f;
    params.bass_energy_gain = 1.00f;
    params.bass_max_energy = 2.00f;
    params.detection_radius = 0.18f;
    params.predation_radius = 0.025f;
    params.move_step = 0.025f;
    params.decay_factor = 0.95f;
    params.noise_scale = 0.20f;
    params.reproduction_energy = 1.45f;
    params.reproduction_cost = 1.00f;
    params.mate_radius = 0.055f;
    params.child_energy = 0.55f;
    params.minnow_reproduction_energy = 1.20f;
    params.minnow_reproduction_cost = 0.55f;
    params.minnow_child_energy = 0.70f;
    params.bass_reproduction_energy = 1.95f;
    params.bass_reproduction_cost = 1.55f;
    params.bass_child_energy = 0.75f;
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
    params.food_energy_gain = 0.22f;
    params.food_eat_radius = 0.025f;
}

bool SimulationApp::parse_option(int argc, char** argv, int* arg_idx) {
    const char* value = nullptr;

    if (std::strcmp(argv[*arg_idx], "--stream") == 0) {
        stream_mode = true;
    } else if (std::strcmp(argv[*arg_idx], "--help") == 0) {
        print_usage(argv[0]);
        std::exit(EXIT_SUCCESS);
    } else if (std::strcmp(argv[*arg_idx], "--steps") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.step_count = std::atoi(value);
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
    } else if (std::strcmp(argv[*arg_idx], "--mate-radius") == 0) {
        if (!read_arg_value(argc, argv, arg_idx, &value)) {
            return false;
        }
        params.mate_radius = static_cast<float>(std::atof(value));
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
    return true;
}

void SimulationApp::normalize_config() {
    if (params.step_count < 0) {
        params.step_count = 0;
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
    if (params.minnow_max_energy <= 0.0f) {
        params.minnow_max_energy = 0.01f;
    }
    if (params.bass_max_energy <= 0.0f) {
        params.bass_max_energy = 0.01f;
    }
    if (params.minnow_reproduction_energy <= 0.0f) {
        params.minnow_reproduction_energy = 0.01f;
    }
    if (params.bass_reproduction_energy <= 0.0f) {
        params.bass_reproduction_energy = 0.01f;
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
}

void SimulationApp::allocate_host() {
    agent_bytes = static_cast<size_t>(agent_count) * sizeof(float);
    species_bytes = static_cast<size_t>(agent_count) * sizeof(int);
    agent_flag_bytes = static_cast<size_t>(agent_count) * sizeof(bool);
    food_bytes = static_cast<size_t>(food_count) * sizeof(float);
    food_flag_bytes = static_cast<size_t>(food_count) * sizeof(int);

    host.pos_x = static_cast<float*>(std::malloc(agent_bytes));
    host.pos_y = static_cast<float*>(std::malloc(agent_bytes));
    host.energy = static_cast<float*>(std::malloc(agent_bytes));
    host.size = static_cast<float*>(std::malloc(agent_bytes));
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

    if (host.pos_x == nullptr || host.pos_y == nullptr ||
        host.energy == nullptr || host.size == nullptr ||
        host.alive == nullptr || host.was_alive == nullptr ||
        host.exists == nullptr || host.reproduced == nullptr || host.species == nullptr ||
        host.dir_x == nullptr || host.dir_y == nullptr ||
        (food_count > 0 && (host.food_x == nullptr ||
                            host.food_y == nullptr ||
                            host.food_active == nullptr))) {
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
        host.size[i] = params.bluegill_min_size;
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
        host.size[i] = bluegill_start_size;
        host.alive[i] = true;
        host.exists[i] = true;
        host.species[i] = SPECIES_BLUEGILL;
    }

    for (int i = 0; i < initial_minnow_count; ++i) {
        int idx = initial_bluegill_count + i;
        float angle = 6.28318530718f * static_cast<float>(i) /
                      static_cast<float>(initial_minnow_count > 0 ? initial_minnow_count : 1);
        host.pos_x[idx] = 0.50f + 0.10f * std::cos(angle + 0.30f);
        host.pos_y[idx] = 0.50f + 0.10f * std::sin(angle + 0.30f);
        host.energy[idx] = 1.05f;
        host.size[idx] = minnow_start_size;
        host.alive[idx] = true;
        host.exists[idx] = true;
        host.species[idx] = SPECIES_MINNOW;
    }

    for (int i = 0; i < initial_bass_count; ++i) {
        int idx = initial_bluegill_count + initial_minnow_count + i;
        float angle = 6.28318530718f * static_cast<float>(i) /
                      static_cast<float>(initial_bass_count > 0 ? initial_bass_count : 1);
        host.pos_x[idx] = 0.50f + 0.32f * std::cos(angle);
        host.pos_y[idx] = 0.50f + 0.32f * std::sin(angle);
        host.energy[idx] = 1.25f;
        host.size[idx] = bass_start_size;
        host.alive[idx] = true;
        host.exists[idx] = true;
        host.species[idx] = SPECIES_BASS;
    }

    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
        host.food_x[food_idx] = static_cast<float>((food_idx * 37) % 100) / 99.0f;
        host.food_y[food_idx] = static_cast<float>((food_idx * 61 + 17) % 100) / 99.0f;
        host.food_active[food_idx] = 1;
    }
}

void SimulationApp::allocate_device() {
    check_cuda(cudaMalloc(&device.pos_x, agent_bytes), "cudaMalloc(device.pos_x)");
    check_cuda(cudaMalloc(&device.pos_y, agent_bytes), "cudaMalloc(device.pos_y)");
    check_cuda(cudaMalloc(&device.energy, agent_bytes), "cudaMalloc(device.energy)");
    check_cuda(cudaMalloc(&device.size, agent_bytes), "cudaMalloc(device.size)");
    check_cuda(cudaMalloc(&device.alive, agent_flag_bytes), "cudaMalloc(device.alive)");
    check_cuda(cudaMalloc(&device.species, species_bytes), "cudaMalloc(device.species)");
    check_cuda(cudaMalloc(&device.dir_x, agent_bytes), "cudaMalloc(device.dir_x)");
    check_cuda(cudaMalloc(&device.dir_y, agent_bytes), "cudaMalloc(device.dir_y)");
    if (food_count > 0) {
        check_cuda(cudaMalloc(&device.food_x, food_bytes), "cudaMalloc(device.food_x)");
        check_cuda(cudaMalloc(&device.food_y, food_bytes), "cudaMalloc(device.food_y)");
        check_cuda(cudaMalloc(&device.food_active, food_flag_bytes), "cudaMalloc(device.food_active)");
    }
}

void SimulationApp::copy_all_host_to_device() {
    check_cuda(cudaMemcpy(device.pos_x, host.pos_x, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.pos_x, host.pos_x)");
    check_cuda(cudaMemcpy(device.pos_y, host.pos_y, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.pos_y, host.pos_y)");
    check_cuda(cudaMemcpy(device.energy, host.energy, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.energy, host.energy)");
    check_cuda(cudaMemcpy(device.size, host.size, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.size, host.size)");
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

void SimulationApp::copy_kernel_results_to_host() {
    check_cuda(cudaMemcpy(host.pos_x, device.pos_x, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.pos_x, device.pos_x)");
    check_cuda(cudaMemcpy(host.pos_y, device.pos_y, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.pos_y, device.pos_y)");
    check_cuda(cudaMemcpy(host.energy, device.energy, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.energy, device.energy)");
    check_cuda(cudaMemcpy(host.size, device.size, agent_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(host.size, device.size)");
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
    check_cuda(cudaMemcpy(device.size, host.size, agent_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(device.size, host.size)");
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

void SimulationApp::print_initial_output() {
    if (stream_mode) {
        std::printf("step,type,species,id,x,y,energy,active,size\n");
        std::fflush(stdout);
        stream_step_state(0,
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
                          food_count);
    } else {
        std::printf("Bluegill: %d | Minnow: %d | Bass: %d | Food: %d/%d | Capacity: %d\n\n",
                    count_alive_species(host.alive, host.species, agent_count, SPECIES_BLUEGILL),
                    count_alive_species(host.alive, host.species, agent_count, SPECIES_MINNOW),
                    count_alive_species(host.alive, host.species, agent_count, SPECIES_BASS),
                    count_active_food(host.food_active, food_count),
                    food_count,
                    agent_count);
        print_state("Initial state",
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
                    food_count);
    }
}

void SimulationApp::run_step(int step, int blocks, int threads_per_block) {
    std::memcpy(host.was_alive, host.alive, agent_flag_bytes);

    simulate_agents_kernel<<<blocks, threads_per_block>>>(device.pos_x,
                                                          device.pos_y,
                                                          device.energy,
                                                          device.size,
                                                          device.alive,
                                                          device.species,
                                                          device.dir_x,
                                                          device.dir_y,
                                                          device.food_x,
                                                          device.food_y,
                                                          device.food_active,
                                                          agent_count,
                                                          food_count,
                                                          step,
                                                          params);
    check_cuda(cudaGetLastError(), "simulate_agents_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    copy_kernel_results_to_host();

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
    process_predation(host.pos_x,
                      host.pos_y,
                      host.energy,
                      host.size,
                      host.alive,
                      host.exists,
                      host.species,
                      agent_count,
                      params);
    reproduce_bluegill(host.pos_x,
                       host.pos_y,
                       host.energy,
                       host.size,
                       host.alive,
                       host.exists,
                       host.species,
                       host.dir_x,
                       host.dir_y,
                       host.reproduced,
                       agent_count,
                       params);
    reproduce_minnow(host.pos_x,
                     host.pos_y,
                     host.energy,
                     host.size,
                     host.alive,
                     host.exists,
                     host.species,
                     host.dir_x,
                     host.dir_y,
                     host.reproduced,
                     agent_count,
                     params);
    reproduce_bass(host.pos_x,
                   host.pos_y,
                   host.energy,
                   host.size,
                   host.alive,
                   host.exists,
                   host.species,
                   host.dir_x,
                   host.dir_y,
                   host.reproduced,
                   agent_count,
                   params);
    respawn_food(host.food_x,
                 host.food_y,
                 host.food_active,
                 food_count,
                 params);

    copy_cpu_results_to_device();

    if (stream_mode) {
        stream_step_state(step + 1,
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
                          food_count);
    } else {
        char label[64];
        std::snprintf(label, sizeof(label), "Step %d", step + 1);
        print_state(label,
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
                    food_count);
    }
}

void SimulationApp::run() {
    allocate_host();
    initialize_host();
    allocate_device();
    copy_all_host_to_device();

    const int threads_per_block = 32;
    const int blocks = (agent_count + threads_per_block - 1) / threads_per_block;

    print_initial_output();

    for (int step = 0; step < params.step_count; ++step) {
        run_step(step, blocks, threads_per_block);
    }
}

void SimulationApp::free_host() {
    std::free(host.pos_x);
    std::free(host.pos_y);
    std::free(host.energy);
    std::free(host.size);
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

    host.pos_x = nullptr;
    host.pos_y = nullptr;
    host.energy = nullptr;
    host.size = nullptr;
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
    if (device.size != nullptr) {
        check_cuda(cudaFree(device.size), "cudaFree(device.size)");
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

    device.pos_x = nullptr;
    device.pos_y = nullptr;
    device.energy = nullptr;
    device.size = nullptr;
    device.alive = nullptr;
    device.species = nullptr;
    device.dir_x = nullptr;
    device.dir_y = nullptr;
    device.food_x = nullptr;
    device.food_y = nullptr;
    device.food_active = nullptr;
}
