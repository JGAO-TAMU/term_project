#ifndef SIMULATION_APP_H
#define SIMULATION_APP_H

#include <cstddef>

#include "sim_types.h"

class SimulationApp {
public:
    SimulationApp();
    ~SimulationApp();

    bool configure(int argc, char** argv);
    void run();

private:
    void apply_defaults();
    bool parse_option(int argc, char** argv, int* arg_idx);
    void normalize_config();
    void allocate_host();
    void initialize_host();
    void allocate_device();
    void copy_all_host_to_device();
    void build_spatial_grids();
    void copy_spatial_grids_to_device();
    void copy_kernel_results_to_host();
    void copy_cpu_results_to_device();
    void initialize_report() const;
    void append_report_state(const char* label) const;
    void print_initial_output();
    void run_tick(int tick,
                  int blocks,
                  int threads_per_block,
                  int* births,
                  int* deaths,
                  int* predation_count);
    void print_run_summary(double program_ms) const;
    void write_report(double program_ms) const;
    void free_host();
    void free_device();

    int initial_bluegill_count;
    int initial_minnow_count;
    int initial_bass_count;
    int agent_count;
    int food_count;
    int cover_count;
    bool stream_mode;
    bool stats_mode;
    SimParams params;
    size_t agent_bytes;
    size_t species_bytes;
    size_t agent_flag_bytes;
    size_t food_bytes;
    size_t food_flag_bytes;
    size_t cover_bytes;
    size_t agent_grid_head_bytes;
    size_t food_grid_head_bytes;
    float total_kernel_ms;
    float max_kernel_ms;
    int kernel_launch_count;
    HostBuffers host;
    DeviceBuffers device;
};

#endif
