#include <cstdlib>

#include "simulation_app.h"

int main(int argc, char** argv) {
    SimulationApp app;
    if (!app.configure(argc, argv)) {
        return EXIT_FAILURE;
    }

    app.run();
    return EXIT_SUCCESS;
}
