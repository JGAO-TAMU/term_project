#ifndef SIM_MATH_H
#define SIM_MATH_H

#include <cmath>
#include <cstdlib>

#include "sim_types.h"

inline float random_unit_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

inline float clamp01_host(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

inline void normalize_direction(float* x, float* y) {
    float length = std::sqrt((*x) * (*x) + (*y) * (*y));
    if (length > 0.0f) {
        *x /= length;
        *y /= length;
    } else {
        *x = 1.0f;
        *y = 0.0f;
    }
}

inline const char* species_name(int species) {
    if (species == SPECIES_BASS) {
        return "bass";
    }
    if (species == SPECIES_MINNOW) {
        return "minnow";
    }
    return "bluegill";
}

#endif
