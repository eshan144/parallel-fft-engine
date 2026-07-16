#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <vector>
#include <complex>

struct SinusoidComponent {
    float frequency;
    float amplitude;
    float phase;
};

std::vector<std::complex<float>> generate_signal(
    int N,
    const std::vector<SinusoidComponent>& components,
    unsigned int seed = 42);

#endif
