#include "signal_generator.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<std::complex<float>> generate_signal(
    int N,
    const std::vector<SinusoidComponent>& components,
    unsigned int seed) {

    (void)seed;

    std::vector<std::complex<float>> signal(N, std::complex<float>(0.0f, 0.0f));

    for (const auto& comp : components) {
        for (int n = 0; n < N; n++) {
            const float angle = 2.0f * static_cast<float>(M_PI) * comp.frequency * n / N + comp.phase;
            const std::complex<float> sample(
                comp.amplitude * std::cos(angle),
                comp.amplitude * std::sin(angle)
            );
            signal[n] += sample;
        }
    }

    return signal;
}
