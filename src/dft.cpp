#include "dft.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void dft_naive(const std::vector<std::complex<float>>& input,
               std::vector<std::complex<float>>& output) {
    const int N = static_cast<int>(input.size());
    output.resize(N);

    for (int k = 0; k < N; k++) {
        std::complex<float> sum(0.0f, 0.0f);

        for (int n = 0; n < N; n++) {
            const float angle = -2.0f * static_cast<float>(M_PI) * k * n / N;
            const std::complex<float> twiddle(std::cos(angle), std::sin(angle));
            sum += input[n] * twiddle;
        }

        output[k] = sum;
    }
}
