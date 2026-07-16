#include "fft.h"
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int bit_reverse(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static int log2_int(int n) {
    int result = 0;
    while (n > 1) {
        n >>= 1;
        result++;
    }
    return result;
}

void fft_radix2(const std::vector<std::complex<float>>& input,
                std::vector<std::complex<float>>& output) {
    const int N = static_cast<int>(input.size());

    if (!is_power_of_two(N)) {
        throw std::invalid_argument("FFT size must be a power of two");
    }

    const int log2_N = log2_int(N);

    output.resize(N);
    for (int i = 0; i < N; i++) {
        output[i] = input[i];
    }

    for (int i = 0; i < N; i++) {
        int j = bit_reverse(i, log2_N);
        if (i < j) {
            std::swap(output[i], output[j]);
        }
    }

    for (int s = 1; s <= log2_N; s++) {
        const int m = 1 << s;
        const int m_half = m >> 1;

        std::vector<std::complex<float>> twiddles(m_half);
        for (int j = 0; j < m_half; j++) {
            const float angle = -2.0f * static_cast<float>(M_PI) * j / m;
            twiddles[j] = std::complex<float>(std::cos(angle), std::sin(angle));
        }

        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < m_half; j++) {
                const std::complex<float> temp = output[k + j + m_half] * twiddles[j];
                output[k + j + m_half] = output[k + j] - temp;
                output[k + j] = output[k + j] + temp;
            }
        }
    }
}
