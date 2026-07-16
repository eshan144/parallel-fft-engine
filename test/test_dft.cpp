#include "dft.h"
#include "signal_generator.h"
#include <iostream>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float EPSILON = 1e-4f;

bool test_impulse() {
    const int N = 64;
    std::vector<std::complex<float>> input(N, std::complex<float>(0.0f, 0.0f));
    input[0] = std::complex<float>(1.0f, 0.0f);

    std::vector<std::complex<float>> output;
    dft_naive(input, output);

    for (int k = 0; k < N; k++) {
        if (std::abs(output[k] - std::complex<float>(1.0f, 0.0f)) > EPSILON) {
            std::cout << "FAIL: impulse test at k=" << k
                      << " expected (1,0) got (" << output[k].real() << "," << output[k].imag() << ")\n";
            return false;
        }
    }
    return true;
}

bool test_dc() {
    const int N = 64;
    std::vector<std::complex<float>> input(N, std::complex<float>(1.0f, 0.0f));

    std::vector<std::complex<float>> output;
    dft_naive(input, output);

    if (std::abs(output[0] - std::complex<float>(static_cast<float>(N), 0.0f)) > EPSILON) {
        std::cout << "FAIL: DC test at k=0 expected (" << N << ",0) got ("
                  << output[0].real() << "," << output[0].imag() << ")\n";
        return false;
    }

    for (int k = 1; k < N; k++) {
        if (std::abs(output[k]) > EPSILON * N) {
            std::cout << "FAIL: DC test at k=" << k << " expected (0,0) got magnitude "
                      << std::abs(output[k]) << "\n";
            return false;
        }
    }
    return true;
}

bool test_real_sinusoid() {
    const int N = 256;
    const float freq = 10.0f;

    std::vector<std::complex<float>> input(N);
    for (int n = 0; n < N; n++) {
        const float angle = 2.0f * static_cast<float>(M_PI) * freq * n / N;
        input[n] = std::complex<float>(std::cos(angle), 0.0f);
    }

    std::vector<std::complex<float>> output;
    dft_naive(input, output);

    const float expected_magnitude = N / 2.0f;
    const int pos_bin = static_cast<int>(freq);
    const int neg_bin = N - static_cast<int>(freq);

    if (std::abs(std::abs(output[pos_bin]) - expected_magnitude) > EPSILON * N) {
        std::cout << "FAIL: real sinusoid test at k=" << pos_bin
                  << " expected magnitude " << expected_magnitude
                  << " got " << std::abs(output[pos_bin]) << "\n";
        return false;
    }

    if (std::abs(std::abs(output[neg_bin]) - expected_magnitude) > EPSILON * N) {
        std::cout << "FAIL: real sinusoid test at k=" << neg_bin
                  << " expected magnitude " << expected_magnitude
                  << " got " << std::abs(output[neg_bin]) << "\n";
        return false;
    }

    return true;
}

bool test_complex_exponential() {
    const int N = 256;
    const int freq = 15;

    std::vector<std::complex<float>> input(N);
    for (int n = 0; n < N; n++) {
        const float angle = 2.0f * static_cast<float>(M_PI) * freq * n / N;
        input[n] = std::complex<float>(std::cos(angle), std::sin(angle));
    }

    std::vector<std::complex<float>> output;
    dft_naive(input, output);

    if (std::abs(output[freq]) - N > EPSILON * N) {
        std::cout << "FAIL: complex exponential test at k=" << freq
                  << " expected magnitude " << N
                  << " got " << std::abs(output[freq]) << "\n";
        return false;
    }

    for (int k = 0; k < N; k++) {
        if (k != freq && std::abs(output[k]) > EPSILON * N) {
            std::cout << "FAIL: complex exponential test at k=" << k
                      << " expected ~0 got magnitude " << std::abs(output[k]) << "\n";
            return false;
        }
    }

    return true;
}

bool test_parseval() {
    const int N = 128;
    std::vector<SinusoidComponent> components = {
        {5.0f, 1.5f, 0.0f},
        {12.0f, 2.0f, static_cast<float>(M_PI) / 4.0f}
    };

    auto input = generate_signal(N, components, 42);

    std::vector<std::complex<float>> output;
    dft_naive(input, output);

    float time_energy = 0.0f;
    for (int n = 0; n < N; n++) {
        time_energy += std::norm(input[n]);
    }

    float freq_energy = 0.0f;
    for (int k = 0; k < N; k++) {
        freq_energy += std::norm(output[k]);
    }
    freq_energy /= N;

    const float relative_error = std::abs(time_energy - freq_energy) / time_energy;
    if (relative_error > 1e-3f) {
        std::cout << "FAIL: Parseval test - time energy=" << time_energy
                  << " freq energy=" << freq_energy
                  << " relative error=" << relative_error << "\n";
        return false;
    }

    return true;
}

bool test_linearity() {
    const int N = 128;

    std::vector<SinusoidComponent> comp_x = {{7.0f, 1.0f, 0.0f}};
    std::vector<SinusoidComponent> comp_y = {{11.0f, 1.5f, 0.0f}};

    auto x = generate_signal(N, comp_x, 42);
    auto y = generate_signal(N, comp_y, 43);

    const float a = 2.0f;
    const float b = 3.0f;

    std::vector<std::complex<float>> ax_plus_by(N);
    for (int n = 0; n < N; n++) {
        ax_plus_by[n] = a * x[n] + b * y[n];
    }

    std::vector<std::complex<float>> dft_x, dft_y, dft_combined;
    dft_naive(x, dft_x);
    dft_naive(y, dft_y);
    dft_naive(ax_plus_by, dft_combined);

    for (int k = 0; k < N; k++) {
        const std::complex<float> expected = a * dft_x[k] + b * dft_y[k];
        const float error = std::abs(dft_combined[k] - expected);
        if (error > EPSILON * N) {
            std::cout << "FAIL: linearity test at k=" << k
                      << " error=" << error << "\n";
            return false;
        }
    }

    return true;
}

int main() {
    std::cout << "Running DFT Correctness Tests\n";
    std::cout << "==============================\n\n";

    int passed = 0;
    int total = 6;

    std::cout << "Test 1: Impulse response... ";
    if (test_impulse()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 2: DC signal... ";
    if (test_dc()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 3: Real sinusoid... ";
    if (test_real_sinusoid()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 4: Complex exponential... ";
    if (test_complex_exponential()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 5: Parseval's theorem... ";
    if (test_parseval()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 6: Linearity... ";
    if (test_linearity()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "\n==============================\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed\n";

    return (passed == total) ? 0 : 1;
}
