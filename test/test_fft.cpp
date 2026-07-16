#include "dft.h"
#include "fft.h"
#include "signal_generator.h"
#include <iostream>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float EPSILON = 1e-4f;

bool test_fft_impulse() {
    const int N = 64;
    std::vector<std::complex<float>> input(N, std::complex<float>(0.0f, 0.0f));
    input[0] = std::complex<float>(1.0f, 0.0f);

    std::vector<std::complex<float>> output;
    fft_radix2(input, output);

    for (int k = 0; k < N; k++) {
        if (std::abs(output[k] - std::complex<float>(1.0f, 0.0f)) > EPSILON) {
            std::cout << "FAIL: FFT impulse test at k=" << k
                      << " expected (1,0) got (" << output[k].real() << "," << output[k].imag() << ")\n";
            return false;
        }
    }
    return true;
}

bool test_fft_dc() {
    const int N = 64;
    std::vector<std::complex<float>> input(N, std::complex<float>(1.0f, 0.0f));

    std::vector<std::complex<float>> output;
    fft_radix2(input, output);

    if (std::abs(output[0] - std::complex<float>(static_cast<float>(N), 0.0f)) > EPSILON * N) {
        std::cout << "FAIL: FFT DC test at k=0 expected (" << N << ",0) got ("
                  << output[0].real() << "," << output[0].imag() << ")\n";
        return false;
    }

    for (int k = 1; k < N; k++) {
        if (std::abs(output[k]) > EPSILON * N) {
            std::cout << "FAIL: FFT DC test at k=" << k << " expected (0,0) got magnitude "
                      << std::abs(output[k]) << "\n";
            return false;
        }
    }
    return true;
}

bool test_fft_real_sinusoid() {
    const int N = 256;
    const float freq = 10.0f;

    std::vector<std::complex<float>> input(N);
    for (int n = 0; n < N; n++) {
        const float angle = 2.0f * static_cast<float>(M_PI) * freq * n / N;
        input[n] = std::complex<float>(std::cos(angle), 0.0f);
    }

    std::vector<std::complex<float>> output;
    fft_radix2(input, output);

    const float expected_magnitude = N / 2.0f;
    const int pos_bin = static_cast<int>(freq);
    const int neg_bin = N - static_cast<int>(freq);

    if (std::abs(std::abs(output[pos_bin]) - expected_magnitude) > EPSILON * N) {
        std::cout << "FAIL: FFT real sinusoid test at k=" << pos_bin
                  << " expected magnitude " << expected_magnitude
                  << " got " << std::abs(output[pos_bin]) << "\n";
        return false;
    }

    if (std::abs(std::abs(output[neg_bin]) - expected_magnitude) > EPSILON * N) {
        std::cout << "FAIL: FFT real sinusoid test at k=" << neg_bin
                  << " expected magnitude " << expected_magnitude
                  << " got " << std::abs(output[neg_bin]) << "\n";
        return false;
    }

    return true;
}

bool test_fft_complex_exponential() {
    const int N = 256;
    const int freq = 15;

    std::vector<std::complex<float>> input(N);
    for (int n = 0; n < N; n++) {
        const float angle = 2.0f * static_cast<float>(M_PI) * freq * n / N;
        input[n] = std::complex<float>(std::cos(angle), std::sin(angle));
    }

    std::vector<std::complex<float>> output;
    fft_radix2(input, output);

    if (std::abs(output[freq]) - N > EPSILON * N) {
        std::cout << "FAIL: FFT complex exponential test at k=" << freq
                  << " expected magnitude " << N
                  << " got " << std::abs(output[freq]) << "\n";
        return false;
    }

    for (int k = 0; k < N; k++) {
        if (k != freq && std::abs(output[k]) > EPSILON * N) {
            std::cout << "FAIL: FFT complex exponential test at k=" << k
                      << " expected ~0 got magnitude " << std::abs(output[k]) << "\n";
            return false;
        }
    }

    return true;
}

bool test_fft_parseval() {
    const int N = 128;
    std::vector<SinusoidComponent> components = {
        {5.0f, 1.5f, 0.0f},
        {12.0f, 2.0f, static_cast<float>(M_PI) / 4.0f}
    };

    auto input = generate_signal(N, components, 42);

    std::vector<std::complex<float>> output;
    fft_radix2(input, output);

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
        std::cout << "FAIL: FFT Parseval test - time energy=" << time_energy
                  << " freq energy=" << freq_energy
                  << " relative error=" << relative_error << "\n";
        return false;
    }

    return true;
}

bool test_fft_linearity() {
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

    std::vector<std::complex<float>> fft_x, fft_y, fft_combined;
    fft_radix2(x, fft_x);
    fft_radix2(y, fft_y);
    fft_radix2(ax_plus_by, fft_combined);

    for (int k = 0; k < N; k++) {
        const std::complex<float> expected = a * fft_x[k] + b * fft_y[k];
        const float error = std::abs(fft_combined[k] - expected);
        if (error > EPSILON * N) {
            std::cout << "FAIL: FFT linearity test at k=" << k
                      << " error=" << error << "\n";
            return false;
        }
    }

    return true;
}

bool test_fft_vs_dft(int N, float& max_error, float& rel_l2_error) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);

    std::vector<std::complex<float>> dft_output, fft_output;
    dft_naive(input, dft_output);
    fft_radix2(input, fft_output);

    max_error = 0.0f;
    float error_norm_squared = 0.0f;
    float dft_norm_squared = 0.0f;

    for (int k = 0; k < N; k++) {
        const float error = std::abs(dft_output[k] - fft_output[k]);
        max_error = std::max(max_error, error);

        error_norm_squared += error * error;
        dft_norm_squared += std::norm(dft_output[k]);
    }

    const float error_l2_norm = std::sqrt(error_norm_squared);
    const float dft_l2_norm = std::sqrt(dft_norm_squared);
    rel_l2_error = error_l2_norm / dft_l2_norm;

    const float abs_tolerance = 0.1f;
    const float rel_tolerance = 1e-4f;

    if (max_error > abs_tolerance || rel_l2_error > rel_tolerance) {
        std::cout << "FAIL: FFT vs DFT comparison for N=" << N
                  << " max_error=" << max_error << " (tolerance=" << abs_tolerance << ")"
                  << " rel_l2_error=" << rel_l2_error << " (tolerance=" << rel_tolerance << ")\n";
        return false;
    }

    return true;
}

bool test_fft_invalid_size() {
    try {
        std::vector<std::complex<float>> input(100, std::complex<float>(1.0f, 0.0f));
        std::vector<std::complex<float>> output;
        fft_radix2(input, output);
        std::cout << "FAIL: FFT should reject non-power-of-two size\n";
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

int main() {
    std::cout << "Running FFT Correctness Tests\n";
    std::cout << "==============================\n\n";

    int passed = 0;
    int total = 12;

    std::cout << "Test 1: FFT impulse response... ";
    if (test_fft_impulse()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 2: FFT DC signal... ";
    if (test_fft_dc()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 3: FFT real sinusoid... ";
    if (test_fft_real_sinusoid()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 4: FFT complex exponential... ";
    if (test_fft_complex_exponential()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 5: FFT Parseval's theorem... ";
    if (test_fft_parseval()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 6: FFT linearity... ";
    if (test_fft_linearity()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "Test 7: FFT invalid size rejection... ";
    if (test_fft_invalid_size()) {
        std::cout << "PASS\n";
        passed++;
    } else {
        std::cout << "FAIL\n";
    }

    std::cout << "\nFFT vs DFT Numerical Error Comparison:\n";
    std::cout << "Note: Maximum absolute error may grow with transform magnitude and accumulated\n";
    std::cout << "      floating-point rounding. Relative L2 error provides scale-aware comparison.\n\n";
    std::cout << std::setw(8) << "N"
              << std::setw(18) << "Max Abs Error"
              << std::setw(18) << "Rel L2 Error"
              << std::setw(10) << "Result"
              << "\n";
    std::cout << std::string(54, '-') << "\n";

    const int test_sizes[] = {8, 16, 64, 256, 1024};
    for (int N : test_sizes) {
        float max_error = 0.0f;
        float rel_l2_error = 0.0f;
        bool pass = test_fft_vs_dft(N, max_error, rel_l2_error);

        std::cout << std::setw(8) << N
                  << std::scientific << std::setprecision(6)
                  << std::setw(18) << max_error
                  << std::setw(18) << rel_l2_error
                  << std::setw(10) << (pass ? "PASS" : "FAIL")
                  << "\n";

        if (pass) {
            passed++;
        }
    }

    std::cout << "\n==============================\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed\n";

    return (passed == total) ? 0 : 1;
}
