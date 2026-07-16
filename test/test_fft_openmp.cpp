#include "dft.h"
#include "fft.h"
#include "signal_generator.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const float EPSILON = 1e-4f;

struct ErrorMetrics {
    float max_abs_error;
    float rel_l2_error;
};

ErrorMetrics compare_transforms(const std::vector<std::complex<float>>& reference,
                                 const std::vector<std::complex<float>>& test) {
    ErrorMetrics metrics;
    metrics.max_abs_error = 0.0f;

    float error_norm_squared = 0.0f;
    float ref_norm_squared = 0.0f;

    const int N = static_cast<int>(reference.size());
    for (int k = 0; k < N; k++) {
        const float error = std::abs(reference[k] - test[k]);
        metrics.max_abs_error = std::max(metrics.max_abs_error, error);

        error_norm_squared += error * error;
        ref_norm_squared += std::norm(reference[k]);
    }

    const float error_l2_norm = std::sqrt(error_norm_squared);
    const float ref_l2_norm = std::sqrt(ref_norm_squared);
    metrics.rel_l2_error = error_l2_norm / ref_l2_norm;

    return metrics;
}

bool test_openmp_vs_serial_fft(int N, int num_threads, float& max_abs, float& rel_l2) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);

    std::vector<std::complex<float>> serial_output, openmp_output;

    fft_radix2(input, serial_output);

    omp_set_num_threads(num_threads);
    fft_radix2_openmp(input, openmp_output);

    ErrorMetrics metrics = compare_transforms(serial_output, openmp_output);
    max_abs = metrics.max_abs_error;
    rel_l2 = metrics.rel_l2_error;

    const float abs_tolerance = 0.1f;
    const float rel_tolerance = 1e-4f;

    return (max_abs <= abs_tolerance && rel_l2 <= rel_tolerance);
}

bool test_openmp_vs_dft(int N, int num_threads, float& max_abs, float& rel_l2) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);

    std::vector<std::complex<float>> dft_output, openmp_output;

    dft_naive(input, dft_output);

    omp_set_num_threads(num_threads);
    fft_radix2_openmp(input, openmp_output);

    ErrorMetrics metrics = compare_transforms(dft_output, openmp_output);
    max_abs = metrics.max_abs_error;
    rel_l2 = metrics.rel_l2_error;

    const float abs_tolerance = 0.1f;
    const float rel_tolerance = 1e-4f;

    return (max_abs <= abs_tolerance && rel_l2 <= rel_tolerance);
}

int main() {
    std::cout << "OpenMP FFT Correctness Tests\n";
    std::cout << "=============================\n\n";

    const int max_threads = omp_get_max_threads();
    std::cout << "Maximum OpenMP threads available: " << max_threads << "\n\n";

    std::vector<int> thread_counts = {1, 2, 4, 8};
    if (max_threads > 8 && max_threads != 8) {
        thread_counts.push_back(max_threads);
    }

    std::vector<int> test_sizes = {8, 16, 64, 256, 1024, 4096};

    int total_tests = 0;
    int passed_tests = 0;

    std::cout << "OpenMP FFT vs Serial FFT Comparison:\n";
    std::cout << "-------------------------------------\n";
    std::cout << std::setw(8) << "N"
              << std::setw(10) << "Threads"
              << std::setw(18) << "Max Abs Error"
              << std::setw(18) << "Rel L2 Error"
              << std::setw(10) << "Result"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (int N : test_sizes) {
        for (int threads : thread_counts) {
            if (threads > max_threads) continue;

            float max_abs = 0.0f;
            float rel_l2 = 0.0f;
            bool pass = test_openmp_vs_serial_fft(N, threads, max_abs, rel_l2);

            std::cout << std::setw(8) << N
                      << std::setw(10) << threads
                      << std::scientific << std::setprecision(6)
                      << std::setw(18) << max_abs
                      << std::setw(18) << rel_l2
                      << std::setw(10) << (pass ? "PASS" : "FAIL")
                      << "\n";

            total_tests++;
            if (pass) passed_tests++;
        }
    }

    std::cout << "\n\nOpenMP FFT vs DFT Comparison (selected sizes):\n";
    std::cout << "----------------------------------------------\n";
    std::cout << std::setw(8) << "N"
              << std::setw(10) << "Threads"
              << std::setw(18) << "Max Abs Error"
              << std::setw(18) << "Rel L2 Error"
              << std::setw(10) << "Result"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    std::vector<int> dft_test_sizes = {8, 16, 64, 256, 1024};
    for (int N : dft_test_sizes) {
        for (int threads : thread_counts) {
            if (threads > max_threads) continue;

            float max_abs = 0.0f;
            float rel_l2 = 0.0f;
            bool pass = test_openmp_vs_dft(N, threads, max_abs, rel_l2);

            std::cout << std::setw(8) << N
                      << std::setw(10) << threads
                      << std::scientific << std::setprecision(6)
                      << std::setw(18) << max_abs
                      << std::setw(18) << rel_l2
                      << std::setw(10) << (pass ? "PASS" : "FAIL")
                      << "\n";

            total_tests++;
            if (pass) passed_tests++;
        }
    }

    std::cout << "\n=============================\n";
    std::cout << "Results: " << passed_tests << "/" << total_tests << " tests passed\n";

    std::cout << "\nNote: OpenMP FFT performs identical arithmetic to serial FFT.\n";
    std::cout << "Minor floating-point differences may occur due to non-associativity\n";
    std::cout << "and thread scheduling, but relative L2 error remains within tolerance.\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
