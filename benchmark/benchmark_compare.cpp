#include "dft.h"
#include "fft.h"
#include "signal_generator.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

struct BenchmarkResult {
    int N;
    int iterations;
    double min_time_ms;
    double mean_time_ms;
    double median_time_ms;
    double max_time_ms;
    double transforms_per_sec;
};

BenchmarkResult benchmark_dft(int N, int iterations) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);
    std::vector<std::complex<float>> output;

    dft_naive(input, output);

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        Timer timer;
        dft_naive(input, output);
        times.push_back(timer.elapsed_ms());
    }

    std::sort(times.begin(), times.end());

    double min_time = times.front();
    double max_time = times.back();
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean_time = sum / times.size();

    double median_time;
    if (times.size() % 2 == 0) {
        median_time = (times[times.size()/2 - 1] + times[times.size()/2]) / 2.0;
    } else {
        median_time = times[times.size()/2];
    }

    double transforms_per_sec = 1000.0 / min_time;

    BenchmarkResult result;
    result.N = N;
    result.iterations = iterations;
    result.min_time_ms = min_time;
    result.mean_time_ms = mean_time;
    result.median_time_ms = median_time;
    result.max_time_ms = max_time;
    result.transforms_per_sec = transforms_per_sec;

    return result;
}

BenchmarkResult benchmark_fft(int N, int iterations) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);
    std::vector<std::complex<float>> output;

    fft_radix2(input, output);

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        Timer timer;
        fft_radix2(input, output);
        times.push_back(timer.elapsed_ms());
    }

    std::sort(times.begin(), times.end());

    double min_time = times.front();
    double max_time = times.back();
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean_time = sum / times.size();

    double median_time;
    if (times.size() % 2 == 0) {
        median_time = (times[times.size()/2 - 1] + times[times.size()/2]) / 2.0;
    } else {
        median_time = times[times.size()/2];
    }

    double transforms_per_sec = 1000.0 / min_time;

    BenchmarkResult result;
    result.N = N;
    result.iterations = iterations;
    result.min_time_ms = min_time;
    result.mean_time_ms = mean_time;
    result.median_time_ms = median_time;
    result.max_time_ms = max_time;
    result.transforms_per_sec = transforms_per_sec;

    return result;
}

void print_header() {
    std::cout << std::setw(8) << "N"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "Median (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(15) << "Xforms/sec"
              << "\n";
    std::cout << std::string(81, '-') << "\n";
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(3)
              << std::setw(8) << r.N
              << std::setw(10) << r.iterations
              << std::setw(12) << r.min_time_ms
              << std::setw(12) << r.mean_time_ms
              << std::setw(12) << r.median_time_ms
              << std::setw(12) << r.max_time_ms
              << std::setprecision(2)
              << std::setw(15) << r.transforms_per_sec
              << "\n";
}

int main() {
    std::cout << "DFT vs FFT Algorithmic Speedup Benchmark\n";
    std::cout << "=========================================\n\n";

    const std::vector<int> sizes = {64, 256, 1024, 4096, 8192, 16384};

    std::cout << "DFT (Naive O(N^2) Implementation):\n";
    print_header();

    std::vector<BenchmarkResult> dft_results;
    for (int N : sizes) {
        int iterations = (N <= 1024) ? 5 : (N <= 4096) ? 3 : 2;
        auto result = benchmark_dft(N, iterations);
        dft_results.push_back(result);
        print_result(result);
    }

    std::cout << "\n\nFFT (Serial Radix-2 O(N log N) Implementation):\n";
    print_header();

    std::vector<BenchmarkResult> fft_results;
    for (int N : sizes) {
        int iterations = 10;
        auto result = benchmark_fft(N, iterations);
        fft_results.push_back(result);
        print_result(result);
    }

    std::cout << "\n\nAlgorithmic Speedup Analysis:\n";
    std::cout << "=========================================\n";
    std::cout << std::setw(8) << "N"
              << std::setw(15) << "DFT Mean (ms)"
              << std::setw(15) << "FFT Mean (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(18) << "Expected (N/logN)"
              << "\n";
    std::cout << std::string(71, '-') << "\n";

    for (size_t i = 0; i < sizes.size(); i++) {
        const int N = sizes[i];
        const double dft_mean = dft_results[i].mean_time_ms;
        const double fft_mean = fft_results[i].mean_time_ms;
        const double speedup = dft_mean / fft_mean;
        const double log2_N = std::log2(static_cast<double>(N));
        const double expected_speedup = N / log2_N;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << N
                  << std::setw(15) << dft_mean
                  << std::setw(15) << fft_mean
                  << std::setw(15) << speedup
                  << std::setw(18) << expected_speedup
                  << "\n";
    }

    std::cout << "\n\nComplexity Validation:\n";
    std::cout << "=========================================\n";
    std::cout << "DFT runtime scaling (when N doubles):\n";
    for (size_t i = 1; i < dft_results.size(); i++) {
        const double ratio_N = static_cast<double>(dft_results[i].N) / dft_results[i-1].N;
        const double ratio_time = dft_results[i].mean_time_ms / dft_results[i-1].mean_time_ms;
        const double expected_ratio = ratio_N * ratio_N;
        std::cout << "  N=" << dft_results[i-1].N << " to N=" << dft_results[i].N
                  << ": time ratio = " << std::fixed << std::setprecision(2) << ratio_time
                  << " (expected ~" << expected_ratio << " for O(N^2))\n";
    }

    std::cout << "\nFFT runtime scaling (when N doubles):\n";
    for (size_t i = 1; i < fft_results.size(); i++) {
        const double ratio_N = static_cast<double>(fft_results[i].N) / fft_results[i-1].N;
        const double ratio_time = fft_results[i].mean_time_ms / fft_results[i-1].mean_time_ms;
        const double log_ratio = (std::log2(fft_results[i].N)) / (std::log2(fft_results[i-1].N));
        const double expected_ratio = ratio_N * log_ratio;
        std::cout << "  N=" << fft_results[i-1].N << " to N=" << fft_results[i].N
                  << ": time ratio = " << std::fixed << std::setprecision(2) << ratio_time
                  << " (expected ~" << expected_ratio << " for O(N log N))\n";
    }

    std::cout << "\n=========================================\n";
    std::cout << "Key Observations:\n";
    std::cout << "- DFT complexity: O(N^2) - N output bins × N input samples per bin\n";
    std::cout << "- FFT complexity: O(N log N) - log2(N) stages × N/2 butterflies per stage\n";
    std::cout << "- Algorithmic speedup grows with N: larger transforms benefit more from FFT\n";
    std::cout << "- Measured speedup approaches theoretical N/log(N) for large N\n";
    std::cout << "- Timing measurements validate but do not prove asymptotic complexity\n";
    std::cout << "- Loop structure and operation count establish theoretical complexity\n";

    return 0;
}
