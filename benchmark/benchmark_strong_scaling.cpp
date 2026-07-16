#include "dft.h"
#include "fft.h"
#include "signal_generator.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

struct BenchmarkResult {
    int N;
    int threads;
    int iterations;
    double min_time_ms;
    double mean_time_ms;
    double median_time_ms;
    double max_time_ms;
    double transforms_per_sec;
};

BenchmarkResult benchmark_openmp_fft(int N, int num_threads, int iterations) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);
    std::vector<std::complex<float>> output;

    omp_set_num_threads(num_threads);

    fft_radix2_openmp(input, output);

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        Timer timer;
        fft_radix2_openmp(input, output);
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
    result.threads = num_threads;
    result.iterations = iterations;
    result.min_time_ms = min_time;
    result.mean_time_ms = mean_time;
    result.median_time_ms = median_time;
    result.max_time_ms = max_time;
    result.transforms_per_sec = transforms_per_sec;

    return result;
}

double benchmark_serial_fft(int N, int iterations) {
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

    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    return sum / times.size();
}

void print_header() {
    std::cout << std::setw(8) << "N"
              << std::setw(10) << "Threads"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "Median (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(15) << "Xforms/sec"
              << "\n";
    std::cout << std::string(91, '-') << "\n";
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(3)
              << std::setw(8) << r.N
              << std::setw(10) << r.threads
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
    std::cout << "OpenMP FFT Strong Scaling Benchmark\n";
    std::cout << "====================================\n\n";

    const int max_threads = omp_get_max_threads();
    std::cout << "Maximum OpenMP threads available: " << max_threads << "\n";
    std::cout << "Strong scaling: fixed N, increasing thread count\n\n";

    std::vector<int> thread_counts = {1, 2, 4, 8};
    if (max_threads > 8 && max_threads != 8) {
        thread_counts.push_back(max_threads);
    }

    std::vector<int> test_sizes = {4096, 8192, 16384};

    for (int N : test_sizes) {
        std::cout << "\n========================================\n";
        std::cout << "Transform size N = " << N << "\n";
        std::cout << "========================================\n\n";

        print_header();

        std::vector<BenchmarkResult> results;
        for (int threads : thread_counts) {
            if (threads > max_threads) continue;

            int iterations = 10;
            auto result = benchmark_openmp_fft(N, threads, iterations);
            results.push_back(result);
            print_result(result);
        }

        std::cout << "\nParallel Speedup and Efficiency for N=" << N << ":\n";
        std::cout << std::string(60, '-') << "\n";
        std::cout << std::setw(10) << "Threads"
                  << std::setw(15) << "Mean Time (ms)"
                  << std::setw(15) << "Speedup (S_p)"
                  << std::setw(20) << "Efficiency (E_p)"
                  << "\n";
        std::cout << std::string(60, '-') << "\n";

        const double baseline_time = results[0].mean_time_ms;
        for (const auto& r : results) {
            double speedup = baseline_time / r.mean_time_ms;
            double efficiency = speedup / r.threads;

            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(10) << r.threads
                      << std::setw(15) << r.mean_time_ms
                      << std::setw(15) << speedup
                      << std::setw(19) << (efficiency * 100.0) << "%"
                      << "\n";
        }
    }

    std::cout << "\n\n========================================\n";
    std::cout << "OpenMP Overhead Analysis\n";
    std::cout << "========================================\n\n";
    std::cout << "Comparing Serial FFT (fft_radix2) vs OpenMP FFT with 1 thread:\n\n";
    std::cout << std::setw(8) << "N"
              << std::setw(20) << "Serial FFT (ms)"
              << std::setw(20) << "OpenMP 1-thread (ms)"
              << std::setw(15) << "Overhead"
              << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (int N : test_sizes) {
        double serial_time = benchmark_serial_fft(N, 10);

        omp_set_num_threads(1);
        std::vector<SinusoidComponent> components = {
            {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
            {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
            {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
        };
        auto input = generate_signal(N, components, 42);
        std::vector<std::complex<float>> output;

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            Timer timer;
            fft_radix2_openmp(input, output);
            times.push_back(timer.elapsed_ms());
        }
        std::sort(times.begin(), times.end());
        double sum = 0.0;
        for (double t : times) sum += t;
        double openmp_1thread_time = sum / times.size();

        double overhead_pct = ((openmp_1thread_time - serial_time) / serial_time) * 100.0;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << N
                  << std::setw(20) << serial_time
                  << std::setw(20) << openmp_1thread_time
                  << std::setw(14) << overhead_pct << "%"
                  << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Scaling Analysis:\n";
    std::cout << "- Parallel speedup S_p = T_1 / T_p (baseline: OpenMP FFT with 1 thread)\n";
    std::cout << "- Parallel efficiency E_p = S_p / p\n";
    std::cout << "- Ideal linear scaling: S_p = p, E_p = 100%\n";
    std::cout << "- OpenMP overhead measured by comparing serial FFT vs OpenMP 1-thread\n";
    std::cout << "- Efficiency degradation may result from a combination of:\n";
    std::cout << "    - OpenMP runtime and barrier synchronization overhead\n";
    std::cout << "    - Memory-system effects (bandwidth, cache contention)\n";
    std::cout << "    - Limited work per thread at smaller problem sizes\n";
    std::cout << "    - Hardware resource contention (shared caches, memory channels)\n";
    std::cout << "  Additional profiling would be required to identify the dominant cause(s)\n";
    std::cout << "- Amdahl's law applies: serial fraction limits maximum speedup\n";

    return 0;
}
