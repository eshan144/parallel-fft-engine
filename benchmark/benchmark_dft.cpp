#include "dft.h"
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
    double n_squared_work_per_sec;
};

BenchmarkResult benchmark_size(int N, int min_iterations = 3) {
    std::vector<SinusoidComponent> components = {
        {static_cast<float>(N) / 16.0f, 1.0f, 0.0f},
        {static_cast<float>(N) / 8.0f, 0.7f, 1.0f},
        {static_cast<float>(N) / 32.0f, 0.5f, 0.5f}
    };

    auto input = generate_signal(N, components, 42);
    std::vector<std::complex<float>> output;

    dft_naive(input, output);

    std::vector<double> times;
    times.reserve(min_iterations);

    for (int i = 0; i < min_iterations; i++) {
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

    double n_squared = static_cast<double>(N) * static_cast<double>(N);
    double n_squared_per_sec = n_squared * transforms_per_sec;

    BenchmarkResult result;
    result.N = N;
    result.iterations = min_iterations;
    result.min_time_ms = min_time;
    result.mean_time_ms = mean_time;
    result.median_time_ms = median_time;
    result.max_time_ms = max_time;
    result.transforms_per_sec = transforms_per_sec;
    result.n_squared_work_per_sec = n_squared_per_sec;

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
              << std::setw(15) << "N^2/sec"
              << "\n";
    std::cout << std::string(96, '-') << "\n";
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
              << std::scientific
              << std::setw(15) << r.n_squared_work_per_sec
              << "\n";
}

int main() {
    std::cout << "DFT Naive Implementation Benchmark\n";
    std::cout << "===================================\n\n";

    std::vector<int> sizes = {64, 256, 1024, 4096, 8192};

    print_header();

    std::vector<BenchmarkResult> results;

    for (int N : sizes) {
        auto result = benchmark_size(N, 5);
        results.push_back(result);
        print_result(result);

        if (N == 8192 && result.min_time_ms > 5000.0) {
            std::cout << "\nNote: N=8192 took > 5 seconds. Skipping N=16384.\n";
            break;
        }
    }

    if (!results.empty() && results.back().N == 8192 && results.back().min_time_ms <= 5000.0) {
        std::cout << "\nN=8192 completed in reasonable time. Running N=16384...\n";
        auto result = benchmark_size(16384, 3);
        results.push_back(result);
        print_result(result);
    }

    std::cout << "\n===================================\n";
    std::cout << "Benchmark complete.\n";
    std::cout << "\nNote: Time complexity is O(N^2) due to nested loops.\n";
    std::cout << "Each output bin requires N complex multiply-adds.\n";
    std::cout << "Each inner loop iteration computes sin/cos fresh (no twiddle reuse).\n";
    std::cout << "\nCache behavior:\n";
    std::cout << "- Input array exhibits temporal reuse across outer loop iterations\n";
    std::cout << "- For small N (< ~4K samples, ~32 KiB), data may remain in L1/L2\n";
    std::cout << "- For large N, actual cache residency depends on cache hierarchy\n";
    std::cout << "- Must use profiling tools (perf, VTune) to measure cache miss rates\n";

    return 0;
}
