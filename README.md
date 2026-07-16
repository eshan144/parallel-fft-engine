# High-Performance Parallel FFT Engine

A pedagogical HPC and performance-engineering project implementing a radix-2 Cooley-Tukey FFT with progressive optimization stages. Each stage builds measurably on the previous, establishing clear performance baselines for comparison.

**Project status:** Complete ✓

**Platform:** Windows · MSVC 2022 · OpenMP · Intel Core i7-9700 (8 cores) · 16 GB RAM

---

## Motivation

The Fast Fourier Transform (FFT) is one of the most widely used algorithms in engineering and scientific computing — signal processing, image analysis, communications, medical imaging, and numerical simulation all depend on efficient spectral analysis.

This project studies both *algorithmic* and *parallel* optimization of the radix-2 Cooley-Tukey FFT. The objective was to build a complete, reproducible performance engineering pipeline: implement, test, benchmark, analyse, and present — in a way that demonstrates systematic methodology for technical interviews and portfolio review.

---

## Architecture

```
                    ┌─────────────────┐
                    │ Signal Generator │
                    └────────┬────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Input buffer │
                      └──────┬───────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │ DFT      │  │ FFT      │  │ FFT (OpenMP) │
        │ O(N²)    │  │ O(N log) │  │ O(N log/p)   │
        └────┬─────┘  └────┬─────┘  └──────┬───────┘
             │             │               │
             ▼             ▼               ▼
           Tests        Benchmarks      Correctness
```

---

## Project Structure

```
├── CMakeLists.txt
├── README.md                               ← This file (primary documentation)
├── docs/
│   └── engineering_analysis.md             ← Detailed engineering analysis
├── include/
│   ├── dft.h                               ← Naive O(N²) DFT
│   ├── fft.h                               ← Serial + OpenMP FFT
│   ├── signal_generator.h                  ← Synthetic signal generation
│   └── timer.h                             ← High-resolution stopwatch
├── src/
│   ├── dft.cpp                             ← DFT implementation
│   ├── fft.cpp                             ← Serial radix-2 FFT
│   ├── fft_openmp.cpp                      ← OpenMP parallel FFT
│   └── signal_generator.cpp
├── test/
│   ├── test_dft.cpp                        ← DFT correctness (6 tests)
│   ├── test_fft.cpp                        ← Serial FFT correctness (12 tests)
│   └── test_fft_openmp.cpp                 ← OpenMP correctness (44 tests)
├── benchmark/
│   ├── benchmark_compare.cpp               ← DFT vs FFT speedup comparison
│   ├── benchmark_dft.cpp                   ← Standalone DFT benchmark
│   └── benchmark_strong_scaling.cpp        ← OpenMP strong-scaling analysis
├── tools/
│   ├── benchmark_runner.py                 ← Automated benchmark execution & parsing
│   ├── plotting.py                         ← Publication-quality figure generation
│   └── requirements.txt                    ← Python dependencies
└── results/
    └── figures/                            ← Generated PNG figures
```

---

## Background (Brief)

**Discrete Fourier Transform (DFT).** Maps N complex time-domain samples to N frequency-domain coefficients. A direct evaluation is O(N²) — each of N output bins requires N multiply-adds.

**Cooley-Tukey Radix-2 FFT.** For N a power of two, the transform decomposes into log₂(N) stages of N/2 independent *butterfly* operations. Complexity drops to O(N log N) — at N=16,384, this reduces ~270 million operations to ~230,000.

**Parallel decomposition.** Butterflies within a single stage are data-independent (each operates on a distinct pair of elements), making them embarrassingly parallel. Stages are sequential, requiring synchronization between them.

---

## Build Instructions

### Prerequisites

- CMake ≥ 3.15
- C++17 compiler with OpenMP support
  - Windows: Visual Studio 2022 with OpenMP
  - Linux: GCC or Clang with `-fopenmp`

### Configure and Build (Release)

```bash
# Windows (x64)
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Running Tests

All 62 correctness tests must pass.

```bash
# DFT correctness — 6 tests
./build/Release/test_dft

# FFT correctness (serial) — 12 tests
./build/Release/test_fft

# OpenMP FFT correctness — 44 tests
./build/Release/test_fft_openmp
```

---

## Running Benchmarks

```bash
# Algorithmic speedup: DFT vs serial FFT
./build/Release/benchmark_compare

# OpenMP strong-scaling analysis
./build/Release/benchmark_strong_scaling

# Standalone DFT benchmark
./build/Release/benchmark_dft
```

---

## Running the Python Plotting Scripts

```bash
# Install dependencies
pip install -r tools/requirements.txt

# Run benchmarks and generate all figures
python tools/plotting.py

# Re-plot from cached results (skip re-running benchmarks)
python tools/plotting.py --no-run
```

Figures are written to `results/figures/` as 300 DPI PNG.

---

## Benchmark Highlights

| Metric | Result |
|---|---|
| Algorithmic speedup (DFT → FFT, N=16,384) | ~7,000× |
| Best parallel speedup (OpenMP, 8 threads) | ~2.1× |
| OpenMP 1-thread overhead vs serial FFT | 37–48% |
| Strong-scaling efficiency (2 threads) | ~71% |
| Strong-scaling efficiency (8 threads) | ~27% |
| Correctness tests passing | 62 / 62 |

### Runtime vs Transform Size

![Runtime vs Size](results/figures/runtime_vs_size.png)

The DFT follows a clear O(N²) slope (4× time when N doubles), while both FFT variants follow O(N log N) (2× time when N doubles). At N=16,384, the serial FFT completes in ~0.54 ms — over 7,000× faster than the DFT's 3.8 s.

### Algorithmic Speedup

![Algorithmic Speedup](results/figures/algorithmic_speedup.png)

The measured speedup exceeds the N/log₂(N) ratio because the DFT evaluates 2N² transcendental functions per transform, while the FFT evaluates N log₂(N) twiddle factors once and reuses them.

### Overall Speedup

![Overall Speedup](results/figures/overall_speedup.png)

Algorithmic optimization (O(N²) → O(N log N)) produces orders of magnitude more speedup than parallelization alone — a core lesson in performance engineering.

### Strong Scaling

![Strong Scaling](results/figures/strong_scaling.png)

Dashed lines show ideal ×p scaling. The OpenMP FFT tracks closest to ideal at larger problem sizes, where there is more work per thread to amortize synchronization costs.

### Parallel Efficiency

![Parallel Efficiency](results/figures/parallel_efficiency.png)

Efficiency degrades from ~100% at 1 thread to ~22–27% at 8 threads. This is consistent with a combination of barrier synchronization overhead, OpenMP runtime costs, and limited work per thread.

---

## Key Engineering Decisions

1. **Persistent parallel region.** A single `#pragma omp parallel` block encloses all FFT stages rather than creating a new region per stage, reducing thread-launch overhead from log₂(N) times to once.

2. **Static scheduling.** `schedule(static)` provides deterministic workload distribution with negligible scheduling overhead and good spatial locality.

3. **Serial twiddle precomputation.** Twiddle factors are computed by a single thread (`#pragma omp single`), avoiding redundant computation at the cost of a brief synchronization point.

4. **Kernel-only timing.** The timer measures only the transform itself, excluding signal generation and output processing, using `std::chrono::steady_clock` with microsecond resolution.

5. **Statistical aggregation.** Each configuration runs 10 times; min, mean, median, and max are reported. The minimum approximates best-case execution without OS interference.

---

## Lessons Learned

| Area | Key Takeaway |
|------|-------------|
| Algorithmic optimization | Changing complexity class (O(N²) → O(N log N)) yields ~7,000× speedup — far more than any constant-factor or parallel improvement |
| Parallel overhead | OpenMP parallel regions are not free: 37–48% overhead at 1 thread vs serial |
| Barrier cost | Implicit barriers between FFT stages limit scaling; each thread finishes its assigned butterflies at a different time |
| Benchmarking methodology | Warm-up iterations, multiple timing metrics, and kernel-only timing are essential for reliable measurements |
| Numerical correctness | The OpenMP variant produces bit-identical results to serial FFT — verified across 44 test cases |

---

## Conclusions

1. **Algorithmic optimization dominates.** The FFT's O(N log N) complexity outperforms the DFT's O(N²) by orders of magnitude. This is always the first optimization to pursue.

2. **Parallel speedup saturates.** Amdahl's Law is visible in the data: synchronization, OpenMP overhead, and limited per-thread work bound the achievable speedup.

3. **Measurement matters.** Statistical timing (min/mean/median), warm-up, and kernel-only measurement produce trustworthy benchmarks.

4. **This is repeatable.** The Python automation pipeline runs all benchmarks and generates all figures from scratch with a single command.

---

## Detailed Analysis

For the full engineering discussion — including Amdahl's Law, profiling methodology, cache/locality analysis, and benchmark tables — see the project report.

---

*Intel, Core i7, and VTune are trademarks of Intel Corporation. Microsoft, MSVC, and Visual Studio are trademarks of Microsoft Corporation.*
