"""
plotting.py — Publication-quality figures for the Parallel FFT Engine report.

Usage:
    python tools/plotting.py                    # run benchmarks, generate all figures
    python tools/plotting.py --no-run           # use cached results only
    python tools/plotting.py --build-dir build/Release

Assumes benchmark_runner.py is in the same directory.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
#  Styling — publication-quality defaults
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (7, 4.5),
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    }
)

# Colorblind-friendly palette (Wong, 2011 / Paul Tol)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
MARKERS = ["o", "s", "D", "^", "v", "<", ">"]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_results(build_dir: Path, no_run: bool = False) -> Dict[str, Any]:
    """
    Load benchmark results — either by running the executables or from cache.
    """
    tools_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools_dir))
    import benchmark_runner  # noqa: E402

    cache_path = RESULTS_DIR / "results_cache.json"

    if no_run:
        if cache_path.exists():
            print(f"Loading cached results from {cache_path}")
            with open(cache_path) as f:
                return json.load(f)
        print(f"No cached results found at {cache_path}. Run without --no-run first.")
        sys.exit(1)

    results = benchmark_runner.run_all(build_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults cached to {cache_path}")
    return results


# ---------------------------------------------------------------------------
#  Figure 1 — Runtime vs FFT Size
# ---------------------------------------------------------------------------

def plot_runtime_vs_size(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot transform time vs N for DFT, serial FFT, and OpenMP FFT (1 thread).
    """
    fig, ax = plt.subplots()

    dft = results.get("compare", {}).get("dft_results", [])
    fft = results.get("compare", {}).get("fft_results", [])
    ss_timing = results.get("strong_scaling", {}).get("timing_results", {})

    # DFT
    if dft:
        ns = [int(r["n"]) for r in dft]
        means = [r["mean_(ms)"] for r in dft]
        ax.loglog(ns, means, "o-", color=COLORS[0], label="DFT (O(N²))")

    # Serial FFT
    if fft:
        ns = [int(r["n"]) for r in fft]
        means = [r["mean_(ms)"] for r in fft]
        ax.loglog(ns, means, "s-", color=COLORS[1], label="FFT serial (O(N log N))")

    # OpenMP FFT 1-thread
    omp_ns: List[int] = []
    omp_means: List[float] = []
    for n_val in sorted(ss_timing.keys()):
        for row in ss_timing[n_val]:
            if abs(row["threads"] - 1) < 0.5:
                omp_ns.append(int(n_val))
                omp_means.append(row["mean_(ms)"])
    if omp_ns:
        pairs = sorted(zip(omp_ns, omp_means))
        omp_ns, omp_means = zip(*pairs)
        ax.loglog(omp_ns, omp_means, "D-", color=COLORS[2],
                  label="FFT OpenMP (1 thread)")

    ax.set_xlabel("Transform size N")
    ax.set_ylabel("Mean execution time (ms)")
    ax.set_title("Runtime vs Transform Size")
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.legend()

    path = output_dir / "runtime_vs_size.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
#  Figure 2 — Algorithmic Speedup
# ---------------------------------------------------------------------------

def plot_algorithmic_speedup(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot measured algorithmic speedup with theoretical N/log₂(N) reference."""
    spd = results.get("compare", {}).get("speedup_results", [])
    if not spd:
        print("  WARNING: No speedup data to plot.")
        return

    fig, ax = plt.subplots()
    ns = [int(r["n"]) for r in spd]
    measured = [r["speedup"] for r in spd]
    expected = [r["expected_(n/logn)"] for r in spd]

    ax.semilogx(ns, measured, "o-", color=COLORS[1], label="Measured speedup")
    ax.semilogx(ns, expected, "s--", color=COLORS[3], alpha=0.7,
                label="Theoretical N / log₂(N)")

    ax.set_xlabel("Transform size N")
    ax.set_ylabel("Speedup (DFT mean / FFT mean)")
    ax.set_title("Algorithmic Speedup: DFT → FFT")
    ax.legend()
    ax.grid(True, which="major", linestyle="--", alpha=0.6)

    path = output_dir / "algorithmic_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
#  Figure 3 — Strong Scaling
# ---------------------------------------------------------------------------

def plot_strong_scaling(results: Dict[str, Any], output_dir: Path) -> None:
    """Mean execution time vs thread count, one curve per N, with ideal lines."""
    timing = results.get("strong_scaling", {}).get("timing_results", {})
    if not timing:
        print("  WARNING: No strong scaling data to plot.")
        return

    fig, ax = plt.subplots()

    for idx, n_val in enumerate(sorted(timing.keys())):
        rows = sorted(timing[n_val], key=lambda r: r["threads"])
        threads = [int(r["threads"]) for r in rows]
        means = [r["mean_(ms)"] for r in rows]
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(threads, means, marker=marker, color=color, label=f"N={n_val}",
                zorder=3)

        # Ideal scaling (based on 1-thread mean)
        ideal = [means[0] / p for p in threads]
        ax.plot(threads, ideal, "--", color=color, alpha=0.3, linewidth=1.2)

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Mean execution time (ms)")
    ax.set_title("Strong Scaling: OpenMP FFT")
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0.5)

    path = output_dir / "strong_scaling.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
#  Figure 4 — Parallel Efficiency
# ---------------------------------------------------------------------------

def plot_parallel_efficiency(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Parallel efficiency vs thread count for each N.

    Efficiency is computed directly from the timing data:
        speedup(p)  = T(1) / T(p)
        efficiency  = speedup(p) / p * 100
    """
    timing = results.get("strong_scaling", {}).get("timing_results", {})
    if not timing:
        print("  WARNING: No timing data to compute efficiency from.")
        return

    fig, ax = plt.subplots()

    for idx, n_val in enumerate(sorted(timing.keys())):
        rows = sorted(timing[n_val], key=lambda r: r["threads"])
        threads = [int(r["threads"]) for r in rows]
        t1 = rows[0]["mean_(ms)"]  # 1-thread baseline
        eff = [(t1 / r["mean_(ms)"] / r["threads"]) * 100 for r in rows]
        ax.plot(threads, eff, marker=MARKERS[idx % len(MARKERS)],
                color=COLORS[idx % len(COLORS)], label=f"N={n_val}")

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4,
               label="Ideal (100%)")
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("Parallel Efficiency vs Thread Count")
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0.5)
    ax.set_ylim(0, 110)

    path = output_dir / "parallel_efficiency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
#  Figure 5 — Overall Speedup
# ---------------------------------------------------------------------------

def plot_overall_speedup(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Side-by-side bar chart comparing DFT, serial FFT, and best parallel
    execution time for each transform size.
    """
    dft = results.get("compare", {}).get("dft_results", [])
    fft = results.get("compare", {}).get("fft_results", [])
    ss_timing = results.get("strong_scaling", {}).get("timing_results", {})

    if not dft or not fft:
        print("  WARNING: Missing DFT or FFT data for overall speedup plot.")
        return

    # Only sizes present in both
    dft_ns = {int(r["n"]) for r in dft}
    fft_ns = {int(r["n"]) for r in fft}
    common = sorted(dft_ns & fft_ns)

    dft_lookup = {int(r["n"]): r["mean_(ms)"] for r in dft}
    fft_lookup = {int(r["n"]): r["mean_(ms)"] for r in fft}

    # Best parallel mean time across thread counts
    par_best: Dict[int, float] = {}
    for n_val_str, rows in ss_timing.items():
        n_val = int(n_val_str) if not isinstance(n_val_str, int) else n_val_str
        if rows:
            par_best[n_val] = min(r["mean_(ms)"] for r in rows)

    fig, ax = plt.subplots()
    x = np.arange(len(common))
    width = 0.25

    dft_vals = [dft_lookup[n] for n in common]
    fft_vals = [fft_lookup[n] for n in common]
    par_vals = [par_best.get(n, float("nan")) for n in common]

    ax.bar(x - width, dft_vals, width, color=COLORS[0], label="DFT", zorder=3)
    ax.bar(x, fft_vals, width, color=COLORS[1], label="Serial FFT", zorder=3)
    ax.bar(x + width, par_vals, width, color=COLORS[2],
           label="Parallel FFT (best)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in common])
    ax.set_yscale("log")
    ax.set_xlabel("Transform size N")
    ax.set_ylabel("Mean execution time (ms)")
    ax.set_title("Overall Speedup: DFT → Serial FFT → Parallel FFT")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    path = output_dir / "overall_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
#  Top-level driver
# ---------------------------------------------------------------------------

def generate_all_figures(results: Dict[str, Any],
                         output_dir: Optional[Path] = None) -> None:
    """Generate all five figures from parsed benchmark *results*."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures ...\n")

    print("  [1/5] Runtime vs Transform Size ...")
    plot_runtime_vs_size(results, output_dir)

    print("  [2/5] Algorithmic Speedup ...")
    plot_algorithmic_speedup(results, output_dir)

    print("  [3/5] Strong Scaling ...")
    plot_strong_scaling(results, output_dir)

    print("  [4/5] Parallel Efficiency ...")
    plot_parallel_efficiency(results, output_dir)

    print("  [5/5] Overall Speedup ...")
    plot_overall_speedup(results, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the Parallel FFT Engine."
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Path to build/Release directory (default: ./build/Release)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip benchmark execution; use cached results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(FIGURES_DIR),
        help=f"Output directory for figures (default: {FIGURES_DIR})",
    )

    args = parser.parse_args()

    if args.build_dir:
        build_dir = Path(args.build_dir).resolve()
    else:
        build_dir = Path(__file__).resolve().parent.parent / "build" / "Release"

    output_dir = Path(args.output_dir).resolve()

    # Remove outdated cache so fresh benchmarks run
    cache_path = RESULTS_DIR / "results_cache.json"
    if not args.no_run and cache_path.exists():
        cache_path.unlink()

    results = load_results(build_dir, no_run=args.no_run)
    generate_all_figures(results, output_dir)


if __name__ == "__main__":
    main()
