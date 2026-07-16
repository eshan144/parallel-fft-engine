"""
benchmark_runner.py — Automated benchmark execution for the Parallel FFT Engine.

Runs benchmark executables, captures stdout, and parses structured timing data
from the output tables. Can be used standalone or imported by plotting.py.

Usage:
    python tools/benchmark_runner.py
    python tools/benchmark_runner.py --compare
    python tools/benchmark_runner.py --strong-scaling
    python tools/benchmark_runner.py --dft
    python tools/benchmark_runner.py --build-dir build/Release
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
#  Executable discovery
# ---------------------------------------------------------------------------

def find_exe(build_dir: Path, name: str) -> Path:
    """Locate a benchmark executable, appending ``.exe`` on Windows."""
    exe = build_dir / name
    if sys.platform == "win32" and exe.suffix != ".exe":
        exe = exe.with_suffix(".exe")
    if not exe.exists():
        raise FileNotFoundError(
            f"Executable not found: {exe}\n"
            f"  Build the project first:  cmake --build build --config Release"
        )
    return exe


# ---------------------------------------------------------------------------
#  Running benchmarks
# ---------------------------------------------------------------------------

def run_benchmark(build_dir: Path, exe_name: str) -> str:
    """Run a benchmark executable and return its full stdout as a string."""
    exe_path = find_exe(build_dir, exe_name)
    print(f"  Running {exe_path.name} ... ", end="", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [str(exe_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr.strip()}")
            return ""
        print(f"done ({elapsed:.1f}s)")
        return result.stdout
    except subprocess.TimeoutExpired:
        print("TIMEOUT (>600s)")
        return ""
    except FileNotFoundError as e:
        print(f"SKIPPED ({e})")
        return ""
    except Exception as e:
        print(f"ERROR ({e})")
        return ""


# ---------------------------------------------------------------------------
#  Core table parser
# ---------------------------------------------------------------------------

def _find_section(lines: List[str], start: int, keywords: List[str]) -> int:
    """Return the first line index (≥ *start*) containing all *keywords*."""
    for i in range(start, len(lines)):
        if all(kw in lines[i] for kw in keywords):
            return i
    return -1


def _parse_numeric_rows(lines: List[str], data_start: int) -> List[List[float]]:
    """
    Read consecutive whitespace-split rows starting at *data_start*
    and convert each token to float.

    Returns a list of float vectors (one per data row).  Stops at the
    first blank line or line that fails conversion.
    """
    rows: List[List[float]] = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line:
            break
        parts = line.split()
        try:
            row = [float(p.replace("%", "").replace(",", "")) for p in parts]
            rows.append(row)
        except ValueError:
            break
    return rows


def _find_dash_index(lines: List[str], header_idx: int) -> int:
    """Find the dash-separator line following *header_idx*."""
    idx = header_idx + 1
    while idx < len(lines) and not lines[idx].strip().startswith("---"):
        idx += 1
    return idx if idx < len(lines) else -1


# ---------------------------------------------------------------------------
#  Parsing benchmark_compare.exe  (DFT + serial FFT + speedup)
# ---------------------------------------------------------------------------

# Benchmark_compare outputs three tables in sequence:
#
# Table 1 — DFT timing:
#   cols: N, Iters, Min (ms), Mean (ms), Median (ms), Max (ms), Xforms/sec
#   data: integer, integer, float ×5
#
# Table 2 — Serial FFT timing:   (same column layout as Table 1)
#
# Table 3 — Algorithmic speedup:
#   cols: N, DFT Mean (ms), FFT Mean (ms), Speedup, Expected (N/logN)
#   data: integer, float ×4
#
# We locate each table by unique header keywords and define the number of
# data columns explicitly to avoid ambiguity from multi-word header labels.

# Column mappings:  position → dict key
_COL_DFT_FFT = {
    0: "n",
    1: "iters",
    2: "min_(ms)",
    3: "mean_(ms)",
    4: "median_(ms)",
    5: "max_(ms)",
    6: "xforms_per_sec",
}

_COL_SPEEDUP = {
    0: "n",
    1: "dft_mean_(ms)",
    2: "fft_mean_(ms)",
    3: "speedup",
    4: "expected_(n/logn)",
}


def _tag_rows(rows: List[List[float]], col_map: Dict[int, str]) -> List[Dict[str, float]]:
    """Convert numeric rows into dicts using the column-position map."""
    result = []
    for r in rows:
        d = {}
        for pos, name in col_map.items():
            if pos < len(r):
                d[name] = r[pos]
        if d:
            result.append(d)
    return result


def parse_compare_output(stdout: str) -> Dict[str, Any]:
    """
    Parse the output of benchmark_compare.exe.

    Returns
    -------
    dict with keys:
        dft_results      List[Dict]  — DFT timing rows
        fft_results      List[Dict]  — Serial FFT timing rows
        speedup_results  List[Dict]  — Algorithmic speedup rows
    """
    if not stdout.strip():
        return {"dft_results": [], "fft_results": [], "speedup_results": []}

    lines = stdout.splitlines(keepends=False)

    # Table 1 — DFT (after line with "Naive" or "DFT")
    result: Dict[str, Any] = {"dft_results": [], "fft_results": [], "speedup_results": []}

    i_dft = _find_section(lines, 0, ["DFT", "Naive"])
    if i_dft >= 0:
        hdr = _find_section(lines, i_dft, ["Xforms/sec"])
        if hdr >= 0:
            dash = _find_dash_index(lines, hdr)
            if dash >= 0:
                rows = _parse_numeric_rows(lines, dash + 1)
                # DFT table has 7 columns
                result["dft_results"] = _tag_rows(rows, _COL_DFT_FFT)

    # Table 2 — Serial FFT (after line with "Serial Radix-2")
    i_fft = _find_section(lines, 0, ["Serial Radix-2"])
    if i_fft >= 0:
        hdr = _find_section(lines, i_fft, ["Xforms/sec"])
        if hdr >= 0:
            dash = _find_dash_index(lines, hdr)
            if dash >= 0:
                rows = _parse_numeric_rows(lines, dash + 1)
                result["fft_results"] = _tag_rows(rows, _COL_DFT_FFT)

    # Table 3 — Algorithmic speedup
    i_spd = _find_section(lines, 0, ["Algorithmic Speedup"])
    if i_spd >= 0:
        hdr = _find_section(lines, i_spd, ["N/logN"])
        if hdr >= 0:
            dash = _find_dash_index(lines, hdr)
            if dash >= 0:
                rows = _parse_numeric_rows(lines, dash + 1)
                result["speedup_results"] = _tag_rows(rows, _COL_SPEEDUP)

    return result


# ---------------------------------------------------------------------------
#  Parsing benchmark_strong_scaling.exe
# ---------------------------------------------------------------------------

# Timing table (per N):
#   cols: N, Threads, Iters, Min (ms), Mean (ms), Median (ms), Max (ms), Xforms/sec
#   data: integer, integer, integer, float ×5

# Speedup / efficiency table (per N):
#   cols: Threads, Mean Time (ms), Speedup (S_p), Efficiency (E_p)
#   data: integer, float, float, float (%)

# Overhead analysis table:
#   cols: N, Serial FFT (ms), OpenMP 1-thread (ms), Overhead
#   data: integer, float, float, float (%)

_COL_TIMING = {
    0: "n",
    1: "threads",
    2: "iters",
    3: "min_(ms)",
    4: "mean_(ms)",
    5: "median_(ms)",
    6: "max_(ms)",
    7: "xforms_per_sec",
}

_COL_SP = {
    0: "threads",
    1: "mean_time_(ms)",
    2: "speedup",
    3: "efficiency_(e_p)",
}

_COL_OVERHEAD = {
    0: "n",
    1: "serial_fft_(ms)",
    2: "openmp_1thread_(ms)",
    3: "overhead",
}


def parse_strong_scaling_output(stdout: str) -> Dict[str, Any]:
    """
    Parse the output of benchmark_strong_scaling.exe.

    Returns
    -------
    dict with keys:
        timing_results   Dict[int, List[Dict]]
             N → list of timing rows
        speedup_results  Dict[int, List[Dict]]
             N → list of speedup/efficiency rows
        overhead_results List[Dict]
             OpenMP overhead comparison rows
        max_threads      int
    """
    if not stdout.strip():
        return {
            "timing_results": {},
            "speedup_results": {},
            "overhead_results": [],
            "max_threads": 0,
        }

    lines = stdout.splitlines(keepends=False)

    # Extract max threads
    max_threads = 0
    for line in lines:
        if "Maximum OpenMP threads available" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    max_threads = int(parts[-1].strip())
                except ValueError:
                    pass
            break

    timing: Dict[int, Any] = {}
    speedup: Dict[int, Any] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Transform size N ="):
            try:
                n_val = int(line.split("=")[-1].strip())
            except ValueError:
                i += 1
                continue

            # --- Timing table (8 columns) ---
            hdr = _find_section(lines, i, ["Xforms/sec"])
            if hdr >= 0:
                dash = _find_dash_index(lines, hdr)
                if dash >= 0:
                    rows = _parse_numeric_rows(lines, dash + 1)
                    timing[n_val] = _tag_rows(rows, _COL_TIMING)

            # --- Speedup/efficiency table (4 columns) ---
            sp_hdr = _find_section(lines, i, ["Speedup", "Efficiency"])
            if sp_hdr >= 0:
                dash = _find_dash_index(lines, sp_hdr)
                if dash >= 0:
                    rows = _parse_numeric_rows(lines, dash + 1)
                    speedup[n_val] = _tag_rows(rows, _COL_SP)

            # Jump past this section
            i += 20
        else:
            i += 1

    # --- Overhead analysis table ---
    overhead: List[Dict] = []
    ov_hdr = _find_section(lines, 0, ["Overhead"])
    if ov_hdr >= 0:
        dash = _find_dash_index(lines, ov_hdr)
        if dash >= 0:
            rows = _parse_numeric_rows(lines, dash + 1)
            overhead = _tag_rows(rows, _COL_OVERHEAD)

    return {
        "timing_results": timing,
        "speedup_results": speedup,
        "overhead_results": overhead,
        "max_threads": max_threads,
    }


# ---------------------------------------------------------------------------
#  Parsing benchmark_dft.exe
# ---------------------------------------------------------------------------

_COL_DFT = {
    0: "n",
    1: "iters",
    2: "min_(ms)",
    3: "mean_(ms)",
    4: "median_(ms)",
    5: "max_(ms)",
    6: "xforms_per_sec",
    7: "n2_per_sec",
}


def parse_dft_output(stdout: str) -> Dict[str, Any]:
    """Parse the output of benchmark_dft.exe."""
    if not stdout.strip():
        return {"dft_results": []}

    lines = stdout.splitlines(keepends=False)
    hdr = _find_section(lines, 0, ["N^2/sec"])
    if hdr < 0:
        return {"dft_results": []}
    dash = _find_dash_index(lines, hdr)
    if dash < 0:
        return {"dft_results": []}
    rows = _parse_numeric_rows(lines, dash + 1)
    return {"dft_results": _tag_rows(rows, _COL_DFT)}


# ---------------------------------------------------------------------------
#  High-level driver
# ---------------------------------------------------------------------------

def run_all(build_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run all three benchmark executables and return parsed results.

    Parameters
    ----------
    build_dir : Path or None
        Directory containing benchmark executables.  Defaults to
        ``./build/Release`` relative to the project root (parent of tools/).

    Returns
    -------
    dict with keys: compare, strong_scaling, dft
    """
    if build_dir is None:
        script_dir = Path(__file__).resolve().parent
        build_dir = script_dir.parent / "build" / "Release"

    build_dir = Path(build_dir).resolve()
    print(f"Benchmark build directory: {build_dir}\n")

    results: Dict[str, Any] = {}

    print("--- Running: benchmark_compare.exe ---")
    stdout = run_benchmark(build_dir, "benchmark_compare")
    results["compare"] = parse_compare_output(stdout)

    print("\n--- Running: benchmark_strong_scaling.exe ---")
    stdout = run_benchmark(build_dir, "benchmark_strong_scaling")
    results["strong_scaling"] = parse_strong_scaling_output(stdout)

    print("\n--- Running: benchmark_dft.exe ---")
    stdout = run_benchmark(build_dir, "benchmark_dft")
    results["dft"] = parse_dft_output(stdout)

    return results


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Parallel FFT Engine benchmarks and parse results."
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Path to build/Release directory (default: ./build/Release)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run only benchmark_compare (DFT vs FFT)",
    )
    parser.add_argument(
        "--strong-scaling",
        action="store_true",
        help="Run only benchmark_strong_scaling",
    )
    parser.add_argument(
        "--dft",
        action="store_true",
        help="Run only benchmark_dft",
    )

    args = parser.parse_args()
    any_specific = any([args.compare, args.strong_scaling, args.dft])

    if args.build_dir:
        build_dir = Path(args.build_dir).resolve()
    else:
        build_dir = Path(__file__).resolve().parent.parent / "build" / "Release"

    results: Dict[str, Any] = {}

    if not any_specific or args.compare:
        print("--- Running: benchmark_compare.exe ---\n")
        stdout = run_benchmark(build_dir, "benchmark_compare")
        results["compare"] = parse_compare_output(stdout)

    if not any_specific or args.strong_scaling:
        print("\n--- Running: benchmark_strong_scaling.exe ---\n")
        stdout = run_benchmark(build_dir, "benchmark_strong_scaling")
        results["strong_scaling"] = parse_strong_scaling_output(stdout)

    if not any_specific or args.dft:
        print("\n--- Running: benchmark_dft.exe ---\n")
        stdout = run_benchmark(build_dir, "benchmark_dft")
        results["dft"] = parse_dft_output(stdout)

    # Summary
    print("\n" + "=" * 60)
    print("Parsed results summary")
    print("=" * 60)

    if "compare" in results:
        dft = results["compare"].get("dft_results", [])
        fft = results["compare"].get("fft_results", [])
        spd = results["compare"].get("speedup_results", [])
        print(f"\n  DFT vs FFT comparison:  {len(dft)} DFT sizes, {len(fft)} FFT sizes")
        if spd:
            last = spd[-1]
            n_val = int(last.get("n", 0))
            sp_val = last.get("speedup", 0)
            print(f"  Algorithmic speedup:    {len(spd)} sizes "
                  f"(best: {sp_val:.0f}× at N={n_val})")

    if "strong_scaling" in results:
        ss = results["strong_scaling"]
        timing = ss.get("timing_results", {})
        sp_vals = ss.get("speedup_results", {})
        print(f"\n  Strong scaling:         {len(timing)} transform sizes")
        for n_val in sorted(timing.keys()):
            t_rows = timing.get(n_val, [])
            sp_rows = sp_vals.get(n_val, [])
            if t_rows:
                best = max(t_rows, key=lambda r: r.get("speedup", 0)) \
                    if "speedup" in t_rows[0] else t_rows[-1]
                sp = best.get("speedup", 0)
                print(f"    N={n_val}: {len(t_rows)} thread configs, "
                      f"speedup={sp:.2f}×")
        overhead = ss.get("overhead_results", [])
        if overhead:
            print(f"  Overhead rows:          {len(overhead)}")

    if "dft" in results:
        dft_rows = results["dft"].get("dft_results", [])
        print(f"\n  DFT standalone:         {len(dft_rows)} sizes")


if __name__ == "__main__":
    main()
