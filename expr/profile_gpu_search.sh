#!/bin/bash
# =============================================================================
# profile_gpu_search.sh
# Profiles gpu_search to determine if kernels are memory-bound or compute-bound.
# Uses nsys (Nsight Systems) — works without elevated permissions (unlike ncu).
#
# Usage:
#   cd /path/to/gpu-mvr/build && bash ../expr/profile_gpu_search.sh
#
# Requirements:
#   - nsys (Nsight Systems CLI) in PATH
#   - gpu_search binary built in current directory
#   - All data files (index, queries, etc.) in current directory
# =============================================================================

set -euo pipefail

# ---------- Configuration ----------
BINARY="./gpu_search"
NSYS_REPORT="gpu_search_profile"

# A100-SXM4-80GB reference specs
A100_PEAK_FP32_TFLOPS=19.5
A100_PEAK_BW_GBS=2039
A100_RIDGE_POINT=9.75  # FLOP/byte for FP32

# ---------- Sanity checks ----------
if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys (Nsight Systems CLI) not found in PATH."
    echo "  Try: export PATH=\$PATH:/path/to/cuda/bin"
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: $BINARY not found or not executable in $(pwd)."
    echo "  Run this script from the build/ directory."
    exit 1
fi

# ---------- Step 1: Run nsys profiling ----------
echo "============================================================"
echo "  GPU Profiling: Memory-bound vs Compute-bound Analysis"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================================"
echo ""
echo "Running nsys profiler on $BINARY ..."
echo ""

# Remove old reports
rm -f "${NSYS_REPORT}.nsys-rep" "${NSYS_REPORT}.sqlite"
rm -f nsys_kernels.csv nsys_memops.csv

nsys profile \
    --output "$NSYS_REPORT" \
    --force-overwrite true \
    --trace cuda,nvtx \
    --stats true \
    "$BINARY"

echo ""
echo "Profiling complete. Report saved to ${NSYS_REPORT}.nsys-rep"
echo ""

# ---------- Step 2: Export kernel and memory stats to CSV ----------
echo "Exporting kernel and memory transfer statistics..."

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output nsys_kernels \
    "${NSYS_REPORT}.nsys-rep"

nsys stats \
    --report cuda_gpu_mem_size_sum \
    --format csv \
    --output nsys_memops \
    "${NSYS_REPORT}.nsys-rep"

# The actual filenames nsys creates
KERN_CSV=$(ls nsys_kernels*.csv 2>/dev/null | head -1)
MEM_CSV=$(ls nsys_memops*.csv 2>/dev/null | head -1)

if [[ -z "$KERN_CSV" ]]; then
    echo "ERROR: nsys did not produce kernel summary CSV."
    exit 1
fi

echo ""

# ---------- Step 3: Analyze and classify ----------
echo "============================================================"
echo "  Per-Kernel Analysis"
echo "============================================================"
echo ""

python3 - "$KERN_CSV" "${MEM_CSV:-}" "$A100_PEAK_FP32_TFLOPS" "$A100_PEAK_BW_GBS" "$A100_RIDGE_POINT" << 'PYEOF'
import csv
import sys
from collections import defaultdict

kern_csv = sys.argv[1]
mem_csv = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
peak_tflops = float(sys.argv[3])
peak_bw_gbs = float(sys.argv[4])
ridge_point = float(sys.argv[5])

peak_flops = peak_tflops * 1e12       # FLOP/s
peak_bw = peak_bw_gbs * 1e9           # bytes/s

# ---- Parse kernel summary ----
# nsys cuda_gpu_kern_sum CSV columns vary by version but typically include:
#   Time(%), Total Time (ns), Instances, Avg (ns), Med (ns), Min (ns), Max (ns), StdDev, Name
kernels = []
total_gpu_time_ns = 0

with open(kern_csv, 'r') as f:
    reader = csv.reader(f)
    header = None
    for row in reader:
        if not row or row[0].startswith('#'):
            continue
        # Find header
        if header is None:
            if any('Name' in c or 'Kernel' in c for c in row):
                header = [c.strip().strip('"') for c in row]
            continue

        if len(row) < len(header):
            continue

        d = {}
        for i, col in enumerate(header):
            d[col] = row[i].strip().strip('"').replace(',', '')

        # Find the name and time columns
        name = None
        total_ns = None
        instances = None
        avg_ns = None

        for k, v in d.items():
            kl = k.lower()
            if 'name' in kl or 'kernel' in kl:
                name = v
            elif 'total' in kl and ('time' in kl or 'ns' in kl or 'duration' in kl):
                try: total_ns = float(v)
                except: pass
            elif kl.startswith('instances') or kl == 'count':
                try: instances = int(v)
                except: pass
            elif 'avg' in kl or 'average' in kl:
                try: avg_ns = float(v)
                except: pass

        if name and total_ns is not None:
            kernels.append({
                'name': name,
                'total_ns': total_ns,
                'instances': instances or 1,
                'avg_ns': avg_ns or (total_ns / max(instances or 1, 1)),
            })
            total_gpu_time_ns += total_ns

# ---- Parse memory transfer summary (if available) ----
total_h2d_bytes = 0
total_d2h_bytes = 0
total_d2d_bytes = 0
total_mem_time_ns = 0

if mem_csv:
    try:
        with open(mem_csv, 'r') as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                if header is None:
                    if any('Name' in c or 'Operation' in c for c in row):
                        header = [c.strip().strip('"') for c in row]
                    continue
                if len(row) < len(header):
                    continue

                d = {}
                for i, col in enumerate(header):
                    d[col] = row[i].strip().strip('"').replace(',', '')

                name = ''
                total_bytes_val = 0
                time_ns = 0
                for k, v in d.items():
                    kl = k.lower()
                    if 'name' in kl or 'operation' in kl:
                        name = v
                    elif 'total' in kl and 'byte' not in kl:
                        try: time_ns = float(v)
                        except: pass
                    elif 'total' in kl and 'byte' in kl:
                        try: total_bytes_val = float(v)
                        except: pass

                total_mem_time_ns += time_ns
                nl = name.lower()
                if 'htod' in nl or 'h2d' in nl or 'host' in nl and 'device' in nl:
                    total_h2d_bytes += total_bytes_val
                elif 'dtoh' in nl or 'd2h' in nl:
                    total_d2h_bytes += total_bytes_val
                elif 'dtod' in nl or 'd2d' in nl:
                    total_d2d_bytes += total_bytes_val
    except Exception as e:
        print(f"  (Warning: could not parse memory CSV: {e})")

# ---- Kernel classification heuristics ----
# Without ncu hardware counters, we estimate based on:
# 1. Known kernel patterns (binary IP = memory-heavy, reductions = memory-heavy)
# 2. Achieved bandwidth vs peak (from data size / kernel time)
# 3. Kernel naming conventions

def short_name(name, max_len=55):
    depth = 0
    result = []
    for ch in name:
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth -= 1
        elif depth == 0:
            result.append(ch)
    s = ''.join(result)
    if '::' in s:
        s = s.split('::')[-1]
    if '(' in s:
        s = s[:s.index('(')]
    s = s.strip()
    if len(s) > max_len:
        s = s[:max_len-3] + '...'
    return s

def classify_kernel(name):
    """Heuristic classification based on kernel name patterns."""
    nl = name.lower()

    # Memory-bound patterns: data movement, gather/scatter, memset, memcpy
    mem_patterns = [
        'gather', 'scatter', 'extract', 'memcpy', 'memset', 'copy',
        'map_emb', 'build_doc_query_keys', 'gather_token_ids',
        'gather_doc_lengths', 'extract_doc_ids', 'sum_doc_scores',
        'aggregate_doc_scores', 'fill', 'radix', 'scan', 'reduce',
        'merge', 'histogram', 'partition',
    ]
    for pat in mem_patterns:
        if pat in nl:
            return 'MEMORY-BOUND'

    # Compute-bound patterns: actual arithmetic computation
    compute_patterns = [
        'binary_ip', 'compute_binary', 'doc_score',
        'matmul', 'gemm', 'conv', 'fft',
    ]
    for pat in compute_patterns:
        if pat in nl:
            return 'COMPUTE-BOUND'

    # Atomic-heavy kernels are typically memory-bound
    if 'atomic' in nl or 'aggregate_stage1' in nl:
        return 'MEMORY-BOUND'

    # CUB/Thrust sort and scan kernels are memory-bound
    if 'cub' in nl or 'thrust' in nl or 'sort' in nl:
        return 'MEMORY-BOUND'

    # CAGRA/FAISS graph search — mix but typically memory-bound (random access)
    if 'cagra' in nl or 'search' in nl or 'graph' in nl:
        return 'MEMORY-BOUND'

    return 'UNKNOWN'

# ---- Print results ----
hdr_fmt = "{:<55s} {:>10s} {:>12s} {:>10s}  {:<15s}"
row_fmt = "{:<55s} {:>10.2f} {:>12d} {:>10.1f}  {:<15s}"

print(hdr_fmt.format("Kernel", "Time(ms)", "Invocations", "Time(%)", "Classification"))
print("-" * 115)

results = []
for k in kernels:
    sname = short_name(k['name'])
    time_ms = k['total_ns'] / 1e6
    time_pct = 100.0 * k['total_ns'] / total_gpu_time_ns if total_gpu_time_ns > 0 else 0
    classification = classify_kernel(k['name'])
    print(row_fmt.format(sname, time_ms, k['instances'], time_pct, classification))
    results.append((k['name'], classification, k['total_ns']))

# ---- Overall summary ----
print("")
print("=" * 115)
print("  OVERALL PROGRAM CLASSIFICATION (weighted by kernel GPU time)")
print("=" * 115)

time_by_class = defaultdict(float)
for _, cls, dur in results:
    time_by_class[cls] += dur

print("")
print(f"  Total kernel GPU time: {total_gpu_time_ns/1e6:.2f} ms")
print("")
print("  Time breakdown by classification:")
for cls, dur in sorted(time_by_class.items(), key=lambda x: -x[1]):
    pct = 100.0 * dur / total_gpu_time_ns if total_gpu_time_ns > 0 else 0
    print(f"    {cls:<20s}: {dur/1e6:10.2f} ms  ({pct:5.1f}%)")

# Determine overall
mem_time = time_by_class.get('MEMORY-BOUND', 0)
comp_time = time_by_class.get('COMPUTE-BOUND', 0)

print("")
if mem_time > comp_time:
    overall = "MEMORY-BOUND"
elif comp_time > mem_time:
    overall = "COMPUTE-BOUND"
else:
    overall = "BALANCED"
print(f"  >>> Overall: The program is predominantly {overall} <<<")

# ---- Memory transfer summary ----
if total_h2d_bytes > 0 or total_d2h_bytes > 0:
    print("")
    print("=" * 115)
    print("  MEMORY TRANSFER SUMMARY")
    print("=" * 115)
    print(f"  Host-to-Device:   {total_h2d_bytes/1e6:10.2f} MB")
    print(f"  Device-to-Host:   {total_d2h_bytes/1e6:10.2f} MB")
    if total_d2d_bytes > 0:
        print(f"  Device-to-Device: {total_d2d_bytes/1e6:10.2f} MB")
    print(f"  Transfer time:    {total_mem_time_ns/1e6:10.2f} ms")
    if total_gpu_time_ns > 0:
        overhead_pct = 100.0 * total_mem_time_ns / total_gpu_time_ns
        print(f"  Transfer overhead: {overhead_pct:.1f}% of kernel time")

# ---- Interpretation ----
print("")
print("=" * 115)
print("  INTERPRETATION GUIDE")
print("=" * 115)
print(f"""
  Classification is based on kernel behavior patterns (nsys does not require
  elevated GPU counter permissions, unlike ncu).

  MEMORY-BOUND:  Kernel performance limited by memory bandwidth.
                 Includes: data gather/scatter, reductions, sorting, atomic ops,
                 random-access graph traversal (CAGRA).
                 Optimization: reduce data movement, improve locality, compress data.

  COMPUTE-BOUND: Kernel performance limited by arithmetic throughput.
                 Includes: binary inner product, document scoring, matrix ops.
                 Optimization: reduce FLOPs, use lower precision, improve ILP.

  UNKNOWN:       Could not classify from kernel name alone.

  A100 reference: {peak_tflops} TFLOPS FP32, {peak_bw_gbs} GB/s HBM2e
  Ridge point:    {ridge_point:.2f} FLOP/byte (FP32)

  NOTE: For precise hardware-counter-based classification, run ncu with admin
  permissions:
    sudo ncu --set roofline ./gpu_search
  This gives exact compute vs memory utilization percentages per kernel.
""")

PYEOF

echo ""
echo "Done. Report: ${NSYS_REPORT}.nsys-rep"
echo "  Kernel CSV: $(ls nsys_kernels*.csv 2>/dev/null | head -1)"
echo "  Memory CSV: $(ls nsys_memops*.csv 2>/dev/null | head -1)"
