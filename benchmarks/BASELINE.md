# RTX 4080 Laptop GPU baseline

This document preserves the pre-optimization Milestone 3B measurements. See
[`MILESTONE_4A.md`](MILESTONE_4A.md) for the measured baseline-versus-row-kernel
comparison.

This is the unoptimized Milestone 3B baseline for
`launch_hybrid_attention_mask()`. The CUDA kernel and public launcher were not
changed for these measurements.

## Measured facts

### Environment

| Item | Value |
| --- | --- |
| Repository revision | `52cad1d54d4ae961d5742481ac77deb616e67ffb` |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9 |
| Device memory | 12,282 MiB |
| CUDA-reported memory properties | 192-bit bus, 9001 MHz memory clock |
| Calculated nominal memory-bandwidth peak | 432.048 GB/s (402.376 GiB/s) |
| CUDA-reported L2 size | 50,331,648 bytes (48 MiB) |
| Power limit during this session | 175 W |
| NVIDIA driver | 555.97 |
| CUDA toolkit / compiler | CUDA 12.5 / nvcc 12.5.82 |
| Host | Ubuntu on WSL2, Linux 6.6.87.2-microsoft-standard-WSL2 |
| Build | CMake 4.4.2, `Release`, GCC 13.3.0 |
| Measurement date | 2026-08-16 |

No other GPU process was shown by `nvidia-smi` immediately before the
measurement work.

The nominal peak was derived from the CUDA device properties as
`2 × 9.001×10^9 clocks/s × (192 bits / 8 bits/byte) = 432.048×10^9
bytes/s`. The factor of two accounts for double-data-rate memory. `GB/s` is
decimal (`10^9` bytes/s), so 432.048 GB/s converts to 402.376 GiB/s after
division by `2^30`. This is a calculated specification-level comparator, not a
measurement of achieved DRAM bandwidth.

### Methodology

- The dedicated executable is `cuda_baseline_benchmark`, built from
  `benchmarks/cuda_baseline_benchmark.cu` and linked to the existing baseline
  library.
- Every row uses 1,000 untimed warmup launches followed by 1,000 timed
  launches on a non-blocking CUDA stream.
- Device allocation, CUDA context/process startup, head-mode host-to-device
  copy, warmups, and CSV output occur outside all timed intervals. The output
  is not copied back during benchmarking.
- Each sample records a CUDA event immediately before calling
  `launch_hybrid_attention_mask()` and another event immediately after it on
  the same stream. `cudaEventElapsedTime()` measures the GPU-stream interval;
  the host synchronization wait is outside that interval.
- `median_us` is the ordinary sample median. `mean_us` is the arithmetic mean,
  and `p95_us` uses the nearest-rank 95th percentile. Derived rates use the
  median.
- Effective output GiB/s is calculated as
  `output_elements × 4 / median_seconds / 2^30`. It is a logical output rate,
  not measured DRAM bandwidth, and it does not account for cache, compression,
  memory transactions, or head-mode reads.
- Within a size, dimensions and sliding-window width stay fixed so the three
  head-mode patterns are directly comparable. Alternating cases begin with a
  full-attention head at head 0.

Reproduction command:

```bash
source .venv/bin/activate
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu --parallel
build-gpu/cuda_baseline_benchmark \
  --case all \
  --warmups 1000 \
  --iterations 1000 \
  --output benchmarks/results/rtx4080_laptop_baseline.csv
```

### Results

`KV` is `past_len + seq_len`. Exact machine-readable values are in
[`results/rtx4080_laptop_baseline.csv`](results/rtx4080_laptop_baseline.csv).

| Case | B×H×Q×KV | Window | Output elements | median_us | mean_us | min_us | p95_us | Elements/s | Effective output GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small, all-full | 1×8×128×256 | 64 | 262,144 | 6.144 | 6.280 | 5.120 | 6.880 | 42.667G | 158.946 |
| small, all-sliding | 1×8×128×256 | 64 | 262,144 | 6.144 | 6.257 | 5.120 | 6.976 | 42.667G | 158.946 |
| small, alternating | 1×8×128×256 | 64 | 262,144 | 6.144 | 6.303 | 5.120 | 7.008 | 42.667G | 158.946 |
| medium, all-full | 1×16×512×1024 | 256 | 8,388,608 | 76.800 | 75.693 | 70.400 | 78.848 | 109.227G | 406.901 |
| medium, all-sliding | 1×16×512×1024 | 256 | 8,388,608 | 73.728 | 73.037 | 69.312 | 75.776 | 113.778G | 423.855 |
| medium, alternating | 1×16×512×1024 | 256 | 8,388,608 | 71.552 | 72.022 | 69.248 | 74.752 | 117.238G | 436.745 |
| large, all-full | 2×16×1024×2048 | 512 | 67,108,864 | 635.904 | 634.648 | 625.664 | 647.168 | 105.533G | 393.141 |
| large, all-sliding | 2×16×1024×2048 | 512 | 67,108,864 | 635.904 | 634.551 | 626.368 | 649.216 | 105.533G | 393.141 |
| large, alternating | 2×16×1024×2048 | 512 | 67,108,864 | 635.904 | 634.409 | 625.664 | 647.168 | 105.533G | 393.141 |

An immediately subsequent independent 1,000/1,000 repeat changed median time
by at most 0.52% for small cases, 4.00% for medium cases, and 0.17% for large
cases. The exact repeat was a stability check and did not replace the primary
CSV.

## Interpretation and hypotheses

- The small cases show a fixed-cost regime: their approximately 6.1 us
  CUDA-event intervals produce much lower effective output rates than the
  medium and large cases despite the same kernel and output format. No control
  designed to isolate launch overhead was measured, so the 6.1 us interval
  must not be reported as measured launch overhead.
- The 256 MiB large output is larger than L2 and reaches 393.141 effective
  output GiB/s, or 97.70% of the 402.376 GiB/s calculated nominal peak. This
  ratio compares a logical output rate with a specification-level peak; it is
  not measured DRAM throughput or utilization. Together with the coarse
  device-level evidence in `PROFILE.md`, it supports the strong inference that
  dense output stores are likely the dominant large-case constraint, not a
  directly measured NCU conclusion.
- The medium output is 32 MiB and fits within the reported 48 MiB L2. Its
  effective output rate can exceed the calculated nominal peak, illustrating
  why this metric must not be read as direct DRAM throughput; caching and/or
  write compression are plausible.
- Head-mode differences are small or inconsistent across the primary and
  repeat runs. There is no timing evidence that full/sliding mode branching is
  the primary baseline bottleneck.

`PROFILE.md` ranks kernel-level Milestone 4 experiments and lists output-format
or consumer changes separately as architectural alternatives.
