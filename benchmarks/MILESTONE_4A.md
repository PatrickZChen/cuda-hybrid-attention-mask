# Milestone 4A: row-oriented CUDA kernel

Milestone 4A preserves the dense float32 `[batch, head, query, key]` output
and exact `0.0f`/`-inf` mask semantics while reducing indexing work. The
original flat grid-stride kernel remains compiled into the library for direct
A/B measurement. The row-oriented kernel is the public default because it
passed correctness and memory-safety validation and improved the median in all
nine measured cases.

## Kernel change

The baseline assigns threads to flattened output elements. Every element
reconstructs key, query, head, and batch coordinates with repeated 64-bit
division and modulo.

The optimized kernel launches a 3D grid with
`grid = [seq_len, num_heads, batch_size]`. One 256-thread block owns one
`[batch, head, query]` row, computes the row offset, head mode, causal bound,
and sliding-window bound once, and has its threads stride over contiguous keys.
There is no division or modulo in the per-key loop. Inputs that cannot be
represented by the CUDA 3D grid limits use the preserved baseline kernel, so
the existing public launcher signature and valid-input coverage are retained.

This milestone does not change the mask representation or dtype, fuse a
consumer, pack output, use Tensor Cores, or add PTX.

## Measurement environment

| Item | Value |
| --- | --- |
| Base repository revision | `f8491fcbdb40c269bd459ea02bf97e5f12e8507c` |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, compute capability 8.9 |
| Device memory | 12,282 MiB |
| CUDA-reported memory properties | 192-bit bus, 9001 MHz memory clock |
| CUDA-reported L2 size | 50,331,648 bytes (48 MiB) |
| NVIDIA driver | 555.97 |
| CUDA toolkit / compiler | CUDA 12.5 / nvcc 12.5.82 |
| Build | CMake `Release`, 256 threads per block |
| Measurement date | 2026-08-16 |

`nvidia-smi` showed no other GPU process immediately before the primary run.

## Methodology

- The A/B executable remains named `cuda_baseline_benchmark` for command-line
  compatibility. It links both internal launchers while production callers
  continue to use only `launch_hybrid_attention_mask()`.
- Before timing each case, it launches both implementations into separate
  buffers and compares every float byte-for-byte in chunks. A mismatch fails
  the benchmark.
- Each implementation receives 1,000 untimed warmup launches and 1,000 timed
  launches on the same non-blocking CUDA stream. Launch order alternates for
  every warmup and timed pair to reduce fixed ordering bias.
- Each sample uses CUDA events immediately around one kernel launch. Allocation,
  setup copies, correctness copies, warmups, host synchronization waits, and
  CSV output are outside timed intervals.
- `speedup = baseline_median_us / optimized_median_us`. Values above 1.0 favor
  the row kernel.
- Effective output GiB/s is
  `output_elements × 4 / median_seconds / 2^30`. It is a logical float32 output
  rate, not measured DRAM bandwidth. The 32 MiB medium output fits within the
  reported 48 MiB L2, so its logical rate can exceed nominal DRAM bandwidth.

Reproduction command:

```bash
source .venv/bin/activate
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu --parallel
build-gpu/cuda_baseline_benchmark \
  --case all \
  --warmups 1000 \
  --iterations 1000 \
  --output benchmarks/results/rtx4080_laptop_milestone4a.csv
```

## Primary A/B results

`KV = past_len + seq_len`. Machine-readable distribution statistics are in
[`results/rtx4080_laptop_milestone4a.csv`](results/rtx4080_laptop_milestone4a.csv).

| Case | B×H×Q×KV | Baseline median (µs) | Optimized median (µs) | Speedup | Baseline output GiB/s | Optimized output GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, all-full | 1×8×128×256 | 6.144 | 5.120 | 1.200× | 158.946 | 190.735 |
| Small, all-sliding | 1×8×128×256 | 6.848 | 5.120 | 1.338× | 142.606 | 190.735 |
| Small, alternating | 1×8×128×256 | 6.144 | 5.120 | 1.200× | 158.946 | 190.735 |
| Medium, all-full | 1×16×512×1024 | 74.752 | 57.168 | 1.308× | 418.049 | 546.634 |
| Medium, all-sliding | 1×16×512×1024 | 70.464 | 58.368 | 1.207× | 443.489 | 535.396 |
| Medium, alternating | 1×16×512×1024 | 71.552 | 58.272 | 1.228× | 436.745 | 536.278 |
| Large, all-full | 2×16×1024×2048 | 636.928 | 633.632 | 1.005× | 392.509 | 394.551 |
| Large, all-sliding | 2×16×1024×2048 | 632.624 | 628.736 | 1.006× | 395.179 | 397.623 |
| Large, alternating | 2×16×1024×2048 | 636.768 | 632.832 | 1.006× | 392.608 | 395.050 |

The geometric-mean speedup across all nine primary rows is **1.160×**. By
size, the three-pattern geometric means are 1.244× small, 1.247× medium, and
1.006× large. The alternating-head representative cases improve by 1.200×,
1.228×, and 1.006× for small, medium, and large respectively.

An immediately subsequent independent 1,000/1,000 repeat also had no median
regressions. Its speedups were 1.200× for every small pattern, 1.299–1.432× for
medium patterns, and 1.005–1.006× for large patterns; its nine-case geometric
mean was 1.184×. Medium timings varied more than small and large timings, so
the primary table is retained as measured rather than replaced by the faster
repeat.

## Regressions and default decision

No measured case regressed in either full run. The large-case improvement is
only 0.5–0.6%, which is consistent with the earlier evidence that writing the
256 MiB dense output dominates that regime; the indexing optimization does
not materially reduce those mandatory stores. This marginal result is
reported rather than generalized into a large-shape performance claim.

The row-oriented launcher is the default because:

1. It matches the preserved baseline byte-for-byte for all nine benchmark
   cases.
2. The public path passes all 41 PyTorch-oracle and validation tests.
3. Compute Sanitizer memcheck reports `ERROR SUMMARY: 0 errors` on the
   representative mixed-head case
   `B=1, H=16, Q=128, past=128, window=32`.
4. Both full A/B runs show a positive median speedup for every measured case.

This concludes Milestone 4A; no additional kernel or representation
optimization is included.
