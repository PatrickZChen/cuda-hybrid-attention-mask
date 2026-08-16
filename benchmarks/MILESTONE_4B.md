# Milestone 4B: uniform-head fast-path evaluation

Milestone 4B evaluated dedicated kernels for all-full and all-sliding head
patterns against the existing row-oriented hybrid kernel. The specialization
was **rejected**. No candidate kernel or dispatch code is retained; the public
API, dense float32 output, and exact `0.0f`/`-inf` semantics remain unchanged.

## Candidate and method

The all-full candidate removed the per-row head-mode load and sliding-window
predicate. The all-sliding candidate removed the per-row head-mode load and
full-attention alternative. Both retained the row grid
`[seq_len, num_heads, batch_size]`, 256 threads per block, contiguous key
writes, and the same output values as the hybrid row kernel.

For each all-full and all-sliding case, the benchmark first compared the row
and candidate output buffers byte-for-byte. It then ran 1,000 untimed warmups
and 1,000 CUDA-event samples per implementation on the same non-blocking
stream, alternating launch order. Allocation, copies, comparison, warmups,
host waits, and CSV output were outside the timed intervals. `Speedup` is
`row_median_us / specialized_median_us`; values above 1 favor specialization.

Measurements were made on the NVIDIA GeForce RTX 4080 Laptop GPU used for
Milestone 4A (compute capability 8.9, CUDA 12.5, Release build) on
2026-08-16. The CSV files contain the full distribution statistics.

## A/B results

Primary run: [`results/rtx4080_laptop_milestone4b_run1.csv`](results/rtx4080_laptop_milestone4b_run1.csv).
Independent repeat: [`results/rtx4080_laptop_milestone4b_run2.csv`](results/rtx4080_laptop_milestone4b_run2.csv).

| Case | B×H×Q×KV | Row run 1 (µs) | Specialized run 1 (µs) | Run 1 speedup | Row run 2 (µs) | Specialized run 2 (µs) | Run 2 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, all-full | 1×8×128×256 | 5.120 | 4.928 | 1.039× | 4.928 | 4.800 | 1.027× |
| Small, all-sliding | 1×8×128×256 | 5.120 | 5.120 | 1.000× | 4.832 | 4.768 | 1.013× |
| Medium, all-full | 1×16×512×1024 | 50.176 | 44.976 | 1.116× | 49.040 | 49.664 | 0.987× |
| Medium, all-sliding | 1×16×512×1024 | 50.112 | 49.616 | 1.010× | 51.184 | 46.064 | 1.111× |
| Large, all-full | 2×16×1024×2048 | 627.712 | 627.712 | 1.000× | 627.712 | 627.712 | 1.000× |
| Large, all-sliding | 2×16×1024×2048 | 627.712 | 627.712 | 1.000× | 628.480 | 627.712 | 1.001× |

## Decision

The results do not establish a stable, meaningful gain. Small cases improved
by at most 3.9% and include a tie; both large cases are ties within event-timer
granularity. More importantly, the apparent medium-size advantage is
inconsistent: all-full changed from an 11.6% improvement to a 1.3% regression,
while all-sliding changed from 1.0% to 11.1% improvement. A production fast
path would also need a compatible way to establish that device-resident head
modes are uniform. The current public API provides no host-side uniformity
signal, so detecting it would require additional work not represented by these
kernel-only timings.

Accordingly, the row-oriented hybrid kernel remains the production default and
comparison baseline. The experimental kernels and A/B harness changes were
removed after measurement.

## Validation

- The benchmark performed byte-for-byte row-versus-specialized comparisons for
  all six measured cases before timing; no mismatch occurred.
- The unchanged public implementation passed the complete 41-test GPU suite.
- Compute Sanitizer memcheck passed the representative mixed-head case
  `B=1, H=16, Q=128, past=128, window=32` with `ERROR SUMMARY: 0 errors`.
- `git diff --check` passed.

This concludes Milestone 4B.
