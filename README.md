# CUDA Hybrid Attention Mask

A clean-room project for hybrid attention-mask generation in transformer workloads, supporting both full-attention and sliding-window attention patterns.

The project includes a reliable PyTorch correctness reference, a preserved
baseline CUDA implementation, and a row-oriented CUDA kernel that removes
per-element coordinate reconstruction while retaining the same dense float32
API and semantics.

## Overview

Modern transformer architectures may use different attention patterns across heads. Some heads attend over the complete available context, while others operate within a restricted sliding window.

The PyTorch implementation defines the mask semantics. A standalone CUDA
runner enables direct elementwise comparison between the default CUDA output
and that reference.

The project covers:

- Full causal attention masks
- Sliding-window causal attention masks
- Per-head hybrid attention patterns
- Optional prefix / past-key-value context
- Correctness validation with a Python/PyTorch reference
- Preserved flat baseline and default row-oriented CUDA mask kernels
- CUDA-event A/B benchmarking across full, sliding, and mixed head modes
- Hardware profiling notes with measured facts separated from interpretation

## Mask Semantics

The additive output mask has shape `[batch_size, num_heads, seq_len, past_len + seq_len]`. Allowed positions contain `0`; masked positions contain `-inf`.

For a current-sequence query position `q`, its absolute position is:

```text
absolute_q = past_len + q
```

For a key position `k` in the complete cached-plus-current KV sequence:

### Full causal attention

A position may attend to keys satisfying:

```text
k <= absolute_q
```

### Sliding-window causal attention

For window size `W`, a position may attend to keys satisfying:

```text
absolute_q - W + 1 <= k <= absolute_q
```

Each head independently selects full or sliding-window causal attention. The same per-head mask is replicated for every batch element.

## Current status

Implemented now:

- Readable PyTorch reference implementation
- Per-head full-versus-sliding configuration
- Cached prefix (`past_len`), floating-point dtype, and device support
- Pytest correctness and input-validation coverage
- Preserved baseline float32 kernel using one-dimensional grid-stride indexing
- Default row-oriented kernel with one CUDA block per `[batch, head, query]` row
- Reusable asynchronous CUDA launcher with argument and launch-error reporting
- Standalone CUDA runner with deterministic `0`/`-inf` output
- Parameterized integration tests that compare CUDA output to the PyTorch oracle
- Dedicated CUDA-event A/B benchmark with byte-for-byte output validation,
  alternating launch order, warmups, and per-launch distribution statistics
- RTX 4080 Laptop GPU baseline, profiling, and Milestone 4A/4B A/B reports

Validation status:

- The Python reference and CUDA integration tests pass on an NVIDIA CUDA host.
- The default row-oriented CUDA output matches the PyTorch reference
  elementwise across the integration suite.
- Compute Sanitizer memcheck passes on a representative mixed-head case with `ERROR SUMMARY: 0 errors`.
- The nine-case A/B benchmark validates baseline and optimized outputs
  byte-for-byte before timing.
- On the primary run, the large alternating case improves from 636.768 µs to
  632.832 µs; the nine-case geometric-mean speedup is 1.160×.

Nsight Compute counter collection remains unavailable until the host enables
performance-counter access. All performance numbers below are specific to the
recorded hardware and software environment.

## GPU Validation

Hardware:
NVIDIA GeForce RTX 4080 Laptop GPU

CUDA Toolkit:
12.5

Validation:
Row-oriented CUDA output validated elementwise against the PyTorch reference;
the benchmark also validates it byte-for-byte against the preserved baseline.

Compute Sanitizer memcheck:
Representative mixed-head case passed with `ERROR SUMMARY: 0 errors`.

Final pytest result:
41 passed, 0 failed, 0 skipped.

## RTX 4080 Laptop GPU Milestone 4A A/B results

Release build, 1,000 warmups and 1,000 CUDA-event samples per implementation
and case, with baseline/optimized launch order alternated. The compact table
shows alternating full/sliding heads. The complete nine-case primary matrix,
methodology, and independent repeat summary are in
[`benchmarks/MILESTONE_4A.md`](benchmarks/MILESTONE_4A.md); exact primary data
are in
[`benchmarks/results/rtx4080_laptop_milestone4a.csv`](benchmarks/results/rtx4080_laptop_milestone4a.csv).

| Size | B×H×Q×KV | Baseline median (µs) | Optimized median (µs) | Speedup | Optimized output GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 1×8×128×256 | 6.144 | 5.120 | 1.200× | 190.735 |
| Medium | 1×16×512×1024 | 71.552 | 58.272 | 1.228× | 536.278 |
| Large | 2×16×1024×2048 | 636.768 | 632.832 | 1.006× | 395.050 |

The primary geometric-mean speedup is 1.160× across all nine cases, and no
case regressed by median in either the primary run or an independent repeat.
The large cases improve by only 0.5–0.6%, consistent with dense output stores
dominating that regime. Effective output GiB/s is logical float32 output size
divided by median time; it is not measured DRAM bandwidth. In particular, the
32 MiB medium output fits in the reported 48 MiB L2 and can exceed nominal DRAM
bandwidth as a logical rate. See [`benchmarks/PROFILE.md`](benchmarks/PROFILE.md)
for the baseline evidence and profiler limitations.

## Setup and tests

Create an isolated Python environment, then install the test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch pytest numpy
python -m pytest -q
```

## CUDA build and integration tests

CMake configures without CUDA targets when no CUDA compiler is installed. On a machine with an NVIDIA CUDA toolkit:

```bash
cmake -S . -B build
cmake --build build
HYBRID_MASK_CUDA_RUNNER=build/cuda_mask_runner python -m pytest tests/test_cuda_integration.py -q
```

To reproduce the RTX 4080 Laptop GPU benchmark after a Release build:

```bash
build/cuda_baseline_benchmark \
  --case all \
  --warmups 1000 \
  --iterations 1000 \
  --output benchmarks/results/rtx4080_laptop_milestone4a.csv
```

The runner accepts five dimensions followed by one mode per head, where `1` selects full causal attention and `0` selects sliding-window causal attention:

```bash
build/cuda_mask_runner 1 2 4 0 2 1 0
```

It prints the flattened row-major float32 mask as whitespace-delimited `0` and `-inf` values. Diagnostics are written to standard error.

## Reference API

```python
import torch

from reference import create_hybrid_attention_mask

mask = create_hybrid_attention_mask(
    batch_size=2,
    num_heads=4,
    seq_len=8,
    past_len=16,
    sliding_window=4,
    full_attention_heads=torch.tensor([True, False, True, False]),
    dtype=torch.float32,
    device="cpu",
)
```

## Milestone boundary

Milestone 4B rejected dedicated all-full and all-sliding kernels because the
measured gains were not stable or meaningful; the row-oriented hybrid kernel
remains the default. The dense float32 representation and public API are
unchanged; fusion, packed masks, Tensor Cores, PTX, and additional kernel
optimizations are not included. See
[`benchmarks/MILESTONE_4B.md`](benchmarks/MILESTONE_4B.md) for the real A/B
measurements and decision.
