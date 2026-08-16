# CUDA Hybrid Attention Mask

A clean-room project for hybrid attention-mask generation in transformer workloads, supporting both full-attention and sliding-window attention patterns.

The project now includes a reliable PyTorch correctness reference, a readable baseline CUDA implementation, and a reproducible hardware-specific performance baseline. Kernel optimization remains intentionally deferred.

## Overview

Modern transformer architectures may use different attention patterns across heads. Some heads attend over the complete available context, while others operate within a restricted sliding window.

The PyTorch implementation defines the mask semantics. A standalone CUDA runner enables direct elementwise comparison between the baseline CUDA output and that reference.

The project covers:

- Full causal attention masks
- Sliding-window causal attention masks
- Per-head hybrid attention patterns
- Optional prefix / past-key-value context
- Correctness validation with a Python/PyTorch reference
- Baseline CUDA mask generation and integration-test infrastructure
- CUDA-event baseline benchmarking across full, sliding, and mixed head modes
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
- Baseline float32 CUDA kernel using simple one-dimensional grid-stride indexing
- Reusable asynchronous CUDA launcher with argument and launch-error reporting
- Standalone CUDA runner with deterministic `0`/`-inf` output
- Parameterized integration tests that compare CUDA output to the PyTorch oracle
- Dedicated CUDA-event benchmark with warmups and per-launch distribution statistics
- RTX 4080 Laptop GPU baseline CSV and profiling report

Validation status:

- The Python reference and CUDA integration tests pass on an NVIDIA CUDA host.
- The baseline CUDA output matches the PyTorch reference elementwise across the integration suite.
- Compute Sanitizer memcheck passes on a representative mixed-head case with `ERROR SUMMARY: 0 errors`.
- The large alternating baseline produces 67,108,864 elements in a median 635.904 us on the measured RTX 4080 Laptop GPU.

Planned later:

- Milestone 4 kernel optimization, guided by the baseline evidence
- Nsight Compute counter collection after the host enables performance-counter access

All performance numbers below are specific to the recorded hardware and software environment. No optimized CUDA techniques are included in this baseline.

## GPU Validation

Hardware:
NVIDIA GeForce RTX 4080 Laptop GPU

CUDA Toolkit:
12.5

Validation:
Baseline CUDA output validated elementwise against the PyTorch reference.

Compute Sanitizer memcheck:
Representative mixed-head case passed with `ERROR SUMMARY: 0 errors`.

Final pytest result:
41 passed, 0 failed, 0 skipped.

## RTX 4080 Laptop GPU performance baseline

Release build, 1,000 warmups and 1,000 CUDA-event samples per row. The compact table shows alternating full/sliding heads; the complete nine-case matrix is in [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md) and [`benchmarks/results/rtx4080_laptop_baseline.csv`](benchmarks/results/rtx4080_laptop_baseline.csv).

| Size | B×H×Q×KV | Output elements | median_us | Effective output GiB/s |
| --- | ---: | ---: | ---: | ---: |
| Small | 1×8×128×256 | 262,144 | 6.144 | 158.946 |
| Medium | 1×16×512×1024 | 8,388,608 | 71.552 | 436.745 |
| Large | 2×16×1024×2048 | 67,108,864 | 635.904 | 393.141 |

Effective output GiB/s is the logical float32 output size (four bytes per element) divided by the median CUDA-event time and by `2^30`; it is not measured DRAM bandwidth. The comparison peak is a nominal value calculated from CUDA device properties: double-data-rate factor `2 × 9.001 GHz × (192 bits / 8) = 432.048 GB/s`, which is `402.376 GiB/s`. Thus, the reported 97.70% compares `393.141 GiB/s` with `402.376 GiB/s`; it is not a measured DRAM-utilization percentage. On a sustained large run, coarse, GPU-wide NVIDIA-SMI samples reported `mem=100%` and `sm=91–94%`. Taken together, these observations support the strong inference that dense output stores are likely the dominant large-case constraint, but this is not a directly measured Nsight Compute conclusion. Nsight Compute 2024.2.1 is installed, but the driver denied access to hardware performance counters (`ERR_NVGPUCTRPERM`), so achieved DRAM throughput, occupancy, and hardware branch metrics are not claimed. See [`benchmarks/PROFILE.md`](benchmarks/PROFILE.md) for the exact evidence and limitations.

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
  --output benchmarks/results/rtx4080_laptop_baseline.csv
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

## Next milestone

Milestone 4 kernel-level experiments are ranked in [`benchmarks/PROFILE.md`](benchmarks/PROFILE.md), beginning with reducing the repeated 64-bit division, modulo, and coordinate-reconstruction work in the flat indexing path. Eliminating mask materialization and adopting a packed representation are classified separately as architectural alternatives because they change the output or consumer contract. The kernel in this milestone remains the original readable implementation.
