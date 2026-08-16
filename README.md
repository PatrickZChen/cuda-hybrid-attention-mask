# CUDA Hybrid Attention Mask

A clean-room project for hybrid attention-mask generation in transformer workloads, supporting both full-attention and sliding-window attention patterns.

The project now includes a reliable PyTorch correctness reference and a readable baseline CUDA implementation. Performance optimization is intentionally deferred until the baseline is compiled and validated on NVIDIA hardware.

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

Validation status:

- The Python reference and CUDA integration tests pass on an NVIDIA CUDA host.
- The baseline CUDA output matches the PyTorch reference elementwise across the integration suite.
- Compute Sanitizer memcheck passes on a representative mixed-head case with `ERROR SUMMARY: 0 errors`.

Planned later:

- GPU profiling and kernel optimization
- GPU benchmarking

No performance claims or optimized CUDA techniques are included in this baseline.

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

Profiling, benchmarking, and optimization remain deferred.
