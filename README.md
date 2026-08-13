# CUDA Hybrid Attention Mask

A clean-room project for hybrid attention-mask generation in transformer workloads, supporting both full-attention and sliding-window attention patterns.

The current milestone focuses on a reliable PyTorch correctness reference. CUDA implementation and performance work are planned for later milestones.

## Overview

Modern transformer architectures may use different attention patterns across heads. Some heads attend over the complete available context, while others operate within a restricted sliding window.

Milestone 1 implements the mask semantics in PyTorch and verifies them with correctness tests.

The project covers:

- Full causal attention masks
- Sliding-window causal attention masks
- Per-head hybrid attention patterns
- Optional prefix / past-key-value context
- Correctness validation with a Python/PyTorch reference
- CUDA latency and throughput benchmarking in future milestones

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

Planned later:

- Baseline CUDA implementation
- Optimized CUDA kernels
- GPU benchmarking and profiling

No CUDA kernels or performance claims are included in Milestone 1.

## Setup and tests

Create an isolated Python environment, then install the two test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch pytest
python -m pytest -q
```

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
