# CUDA Hybrid Attention Mask

CUDA implementations of hybrid attention-mask generation for transformer workloads, supporting both full-attention and sliding-window attention patterns.

The project focuses on correctness, memory-efficient GPU execution, and performance benchmarking against a reference implementation.

## Overview

Modern transformer architectures may use different attention patterns across heads. Some heads attend over the complete available context, while others operate within a restricted sliding window.

This project implements GPU kernels that generate these masks efficiently in CUDA.

The implementation will cover:

- Full causal attention masks
- Sliding-window causal attention masks
- Per-head hybrid attention patterns
- Optional prefix / past-key-value context
- Correctness validation against a Python/PyTorch reference
- CUDA latency and throughput benchmarking

## Mask Semantics

For a query position `q` and key position `k`:

### Full causal attention

A position may attend to keys satisfying:

```text
k <= q
