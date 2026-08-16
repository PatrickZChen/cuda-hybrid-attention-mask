#pragma once

#include "attention_mask.cuh"

namespace hybrid_attention_mask::detail {

// Internal launchers used by the A/B benchmark. These are not part of the
// public API; production callers should use launch_hybrid_attention_mask().
cudaError_t launch_hybrid_attention_mask_baseline(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream = nullptr);

cudaError_t launch_hybrid_attention_mask_row(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream = nullptr);

}  // namespace hybrid_attention_mask::detail
