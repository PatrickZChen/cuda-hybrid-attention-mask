#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace hybrid_attention_mask {

// Device head-mode values. There must be exactly one value per attention head.
inline constexpr std::uint8_t kSlidingWindowCausal = 0;
inline constexpr std::uint8_t kFullCausal = 1;

// Launches an asynchronous float32 additive-mask kernel on `stream`.
//
// `output` and `head_is_full` must point to device memory. The output has
// row-major shape [batch_size, num_heads, seq_len, past_len + seq_len].
// Allowed positions receive 0.0f and masked positions receive -infinity.
//
// Returns argument-validation or kernel-launch errors. This function does not
// synchronize the stream; callers that need completion must do so explicitly.
cudaError_t launch_hybrid_attention_mask(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream = nullptr);

}  // namespace hybrid_attention_mask
