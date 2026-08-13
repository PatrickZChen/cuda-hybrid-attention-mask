#include "attention_mask.cuh"

#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <math_constants.h>

namespace hybrid_attention_mask {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kMaxGridBlocks = 65535;

bool checked_multiply(
    std::int64_t left, std::int64_t right, std::int64_t* result) {
    if (left > std::numeric_limits<std::int64_t>::max() / right) {
        return false;
    }
    *result = left * right;
    return true;
}

__global__ void hybrid_attention_mask_kernel(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t total_elements,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    std::int64_t kv_len) {
    const std::uint64_t first_output_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t grid_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const std::uint64_t element_count =
        static_cast<std::uint64_t>(total_elements);

    for (std::uint64_t output_index = first_output_index;
         output_index < element_count;
         output_index += grid_stride) {
        // Unflatten row-major [batch, head, query, key] coordinates.
        std::int64_t remaining = static_cast<std::int64_t>(output_index);
        const std::int64_t key = remaining % kv_len;
        remaining /= kv_len;
        const std::int64_t query = remaining % seq_len;
        remaining /= seq_len;
        const std::int64_t head = remaining % num_heads;
        const std::int64_t batch = remaining / num_heads;
        (void)batch;

        const std::int64_t absolute_query = past_len + query;
        const bool causal = key <= absolute_query;
        const bool is_full = head_is_full[head] == kFullCausal;
        const bool within_window =
            key >= absolute_query - sliding_window + 1;
        const bool allowed = causal && (is_full || within_window);

        output[output_index] = allowed ? 0.0F : -CUDART_INF_F;
    }
}

}  // namespace

cudaError_t launch_hybrid_attention_mask(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream) {
    if (output == nullptr || head_is_full == nullptr || batch_size <= 0 ||
        num_heads <= 0 || seq_len <= 0 || past_len < 0 ||
        sliding_window <= 0) {
        return cudaErrorInvalidValue;
    }

    if (past_len > std::numeric_limits<std::int64_t>::max() - seq_len) {
        return cudaErrorInvalidValue;
    }
    const std::int64_t kv_len = past_len + seq_len;

    std::int64_t total_elements = batch_size;
    if (!checked_multiply(total_elements, num_heads, &total_elements) ||
        !checked_multiply(total_elements, seq_len, &total_elements) ||
        !checked_multiply(total_elements, kv_len, &total_elements)) {
        return cudaErrorInvalidValue;
    }

    const std::int64_t required_block_count =
        (total_elements - 1) / kThreadsPerBlock + 1;
    const std::int64_t block_count =
        required_block_count < kMaxGridBlocks ? required_block_count
                                              : kMaxGridBlocks;

    hybrid_attention_mask_kernel<<<
        static_cast<unsigned int>(block_count), kThreadsPerBlock, 0, stream>>>(
        output,
        head_is_full,
        total_elements,
        num_heads,
        seq_len,
        past_len,
        sliding_window,
        kv_len);
    return cudaGetLastError();
}

}  // namespace hybrid_attention_mask
