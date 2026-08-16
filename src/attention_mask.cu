#include "attention_mask.cuh"
#include "attention_mask_internal.cuh"

#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <math_constants.h>

namespace hybrid_attention_mask {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kMaxGridBlocks = 65535;
constexpr std::int64_t kMaxGridX = 2147483647;
constexpr std::int64_t kMaxGridY = 65535;
constexpr std::int64_t kMaxGridZ = 65535;

struct LaunchConfiguration {
    std::int64_t total_elements;
    std::int64_t kv_len;
};

bool checked_multiply(
    std::int64_t left, std::int64_t right, std::int64_t* result) {
    if (left > std::numeric_limits<std::int64_t>::max() / right) {
        return false;
    }
    *result = left * right;
    return true;
}

__global__ void hybrid_attention_mask_baseline_kernel(
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

__global__ void hybrid_attention_mask_row_kernel(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    std::int64_t kv_len) {
    // The 3D grid maps one block directly to one [batch, head, query] row.
    // Row-level coordinates and bounds are therefore computed once, while
    // threads walk contiguous keys without per-element division or modulo.
    const std::int64_t query = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t head = static_cast<std::int64_t>(blockIdx.y);
    const std::int64_t batch = static_cast<std::int64_t>(blockIdx.z);
    const std::int64_t row =
        (batch * num_heads + head) * seq_len + query;
    const std::int64_t row_offset = row * kv_len;
    const std::int64_t absolute_query = past_len + query;
    const std::int64_t window_start =
        absolute_query - sliding_window + 1;
    const bool is_full = head_is_full[head] == kFullCausal;

    for (std::int64_t key = static_cast<std::int64_t>(threadIdx.x);
         key < kv_len;
         key += static_cast<std::int64_t>(blockDim.x)) {
        const bool causal = key <= absolute_query;
        const bool within_window = key >= window_start;
        const bool allowed = causal && (is_full || within_window);
        output[row_offset + key] = allowed ? 0.0F : -CUDART_INF_F;
    }
}

cudaError_t validate_launch(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    LaunchConfiguration* configuration) {
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

    *configuration = {total_elements, kv_len};
    return cudaSuccess;
}

cudaError_t launch_baseline_validated(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t total_elements,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    std::int64_t kv_len,
    cudaStream_t stream) {
    const std::int64_t required_block_count =
        (total_elements - 1) / kThreadsPerBlock + 1;
    const std::int64_t block_count =
        required_block_count < kMaxGridBlocks ? required_block_count
                                              : kMaxGridBlocks;

    hybrid_attention_mask_baseline_kernel<<<
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

bool supports_direct_row_grid(
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len) {
    return seq_len <= kMaxGridX && num_heads <= kMaxGridY &&
           batch_size <= kMaxGridZ;
}

}  // namespace

namespace detail {

cudaError_t launch_hybrid_attention_mask_baseline(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream) {
    LaunchConfiguration configuration{};
    const cudaError_t validation_error = validate_launch(
        output,
        head_is_full,
        batch_size,
        num_heads,
        seq_len,
        past_len,
        sliding_window,
        &configuration);
    if (validation_error != cudaSuccess) {
        return validation_error;
    }

    return launch_baseline_validated(
        output,
        head_is_full,
        configuration.total_elements,
        num_heads,
        seq_len,
        past_len,
        sliding_window,
        configuration.kv_len,
        stream);
}

cudaError_t launch_hybrid_attention_mask_row(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream) {
    LaunchConfiguration configuration{};
    const cudaError_t validation_error = validate_launch(
        output,
        head_is_full,
        batch_size,
        num_heads,
        seq_len,
        past_len,
        sliding_window,
        &configuration);
    if (validation_error != cudaSuccess) {
        return validation_error;
    }

    if (!supports_direct_row_grid(batch_size, num_heads, seq_len)) {
        return launch_baseline_validated(
            output,
            head_is_full,
            configuration.total_elements,
            num_heads,
            seq_len,
            past_len,
            sliding_window,
            configuration.kv_len,
            stream);
    }

    const dim3 block_count(
        static_cast<unsigned int>(seq_len),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(batch_size));
    hybrid_attention_mask_row_kernel<<<
        block_count, kThreadsPerBlock, 0, stream>>>(
        output,
        head_is_full,
        num_heads,
        seq_len,
        past_len,
        sliding_window,
        configuration.kv_len);
    return cudaGetLastError();
}

}  // namespace detail

cudaError_t launch_hybrid_attention_mask(
    float* output,
    const std::uint8_t* head_is_full,
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::int64_t sliding_window,
    cudaStream_t stream) {
    return detail::launch_hybrid_attention_mask_row(
        output,
        head_is_full,
        batch_size,
        num_heads,
        seq_len,
        past_len,
        sliding_window,
        stream);
}

}  // namespace hybrid_attention_mask
