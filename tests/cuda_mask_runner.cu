#include "attention_mask.cuh"

#include <charconv>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <system_error>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kCudaUnavailableExitCode = 77;

void print_usage(const char* executable) {
    std::cerr
        << "Usage: " << executable
        << " <batch_size> <num_heads> <seq_len> <past_len> <sliding_window>"
           " <head_mode>...\n"
        << "Each head_mode must be 1 (full causal) or 0 (sliding-window "
           "causal).\n";
}

bool parse_int64(std::string_view text, std::int64_t* value) {
    if (text.empty()) {
        return false;
    }
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, *value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool checked_multiply(
    std::size_t left, std::size_t right, std::size_t* result) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    *result = left * right;
    return true;
}

bool calculate_output_size(
    std::int64_t batch_size,
    std::int64_t num_heads,
    std::int64_t seq_len,
    std::int64_t past_len,
    std::size_t* element_count,
    std::size_t* byte_count) {
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || past_len < 0 ||
        past_len > std::numeric_limits<std::int64_t>::max() - seq_len) {
        return false;
    }

    const std::int64_t kv_len = past_len + seq_len;
    const std::int64_t dimensions[] = {
        batch_size, num_heads, seq_len, kv_len};
    std::size_t elements = 1;
    for (const std::int64_t dimension : dimensions) {
        const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
        if (unsigned_dimension > std::numeric_limits<std::size_t>::max() ||
            !checked_multiply(
                elements, static_cast<std::size_t>(dimension), &elements)) {
            return false;
        }
    }

    std::size_t bytes = 0;
    if (!checked_multiply(elements, sizeof(float), &bytes)) {
        return false;
    }
    *element_count = elements;
    *byte_count = bytes;
    return true;
}

bool report_cuda_error(const char* operation, cudaError_t error) {
    if (error == cudaSuccess) {
        return false;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(error) << '\n';
    return true;
}

bool free_device_memory(void* pointer, const char* label) {
    if (pointer == nullptr) {
        return true;
    }
    const cudaError_t error = cudaFree(pointer);
    if (error != cudaSuccess) {
        std::cerr << "cudaFree(" << label << ") failed: "
                  << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

int probe_cuda_device() {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "CUDA device unavailable: " << cudaGetErrorString(error)
                  << '\n';
        return kCudaUnavailableExitCode;
    }
    if (device_count == 0) {
        std::cerr << "CUDA device unavailable: no CUDA-capable device found\n";
        return kCudaUnavailableExitCode;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--probe") {
        return probe_cuda_device();
    }

    if (argc < 6) {
        print_usage(argv[0]);
        return 2;
    }

    std::int64_t batch_size = 0;
    std::int64_t num_heads = 0;
    std::int64_t seq_len = 0;
    std::int64_t past_len = 0;
    std::int64_t sliding_window = 0;
    if (!parse_int64(argv[1], &batch_size) ||
        !parse_int64(argv[2], &num_heads) ||
        !parse_int64(argv[3], &seq_len) ||
        !parse_int64(argv[4], &past_len) ||
        !parse_int64(argv[5], &sliding_window) || batch_size <= 0 ||
        num_heads <= 0 || seq_len <= 0 || past_len < 0 ||
        sliding_window <= 0) {
        std::cerr << "Invalid dimensions: batch_size, num_heads, seq_len, and "
                     "sliding_window must be positive; past_len must be "
                     "non-negative.\n";
        return 2;
    }

    if (num_heads > std::numeric_limits<int>::max() - 6 ||
        argc != 6 + static_cast<int>(num_heads)) {
        std::cerr << "Expected exactly one head_mode per attention head.\n";
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::uint8_t> head_modes(static_cast<std::size_t>(num_heads));
    for (std::int64_t head = 0; head < num_heads; ++head) {
        const std::string_view mode(argv[6 + static_cast<int>(head)]);
        if (mode == "0") {
            head_modes[static_cast<std::size_t>(head)] =
                hybrid_attention_mask::kSlidingWindowCausal;
        } else if (mode == "1") {
            head_modes[static_cast<std::size_t>(head)] =
                hybrid_attention_mask::kFullCausal;
        } else {
            std::cerr << "Invalid head_mode at index " << head
                      << ": expected 0 or 1.\n";
            return 2;
        }
    }

    std::size_t output_elements = 0;
    std::size_t output_bytes = 0;
    if (!calculate_output_size(
            batch_size,
            num_heads,
            seq_len,
            past_len,
            &output_elements,
            &output_bytes)) {
        std::cerr << "Output size is invalid or overflows host address space.\n";
        return 2;
    }

    std::uint8_t* device_head_modes = nullptr;
    float* device_output = nullptr;
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_head_modes), head_modes.size());
    if (report_cuda_error("cudaMalloc(head modes)", error)) {
        return 1;
    }

    error = cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes);
    if (report_cuda_error("cudaMalloc(output)", error)) {
        free_device_memory(device_head_modes, "head modes");
        return 1;
    }

    error = cudaMemcpy(
        device_head_modes,
        head_modes.data(),
        head_modes.size(),
        cudaMemcpyHostToDevice);
    if (report_cuda_error("cudaMemcpy(head modes, host to device)", error)) {
        free_device_memory(device_output, "output");
        free_device_memory(device_head_modes, "head modes");
        return 1;
    }

    error = hybrid_attention_mask::launch_hybrid_attention_mask(
        device_output,
        device_head_modes,
        batch_size,
        num_heads,
        seq_len,
        past_len,
        sliding_window);
    if (report_cuda_error("kernel launch", error)) {
        free_device_memory(device_output, "output");
        free_device_memory(device_head_modes, "head modes");
        return 1;
    }

    error = cudaDeviceSynchronize();
    if (report_cuda_error("cudaDeviceSynchronize", error)) {
        free_device_memory(device_output, "output");
        free_device_memory(device_head_modes, "head modes");
        return 1;
    }

    std::vector<float> host_output(output_elements);
    error = cudaMemcpy(
        host_output.data(),
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost);
    if (report_cuda_error("cudaMemcpy(output, device to host)", error)) {
        free_device_memory(device_output, "output");
        free_device_memory(device_head_modes, "head modes");
        return 1;
    }

    const bool output_freed = free_device_memory(device_output, "output");
    const bool head_modes_freed =
        free_device_memory(device_head_modes, "head modes");
    if (!output_freed || !head_modes_freed) {
        return 1;
    }

    for (std::size_t index = 0; index < host_output.size(); ++index) {
        if (index != 0) {
            std::cout << ' ';
        }
        const float value = host_output[index];
        if (value == 0.0F) {
            std::cout << '0';
        } else if (std::isinf(value) && value < 0.0F) {
            std::cout << "-inf";
        } else {
            std::cerr << "Kernel returned an unexpected value at flat index "
                      << index << ": " << value << '\n';
            return 1;
        }
    }
    std::cout << '\n';
    return 0;
}
