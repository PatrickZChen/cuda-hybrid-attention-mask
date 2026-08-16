#include "attention_mask.cuh"
#include "attention_mask_internal.cuh"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kDefaultWarmups = 1000;
constexpr int kDefaultIterations = 1000;

enum class HeadPattern {
    kAllFull,
    kAllSliding,
    kAlternating,
};

struct BenchmarkCase {
    const char* name;
    const char* size;
    HeadPattern pattern;
    std::int64_t batch_size;
    std::int64_t num_heads;
    std::int64_t seq_len;
    std::int64_t past_len;
    std::int64_t sliding_window;
};

struct Options {
    std::string case_name = "all";
    std::string output_path;
    int warmups = kDefaultWarmups;
    int iterations = kDefaultIterations;
    bool list_cases = false;
};

struct TimingStatistics {
    double median_us;
    double mean_us;
    double min_us;
    double p95_us;
    double elements_per_second;
    double effective_output_gib_per_second;
};

struct BenchmarkResult {
    const BenchmarkCase* benchmark_case;
    int warmups;
    int iterations;
    std::uint64_t output_elements;
    TimingStatistics baseline;
    TimingStatistics optimized;
    double speedup;
};

using KernelLauncher = cudaError_t (*)(
    float*,
    const std::uint8_t*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    cudaStream_t);

constexpr KernelLauncher kBaselineLauncher =
    hybrid_attention_mask::detail::launch_hybrid_attention_mask_baseline;
constexpr KernelLauncher kRowLauncher =
    hybrid_attention_mask::detail::launch_hybrid_attention_mask_row;

constexpr BenchmarkCase kCases[] = {
    {"small-all-full", "small", HeadPattern::kAllFull, 1, 8, 128, 128, 64},
    {"small-all-sliding",
     "small",
     HeadPattern::kAllSliding,
     1,
     8,
     128,
     128,
     64},
    {"small-alternating",
     "small",
     HeadPattern::kAlternating,
     1,
     8,
     128,
     128,
     64},
    {"medium-all-full",
     "medium",
     HeadPattern::kAllFull,
     1,
     16,
     512,
     512,
     256},
    {"medium-all-sliding",
     "medium",
     HeadPattern::kAllSliding,
     1,
     16,
     512,
     512,
     256},
    {"medium-alternating",
     "medium",
     HeadPattern::kAlternating,
     1,
     16,
     512,
     512,
     256},
    {"large-all-full",
     "large",
     HeadPattern::kAllFull,
     2,
     16,
     1024,
     1024,
     512},
    {"large-all-sliding",
     "large",
     HeadPattern::kAllSliding,
     2,
     16,
     1024,
     1024,
     512},
    {"large-alternating",
     "large",
     HeadPattern::kAlternating,
     2,
     16,
     1024,
     1024,
     512},
};

const char* head_pattern_name(HeadPattern pattern) {
    switch (pattern) {
        case HeadPattern::kAllFull:
            return "all-full";
        case HeadPattern::kAllSliding:
            return "all-sliding";
        case HeadPattern::kAlternating:
            return "alternating";
    }
    return "unknown";
}

void print_usage(const char* executable) {
    std::cerr
        << "Usage: " << executable
        << " [--case <name|all>] [--warmups <count>]"
           " [--iterations <count>] [--output <csv-path>] [--list]\n";
}

bool parse_positive_int(std::string_view text, int* value) {
    if (text.empty()) {
        return false;
    }
    int parsed = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        parsed <= 0) {
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_nonnegative_int(std::string_view text, int* value) {
    if (text.empty()) {
        return false;
    }
    int parsed = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        parsed < 0) {
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_options(int argc, char** argv, Options* options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--list") {
            options->list_cases = true;
            continue;
        }
        if (index + 1 >= argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return false;
        }
        const std::string_view value(argv[++index]);
        if (argument == "--case") {
            options->case_name = value;
        } else if (argument == "--warmups") {
            if (!parse_nonnegative_int(value, &options->warmups)) {
                std::cerr << "Warmup count must be a non-negative integer.\n";
                return false;
            }
        } else if (argument == "--iterations") {
            if (!parse_positive_int(value, &options->iterations)) {
                std::cerr << "Iteration count must be a positive integer.\n";
                return false;
            }
        } else if (argument == "--output") {
            options->output_path = value;
        } else {
            std::cerr << "Unknown option: " << argument << '\n';
            return false;
        }
    }
    return true;
}

bool report_cuda_error(const char* operation, cudaError_t error) {
    if (error == cudaSuccess) {
        return false;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(error) << '\n';
    return true;
}

bool checked_multiply(
    std::uint64_t left, std::uint64_t right, std::uint64_t* result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    *result = left * right;
    return true;
}

bool output_size(
    const BenchmarkCase& benchmark_case,
    std::uint64_t* elements,
    std::size_t* bytes) {
    const std::int64_t kv_len = benchmark_case.past_len + benchmark_case.seq_len;
    std::uint64_t count = 1;
    const std::int64_t dimensions[] = {
        benchmark_case.batch_size,
        benchmark_case.num_heads,
        benchmark_case.seq_len,
        kv_len,
    };
    for (const std::int64_t dimension : dimensions) {
        if (!checked_multiply(
                count, static_cast<std::uint64_t>(dimension), &count)) {
            return false;
        }
    }
    std::uint64_t byte_count = 0;
    if (!checked_multiply(count, sizeof(float), &byte_count) ||
        byte_count > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    *elements = count;
    *bytes = static_cast<std::size_t>(byte_count);
    return true;
}

std::vector<std::uint8_t> make_head_modes(const BenchmarkCase& benchmark_case) {
    std::vector<std::uint8_t> modes(
        static_cast<std::size_t>(benchmark_case.num_heads));
    for (std::int64_t head = 0; head < benchmark_case.num_heads; ++head) {
        bool is_full = false;
        if (benchmark_case.pattern == HeadPattern::kAllFull) {
            is_full = true;
        } else if (benchmark_case.pattern == HeadPattern::kAlternating) {
            is_full = head % 2 == 0;
        }
        modes[static_cast<std::size_t>(head)] =
            is_full ? hybrid_attention_mask::kFullCausal
                    : hybrid_attention_mask::kSlidingWindowCausal;
    }
    return modes;
}

double percentile_95(const std::vector<double>& sorted_samples) {
    const auto rank = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(sorted_samples.size())));
    return sorted_samples[std::max<std::size_t>(1, rank) - 1];
}

cudaError_t launch_kernel(
    KernelLauncher launcher,
    float* output,
    const std::uint8_t* device_modes,
    const BenchmarkCase& benchmark_case,
    cudaStream_t stream) {
    return launcher(
        output,
        device_modes,
        benchmark_case.batch_size,
        benchmark_case.num_heads,
        benchmark_case.seq_len,
        benchmark_case.past_len,
        benchmark_case.sliding_window,
        stream);
}

bool validate_implementations_match(
    const BenchmarkCase& benchmark_case,
    std::uint64_t elements,
    float* baseline_output,
    float* optimized_output,
    const std::uint8_t* device_modes,
    cudaStream_t stream) {
    if (report_cuda_error(
            "baseline validation launch",
            launch_kernel(
                kBaselineLauncher,
                baseline_output,
                device_modes,
                benchmark_case,
                stream)) ||
        report_cuda_error(
            "optimized validation launch",
            launch_kernel(
                kRowLauncher,
                optimized_output,
                device_modes,
                benchmark_case,
                stream)) ||
        report_cuda_error(
            "cudaStreamSynchronize(validation)",
            cudaStreamSynchronize(stream))) {
        return false;
    }

    constexpr std::size_t kComparisonChunkElements = 1U << 20;
    std::vector<float> baseline_chunk(kComparisonChunkElements);
    std::vector<float> optimized_chunk(kComparisonChunkElements);
    std::uint64_t offset = 0;
    while (offset < elements) {
        const auto remaining = static_cast<std::size_t>(
            std::min<std::uint64_t>(elements - offset,
                                    kComparisonChunkElements));
        const std::size_t comparison_bytes = remaining * sizeof(float);
        if (report_cuda_error(
                "cudaMemcpy(baseline validation output)",
                cudaMemcpy(
                    baseline_chunk.data(),
                    baseline_output + offset,
                    comparison_bytes,
                    cudaMemcpyDeviceToHost)) ||
            report_cuda_error(
                "cudaMemcpy(optimized validation output)",
                cudaMemcpy(
                    optimized_chunk.data(),
                    optimized_output + offset,
                    comparison_bytes,
                    cudaMemcpyDeviceToHost))) {
            return false;
        }
        if (std::memcmp(
                baseline_chunk.data(),
                optimized_chunk.data(),
                comparison_bytes) != 0) {
            for (std::size_t index = 0; index < remaining; ++index) {
                if (std::memcmp(
                        &baseline_chunk[index],
                        &optimized_chunk[index],
                        sizeof(float)) != 0) {
                    std::cerr
                        << "Output mismatch for " << benchmark_case.name
                        << " at flat index " << offset + index
                        << ": baseline=" << baseline_chunk[index]
                        << ", optimized=" << optimized_chunk[index] << '\n';
                    return false;
                }
            }
        }
        offset += remaining;
    }
    return true;
}

bool record_sample(
    KernelLauncher launcher,
    float* output,
    const std::uint8_t* device_modes,
    const BenchmarkCase& benchmark_case,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop,
    std::vector<double>* samples_us) {
    if (report_cuda_error(
            "cudaEventRecord(start)", cudaEventRecord(start, stream)) ||
        report_cuda_error(
            "timed kernel launch",
            launch_kernel(
                launcher,
                output,
                device_modes,
                benchmark_case,
                stream)) ||
        report_cuda_error(
            "cudaEventRecord(stop)", cudaEventRecord(stop, stream)) ||
        report_cuda_error(
            "cudaEventSynchronize(stop)", cudaEventSynchronize(stop))) {
        return false;
    }
    float elapsed_ms = 0.0F;
    if (report_cuda_error(
            "cudaEventElapsedTime",
            cudaEventElapsedTime(&elapsed_ms, start, stop))) {
        return false;
    }
    samples_us->push_back(static_cast<double>(elapsed_ms) * 1000.0);
    return true;
}

TimingStatistics summarize_samples(
    std::vector<double> samples_us, std::uint64_t elements) {
    std::sort(samples_us.begin(), samples_us.end());
    const std::size_t middle = samples_us.size() / 2;
    const double median_us = samples_us.size() % 2 == 0
                                 ? (samples_us[middle - 1] + samples_us[middle]) /
                                       2.0
                                 : samples_us[middle];
    const double mean_us =
        std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
        static_cast<double>(samples_us.size());
    const double seconds = median_us / 1'000'000.0;
    const double elements_per_second = static_cast<double>(elements) / seconds;
    constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    return {
        median_us,
        mean_us,
        samples_us.front(),
        percentile_95(samples_us),
        elements_per_second,
        elements_per_second * sizeof(float) / kBytesPerGiB,
    };
}

bool run_case(
    const BenchmarkCase& benchmark_case,
    int warmups,
    int iterations,
    cudaStream_t stream,
    BenchmarkResult* result) {
    std::uint64_t elements = 0;
    std::size_t bytes = 0;
    if (!output_size(benchmark_case, &elements, &bytes)) {
        std::cerr << "Output size overflow for " << benchmark_case.name << '\n';
        return false;
    }

    const std::vector<std::uint8_t> host_modes =
        make_head_modes(benchmark_case);
    std::uint8_t* device_modes = nullptr;
    float* baseline_output = nullptr;
    float* optimized_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    auto cleanup = [&]() {
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (optimized_output != nullptr) {
            cudaFree(optimized_output);
        }
        if (baseline_output != nullptr) {
            cudaFree(baseline_output);
        }
        if (device_modes != nullptr) {
            cudaFree(device_modes);
        }
    };

    if (report_cuda_error(
            "cudaMalloc(head modes)",
            cudaMalloc(
                reinterpret_cast<void**>(&device_modes), host_modes.size())) ||
        report_cuda_error(
            "cudaMalloc(baseline output)",
            cudaMalloc(reinterpret_cast<void**>(&baseline_output), bytes)) ||
        report_cuda_error(
            "cudaMalloc(optimized output)",
            cudaMalloc(reinterpret_cast<void**>(&optimized_output), bytes)) ||
        report_cuda_error(
            "cudaMemcpyAsync(head modes)",
            cudaMemcpyAsync(
                device_modes,
                host_modes.data(),
                host_modes.size(),
                cudaMemcpyHostToDevice,
                stream)) ||
        report_cuda_error("cudaStreamSynchronize(setup)",
                          cudaStreamSynchronize(stream)) ||
        report_cuda_error("cudaEventCreate(start)", cudaEventCreate(&start)) ||
        report_cuda_error("cudaEventCreate(stop)", cudaEventCreate(&stop))) {
        cleanup();
        return false;
    }

    if (!validate_implementations_match(
            benchmark_case,
            elements,
            baseline_output,
            optimized_output,
            device_modes,
            stream)) {
        cleanup();
        return false;
    }

    for (int iteration = 0; iteration < warmups; ++iteration) {
        const KernelLauncher first =
            iteration % 2 == 0 ? kBaselineLauncher : kRowLauncher;
        const KernelLauncher second =
            iteration % 2 == 0 ? kRowLauncher : kBaselineLauncher;
        float* first_output =
            iteration % 2 == 0 ? baseline_output : optimized_output;
        float* second_output =
            iteration % 2 == 0 ? optimized_output : baseline_output;
        if (report_cuda_error(
                "first warmup kernel launch",
                launch_kernel(
                    first,
                    first_output,
                    device_modes,
                    benchmark_case,
                    stream)) ||
            report_cuda_error(
                "second warmup kernel launch",
                launch_kernel(
                    second,
                    second_output,
                    device_modes,
                    benchmark_case,
                    stream))) {
            cleanup();
            return false;
        }
    }
    if (report_cuda_error(
            "cudaStreamSynchronize(warmups)", cudaStreamSynchronize(stream))) {
        cleanup();
        return false;
    }

    std::vector<double> baseline_samples_us;
    std::vector<double> optimized_samples_us;
    baseline_samples_us.reserve(static_cast<std::size_t>(iterations));
    optimized_samples_us.reserve(static_cast<std::size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const bool baseline_first = iteration % 2 == 0;
        if (baseline_first) {
            if (!record_sample(
                    kBaselineLauncher,
                    baseline_output,
                    device_modes,
                    benchmark_case,
                    stream,
                    start,
                    stop,
                    &baseline_samples_us) ||
                !record_sample(
                    kRowLauncher,
                    optimized_output,
                    device_modes,
                    benchmark_case,
                    stream,
                    start,
                    stop,
                    &optimized_samples_us)) {
                cleanup();
                return false;
            }
        } else if (!record_sample(
                       kRowLauncher,
                       optimized_output,
                       device_modes,
                       benchmark_case,
                       stream,
                       start,
                       stop,
                       &optimized_samples_us) ||
                   !record_sample(
                       kBaselineLauncher,
                       baseline_output,
                       device_modes,
                       benchmark_case,
                       stream,
                       start,
                       stop,
                       &baseline_samples_us)) {
            cleanup();
            return false;
        }
    }

    cleanup();

    const TimingStatistics baseline =
        summarize_samples(std::move(baseline_samples_us), elements);
    const TimingStatistics optimized =
        summarize_samples(std::move(optimized_samples_us), elements);

    *result = {
        &benchmark_case,
        warmups,
        iterations,
        elements,
        baseline,
        optimized,
        baseline.median_us / optimized.median_us,
    };
    return true;
}

void write_csv(std::ostream& output, const std::vector<BenchmarkResult>& results) {
    output
        << "case,size,head_pattern,batch_size,num_heads,seq_len,past_len,"
           "sliding_window,warmups,iterations,baseline_median_us,"
           "optimized_median_us,speedup,baseline_mean_us,optimized_mean_us,"
           "baseline_min_us,optimized_min_us,baseline_p95_us,optimized_p95_us,"
           "output_elements,baseline_elements_per_sec,"
           "optimized_elements_per_sec,baseline_effective_output_gib_per_sec,"
           "optimized_effective_output_gib_per_sec\n";
    output << std::fixed << std::setprecision(3);
    for (const BenchmarkResult& result : results) {
        const BenchmarkCase& benchmark_case = *result.benchmark_case;
        output << benchmark_case.name << ',' << benchmark_case.size << ','
               << head_pattern_name(benchmark_case.pattern) << ','
               << benchmark_case.batch_size << ',' << benchmark_case.num_heads
               << ',' << benchmark_case.seq_len << ','
               << benchmark_case.past_len << ','
               << benchmark_case.sliding_window << ',' << result.warmups << ','
               << result.iterations << ',' << result.baseline.median_us << ','
               << result.optimized.median_us << ',' << result.speedup << ','
               << result.baseline.mean_us << ',' << result.optimized.mean_us
               << ',' << result.baseline.min_us << ','
               << result.optimized.min_us << ',' << result.baseline.p95_us
               << ',' << result.optimized.p95_us << ','
               << result.output_elements << ','
               << result.baseline.elements_per_second << ','
               << result.optimized.elements_per_second << ','
               << result.baseline.effective_output_gib_per_second << ','
               << result.optimized.effective_output_gib_per_second << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return 2;
    }

    if (options.list_cases) {
        for (const BenchmarkCase& benchmark_case : kCases) {
            std::cout << benchmark_case.name << '\n';
        }
        return 0;
    }

    std::vector<const BenchmarkCase*> selected_cases;
    for (const BenchmarkCase& benchmark_case : kCases) {
        if (options.case_name == "all" || options.case_name == benchmark_case.name) {
            selected_cases.push_back(&benchmark_case);
        }
    }
    if (selected_cases.empty()) {
        std::cerr << "Unknown benchmark case: " << options.case_name << '\n';
        return 2;
    }

    int device = 0;
    cudaDeviceProp properties{};
    cudaStream_t stream = nullptr;
    if (report_cuda_error("cudaGetDevice", cudaGetDevice(&device)) ||
        report_cuda_error(
            "cudaGetDeviceProperties", cudaGetDeviceProperties(&properties, device)) ||
        report_cuda_error(
            "cudaStreamCreate",
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
        return 1;
    }

    std::vector<BenchmarkResult> results;
    results.reserve(selected_cases.size());
    for (const BenchmarkCase* benchmark_case : selected_cases) {
        BenchmarkResult result{};
        if (!run_case(
                *benchmark_case,
                options.warmups,
                options.iterations,
                stream,
                &result)) {
            cudaStreamDestroy(stream);
            return 1;
        }
        results.push_back(result);
    }

    if (report_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream))) {
        return 1;
    }

    std::cerr << "GPU: " << properties.name << " (compute capability "
              << properties.major << '.' << properties.minor << ")\n";
    const double peak_memory_gb_per_second =
        2.0 * static_cast<double>(properties.memoryClockRate) *
        static_cast<double>(properties.memoryBusWidth / 8) / 1'000'000.0;
    std::cerr << "CUDA device properties: " << properties.memoryBusWidth
              << "-bit memory bus, "
              << static_cast<double>(properties.memoryClockRate) / 1000.0
              << " MHz reported memory clock, " << std::fixed
              << std::setprecision(3) << peak_memory_gb_per_second
              << " GB/s calculated peak, " << properties.l2CacheSize
              << "-byte L2 cache.\n";
    std::cerr
        << "Validated baseline and row-oriented outputs byte-for-byte, then "
           "timed both launchers with alternating order and CUDA events; "
           "setup, copies, warmups, synchronization waits, and CSV output are "
           "outside measured intervals.\n";

    if (options.output_path.empty()) {
        write_csv(std::cout, results);
        return 0;
    }

    std::ofstream output(options.output_path);
    if (!output) {
        std::cerr << "Unable to open output file: " << options.output_path << '\n';
        return 1;
    }
    write_csv(output, results);
    if (!output) {
        std::cerr << "Failed while writing output file: " << options.output_path
                  << '\n';
        return 1;
    }
    std::cerr << "Wrote " << options.output_path << '\n';
    return 0;
}
