cmake_minimum_required(VERSION 3.24)

project(cuda_hybrid_attention_mask LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    add_library(hybrid_attention_mask STATIC src/attention_mask.cu)
    target_include_directories(
        hybrid_attention_mask
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    set_target_properties(
        hybrid_attention_mask
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
    )

    add_executable(cuda_mask_runner tests/cuda_mask_runner.cu)
    target_link_libraries(cuda_mask_runner PRIVATE hybrid_attention_mask)
    set_target_properties(
        cuda_mask_runner
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
    )

    add_executable(
        cuda_baseline_benchmark benchmarks/cuda_baseline_benchmark.cu
    )
    target_link_libraries(
        cuda_baseline_benchmark PRIVATE hybrid_attention_mask
    )
    target_include_directories(
        cuda_baseline_benchmark
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    set_target_properties(
        cuda_baseline_benchmark
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
    )
else()
    message(STATUS "CUDA compiler not found; CUDA targets disabled")
endif()
