cmake_minimum_required(VERSION 3.18)
project(bitcache LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require sm_80+ (Ampere and newer)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)

set(CUDA_FLAGS
    --use_fast_math
    -O3
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler -Wall
)

# Common include directory
include_directories(${CMAKE_SOURCE_DIR}/src)

# Core library with all kernels
add_library(bitcache_kernels STATIC
    src/binary_encode.cu
    src/residual.cu
    src/binary_attention.cu
    src/compensated_attention.cu
    src/reference_attention.cu
)
target_compile_options(bitcache_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
target_include_directories(bitcache_kernels PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(bitcache_kernels PUBLIC CUDA::cudart CUDA::cuda_driver)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_compile_options(test_correctness PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
target_link_libraries(test_correctness PRIVATE bitcache_kernels)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_compile_options(bench_latency PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
target_link_libraries(bench_latency PRIVATE bitcache_kernels)

# Enable CTest
enable_testing()
add_test(NAME correctness COMMAND test_correctness)
