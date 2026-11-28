
cmake_minimum_required(VERSION 3.24)

project(EdgeInferCPP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 4050 Laptop GPU: Ada, compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

add_library(edgeinfer STATIC
    src/kernels/rope.cu
    src/kernels/gqa_attention.cu
)

target_include_directories(edgeinfer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(operator_integration_test
    tests/operator_integration_test.cu
)

target_link_libraries(operator_integration_test
    PRIVATE edgeinfer
)

enable_testing()

add_test(
    NAME OperatorIntegration
    COMMAND operator_integration_test
)
