# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -Wno-deprecated-gpu-targets;
    -std=c++11
)

# Specify binary name and source file to build it from
cuda_add_executable(sort
    utils.h
    radix_sort.cu
)
