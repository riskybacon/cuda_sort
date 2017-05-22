/**
 * Adapted from the Udacity CS344 course Introduction to Parallel Programming
 *
 * Changes:
 *
 * Added type checking to the template parameters
 * Changed indentation, line length and naming conventions
 * Removed autodesk related code
 * Added next_pow2
 */
#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

template<
    typename T,
    typename std::enable_if<
        std::is_arithmetic<T>::value && !std::is_floating_point<T>::value,
        int
     >::type = 0
>
void check_results(const T* const ref, const T* const result, size_t size) {
    for (size_t idx = 0; idx < size; ++idx) {
        if (ref[idx] != result[idx]) {
            std::cerr << "Difference at pos " << idx << std::endl;
            // The + operator is used to convert chars to ints without
            // messing with other types
            std::cerr << "Reference: " << std::setprecision(17) << +ref[idx]
                      << "\nresult      : " << +result[idx] << std::endl;
                      exit(1);
        }
    }
}

template<
    typename T,
    typename std::enable_if<
        std::is_arithmetic<T>::value && std::is_floating_point<T>::value,
        int
     >::type = 0
>
void check_results(const T* const ref, const T* const result, size_t size,
                   double eps1, double eps2) {
    assert(eps1 >= 0 && eps2 >= 0);
    unsigned long long total_diff = 0;
    unsigned num_small_differences = 0;
    for (size_t idx = 0; idx < size; ++idx) {
        //subtract smaller from larger in case of unsigned types
        T smaller = std::min(ref[idx], result[idx]);
        T larger = std::max(ref[idx], result[idx]);
        T diff = larger - smaller;
        if (diff > 0 && diff <= eps1) {
            num_small_differences++;
        } else if (diff > eps1) {
            std::cerr << "Difference at pos " << +idx << " exceeds tolerance of "
                      << eps1 << std::endl;
            std::cerr << "Reference: " << std::setprecision(17) << +ref[idx] <<
                         "\nresult      : " << +result[idx] << std::endl;
        exit(1);
        }
        total_diff += diff * diff;
    }

    double percent_small_differences =
        (double)num_small_differences / (double)size;

    if (percent_small_differences > eps2) {
        std::cerr << "Total percentage of non-zero pixel difference between "
                  << "the reference and result exceeds " << 100.0 * eps2 << "%"
                  << std::endl
                  << "Percentage of non-zero differences: "
                  << 100.0 * percent_small_differences << "%" << std::endl;
        exit(1);
    }
}

/**
 * @return  The next power of two that is greater than x
 */
__host__ __device__
unsigned int next_pow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#endif
