#include "helper.cuh"
#include <cassert>
#include <cstdint>
#include <cstdio>

template <typename T>



__global__ void mm_kernal(const T* mat_1, const T* mat_2, T* mat_3, size_t m, size_t n, size_t p) {
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( (i >= m) || (j >= p)) return;


    // Dot product of ith row and jth column
    T acc_sum{0};
    for(size_t k{0}; k < n; ++k)
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];


    mat_3[i * p + j] = acc_sum;
}

int main() {
    // Test Suite 1: Correct expected value
    unsigned long MAT_DIM = 20;
    const size_t num_tests{2};
    size_t m{MAT_DIM}, n{MAT_DIM}, p{MAT_DIM};
    assert(random_multiple_test_mm_cuda<int32_t>(num_tests, m, n, p));
    assert(random_multiple_test_mm_cuda<float>(num_tests, m, n, p));
    assert(random_multiple_test_mm_cuda<double>(num_tests, m, n, p));
    printf("All tests passed!\n");


    //Test Suite 2: Performance
    const size_t num_measurement_tests{2};
    const size_t num_measurement_warmups{1};
    m = MAT_DIM; n = MAT_DIM; p = MAT_DIM;

    float mm_cuda_int32_latency = measure_latency_mm_cuda<int32_t>(
        m,
        n,
        p,
        num_measurement_tests,
        num_measurement_warmups
    );    
    float mm_cuda_float_latency = measure_latency_mm_cuda<float>(
        m,
        n,
        p,
        num_measurement_tests,
        num_measurement_warmups
    );   
    float mm_cuda_double_latency = measure_latency_mm_cuda<double>(
        m,
        n,
        p,
        num_measurement_tests,
        num_measurement_warmups
    );   


    printf("Matrix Multiplication Runtime\n");
    printf("m: %zu, n: %zu, p: %zu\n", m, n, p);
    printf("INT32: %f ms\n", mm_cuda_int32_latency);
    printf("FLOAT: %f ms\n", mm_cuda_float_latency);
    printf("DOUBLE: %f ms\n", mm_cuda_double_latency);
}
