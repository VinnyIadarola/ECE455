#include "helper.cu"

template <typename T>



__global__ void mm_kernal(const T* mat_1, const T* mat_2, T* mat_3, size_t m, size_t n, size_t p) {
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( (row >= m) || (col >= p)) return;


    // Dot product of ith row and jth column
    T acc_sum{0};
    for(size_t k{0}; k < n; ++k)
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
        //TODO check if there is a difference between accuk in mat_3 vs
        //using an intermediate I assume its faster

    mat_3[i * p + j] = acc_sum;
}

int main() {
    // Test Suite 1: Correct expected value
    const size_t num_test{2};
    assert(random_multiple_test_mm_cuda<int32_t>(num_tests));
    assert(random_multiple_test_mm_cuda<float>(num_tests));
    assert(random_multiple_test_mm_cuda<double>(num_tests));
    printf("All tests passed!\n")


    //Test Suite 2: Performance
    const size_t num_measurement_tests{2};
    const size_t num_measurement_warmups{1};
    size_t m{MAT_DIM}, n{Mat_DIM}, p{MAT_DIM};

    float mm_cuda_int32_latency = measure_latency_mm_cuda<int_32_t>(
        m,
        n,
        p,
        num_measurement_tests,
        num_measurement_warmups
    );    
    float mm_cuda_int32_latency = measure_latency_mm_cuda<float>(
        m,
        n,
        p,
        num_measurement_tests,
        num_measurement_warmups
    );   
    float mm_cuda_int32_latency = measure_latency_mm_cuda<double>(
        m,
        n,
        p,
        num_measurement_tests,
        num_measurement_warmups
    );   


    printf("Matrix Multiplication Runtime\n");
    printf("m: %d, n: %d, p: %d\n", m, n, p);
    printf("INT32: %f ms\n", mm_cuda_int32_latency);
    printf("FLOAT: %f ms\n", mm_cuda_float_latency);
    printf("DOUBLE: %f ms\n", mm_cuda_double_latency);
}




