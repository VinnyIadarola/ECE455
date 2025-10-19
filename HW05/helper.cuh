// mm_cuda_harness.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

// Macro wrapper: converts the CUDA call `val` into a string (#val) and passes it with file and line info to `check`
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

// Checks the return code of CUDA API calls
inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " in " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Create a vector filled with random numbers in [-256, 256]
template <typename T>
std::vector<T> create_rand_vector(size_t n) {
    std::random_device r;                 // Non-deterministic seed
    std::default_random_engine e(r());    // Random engine
    std::uniform_int_distribution<int> uniform_dist(-256, 256);

    std::vector<T> vec(n);
    for (size_t i = 0; i < n; ++i) {
        vec[i] = static_cast<T>(uniform_dist(e));
    }
    return vec;
}

// Naive triple-loop matrix multiplication : C = A * B
// A: m x n, B: n x p, C: m x p (row-major)
template <typename T>
void mm(const T* mat1, const T* mat2, T* mat3, size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T acc_sum = T(0);
            for (size_t k = 0; k < n; ++k) {
                acc_sum += mat1[i * n + k] * mat2[k * p + j];
            }
            mat3[i * p + j] = acc_sum;
        }
    }
}

// Compare two vectors elementwise within an absolute tolerance
template <typename T>
bool allclose(const std::vector<T>& vec1, const std::vector<T>& vec2, double abs_tol) {
    if (vec1.size() != vec2.size()) return false;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::fabs(static_cast<double>(vec1[i]) - static_cast<double>(vec2[i])) > abs_tol) {
            std::cout << "Mismatch at " << i << ": " << vec1[i] << " vs " << vec2[i] << std::endl;
            return false; // First mismatch printed
        }
    }
    return true;
}

/* ------------------------ CUDA implementation ------------------------ */

// Simple (naive) CUDA kernel: each thread computes one C(i,j)
template <typename T>
__global__ void mm_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                          size_t m, size_t n, size_t p) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= p) return;

    T acc = T(0);
    for (size_t k = 0; k < n; ++k) {
        acc += A[row * n + k] * B[k * p + col];
    }
    C[row * p + col] = acc;
}

// Host launcher for the kernel
template <typename T>
void mm_cuda(const T* d_mat1, const T* d_mat2, T* d_mat_out, size_t m, size_t n, size_t p) {
    const dim3 block(16, 16);
    const dim3 grid((unsigned)((p + block.x - 1) / block.x),
                    (unsigned)((m + block.y - 1) / block.y));
    mm_kernel<T><<<grid, block>>>(d_mat1, d_mat2, d_mat_out, m, n, p);
    checkCuda(cudaGetLastError());
}

/* ---------------------- Testing / benchmarking ---------------------- */

// Run one randomized test comparing CPU vs GPU results
template <typename T>
bool random_test_mm_cuda(size_t m, size_t n, size_t p) {
    // --- Allocate and initialize host matrices ---
    const std::vector<T> mat1_vec = create_rand_vector<T>(m * n);
    const std::vector<T> mat2_vec = create_rand_vector<T>(n * p);
          std::vector<T> mat3_vec(m * p); // CPU result
          std::vector<T> mat4_vec(m * p); // GPU result

    const T* mat1 = mat1_vec.data();
    const T* mat2 = mat2_vec.data();
          T* mat3 = mat3_vec.data();
          T* mat4 = mat4_vec.data();

    // --- Compute reference result on CPU ---
    mm(mat1, mat2, mat3, m, n, p);

    // --- Allocate GPU memory ---
    T *d_mat1, *d_mat2, *d_mat4;
    checkCuda(cudaMalloc(&d_mat1, sizeof(T) * mat1_vec.size()));
    checkCuda(cudaMalloc(&d_mat2, sizeof(T) * mat2_vec.size()));
    checkCuda(cudaMalloc(&d_mat4, sizeof(T) * mat4_vec.size()));

    // --- Copy input matrices to device ---
    checkCuda(cudaMemcpy(d_mat1, mat1, sizeof(T) * mat1_vec.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat2, mat2, sizeof(T) * mat2_vec.size(), cudaMemcpyHostToDevice));

    // --- Launch CUDA kernel ---
    mm_cuda<T>(d_mat1, d_mat2, d_mat4, m, n, p);
    checkCuda(cudaDeviceSynchronize());

    // --- Copy result back to host ---
    checkCuda(cudaMemcpy(mat4, d_mat4, sizeof(T) * mat4_vec.size(), cudaMemcpyDeviceToHost));

    // --- Free device memory ---
    checkCuda(cudaFree(d_mat1));
    checkCuda(cudaFree(d_mat2));
    checkCuda(cudaFree(d_mat4));

    // --- Compare CPU vs GPU results ---
    return allclose<T>(mat3_vec, mat4_vec, 1e-4);
}

// Run multiple random tests in a loop (for stress testing)
template <typename T>
bool random_multiple_test_mm_cuda(size_t num_tests, size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < num_tests; ++i) {
        if (!random_test_mm_cuda<T>(m, n, p)) {
            return false; // Stop if any test fails
        }
    }
    return true; // All tests passed
}

// Measure average runtime of mm_cuda using CUDA events
template <typename T>
float measure_latency_mm_cuda(size_t m, size_t n, size_t p, size_t num_tests, size_t num_warmups) {
    cudaEvent_t startEvent, stopEvent;
    float time_ms = 0.0f;

    // --- Create CUDA events for timing ---
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    // --- Host data (so device has something real to compute) ---
    const std::vector<T> hA = create_rand_vector<T>(m * n);
    const std::vector<T> hB = create_rand_vector<T>(n * p);
          std::vector<T> hC(m * p);

    // --- Allocate device matrices once ---
    T *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, sizeof(T) * m * n));
    checkCuda(cudaMalloc(&dB, sizeof(T) * n * p));
    checkCuda(cudaMalloc(&dC, sizeof(T) * m * p));

    // Copy inputs once
    checkCuda(cudaMemcpy(dA, hA.data(), sizeof(T) * m * n, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dB, hB.data(), sizeof(T) * n * p, cudaMemcpyHostToDevice));

    // --- Warm-up runs (not timed) ---
    for (size_t i = 0; i < num_warmups; ++i) {
        mm_cuda<T>(dA, dB, dC, m, n, p);
    }
    checkCuda(cudaDeviceSynchronize());

    // --- Timed runs using CUDA events ---
    checkCuda(cudaEventRecord(startEvent, 0));
    for (size_t i = 0; i < num_tests; ++i) {
        mm_cuda<T>(dA, dB, dC, m, n, p);
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent)); // Wait for GPU
    checkCuda(cudaEventElapsedTime(&time_ms, startEvent, stopEvent)); // in ms

    // (Optional) copy back once to prevent dead-code elimination concerns
    checkCuda(cudaMemcpy(hC.data(), dC, sizeof(T) * m * p, cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    checkCuda(cudaFree(dA));
    checkCuda(cudaFree(dB));
    checkCuda(cudaFree(dC));
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));

    // Return average runtime per test (milliseconds)
    return time_ms / static_cast<float>(num_tests);
}
