#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

// Include the kernel code here or as an external header
#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

template <typename T>
__global__ void square_shared_kernel(const T *in, T *out, size_t N) {
    __shared__ T tile[BLOCK_DIM];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // 1. Load from global to shared memory
    tile[threadIdx.x] = in[idx];
    __syncthreads();
    // 2. Compute in shared memory
    tile[threadIdx.x] = tile[threadIdx.x] * tile[threadIdx.x];
    __syncthreads();
    // 3. Write back to global memory
    out[idx] = tile[threadIdx.x];
}

int main() {
    using T = float;
    const size_t N = 1024;
    std::vector<T> h_in(N), h_out(N);

    // Initialize input
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<T>(i);

    // Allocate device memory
    T *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(T));
    cudaMalloc(&d_out, N * sizeof(T));

    // Copy input to device
    cudaMemcpy(d_in, h_in.data(), N * sizeof(T), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t numBlocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    square_shared_kernel<T><<<numBlocks, BLOCK_DIM>>>(d_in, d_out, N);

    // Copy result back to host
    cudaMemcpy(h_out.data(), d_out, N * sizeof(T), cudaMemcpyDeviceToHost);

    // Verify results
    for (size_t i = 0; i < N; ++i) {
        assert(h_out[i] == h_in[i] * h_in[i]);
    }
    std::cout << "All results are correct!" << std::endl;

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}