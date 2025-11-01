#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__device__ inline int warp_reduce_sum(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    return val;
}

__global__ void reduce_warp(const int* in, int* out, size_t num_elems) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < num_elems) ? in[idx] : 0;

    val = warp_reduce_sum(val);

    // one atomic add per warp (lane 0 of each warp)
    if ((threadIdx.x & 31) == 0)
        atomicAdd(out, val);
}

int main() {
    const size_t N = 1 << 20; // 1M elements
    const int threads = 128;  // multiple of 32
    const int blocks = (N + threads - 1) / threads;

    // host allocation and init
    int *h_in = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; ++i) h_in[i] = 1; // simple data: all ones

    // device allocation
    int *d_in = nullptr;
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));

    // launch
    reduce_warp<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back
    int h_out = 0;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // verify on host
    long long cpu_sum = 0;
    for (size_t i = 0; i < N; ++i) cpu_sum += h_in[i];

    printf("GPU sum = %d, CPU sum = %lld\n", h_out, cpu_sum);

    // cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);

    return 0;
}