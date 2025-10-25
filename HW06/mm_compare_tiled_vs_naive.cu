#define TILE_SIZE 16

template <typename T>
__global__ void mm_tiled(const T* A, const T* B, T* C, int N) {
    __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    T val = 0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = val;
}



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



#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define N 1024

template <typename T>
__global__ void mm_naive(const T* A, const T* B, T* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        T val = 0;
        for (int k = 0; k < N; ++k)
            val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

int main() {
    using T = float;
    size_t bytes = N * N * sizeof(T);

    T *h_A = (T*)malloc(bytes);
    T *h_B = (T*)malloc(bytes);
    T *h_C_naive = (T*)malloc(bytes);
    T *h_C_tiled = (T*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<T>(rand()) / RAND_MAX;
        h_B[i] = static_cast<T>(rand()) / RAND_MAX;
    }

    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(start);
    mm_naive<T><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost);

    // Tiled
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(start);
    mm_tiled<T><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled = 0;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Naive kernel time: %.3f ms\n", ms_naive);
    printf("Tiled kernel time: %.3f ms\n", ms_tiled);

    // Optional: correctness check
    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C_naive[i] - h_C_tiled[i]) > 1e-3) {
            errors++;
            if (errors < 10)
                printf("Mismatch at %d: naive=%f, tiled=%f\n", i, h_C_naive[i], h_C_tiled[i]);
        }
    }
    if (errors == 0)
        printf("Results match.\n");
    else
        printf("Total mismatches: %d\n", errors);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}