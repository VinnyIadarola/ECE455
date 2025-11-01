#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_DIM 256

__global__ void reduce_shared(const int* in, int* out, size_t n_elems) {
    __shared__ int sdata[BLOCK_DIM];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x = (idx < n_elems) ? in[idx] : 0;
    sdata[tid] = x;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

static void checkCuda(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const size_t N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    int *h_in = (int*)malloc(bytes);
    for (size_t i = 0; i < N; ++i) h_in[i] = 1; // expected sum = N

    int *d_in = nullptr, *d_out = nullptr;
    checkCuda(cudaMalloc(&d_in, bytes));
    checkCuda(cudaMalloc(&d_out, sizeof(int)));
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_out, 0, sizeof(int)));

    dim3 block(BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM);

    reduce_shared<<<grid, block>>>(d_in, d_out, N);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    int h_out = 0;
    checkCuda(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    printf("device sum = %d, expected = %zu\n", h_out, N);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}