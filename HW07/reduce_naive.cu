#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <cuda_runtime.h>

static void checkCuda(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// corrected kernel (fixes blockDim.x and missing semicolon)
__global__ void reduce_naive(const int* in, int* out, size_t n_elems) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elems)
        atomicAdd(out, in[idx]);
}

int main(int argc, char** argv) {
    size_t n = 1 << 20; // default 1M elements
    if (argc > 1) n = strtoull(argv[1], nullptr, 0);

    // allocate + init host
    int *h_in = (int*)malloc(n * sizeof(int));
    for (size_t i = 0; i < n; ++i) h_in[i] = 1; // simple values for easy check
    long long host_sum = std::accumulate(h_in, h_in + n, 0LL);

    // device allocations
    int *d_in = nullptr;
    int *d_out = nullptr;
    checkCuda(cudaMalloc(&d_in, n * sizeof(int)), "cudaMalloc d_in");
    checkCuda(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc d_out");

    // copy input and zero output
    checkCuda(cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_in");
    checkCuda(cudaMemset(d_out, 0, sizeof(int)), "cudaMemset d_out");

    // launch kernel
    const int blockSize = 256;
    const int gridSize = (int)((n + blockSize - 1) / blockSize);
    reduce_naive<<<gridSize, blockSize>>>(d_in, d_out, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // copy result back
    int gpu_result = 0;
    checkCuda(cudaMemcpy(&gpu_result, d_out, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H d_out");

    printf("Host sum = %lld\n", host_sum);
    printf("GPU  sum = %d\n", gpu_result);

    // cleanup
    checkCuda(cudaFree(d_in), "cudaFree d_in");
    checkCuda(cudaFree(d_out), "cudaFree d_out");
    free(h_in);

    return (long long)gpu_result == host_sum ? 0 : 1;
}
