#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Split workload in half
    int half = N / 2;
    size_t half_size = size / 2;

    // Async copy first half
    cudaMemcpyAsync(d_A, A, half_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, B, half_size, cudaMemcpyHostToDevice, streamP1);

    // Async copy second half
    cudaMemcpyAsync(d_A + half, A + half, half_size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B + half, B + half, half_size, cudaMemcpyHostToDevice, stream2);

    // Kernel launch parameters
    int threads = 256;
    int blocks_half = (half + threads - 1) / threads;

    // Launch kernel on both streams
    vector_add<<<blocks_half, threads, 0, stream1>>>(d_A, d_B, d_C, half);
    vector_add<<<blocks_half, threads, 0, stream2>>>(d_A + half, d_B + half, d_C + half, half);

    // Async copy result back to host
    cudaMemcpyAsync(C, d_C, half_size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(C + half, d_C + half, half_size, cudaMemcpyDeviceToHost, stream2);

    // Sync streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Output results
    printf("C[0] = %f, C[N-1] = %f\n", C[0], C[N - 1]);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
