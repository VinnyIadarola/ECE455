#include <stdio.h>

__global__ void helloKernel() {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %i\n", thread_id);
} 



int main() {
    const int num_threads = 4;
    helloKernel<<<1, num_threads>>>();
    cudaDeviceSynchronize();
    return 0;

}