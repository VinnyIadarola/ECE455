#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

void multiply_block(
    const std::vector<std::vector<double>> &A,
    const std::vector<std::vector<double>> &B,
    std::vector<std::vector<double>> &C,
    int N, int row
) {
    for (int col = 0; col < N; ++col) {
        for (int k = 0; k < N; k++) {
            C[row][col] += A[row][k] * B[k][col];
        }
    }
}

int main() {
    const int N = 800;
    const int T = std::thread::hardware_concurrency() ?
         std::thread::hardware_concurrency() : 4;

    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    std::vector<std::vector<double>> B(N, std::vector<double>(N));
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto &row : A)
        for (auto &x : row)
            x = dist(rng);
    for (auto &row : B)
        for (auto &x : row)
            x = dist(rng);

    {
        std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < N; ++row) {
            multiply_block(A, B, C, N, row);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        printf("Single-threaded multiplication took %.6f s\n",
               std::chrono::duration<double>(end_time - start_time).count());
    }

    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));
    std::vector<std::thread> threads;
    int chunk = N / T;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < T; ++t) {
        int rs = t * chunk;
        int re = (t == T - 1) ? N : rs + chunk;
        threads.emplace_back([&A, &B, &C, N, rs, re]() {
            for (int row = rs; row < re; ++row) {
                multiply_block(A, B, C, N, row);
            }
        });
    }

    for (auto &th : threads)
        th.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    printf("Parallel multiplication took %.6f s\n",
           std::chrono::duration<double>(end_time - start_time).count());
}
