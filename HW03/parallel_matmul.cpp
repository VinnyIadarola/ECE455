#include <iostream>
#include <vector>

int main() {
    const int N = 1024;
    std::vector<std::vector<int>> arr1(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> arr2(N, std::vector<int>(N, 2));
    std::vector<std::vector<int>> result(N, std::vector<int>(N, 0));

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += arr1[i][k] * arr2[k][j];
            }
            result[i][j] = sum;
        }
    }


    printf("result[0][0] = %d\n", result[0][0]);
    return 0;
}