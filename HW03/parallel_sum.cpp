#include <iostream>
#include <vector>
#include <omp.h>




int main() {
    // Array size
    const int n = 1000000;


    // Reserve the spaces
    std::vector<int> arr(n);

    // Generate random elements
    #pragma omp parallel for
    for(int i = 0; i < n; ++i)
       arr[i] = rand() % 100;

    int sum = 0;


    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; ++i)
        sum = arr[i];



    printf("Sum = %i", sum);
    return 0;
}


