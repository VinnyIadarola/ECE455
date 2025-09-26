#include <iostream>
#include <thread>
#include <vector>


void helloThread(int id, int N) {
    printf("Hello from thread %i of %i\n", id, N);
}

int main() {
    const int N = 5;
    std::vector<std::thread> threads(N);
    

    for (int i = 0; i < N; ++i) {
        threads[i] = std::thread(helloThread, i, N);
    }


    for (std::thread &t : threads) t.join();

    return 0;
}

