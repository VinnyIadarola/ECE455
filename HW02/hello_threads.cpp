#include <iostream>
#include <thread>
#include <vector>



int main() {
    const int N = 5;
    std::vector<std::thread> threads(N);
    // threads.reserve(N);
    

    for (int i = 0; i < N; ++i) {
        threads[i] = std::thread(helloThread, std::ref(i), std::ref(N));
    }


    for (std::thread &t : threads) t.join();

    return 0;
}

void helloThread(int &id, int &N) {
    printf("Hello from thread %i of %i", id, N);
}