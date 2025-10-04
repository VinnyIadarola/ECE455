#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>

const int ITER = 100000;

void inc_no_lock(int &count){
    for (int i = 0; i < ITER; ++i)
        count++;                 // data race (intended for demo)
}

void inc_mutex(int &count, std::mutex &countLock) {  // <-- pass by reference
    for (int i = 0; i < ITER; ++i) {
        std::lock_guard<std::mutex> lk(countLock);
        ++count;
    }
}

void inc_atomic(std::atomic<int> &count) {
    for (int i = 0; i < ITER; ++i)
        count.fetch_add(1, std::memory_order_relaxed);
}

template <typename F>
double run_and_time(int T, F&& fn) {                 // <-- return double
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> ths;
    ths.reserve(T);
    for (int i = 0; i < T; ++i) ths.emplace_back(fn);
    for (auto& t : ths) t.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const int T = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    const int expected = T * ITER;

    { // No lock (incorrect)
        int counter = 0;
        double ms = run_and_time(T, [&]{ inc_no_lock(counter); });
        std::cout << "[No lock] counter = " << counter
                  << " (expected " << expected << "), "
                  << ms << " ms\n";
    }

    { // Mutex
        int counter = 0;
        std::mutex m;
        double ms = run_and_time(T, [&]{ inc_mutex(counter, m); }); // same mutex shared by all threads
        std::cout << "[Mutex] counter = " << counter
                  << " (expected " << expected << "), "
                  << ms << " ms\n";
    }

    { // Atomic
        std::atomic<int> counter{0};
        double ms = run_and_time(T, [&]{ inc_atomic(counter); });
        std::cout << "[Atomic] counter = " << counter.load()
                  << " (expected " << expected << "), "
                  << ms << " ms\n";
    }
}
