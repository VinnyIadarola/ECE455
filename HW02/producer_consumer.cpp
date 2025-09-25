#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

constexpr int MAX_ITEMS = 10;
std::queue q;
std::mutex mtx;
std::condition_variable cv;
bool data_ready = false;

void producer() {
    
    for(int i = 0; i < 100; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] () -> int {return (int)q.size < MAX_ITEMS;});
        q.push(i);
        printf("Produced: %i\n", i);
        lock.unlock();
        cv.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        data_ready = true;
    }

    cv.notify_one();
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !q.empty() || data_ready; });
        if (q.empty() && data_ready) break;
        int item = q.front(); q.pop();
        lock.unlock();
        std::cout << "Consumed: " << item << "\n";
        cv.notify_one();
    }
}


int main() {
    std::thread consumer_thread(consumer);
    std::thread producer_thread(producer);
    consumer_thread.join();
    producer_thread.join();
    return 0;
}