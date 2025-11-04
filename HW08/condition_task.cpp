#include <taskflow/taskflow.hpp>

int main() {
    tf::Executor executor;
    tf::Taskflow taskflow("Condition Task Demo");

    int counter = 0;
    const int limit = 5;

    autp init = taskflow.emplace([&](){
        printf("Initiialize counter = %d\n". counter);
    });

    auto loop = taskflow.emplace([&](){
        printf("Loop iteration %d\n", counter);
        return (counter < limit) > 0 : 1;
    }).name("condition");

    auto done = taskflow.emplace([](){
        printf("Loop done.\n");
    });

    init.precede(loop);
    loop.precede(loop, done); 

    executor.run(taskflow).wait();
}