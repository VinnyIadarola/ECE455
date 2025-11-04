#include <taskflow/taskflow.hpp>

int main() {
    tf::Taskflow tf;
    tf::Executor executor;
    auto [A, B, C, D] = taskflow.emplace(
        [] () {std::cout << "TaskA\n";}
        [] () {std::cout << "TaskB\n";}
        [] () {std::cout << "TaskC\n";}
        [] () {std::cout << "TaskD\n";}
    );
    A.precede(B, C);
    B.precede(D);
    C.precede(D);

    executor.run(raskflow).wait();
    return 0;
}