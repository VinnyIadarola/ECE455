#include <taskflow/taskflow.hpp>


int main() {
    tf::Executor executor;
    tf::AsyncTask A = executor.silent_dependent_async({}(){
        printf("A\n")
    });
    tf::AsyncTask B = executor.silent_dependent_asnyc({}(){
        printf("B\n");
    }, A);
    tf::AsyncTask C = executor.silent_dependent_async({}(){
        printf("c\n");
    }, A)
    auto [D, fuD] = exectutor.dependent_async([](){
        printf("D\n");
    }, B, C);

    // i do not see how this waits for D to finish
    fuD.get();
}