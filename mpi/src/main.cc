#include <iostream>
#include <chrono>
#include "bellman-ford.hh"

template <class Func>
void duration(int rank, Func f)
{
    using namespace std::chrono;

    MPI::COMM_WORLD.Barrier();
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();

    auto elapsed = duration_cast<milliseconds>(end - start).count() / 1000.;
    MPI::COMM_WORLD.Reduce(
        !rank ? MPI::IN_PLACE : &elapsed,
        &elapsed,
        1,
        MPI::DOUBLE,
        MPI::MAX,
        0
    );
    if (!rank)
        std::cerr << "Time elapsed: " << elapsed << "s.\n";
}

int main()
{
    icesp::bellman_ford bf("../dataset/USA-road-d.NY.gr");
    // icesp::bellman_ford bf("../dataset/USA-road-d.USA.gr");
    // icesp::bellman_ford bf("../dataset/USA-road-d.CAL.gr");
    // bf.print();
    // duration(bf.rank, [&]() { bf.compute(0, 2000000, true); });
    duration(bf.rank, [&]() { bf.compute(0, 200000, true); });
}

