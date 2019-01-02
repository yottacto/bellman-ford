#include <iostream>
#include "bellman-ford.hh"

int main()
{
    icesp::bellman_ford bf("../dataset/USA-road-d.NY.gr");
    bf.print();
    // bf.test();
    std::cout << bf.compute(0, 1) << "\n";
}

