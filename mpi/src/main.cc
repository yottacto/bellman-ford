#include "bellman-ford.hh"

int main()
{
    icesp::bellman_ford bf("../dataset/USA-road-d.NY.gr");
    bf.print();
}

