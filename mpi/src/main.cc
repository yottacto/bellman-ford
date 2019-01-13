#include <iostream>
#include "sssp.hh"
#include "util.hh"

int main()
{
    icesp::sssp bf{"../dataset/USA-road-d.NY.gr"};
    // icesp::sssp bf{"../dataset/USA-road-d.USA.gr"};
    // icesp::sssp bf{"../dataset/USA-road-d.CAL.gr"};
    // bf.print();
    // duration(bf.rank, [&]() { bf.compute(0, 2000000); });
    icesp::duration(bf.rank, [&]() { bf.compute<true>(0, 200000); });
    bf.print_statistic();
}

