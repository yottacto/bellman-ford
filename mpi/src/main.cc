#include <iostream>
#include "sssp.hh"
#include "util.hh"
#include "config.hh"

int main()
{
    auto config{icesp::config{}};
    auto path = config.front().path;
    auto s = config.front().s;
    auto t = config.front().t;

    icesp::sssp bf{path};
    bf.compute<true>(s, t);
    bf.print_statistic();
}

