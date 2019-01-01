#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <mpi.h>

namespace icesp
{

struct edge
{
    edge(int from, int to, int cost) : from(from), to(to), cost(cost) {}

    int from;
    int to;
    int cost;
};

struct bellman_ford
{

    bellman_ford(std::string const& path) : path(path)
    {
        if (!MPI::Is_initialized())
            MPI::Init();
        rank = MPI::COMM_WORLD.Get_rank();
        auto fin = std::ifstream{path};
        read_graph(fin);
    }

    ~bellman_ford()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    // TODO pass stream in parameter?
    void read_graph(std::istream& is)
    {
        if (rank) return;
        for (char ch; is >> ch; ) {
            std::string buf;
            if (ch == 'c') {
                std::getline(is, buf);
                continue;
            }
            if (ch == 'p') {
                is >> buf >> n >> m;
                continue;
            }
            int u, v, w;
            is >> u >> v >> w;
            edges.emplace_back(u, v, w);
        }
        std::sort(std::begin(edges), std::end(edges), [](auto const& a, auto const& b) {
            return a.from  < b.from
                || (a.from == b.from && a.to < b.to);
        });
        transfer_graph();
    }

    // distribute graph from rank=0 to all others
    void transfer_graph()
    {
    }

    void print()
    {
        std::cout << "Hi from rank=" << rank << ", path=" << path << "\n";
        if (rank == 0) {
            std::cout << "n=" << n << " m=" << m << "\n";
            std::cout << "edges.size()=" << edges.size() << "\n";
        }
    }

    // graph data path
    std::string path;
    int rank;
    // total number of nodes
    int n;
    // total number of edges
    int m;
    std::vector<edge> edges;
};


} // namespace icesp

