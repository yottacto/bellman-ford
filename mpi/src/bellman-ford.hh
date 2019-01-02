#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <mpi.h>

namespace icesp
{

struct edge
{
    edge(int from, int to, int cost) : from(from), to(to), cost(cost) {}

    static auto constexpr max() noexcept
    {
        return std::numeric_limits<int>::max();
    }

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
        size = MPI::COMM_WORLD.Get_size();
        auto fin = std::ifstream{path};
        read_graph(fin);
    }

    ~bellman_ford()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    template <class T>
    auto inrange(T x, T l, T r)
    {
        return !(x < l) && x < r;
    }

    // TODO pass stream in parameter?
    void read_graph(std::istream& is)
    {
        for (char ch; is >> ch; ) {
            std::string buf;
            if (ch == 'c') {
                std::getline(is, buf);
                continue;
            }
            if (ch == 'p') {
                is >> buf >> n >> m;
                // TODO
                block_size = n / size;
                auto extra = n % size;
                start = rank       * block_size + std::min(rank,     extra);
                end   = (rank + 1) * block_size + std::min(rank + 1, extra);
                block_size = end - start;
                continue;
            }
            int u, v, w;
            is >> u >> v >> w;
            u--; v--;
            if (inrange(u, start, end))
                edges.emplace_back(u, v, w);
        }
        std::sort(std::begin(edges), std::end(edges), [](auto const& a, auto const& b) {
            return a.from  < b.from
                || (a.from == b.from && a.to < b.to);
        });
    }

    auto compute(int s, int t, bool info = false)
    {
        std::vector<int> dist(n, edge::max());
        dist[s] = 0;
        auto iter = 0;
        for (auto relaxed = 0; ; iter++) {
            if (!rank && info)
                std::cerr << "iterating on " << iter << " relaxed=" << relaxed << "\n";
            relaxed = 0;
            for (auto const& e : edges) {
                if (dist[e.from] != edge::max() && dist[e.from] + e.cost < dist[e.to]) {
                    dist[e.to] = dist[e.from] + e.cost;
                    relaxed++;
                }
            }
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &relaxed, 1, MPI::INT, MPI::SUM);
            if (!relaxed) break;

            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, dist.data(), n, MPI::INT, MPI::MIN);
        }
        if (!rank && info)
            std::cout << "distance from [" << s << "] to [" << t << "] is "
                << dist[t] << "\n";
        return dist[t];
    }

    void print(bool all = false)
    {
        int tot_size = edges.size();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &tot_size, 1, MPI::INT, MPI::SUM);
        if (all || !rank) {
            std::cout << "Hi from rank=" << rank << ", start=" << start
                << ", end=" << end << "\n";
            std::cout << "n=" << n << " m=" << m << " "
                << "edges.size()=" << edges.size() << "\n";
            std::cout << "total=" << tot_size << "\n";
        }
    }

    // graph data path
    std::string path;
    int rank;
    int size;
    // total number of nodes
    int n;
    // total number of edges
    int m;
    std::vector<edge> edges;
    int start;
    int end;
    int block_size;
};

} // namespace icesp

