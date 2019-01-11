#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <queue>
#include <unordered_set>
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
        {
            if (!rank)
                std::cerr << "reading graph partition.\n";
            auto fin = std::ifstream{path + ".metis.part.8"};
            read_partition(fin);
        }
        {
            if (!rank)
                std::cerr << "reading graph.\n";
            auto fin = std::ifstream{path};
            read_graph(fin);
        }
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
    void read_partition(std::istream& is)
    {
        for (int u = 0, part; is >> part; u++)
            if (part == rank)
                nodes.emplace(u);
        std::cerr << rank << " has " << nodes.size() << " nodes.\n";
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
                g.resize(n);
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
            if (nodes.count(u)) {
                edges.emplace_back(u, v, w);
                g[u].emplace_back(u, v, w);
            }
        }
    }

    auto compute(int s, int t, bool info = false)
    {
        if (!rank)
            std::cerr << "computing.\n";
        std::vector<int> dist(n, edge::max());
        dist[s] = 0;
        auto iter = 0;
        for (auto relaxed = 0; ; iter++) {
            if (!rank && info)
                std::cerr << "iterating on " << iter << " relaxed=" << relaxed << "\n";
            relaxed = 0;
            std::queue<int> q;
            std::unordered_set<int> inqueue;
            for (auto u : nodes) {
                if (dist[u] == edge::max())
                    continue;
                inqueue.emplace(u);
                q.emplace(u);
            }
            while (!q.empty()) {
                auto u = q.front();
                q.pop();
                // if (!rank) std::cerr << "relaxing " << u << " " << dist[u] << " " << q.size() << "\n";
                for (auto const& e : g[u]) {
                    if (dist[e.from] != edge::max() && dist[e.from] + e.cost < dist[e.to]) {
                        dist[e.to] = dist[e.from] + e.cost;
                        relaxed++;
                        if (nodes.count(e.to) && !inqueue.count(e.to)) {
                            q.emplace(e.to);
                            inqueue.emplace(e.to);
                        }
                    }
                }
                inqueue.erase(u);
            }
            if (!rank)
                std::cerr << "before allreduce dist.\n";
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &relaxed, 1, MPI::INT, MPI::SUM);
            if (!relaxed) {
                std::cerr << "--> " << rank << " stop relaxing at " << iter << "\n";
                break;
            }

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
    std::vector<std::vector<edge>> g;
    int start;
    int end;
    int block_size;

    // nodes belong to this rank
    std::unordered_set<int> nodes;
};

} // namespace icesp

