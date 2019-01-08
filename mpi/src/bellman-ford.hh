#pragma once
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <utility>
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
    auto locality_id(double l)
    {
        return std::distance(
            std::begin(pivots),
            std::lower_bound(
                std::begin(pivots),
                std::end(pivots),
                l
            )
        );
        // return static_cast<int>(l * size);
    }

    bellman_ford(std::string const& path) : path(path)
    {
        if (!MPI::Is_initialized())
            MPI::Init();
        rank = MPI::COMM_WORLD.Get_rank();
        size = MPI::COMM_WORLD.Get_size();
        calc_locality(20);

        split(loc);
        loc_id.reserve(n);
        for (auto l : loc)
            loc_id.emplace_back(locality_id(l));

        std::ifstream fin{path};
        read_graph(fin, [&](auto u, auto v, auto w) {
            if (loc_id[u] == rank)
                edges.emplace_back(u, v, w);
        });
    }

    ~bellman_ford()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    void split(std::vector<double> const& pre)
    {
        loc = pre;
        std::sort(std::begin(loc), std::end(loc));
        pivots.clear();
        pivots.reserve(size);
        auto block = loc.size() / size;
        // FIXME
        for (auto i = 1; i < size; i++)
            pivots.emplace_back(loc[i * block]);
        pivots.emplace_back(2 * loc.back());
    }

    auto cross(double l1, double l2)
    {
        return locality_id(l1) != locality_id(l2);
    }

    void calc_locality(int iter)
    {
        if (rank) {
            MPI::COMM_WORLD.Barrier();
            MPI::COMM_WORLD.Bcast(&n, 1, MPI::INT, 0);
            loc.resize(n);
            MPI::COMM_WORLD.Bcast(loc.data(), n, MPI::DOUBLE, 0);
            return;
        }
        std::ifstream fin{path};
        read_graph(fin, [&](auto u, auto v, auto) {
            if (graph.empty())
                graph.resize(n);
            graph[u].emplace_back(v);
        });

        decltype(loc) now(n);
        {
            std::random_device rd;
            std::mt19937 gen{rd()};
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::generate(std::begin(now), std::end(now), [&]() {
                return dis(gen);
            });
        }

        print_locality_distribution(now);
        auto pre = now;
        for (auto i = 0; i < iter; i++) {
            std::swap(now, pre);
            split(pre);
            auto count_cross_edge = 0;
            for (auto u = 0; u < n; u++) {
                // now[u] = 0;
                now[u] = pre[u];
                for (auto v : graph[u]) {
                    // now[u] += pre[v] / graph[v].size();
                    now[u] += pre[v];
                    if (cross(pre[v], pre[u]))
                        count_cross_edge++;
                }
                now[u] /= (graph[u].size() + 1);
            }
            std::cerr << "iter [" << i << "] "
                << "cross ["
                << std::left << std::setw(6)
                << count_cross_edge << "] ";
            print_locality_distribution(now);
        }
        // TODO do i need move?
        loc = std::move(now);

        // std::sort(std::begin(loc), std::end(loc));
        // for (auto l : loc)
        //     std::cerr << l << ", ";
        // std::cerr << "\n";

        MPI::COMM_WORLD.Barrier();
        MPI::COMM_WORLD.Bcast(&n, 1, MPI::INT, 0);
        MPI::COMM_WORLD.Bcast(loc.data(), n, MPI::DOUBLE, 0);
    }

    void print_locality_distribution(std::vector<double> const& loc)
    {
        std::vector<int> count(size + 1);
        for (auto l : loc)
            count[static_cast<int>(std::min(1., l) * size)]++;
        std::cerr << "count: [";
        for (auto i = 0; i <= size; i++)
            std::cerr << std::left << std::setw(7)
                << count[i] << ", ";
        std::cerr << "]\n";
    }

    template <class T>
    auto inrange(T x, T l, T r)
    {
        return !(x < l) && x < r;
    }

    // TODO pass stream in parameter?
    template <class InsertEdge>
    void read_graph(std::istream& is, InsertEdge insert)
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
            insert(u, v, w);
        }
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
    std::vector<double> loc;
    std::vector<int> loc_id;
    std::vector<double> pivots;
    std::vector<std::vector<int>> graph;
    std::vector<edge> edges;
    int start;
    int end;
    int block_size;
};

} // namespace icesp

