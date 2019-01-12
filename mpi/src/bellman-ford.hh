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

auto operator<(edge const& a, edge const& b)
{
    return a.cost > b.cost;
}

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

        recv_count = max_border_nodes_count + cross_edge_count;
        recv_buf.reserve(recv_count * size * 2);
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
    }

    // TODO pass stream in parameter?
    void read_graph(std::istream& is)
    {
        cross_edge_count = 0;
        for (char ch; is >> ch; ) {
            std::string buf;
            if (ch == 'c') {
                std::getline(is, buf);
                continue;
            }
            if (ch == 'p') {
                is >> buf >> n >> m;
                g.resize(n);
                continue;
            }
            int u, v, w;
            is >> u >> v >> w;
            u--; v--;
            if (nodes.count(u)) {
                edges.emplace_back(u, v, w);
                g[u].emplace_back(u, v, w);
                if (!nodes.count(v)) {
                    border_nodes.emplace(u);
                    cross_edge_count++;
                }
            }
        }
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &cross_edge_count, 1, MPI::INT, MPI::SUM);
        max_border_nodes_count = border_nodes.size();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &max_border_nodes_count, 1, MPI::INT, MPI::MAX);
        cross_edge_count /= 2;
        if (!rank)
            std::cerr << "cross edge: " << cross_edge_count <<
                ", max border nodes: " << max_border_nodes_count << "\n";
    }

    auto compute(int s, int t, bool info = false)
    {
        if (!rank)
            std::cerr << "computing.\n";
        std::vector<int> dist_now(n, edge::max());
        auto dist_pre = dist_now;

        std::priority_queue<edge> pq;
        if (nodes.count(s)) {
            dist_now[s] = dist_pre[s] = 0;
            pq.emplace(s, s, 0);
        }
        auto iter = 0;
        for (auto relaxed = 0; ; iter++) {
            if (!rank && info)
                std::cerr << "iterating on " << iter << ", relaxed=" << relaxed << "\n";
            relaxed = 0;
            while (!pq.empty()) {
                auto u = pq.top().to;
                auto dis = pq.top().cost;
                pq.pop();
                if (dist_now[u] < dis) continue;

                for (auto const& e : g[u]) {
                    auto v = e.to;
                    auto c = e.cost;
                    if (dist_now[u] != edge::max() && dist_now[v] > dist_now[u] + c) {
                        dist_now[v] = dist_now[u] + c;
                        pq.emplace(u, v, dist_now[v]);
                        relaxed++;
                    }
                }
            }

            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &relaxed, 1, MPI::INT, MPI::SUM);
            if (!relaxed)
                break;

            update_dist(dist_now, dist_pre);

            for (auto u : nodes)
                if (dist_now[u] != dist_pre[u])
                    pq.emplace(s, u, dist_now[u]);
            dist_pre = dist_now;
        }
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &dist_now[t], 1, MPI::INT, MPI::MIN);
        if (!rank && info)
            std::cout << "distance from [" << s << "] to [" << t << "] is "
                << dist_now[t] << "\n";
        return dist_now[t];
    }

    void update_dist(std::vector<int>& dist_now, std::vector<int> const& dist_pre)
    {
        recv_buf.clear();
        for (auto i = 0; i < n; i++) {
            if (dist_now[i] == dist_pre[i]
                    || (nodes.count(i) && !border_nodes.count(i)))
                continue;
            recv_buf.emplace_back(i);
            recv_buf.emplace_back(dist_now[i]);
        }

        recv_buf.resize(recv_count * 2 * size, -1);

        std::swap_ranges(
            std::begin(recv_buf),
            std::next(std::begin(recv_buf), recv_count * 2),
            std::next(std::begin(recv_buf), recv_count * 2 * rank)
        );

        MPI::COMM_WORLD.Allgather(
            MPI::IN_PLACE, 0, MPI::DATATYPE_NULL,
            recv_buf.data(), recv_count * 2, MPI::INT
        );

        auto updated = 0;
        for (auto i = 0u; i < recv_buf.size(); i += 2) {
            auto id = recv_buf[i];
            auto value = recv_buf[i + 1];
            if (id != -1) {
                dist_now[id] = std::min(dist_now[id], value);
                updated++;
            }
        }
    }

    void print(bool all = false)
    {
        int tot_size = edges.size();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &tot_size, 1, MPI::INT, MPI::SUM);
        if (all || !rank) {
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
    int block_size;

    // nodes belong to this rank
    std::unordered_set<int> nodes;
    std::unordered_set<int> border_nodes;
    int max_border_nodes_count;
    int cross_edge_count;

    int recv_count;
    std::vector<int> recv_buf;
};

} // namespace icesp

