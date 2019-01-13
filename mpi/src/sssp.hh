#pragma once
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <mpi.h>
#include "util.hh"
#include "timer.hh"

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

struct sssp
{
    sssp(std::string const& path) : path(path)
    {
        if (!MPI::Is_initialized())
            MPI::Init();
        rank = MPI::COMM_WORLD.Get_rank();
        size = MPI::COMM_WORLD.Get_size();

        auto t{timer{}};
        t.restart();
        read_partition(path);
        t.stop();
        print("read partition elapsed ", t.elapsed_seconds(), "s\n");

        t.restart();
        read_graph(path);
        t.stop();
        print("read graph elapsed ", t.elapsed_seconds(), "s\n");

        recv_count = max_border_node_count + cross_edge_count;
        print("recv_count=",            recv_count,            ", ");
        print("max_border_node_count=", max_border_node_count, ", ");
        print("cross_edge_count=",      cross_edge_count,      "\n");
        recv_buf.reserve(recv_count * size * 2);
    }

    ~sssp()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    template <class T>
    auto inrange(T x, T l, T r)
    {
        return !(x < l) && x < r;
    }

    std::string binary_file_name(std::string const& path)
    {
        return path + ".binary." + std::to_string(size)
            + ".rank." + std::to_string(rank);
    }

    // TODO pass stream in parameter?
    void read_partition(std::string const& base_path)
    {
        auto path = base_path + ".metis.part.8";
        auto has_binary = std::ifstream{
            binary_file_name(path),
            std::ios::binary
        }.good();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &has_binary, 1, MPI::BOOL, MPI::LAND);

        if (has_binary) {
            print("reading binary graph partition file.\n");
            auto fin = std::ifstream{binary_file_name(path), std::ios::binary};
            int count;
            bin_read(fin, &count);
            for (int u; count--; ) {
                bin_read(fin, &u);
                nodes.emplace(u);
            }
        } else {
            print("reading normal graph partition file.\n");
            auto fin = std::ifstream{path};
            auto fout = std::ofstream{binary_file_name(path), std::ios::binary};
            for (int u = 0, part; fin >> part; u++)
                if (part == rank)
                    nodes.emplace(u);
            int count = nodes.size();
            bin_write(fout, &count);
            for (auto u : nodes)
                bin_write(fout, &u);
        }
    }

    // TODO pass stream in parameter?
    void read_graph(std::string const& path)
    {
        auto has_binary = std::ifstream{
            binary_file_name(path),
            std::ios::binary
        }.good();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &has_binary, 1, MPI::BOOL, MPI::LAND);

        if (has_binary) {
            print("reading binary graph file\n");
            auto fin = std::ifstream{binary_file_name(path), std::ios::binary};
            bin_read(fin, &n);
            bin_read(fin, &m);
            g.resize(n);
            print("n=", n, ", ");
            print("m=", m, "\n");
            int edge_count;
            bin_read(fin, &edge_count);
            for (int u, v, w; edge_count--; ) {
                bin_read(fin, &u);
                bin_read(fin, &v);
                bin_read(fin, &w);
                g[u].emplace_back(u, v, w);
                // dist_now[u] = dist_now[v] = edge::max();
            }

            int border_node_count;
            bin_read(fin, &border_node_count);
            for (int u; border_node_count--; ) {
                bin_read(fin, &u);
                border_nodes.emplace(u);
            }
            bin_read(fin, &max_border_node_count);
            bin_read(fin, &cross_edge_count);
        } else {
            print("reading normal graph file\n");
            auto fin = std::ifstream{path};
            auto fout = std::ofstream{binary_file_name(path), std::ios::binary};
            auto edge_count = 0;
            for (char ch; fin >> ch; ) {
                std::string buf;
                if (ch == 'c') {
                    std::getline(fin, buf);
                } else if (ch == 'p') {
                    fin >> buf >> n >> m;
                    bin_write(fout, &n);
                    bin_write(fout, &m);
                    g.resize(n);
                } else {
                    int u, v, w;
                    fin >> u >> v >> w;
                    u--; v--;
                    // dist_now[u] = dist_now[v] = edge::max();
                    if (nodes.count(u)) {
                        edge_count++;
                        g[u].emplace_back(u, v, w);
                        if (!nodes.count(v)) {
                            border_nodes.emplace(u);
                            cross_edge_count++;
                        }
                    }
                }
            }

            bin_write(fout, &edge_count);
            for (auto u = 0; u < n; u++)
                for (auto& e : g[u]) {
                    bin_write(fout, &e.from);
                    bin_write(fout, &e.to);
                    bin_write(fout, &e.cost);
                }
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &cross_edge_count, 1, MPI::INT, MPI::SUM);
            max_border_node_count = border_nodes.size();
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &max_border_node_count, 1, MPI::INT, MPI::MAX);
            cross_edge_count /= 2;

            int border_node_count = border_nodes.size();
            bin_write(fout, &border_node_count);
            for (auto u : border_nodes)
                bin_write(fout, &u);
            bin_write(fout, &max_border_node_count);
            bin_write(fout, &cross_edge_count);
        }

        MPI::COMM_WORLD.Barrier();
        print("cross_edge_count=",      cross_edge_count,      ", ");
        print("max_border_node_count=", max_border_node_count, "\n");

        // statistic
        last_updated.resize(n);
        dist_now.resize(n, edge::max());
        dist_pre = dist_now;

        MPI::COMM_WORLD.Barrier();
    }

    template <bool Enabled = false>
    auto compute(int s, int t)
    {
        print<Enabled>("computing.\n");

        std::priority_queue<edge> pq;
        if (nodes.count(s)) {
            dist_now[s] = dist_pre[s] = 0;
            pq.emplace(s, s, 0);
        }

        // statisitc
        updated.clear();
        iter = 0;
        recv_empty = false;
        for (; !recv_empty; iter++) {
            print<Enabled>("iterating on ", iter, ", ");

            auto t{timer{}};
            t.restart();
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
                    }
                }
            }
            t.stop();
            print<Enabled>("dij elapsed ", t.elapsed_seconds(), "s, ");

            t.restart();
            update_dist(dist_now, dist_pre);
            for (auto u : nodes)
                if (dist_now[u] != dist_pre[u]) {
                    pq.emplace(s, u, dist_now[u]);
                    dist_pre[u] = dist_now[u];
                }
            t.stop();
            print<Enabled>("update dist elapsed ", t.elapsed_seconds(), "s, ");

            print<Enabled>("\n");

            if (recv_empty)
                break;
        }

        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &dist_now[t], 1, MPI::INT, MPI::MIN);

        print<Enabled>("distance from [", s);
        print<Enabled>("] to [", t);
        print<Enabled>("] is ", dist_now[t], "\n");
        return dist_now[t];
    }

    template <class T>
    void update_dist(T& dist_now, T& dist_pre)
    {
        recv_buf.clear();

        // statistic
        auto updated_count{0};
        for (auto i = 0; i < n; i++) {
            if (dist_now.at(i) != dist_pre.at(i))
                updated_count++;
            if (dist_now.at(i) == dist_pre.at(i) || (nodes.count(i) && !border_nodes.count(i)))
                continue;
            recv_buf.emplace_back(i);
            recv_buf.emplace_back(dist_now[i]);
        }

        // statistic
        updated.emplace_back(size);
        updated.at(iter).at(rank) = updated_count;

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

        recv_empty = true;
        for (auto i = 0u; i < recv_buf.size(); i += 2) {
            auto id = recv_buf[i];
            auto value = recv_buf[i + 1];
            if (id != -1) {
                recv_empty = false;
                if (!nodes.count(id))
                    dist_pre.at(id) = std::min(dist_pre.at(id), value);
                if (value < dist_now.at(id)) {
                    dist_now.at(id) = value;
                    // statistic
                    last_updated.at(id) = iter;
                }
            }
        }
    }

    template <bool Enabled = true, class T, class U = std::string, class V = std::string>
    void print(T const& x, U const& y = std::string{}, V const& z = std::string{}, bool all = false)
    {
        ::icesp::print<Enabled>(rank, x, y, z, all);
    }

    void summary(bool all = false)
    {
        if (all || !rank) {
            std::cout << "n=" << n << " m=" << m << "\n";
        }
    }

    void print_statistic()
    {
        if (!rank)
            std::cerr << "\nupdaed per iter\n";
        for (auto i = 0; i < iter; i++) {
            MPI::COMM_WORLD.Allgather(
                MPI::IN_PLACE, 0, MPI::DATATYPE_NULL,
                updated[i].data(), 1, MPI::INT
            );
            if (!rank) {
                std::cerr << "iter " << i << ": ";
                for (auto u : updated[i])
                    std::cerr << std::setw(7) << u << " ";
                std::cerr << "\n";
            }
        }

        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, last_updated.data(), n, MPI::INT, MPI::MAX);
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, dist_now.data(), n, MPI::INT, MPI::MIN);
        if (!rank) {
            std::cerr << "min/max distance per iter\n";
            for (auto i = 0; i < iter; i++) {
                auto min = edge::max();
                auto max = 0;
                for (auto u = 0; u < n; u++)
                    if (last_updated[u] == i) {
                        max = std::max(max, dist_now[u]);
                        min = std::min(min, dist_now[u]);
                    }
                std::cerr << "iter " << i << ":"
                    << " min=" << std::setw(7) << min
                    << " max=" << std::setw(7) << max
                    << "\n";
            }

            auto xor_sum = 0;
            auto unreachable = 0;
            for (auto i = 0; i < n; i++)
                if (dist_now[i] == edge::max())
                    unreachable++;
                else
                    xor_sum ^= dist_now[i];
            std::cerr << "\nunreachable nodes: " << unreachable << "\n";
            std::cerr << "all distance xor: " << xor_sum << "\n\n";
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
    std::vector<std::vector<edge>> g;
    // std::unordered_map<int, std::vector<edge>> g;

    // nodes belong to this rank
    std::unordered_set<int> nodes;
    std::unordered_set<int> border_nodes;
    int max_border_node_count;
    int cross_edge_count{0};

    std::vector<int> dist_now;
    std::vector<int> dist_pre;
    // std::unordered_map<int, int> dist_now;
    // std::unordered_map<int, int> dist_pre;

    int recv_count;
    std::vector<int> recv_buf;
    bool recv_empty;

    // statistic
    int iter;
    std::vector<std::vector<int>> updated;
    std::vector<int> last_updated;
};

} // namespace icesp

