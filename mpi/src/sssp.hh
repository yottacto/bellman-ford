#pragma once
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <queue>
#include <mpi.h>
#include "util.hh"
#include "timer.hh"

namespace icesp
{

struct node
{
    bool owned{};
    bool boundary{};
    bool interior{};
    bool updated{};
    int iter{};
    int index_in_recv_buf{};
};

struct edge
{
    edge(int to, int cost) : to(to), cost(cost) {}

    static auto constexpr max() noexcept
    {
        return std::numeric_limits<int>::max();
    }

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

        read_partition(path);
        read_graph(path);

        recv_count = 2 * (max_border_node_count + cross_edge_count);
        print("recv_count=",            recv_count,            ", ");
        print("max_border_node_count=", max_border_node_count, ", ");
        print("cross_edge_count=",      cross_edge_count,      "\n");
        recv_buf.reserve(recv_count * size);
    }

    ~sssp()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    void update_dist()
    {
        recv_buf.resize(recv_count * size, -1);

        std::swap_ranges(
            std::begin(recv_buf),
            std::next(std::begin(recv_buf), recv_count),
            std::next(std::begin(recv_buf), recv_count * rank)
        );

        compute_timer.stop();
        comm_timer.restart();

        MPI::COMM_WORLD.Allgather(
            MPI::IN_PLACE, 0, MPI::DATATYPE_NULL,
            recv_buf.data(), recv_count, MPI::INT
        );

        recv_empty = true;
        for (auto i = 0u; i < recv_buf.size(); i += 2) {
            auto u = recv_buf[i];
            auto value = recv_buf[i + 1];
            if (u != -1) {
                recv_empty = false;
                if (value < dist.at(u)) {
                    dist.at(u) = value;
                    if (info[u].boundary)
                        info[u].updated = true;
                    // statistic
                    last_updated.at(u) = iter;
                }
            }
        }
    }

    void compute_core()
    {
        recv_buf.clear();
        auto len = 0;
        while (!pq.empty()) {
            auto u = pq.top().to;
            auto dis = pq.top().cost;
            pq.pop();
            if (dist[u] < dis)
                continue;

            for (auto const& e : g[u]) {
                auto v = e.to;
                auto c = e.cost;
                if (dist[v] <= dist[u] + c)
                    continue;
                dist[v] = dist[u] + c;
                pq.emplace(v, dist[v]);

                if (!info[v].interior) {
                    if (info[v].iter == iter + 1) {
                        recv_buf[info[v].index_in_recv_buf + 1] = dist[v];
                        if (info[v].boundary)
                            info[v].updated = true;
                    } else {
                        auto p = len;
                        info[v].iter = iter + 1;
                        info[v].index_in_recv_buf =  p;
                        updated_count++;
                        recv_buf.emplace_back(v);
                        recv_buf.emplace_back(dist[v]);
                        len += 2;
                    }
                } else {
                    if (info[v].iter != iter + 1) {
                        info[v].iter = iter + 1;
                        updated_count++;
                    }
                }
            }
        }

        update_dist();

        if (recv_empty)
            return;
        for (auto u : nodes)
            if (info[u].updated) {
                pq.emplace(u, dist[u]);
                info[u].updated = false;
            }
    }

    template <bool Enabled = false>
    auto compute(int s, int t)
    {
        print<Enabled>("computing.\n");

        // statisitc
        updated.clear();
        total_compute = total_comm = 0.;

        total_timer.restart();
        dist.clear();
        dist.resize(n, edge::max());
        for (auto& i : info)
            i.iter = i.updated = false;

        // pq = {}; // server gcc 5.5.0 dont support
        if (info[s].owned) {
            dist[s] = 0;
            pq.emplace(s, 0);
        }

        iter = 0;
        recv_empty = false;
        for (; !recv_empty; iter++) {
            print<Enabled>("iterating on ", iter, ", \n");
            updated_count = 0;

            total_timer.start();
            compute_timer.restart();

            compute_core();

            comm_timer.stop();
            total_timer.stop();


            // statistic
            updated.emplace_back(size);
            updated.at(iter).at(rank) = updated_count;
            auto comm_elapsed = comm_timer.elapsed_seconds();
            auto compute_elapsed = compute_timer.elapsed_seconds();
            total_comm += comm_elapsed;
            total_compute += compute_elapsed;

            if (Enabled) {
                for (auto i = 0; i < size; i++) {
                    if (i == rank)
                        std::cerr << "rank: " << rank
                            << ", compute " << std::setw(5) << compute_elapsed
                            << ", comm " << std::setw(5) << comm_elapsed
                            << std::endl;
                    MPI::COMM_WORLD.Barrier();
                }
            }

            print<Enabled>("\n");
        }

        total_timer.stop();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &dist[t], 1, MPI::INT, MPI::MIN);

        print<Enabled>("distance from [", s);
        print<Enabled>("] to [", t);
        print<Enabled>("] is ", dist[t], "\n");

        print<Enabled>("total compute elapsed ", total_compute, ", ");
        print<Enabled>("total comm elapsed ", total_comm, "\n");
        print<Enabled>("total time elapsed ", total_timer.elapsed_seconds(), "\n");
        return dist[t];
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

    void read_partition(std::string const& base_path)
    {
        auto t{timer{}};
        t.restart();

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
            nodes.reserve(count);
            for (int u; count--; ) {
                bin_read(fin, &u);
                nodes.emplace_back(u);
            }
        } else {
            print("reading normal graph partition file.\n");
            auto fin = std::ifstream{path};
            auto fout = std::ofstream{binary_file_name(path), std::ios::binary};
            for (int u = 0, part; fin >> part; u++)
                if (part == rank)
                    nodes.emplace_back(u);
            int count = nodes.size();
            bin_write(fout, &count);
            for (auto u : nodes)
                bin_write(fout, &u);
        }

        t.stop();
        print("read partition elapsed ", t.elapsed_seconds(), "s\n");
    }

    void read_graph(std::string const& path)
    {
        auto t{timer{}};
        t.restart();
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
            info.resize(n);
            for (auto u : nodes)
                info[u].owned = true;
            print("n=", n, ", ");
            print("m=", m, "\n");
            int edge_count;
            bin_read(fin, &edge_count);
            for (int u, v, w; edge_count--; ) {
                bin_read(fin, &u);
                bin_read(fin, &v);
                bin_read(fin, &w);
                g[u].emplace_back(v, w);
            }

            int border_node_count;
            bin_read(fin, &border_node_count);
            for (int u; border_node_count--; ) {
                bin_read(fin, &u);
                border_nodes.emplace_back(u);
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
                    info.resize(n);
                    for (auto u : nodes)
                        info[u].owned = true;
                } else {
                    int u, v, w;
                    fin >> u >> v >> w;
                    u--; v--;
                    if (info[u].owned) {
                        edge_count++;
                        g[u].emplace_back(v, w);
                        if (!info[v].owned) {
                            border_nodes.emplace_back(u);
                            cross_edge_count++;
                        }
                    }
                }
            }

            bin_write(fout, &edge_count);
            for (auto u = 0; u < n; u++)
                for (auto& e : g[u]) {
                    bin_write(fout, &u);
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

        // statistic
        last_updated.resize(n);
        dist.resize(n, edge::max());

        for (auto u : border_nodes)
            info[u].boundary = true;
        for (auto& i : info)
            i.interior = i.owned && !i.boundary;

        MPI::COMM_WORLD.Barrier();

        t.stop();
        print("read graph elapsed ", t.elapsed_seconds(), "s\n");
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

        // MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, last_updated.data(), n, MPI::INT, MPI::MAX);
        // if (!rank) {
        //     std::cerr << "min/max distance per iter\n";
        //     for (auto i = 0; i < iter; i++) {
        //         auto min = edge::max();
        //         auto max = 0;
        //         for (auto u = 0; u < n; u++)
        //             if (last_updated[u] == i) {
        //                 max = std::max(max, dist[u]);
        //                 min = std::min(min, dist[u]);
        //             }
        //         std::cerr << "iter " << i << ":"
        //             << " min=" << std::setw(7) << min
        //             << " max=" << std::setw(7) << max
        //             << "\n";
        //     }
        // }

        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, dist.data(), n, MPI::INT, MPI::MIN);
        if (!rank) {
            auto xor_sum = 0;
            auto unreachable = 0;
            for (auto i = 0; i < n; i++)
                if (dist[i] == edge::max())
                    unreachable++;
                else
                    xor_sum ^= dist[i];
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
    std::vector<node> info;

    // nodes belong to this rank
    std::vector<int> nodes;
    std::vector<int> border_nodes;
    int max_border_node_count;
    int cross_edge_count{0};

    std::priority_queue<edge> pq;
    std::vector<int> dist;

    int recv_count;
    std::vector<int> recv_buf;
    bool recv_empty;

    // statistic
    int iter;
    std::vector<std::vector<int>> updated;
    std::vector<int> last_updated;
    double total_compute;
    double total_comm;
    timer compute_timer;
    timer comm_timer;
    timer total_timer;
    int updated_count;
};

} // namespace icesp

