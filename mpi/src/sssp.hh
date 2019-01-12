#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <mpi.h>

#include <chrono>

namespace icesp
{

template <class T>
void bin_read(std::istream& i, T* x)
{
    i.read(reinterpret_cast<char*>(x), sizeof(*x));
    if (!i) throw std::runtime_error{"binary read failed"};
}

template <class T>
void bin_write(std::ostream& o, T* x)
{
    o.write(reinterpret_cast<char*>(x), sizeof(*x));
    if (!o) throw std::runtime_error{"binary write failed"};
}

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

        {
            using namespace std::chrono;
            auto start = high_resolution_clock::now();
            read_partition(path);
            auto end = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(end - start).count() / 1000.;
            if (!rank)
                std::cerr << "read partition elapsed <" << elapsed << "s>\n";
        }

        {
            using namespace std::chrono;
            auto start = high_resolution_clock::now();
            read_graph(path);
            auto end = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(end - start).count() / 1000.;
            if (!rank)
                std::cerr << "read graph elapsed <" << elapsed << "s>\n";
        }

        recv_count = max_border_node_count + cross_edge_count;
        if (!rank)
            std::cerr << "recv_count=" << recv_count
                << ", max_border_node_count=" << max_border_node_count
                << ", cross_edge_count=" << cross_edge_count
                << "\n";
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
            if (!rank)
                std::cerr << "reading binary graph partition file.\n";
            auto fin = std::ifstream{binary_file_name(path), std::ios::binary};
            int count;
            bin_read(fin, &count);
            for (int u; count--; ) {
                bin_read(fin, &u);
                nodes.emplace(u);
            }
        } else {
            if (!rank)
                std::cerr << "reading normal graph partition file.\n";
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
            if (!rank)
                std::cerr << "reading binary graph file\n";
            auto fin = std::ifstream{binary_file_name(path), std::ios::binary};
            bin_read(fin, &n);
            bin_read(fin, &m);
            g.resize(n);
            if (!rank)
                std::cerr << "n=" << n << ", m=" << m << "\n";
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
            if (!rank)
                std::cerr << "reading normal graph file\n";
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
            if (!rank)
                std::cerr << "writing edges\n";
            bin_write(fout, &edge_count);
            for (auto u = 0; u < n; u++)
                for (auto& e : g[u]) {
                    bin_write(fout, &e.from);
                    bin_write(fout, &e.to);
                    bin_write(fout, &e.cost);
                }
            if (!rank)
                std::cerr << "allreduce cross edge\n";
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &cross_edge_count, 1, MPI::INT, MPI::SUM);
            max_border_node_count = border_nodes.size();
            if (!rank)
                std::cerr << "allreduce max border node\n";
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &max_border_node_count, 1, MPI::INT, MPI::MAX);
            cross_edge_count /= 2;

            if (!rank)
                std::cerr << "writing border nodes\n";
            int border_node_count = border_nodes.size();
            bin_write(fout, &border_node_count);
            for (auto u : border_nodes)
                bin_write(fout, &u);
            if (!rank)
                std::cerr << "writing max_border_node_count\n";
            bin_write(fout, &max_border_node_count);
            bin_write(fout, &cross_edge_count);
        }

        MPI::COMM_WORLD.Barrier();
        if (!rank)
            std::cerr << "cross edge: " << cross_edge_count <<
                ", max border nodes: " << max_border_node_count << "\n";

        dist_now.resize(n, edge::max());
        dist_pre = dist_now;

        MPI::COMM_WORLD.Barrier();
    }

    auto compute(int s, int t)
    {
        if (!rank)
            std::cerr << "computing.\n";

        std::priority_queue<edge> pq;
        if (nodes.count(s)) {
            dist_now[s] = dist_pre[s] = 0;
            pq.emplace(s, s, 0);
        }
        auto iter = 0;
        for (auto relaxed = 0; ; iter++) {
            if (!rank)
                std::cerr << "iterating on " << iter << ", relaxed=" << relaxed << ", ";
            relaxed = 0;

            using namespace std::chrono;
            auto start = high_resolution_clock::now();
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
            auto end = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(end - start).count() / 1000.;
            if (!rank)
                std::cerr << "dij elapsed <" << elapsed << "s>, ";

            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &relaxed, 1, MPI::INT, MPI::SUM);
            if (!relaxed)
                break;

            {
                using namespace std::chrono;
                auto start = high_resolution_clock::now();
                update_dist(dist_now, dist_pre);
                auto end = high_resolution_clock::now();
                auto elapsed = duration_cast<milliseconds>(end - start).count() / 1000.;
                if (!rank)
                    std::cerr << "update dist elapsed <" << elapsed << "s>\n";
            }

            for (auto u : nodes)
                if (dist_now[u] != dist_pre[u])
                    pq.emplace(s, u, dist_now[u]);
            dist_pre = dist_now;
        }

        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &dist_now[t], 1, MPI::INT, MPI::MIN);
        if (!rank)
            std::cout << "distance from [" << s << "] to [" << t << "] is "
                << dist_now[t] << "\n";
        return dist_now[t];
    }

    template <class T>
    void update_dist(T& dist_now, T const& dist_pre)
    {
        recv_buf.clear();

        // for (auto const& p : dist_now) {
        //     auto i = p.first;
        //     auto d = p.second;
        //     if (d == dist_pre.at(i) || (nodes.count(i) && !border_nodes.count(i)))
        //         continue;
        //     recv_buf.emplace_back(i);
        //     recv_buf.emplace_back(d);
        // }

        for (auto i = 0; i < n; i++) {
            if (dist_now.at(i) == dist_pre.at(i) || (nodes.count(i) && !border_nodes.count(i)))
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
                dist_now.at(id) = std::min(dist_now.at(id), value);
                updated++;
            }
        }
    }

    void print(bool all = false)
    {
        if (all || !rank) {
            std::cout << "n=" << n << " m=" << m << "\n";
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
};

} // namespace icesp

