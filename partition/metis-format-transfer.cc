#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>
#include <unordered_map>
#include <utility>

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

struct graph
{
    graph(std::string const& path) : path(path)
    {
        auto fin = std::ifstream{path};
        read_graph(fin);
        auto fout = std::ofstream{path + ".metis"};
        transfer(fout);
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
                g.resize(n + 1);
                continue;
            }
            int u, v, w;
            is >> u >> v >> w;
            if (!g[u].count(v))
                g[u].emplace(std::make_pair(v, edge{u, v, w}));
            else
                g[u].at(v).cost = std::min(g[u].at(v).cost, w);
        }
        m = 0;
        for (auto i = 1; i <= n; i++)
            m += g[i].size();
        m /= 2;
    }

    // TODO pass stream in parameter?
    void transfer(std::ostream& os)
    {
        os << n << " " << m << " 1\n";
        for (auto i = 1; i <= n; i++) {
            auto first = true;
            for (auto const& e : g[i]) {
                if (!first)
                    os << " ";
                else
                    first = false;
                os << e.first << " " << e.second.cost;
            }
            os << "\n";
        }
    }

    // graph data path
    std::string path;
    // total number of nodes
    int n;
    // total number of edges
    int m;
    // for node <from>, store {<to> -> edge}, because graph may contain
    // duplicate edges
    std::vector<std::unordered_map<int, edge>> g;
};

int main()
{
    graph g("../dataset/USA-road-d.CAL.gr");
}

