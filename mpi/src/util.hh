#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <utility>
#include <mpi.h>
#include "timer.hh"

namespace icesp
{

namespace detail
{

auto constexpr color_rst{"\e[0m"};
auto constexpr color_act{"\e[0;32m"};
auto constexpr color_arg{"\e[1;35m"};

template <class T>
void print_setw(T const& y)
{
    std::cerr << color_arg << std::left << std::setw(6) << y;
};

template <>
void print_setw(std::string const& y)
{
    if (y.empty())
        std::cerr << color_arg << y;
    else
        std::cerr << color_arg << std::left << std::setw(6) << y;
};

template <bool Enabled>
struct print
{
    template <class T, class U, class V>
    static void impl(int rank, T const& x, U const& y, V const& z, bool all)
    {
        if (!rank || all) {
            std::cerr << color_act << x;
            print_setw<U>(y);
            std::cerr << color_rst << z;
        }
    }
};

template <>
struct print<false>
{
    template <class T, class U, class V>
    static void impl(int, T const&, U const&, V const&, bool) {}
};

} // namespace detail

template <bool Enabled = true, class T, class U = std::string, class V = std::string>
void print(int rank, T const& x, U const& y = std::string{}, V const& z = std::string{}, bool all = false)
{
    detail::print<Enabled>::impl(rank, x, y, z, all);
}


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


template <class Func, class... Args>
void duration(int rank, Func f, Args... args)
{
    using namespace std::chrono;

    MPI::COMM_WORLD.Barrier();
    auto t = timer{};
    t.start();
    f(std::forward<Args>(args)...);
    t.stop();
    auto elapsed = t.elapsed_seconds();

    MPI::COMM_WORLD.Reduce(
        !rank ? MPI::IN_PLACE : &elapsed,
        &elapsed,
        1,
        MPI::DOUBLE,
        MPI::MAX,
        0
    );
    if (!rank)
        std::cerr << "Time elapsed: " << elapsed << "s.\n";
}

} // namespace icsp

