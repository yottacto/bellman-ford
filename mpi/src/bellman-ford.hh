#pragma once
#include <iostream>
#include <mpi.h>

namespace icesp
{

struct bellman_ford
{

    bellman_ford(std::string const& path) : path(path)
    {
        if (!MPI::Is_initialized())
            MPI::Init();
        rank = MPI::COMM_WORLD.Get_rank();
    }

    ~bellman_ford()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    void print()
    {
        std::cout << "Hi from rank=" << rank << ", path=" << path << "\n";
    }

    std::string path;
    int rank;
};


} // namespace icesp

