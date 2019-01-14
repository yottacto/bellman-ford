# Parallel Implementation of SSSP Algorithm

## MPI

* each process hold whole `dist` array
* testing out locality
* using [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to do graph partition
* each process using dijkstra now
* using `MPI::COMM_WORLD::Allgather` to reduce all `(index, dist)` pairs

