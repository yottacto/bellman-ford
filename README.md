# Parallel Implementation of Bellman-Ford Algorithm

## MPI

* each process hold whole `dist` array
* testing out locality
* using [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to do graph partition
* each process using dijkstra now

