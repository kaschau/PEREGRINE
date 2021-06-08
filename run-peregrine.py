#!/usr/bin/env python

from mpi4py import MPI
import kokkos

import peregrinepy as pgpy
import time

def simulate():

    # Initialize the parallel information for each rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    myBlocks = pgpy.readers.read_blocks4procs(rank)
    myCompBlocks = pgpy.initialize.grid(myBlocks)

    ts = time.time()
    for b in myCompBlocks:
        pgpy.compute.add3D(b,1.0)
        pgpy.compute.add3D(b,1.0)
        pgpy.compute.add3D(b,1.0)
    print(time.time()-ts, 'took this many seconds')
    print(pgpy.np.max(myCompBlocks[0].x_))
    #return CompBlocks


if __name__ == "__main__":
    kokkos.initialize()
    simulate()
    kokkos.finalize()
