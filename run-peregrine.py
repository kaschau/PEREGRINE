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

    myblocks = pgpy.readers.read_blocks4procs(rank)
    CompBlocks = []
    for nblki in myblocks:
        b = pgpy.block()
        b.nblki = nblki
        CompBlocks.append(b)

    pgpy.readers.read_grid(CompBlocks)

    ts = time.time()
    for b in CompBlocks:
        pgpy.compute.add3D(b,1.0)
        pgpy.compute.add3D(b,1.0)
        pgpy.compute.add3D(b,1.0)
    print(time.time()-ts, 'took this many seconds')
    #return CompBlocks


if __name__ == "__main__":
    kokkos.initialize()
    simulate()
    kokkos.finalize()
