#!/usr/bin/env python

from mpi4py import MPI
import kokkos

import peregrinepy as pgpy
import time

import sys
def simulate(config_file_path):

    # Initialize the parallel information for each rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = pgpy.initialize.init_config(config_file_path)
    myBlocks = pgpy.readers.read_blocks4procs(config)
    myCompBlocks = pgpy.initialize.init_grid(myBlocks,config)

    #ts = time.time()
    #for b in myCompBlocks:
    #    pgpy.compute.add3D(b,1.0)
    #    pgpy.compute.add3D(b,1.0)
    #    pgpy.compute.add3D(b,1.0)
    #print(time.time()-ts, 'took this many seconds')
    print(myCompBlocks[0].x)
    #return CompBlocks


if __name__ == "__main__":
    config_file_path = sys.argv[1]

    kokkos.initialize()
    simulate(config_file_path)
    kokkos.finalize()
