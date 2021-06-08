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
    compBlocks = pgpy.initialize.init_multiblock(config)
    pgpy.initialize.init_grid(compBlocks,config)

    #ts = time.time()
    #for b in myCompBlocks:
    #    pgpy.compute.add3D(b,1.0)
    #    pgpy.compute.add3D(b,1.0)
    #    pgpy.compute.add3D(b,1.0)
    #print(time.time()-ts, 'took this many seconds')
    print(compBlocks[0].x)
    #return CompBlocks


if __name__ == "__main__":
    config_file_path = sys.argv[1]
    try:
        kokkos.initialize()
        simulate(config_file_path)
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
