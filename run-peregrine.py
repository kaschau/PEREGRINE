#!/usr/bin/env python

import sys
sys.path.append('./Lib')

import peregrine as pgc
if pgc.KokkosLocation == 'Default':
    import numpy as np
#elif pgc.KokkosLocation.startswith('Cuda'):
#    import cupy as np
import peregrinepy as pgpy
from mpi4py import MPI

import kokkos
import time

def simulate():

    # Initialize the parallel information for each rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    blk_collection = pgpy.init_grid(rank)
    print(blk_collection)
    sys.exit()


    nb = 100
    nx,ny,nz = 10,10,2
    collection = []
    for i in range(10):
        collection.append(pgc.block())
        b = collection[-1]
        b.nblki = i
        b.nx = nx
        b.ny = ny
        b.nz = nz
        b.x = kokkos.array("python_allocated_view",
                                        [b.nx,b.ny,b.nz],
                                        dtype=kokkos.double,
                                        space=kokkos.HostSpace)
        b.x_np = np.array(b.x, copy=False)
        b.y = kokkos.array("python_allocated_view",
                                        [b.nx,b.ny,b.nz],
                                        dtype=kokkos.double,
                                        space=kokkos.HostSpace)
        b.y_np = np.array(b.y, copy=False)
        b.z = kokkos.array("python_allocated_view",
                                        [b.nx,b.ny,b.nz],
                                        dtype=kokkos.double,
                                        space=kokkos.HostSpace)
        b.z_np = np.array(b.z, copy=False)

    ts = time.time()
    for i,b in enumerate(collection):
        pgc.add3(b,float(i))
        pgc.add3(b,float(i))
        pgc.add3(b,float(i))
        pgc.add3(b,float(i))
    print(time.time()-ts, 'took this many seconds')


if __name__ == "__main__":
    kokkos.initialize()
    simulate()
    kokkos.finalize()
