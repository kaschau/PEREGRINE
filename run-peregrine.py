#!/usr/bin/env python

import sys
sys.path.append('./Lib')
import peregrine as pgc
import peregrinepy as pgpy
from mpi4py import MPI

import kokkos
import time

def simulate():

    # Initialize the parallel information for each rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    myblocks = pgpy.readers.read_blocks4procs(rank)
    collection = []
    for nblki in myblocks:
        b = pgc.block()
        b.nblki = nblki
        collection.append(b)

    pgpy.readers.read_grid(collection)

    return collection
    #print(collection[0].x_np)
    #sys.exit()

    #ts = time.time()
    #for i,b in enumerate(collection):
    #    pgc.add3(b,float(i))
    #    pgc.add3(b,float(i))
    #    pgc.add3(b,float(i))
    #    pgc.add3(b,float(i))
    #print(time.time()-ts, 'took this many seconds')


if __name__ == "__main__":
    kokkos.initialize()
    collection = simulate()
    kokkos.finalize()
