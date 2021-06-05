#!/usr/bin/env python
import numpy as np

#
# The python bindings for generate_view are in ex-generate.cpp
# The declaration and definition of generate_view are in user.hpp and user.cpp
# The generate_view function will return a Kokkos::View and will be converted
# to a numpy array
from perepute import add2,add3,block

# Importing this module is necessary to call kokkos init/finalize and
# import the python bindings to Kokkos::View which generate_view will
# return
#
import kokkos
import time
import sys

def test():
    nb = 100
    nx,ny,nz = 100,100,100
    collection = []
    for i in range(10):
        collection.append(block())
        b = collection[-1]
        b.nblki = 1
        b.nx = 30
        b.ny = 30
        b.nz = 30
        b.x = kokkos.array("python_allocated_view",
                                        [b.nx,b.ny,b.nz],
                                        dtype=kokkos.double,
                                        space=kokkos.HostSpace)

    ts = time.time()
    for i,b in enumerate(collection):
        add3(b,float(i))
        add3(b,float(i))
        add3(b,float(i))
        add3(b,float(i))
    print(time.time()-ts, 'took this many seconds')
    #print(arr)0
    return collection

if __name__ == "__main__":
    kokkos.initialize()
    col = test()
    #kokkos.finalize()
