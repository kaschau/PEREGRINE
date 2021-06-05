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
    nx,ny,nz = 10,10,2
    collection = []
    for i in range(10):
        collection.append(block())
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
        add3(b,float(i))
        add3(b,float(i))
        add3(b,float(i))
        add3(b,float(i))
    print(time.time()-ts, 'took this many seconds')


if __name__ == "__main__":
    kokkos.initialize()
    test()
    kokkos.finalize()
