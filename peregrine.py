#!/usr/bin/env python
import numpy as np

#
# The python bindings for generate_view are in ex-generate.cpp
# The declaration and definition of generate_view are in user.hpp and user.cpp
# The generate_view function will return a Kokkos::View and will be converted
# to a numpy array
from perepute import add2,add3

# Importing this module is necessary to call kokkos init/finalize and
# import the python bindings to Kokkos::View which generate_view will
# return
#
import kokkos
import time

def test():
    nx,ny,nz = 1000,1000,1000
    # get the kokkos view
    view = kokkos.array(
        "python_allocated_view",
        [nx,ny,nz],
        dtype=kokkos.double,
        space=kokkos.HostSpace,
    )
    #for i in range(view.shape[0]):
    #    view[i,] = i * (i % 2)
    # wrap the buffer protocal as numpy array without copying the data
    arr = np.array(view, copy=False)
    #arr[:,:] = 0.0
    #arr[:,1] = 1.0
    # verify type id
    # print("Numpy Array : {} (shape={})".format(type(arr).__name__, arr.shape))
    # demonstrate the data is the same as what was printed by generate_view
    #print(arr)
    ts = time.time()
    add2(view,1.0,0,0,0,nx,ny,nz)
    add2(view,1.0,0,0,0,nx,ny,nz)
    add2(view,1.0,0,0,0,nx,ny,nz)
    add2(view,1.0,0,0,0,nx,ny,nz)
    print(time.time()-ts, 'took this many seconds')
    #print(arr)

if __name__ == "__main__":
    kokkos.initialize()
    test()
    kokkos.finalize()
