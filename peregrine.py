#!/usr/bin/env python
import numpy as np

#
# The python bindings for generate_view are in ex-generate.cpp
# The declaration and definition of generate_view are in user.hpp and user.cpp
# The generate_view function will return a Kokkos::View and will be converted
# to a numpy array
from perepute import generate_view

# Importing this module is necessary to call kokkos init/finalize and
# import the python bindings to Kokkos::View which generate_view will
# return
#
import kokkos


#def main(args):
#    # get the kokkos view
#    view = generate_view2()
#    # verify the type id
#    print("Kokkos View : {} (shape={})".format(type(view).__name__, view.shape))
#    # print data provided by generate_view
#    if view.space != kokkos.CudaSpace:
#        for i in range(view.shape[0]):
#            print(
#                "    view({}) = [{:1.0f}., {:1.0f}.]".format(i, view[i, 0], view[i, 1])
#            )
#    # wrap the buffer protocal as numpy array without copying the data
#    arr = np.array(view, copy=False)
#    # verify type id
#    print("Numpy Array : {} (shape={})".format(type(arr).__name__, arr.shape))
#    # demonstrate the data is the same as what was printed by generate_view
#    if view.space != kokkos.CudaSpace:
#        for i in range(arr.shape[0]):
#            print("     arr({}) = {}".format(i, arr[i]))


def test():
    # get the kokkos view
    view = kokkos.array(
        "python_allocated_view",
        [10,10],
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
    print(view.space)


if __name__ == "__main__":
    kokkos.initialize()
    test()
    kokkos.finalize()
