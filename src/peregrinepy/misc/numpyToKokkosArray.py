import numpy as np
import kokkos


def numpyToKokkosArray(npArray, obj, name, space):
    setattr(
        obj,
        name,
        kokkos.array(
            obj.array[name],
            dtype=kokkos.double,
            space=space,
            dynamic=False,
        ),
    )
