import kokkos
import numpy as np
from itertools import product
from ..compute import KokkosLocation


def createViewMirrorArray(obj, names, shape):

    if type(names) != list:
        names = [names]

    if KokkosLocation in ["OpenMP", "Serial", "Default"]:
        kokkosSpace = kokkos.HostSpace
    elif KokkosLocation in ["Cuda"]:
        kokkosSpace = kokkos.CudaSpace
    else:
        raise ValueError("What space?")

    for name in names:
        setattr(
            obj,
            name,
            kokkos.array(
                name,
                shape=shape,
                dtype=kokkos.double,
                space=kokkosSpace,
                dynamic=False,
            ),
        )
        obj.mirror[name] = kokkos.create_mirror_view(getattr(obj, name), copy=False)
        if obj.array[name] is None:
            obj.array[name] = np.array(obj.mirror[name], copy=False)
        else:
            extents = obj.array[name].shape
            assert all(
                [i == j for i, j in zip(shape, extents)]
            ), f"Requested shape for {name} does not equal existing numpy array shape."
            it = product(*[range(nx) for nx in extents])
            for ijk in it:
                obj.mirror[name][ijk] = obj.array[name][ijk]
            obj.array[name] = None
            obj.array[name] = np.array(obj.mirror[name], copy=False)
            kokkos.deep_copy(getattr(obj, name), obj.mirror[name])
