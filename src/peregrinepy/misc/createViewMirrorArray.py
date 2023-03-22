import kokkos
import numpy as np
from ..compute import KokkosLocation


def createViewMirrorArray(obj, names, shape):
    if type(names) != list:
        names = [names]

    if KokkosLocation in ["OpenMP", "Serial", "Default"]:
        kokkosSpace = kokkos.HostSpace
        kokkosLayout = kokkos.LayoutRight
    elif KokkosLocation in ["Cuda"]:
        kokkosSpace = kokkos.CudaSpace
        kokkosLayout = kokkos.LayoutLeft
    else:
        raise ValueError("What space?")

    for name in names:
        setattr(
            obj,
            name,
            kokkos.array(
                name,
                shape=shape,
                layout=kokkosLayout,
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
            temp = np.copy(obj.array[name])
            obj.array[name] = None
            obj.array[name] = np.array(obj.mirror[name], copy=False)
            obj.array[name][:] = temp[:]
            kokkos.deep_copy(getattr(obj, name), obj.mirror[name])
