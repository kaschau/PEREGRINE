import numpy as np
from ..compute import pgkokkos


def createViewMirrorArray(obj, names, shape):
    if type(names) != list:
        names = [names]

    view = getattr(pgkokkos, f"view{len(shape)}")
    mirror = getattr(pgkokkos, f"mirror{len(shape)}")

    for name in names:
        setattr(
            obj,
            name,
            view(
                name,
                *shape,
            ),
        )
        obj.mirror[name] = mirror(getattr(obj, name))
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
            pgkokkos.deep_copy(getattr(obj, name), obj.mirror[name])
