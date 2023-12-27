import numpy as np
from ..compute import pgkokkos


def createViewMirrorArray(obj, names, shape):
    """
    This function creates the three primary data structures in peregrine:

    - Kokkos View
    - Kokkos Host Mirror View
    - Numpy Array wrapping of the Host Mirror View

    We can create these from scratch with just a name,
    or from existing Numpy Arrays.

    Convention for whatever the input "obj" is, is as follows:

    obj.name -> Kokko View
    obj.mirror[name] -> Kokkos Host Mirror View
    obj.array[name] -> Numpy Array wrapping host mirror

    Where obj.mirror and obj.array are python dictionaries.

    """

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
            # If the numpy array doesnt exist, create it here
            obj.array[name] = np.array(obj.mirror[name], copy=False)
        else:
            # Otherwise, copy the existing numpy array data into the newly
            # created View and Host Mirror Views
            extents = obj.array[name].shape
            assert all(
                [i == j for i, j in zip(shape, extents)]
            ), f"Requested shape for {name} does not equal existing numpy array shape."
            temp = np.copy(obj.array[name])
            obj.array[name] = None
            obj.array[name] = np.array(obj.mirror[name], copy=False)
            obj.array[name][:] = temp[:]
            pgkokkos.deep_copy(getattr(obj, name), obj.mirror[name])
