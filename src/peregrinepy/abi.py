"""PEREGRINE's C ABI from the python side: an array is a pointer and its
extents, a kernel is a C function that takes them."""

import ctypes
import re
from pathlib import Path

import numpy as np


class Library:
    """Every C function the process has loaded, by name. A kernel's argument
    types are declared before its library exists, and looked up when its
    library is loaded and the kernel first called."""

    def __init__(self):
        self._libs = {}
        self._declared = {}
        self._resolved = {}

    def load(self, path, shared=False):
        """Load a library once; the runtime is shared so the kernels resolve
        Kokkos out of it. A kernel compiled for a new case replaces the one
        of the same name from the last."""
        if path not in self._libs:
            mode = ctypes.RTLD_GLOBAL if shared else ctypes.RTLD_LOCAL
            self._libs[path] = ctypes.CDLL(str(path), mode=mode)
        else:
            self._libs[path] = self._libs.pop(path)
        self._resolved.clear()

    def initialize(self):
        """Load the runtime and start Kokkos. Nothing before this touches a
        device, so a pre or post processing run never needs one."""
        self.load(next(Path(__file__).parent.glob("libpgruntime.*")), shared=True)
        self.pgInitialize()
        DeviceArray.order = "F" if self.pgLayoutLeft() else "C"

    def finalize(self):
        self.pgFinalize()

    def declare(self, name, argtypes, restype=None):
        self._declared[name] = (argtypes, restype)
        self._resolved.pop(name, None)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._resolved:
            for lib in reversed(self._libs.values()):
                if hasattr(lib, name):
                    function = getattr(lib, name)
                    break
            else:
                raise AttributeError(
                    f"{name} is in no loaded library; is the runtime initialized?"
                )
            function.argtypes, function.restype = self._declared.get(name, (None, None))
            self._resolved[name] = function
        return self._resolved[name]


lib = Library()
lib.declare("pgInitialize", [])
lib.declare("pgFinalize", [])
lib.declare("pgLayoutLeft", [], ctypes.c_int)
lib.declare("pgAllocate", [ctypes.c_size_t], ctypes.c_void_p)
lib.declare("pgFree", [ctypes.c_void_p])
lib.declare("pgToHost", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])
lib.declare("pgToDevice", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])
lib.declare("pgCopy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])


class View(ctypes.Structure):
    """One array as a kernel receives it."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("extent", ctypes.c_int * 5),
    ]

    @classmethod
    def of(cls, array):
        # an array a face never allocated: the kernel never reads it
        return cls(None, 0, (ctypes.c_int * 5)()) if array is None else array.record


class Dims(ctypes.Structure):
    """A block's shape: cells per direction; the halo depth is compiled in."""

    _fields_ = [(n, ctypes.c_int) for n in ("ni", "nj", "nk")]

    @classmethod
    def of(cls, blk):
        return cls(blk.ni, blk.nj, blk.nk)


class Range(ctypes.Structure):
    """The cells a kernel does: [i0, i1) x [j0, j1) x [k0, k1)."""

    _fields_ = [(n, ctypes.c_int) for n in ("i0", "i1", "j0", "j1", "k0", "k1")]

    @classmethod
    def of(cls, dims, ng, nface):
        """The old nface convention as explicit bounds: -1 the whole block, 0
        the interior, 1..6 a face's halo."""
        ni, nj, nk = dims.ni + 2 * ng - 1, dims.nj + 2 * ng - 1, dims.nk + 2 * ng - 1
        if nface == -1:
            return cls(0, ni, 0, nj, 0, nk)
        if nface == 0:
            return cls(ng, ni - ng, ng, nj - ng, ng, nk - ng)
        halo = {
            1: (0, ng, 0, nj, 0, nk),
            2: (ni - ng, ni, 0, nj, 0, nk),
            3: (0, ni, 0, ng, 0, nk),
            4: (0, ni, nj - ng, nj, 0, nk),
            5: (0, ni, 0, nj, 0, ng),
            6: (0, ni, 0, nj, nk - ng, nk),
        }
        return cls(*halo[nface])


class DeviceArray:
    """An array living where the kernels run. The device copy is the truth;
    get() is a snapshot for the host to use and let go of, set() is how the
    host writes one."""

    # the layout Kokkos uses on this device, known once the runtime is up
    order = None

    def __init__(self, shape, dtype=np.float64):
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = lib.pgAllocate(self.nbytes)
        # the record a kernel receives, fixed for the life of the array
        self.record = View(
            self.ptr,
            len(self.shape),
            (ctypes.c_int * 5)(*self.shape, *[0] * (5 - len(self.shape))),
        )

    def __del__(self):
        if getattr(self, "ptr", None):
            lib.pgFree(self.ptr)

    def __repr__(self):
        return f"<DeviceArray {self.shape}>"

    def get(self):
        """A fresh host copy."""
        host = np.empty(self.shape, self.dtype, order=self.order)
        lib.pgToHost(self.ptr, host.ctypes.data, self.nbytes)
        return host

    def set(self, array):
        """Write a host array to the device."""
        host = np.require(array, self.dtype, self.order)
        assert host.shape == self.shape, (host.shape, self.shape)
        lib.pgToDevice(host.ctypes.data, self.ptr, self.nbytes)

    def copyFrom(self, other):
        """Take another device array's contents, without the host."""
        assert other.shape == self.shape, (other.shape, self.shape)
        lib.pgCopy(self.ptr, other.ptr, self.nbytes)


class HostStorageMixin:
    """Arrays that are numpy arrays: the host reads and writes them in place."""

    def declare(self, *names, kind, components=None):
        """Say an array exists and what shape it will take. Nothing else may
        be put in an array attribute."""
        if isinstance(components, int):
            components = (components,)
        for name in names:
            self.declared[name] = (kind, components)
            setattr(self, name, None)

    def shapeOf(self, name):
        kind, components = self.declared[name]
        return self.shapes[kind] + (components or ())

    def _new(self, shape):
        return np.zeros(shape)

    def hostCopy(self, name):
        """One of these arrays for the host to read: the array itself here, a
        snapshot where it lives on the device."""
        return getattr(self, name)

    def store(self, name, values):
        """Write values into one of these arrays, wherever it lives."""
        getattr(self, name)[...] = values


class DeviceStorageMixin(HostStorageMixin):
    """Arrays that live where the kernels run: the host sees a snapshot and
    writes one back."""

    def _new(self, shape):
        return DeviceArray(shape)

    def hostCopy(self, name):
        return getattr(self, name).get()

    def store(self, name, values):
        getattr(self, name).set(values)
