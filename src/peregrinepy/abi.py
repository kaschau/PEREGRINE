"""PEREGRINE's C ABI from the python side: an array is a pointer and its
extents, a kernel is a C function that takes them."""

import ctypes
from pathlib import Path

import numpy as np

lib = ctypes.CDLL(str(next(Path(__file__).parent.glob("libcompute.*"))))
lib.pgLayoutLeft.restype = ctypes.c_int
lib.pgAllocate.restype = ctypes.c_void_p
lib.pgAllocate.argtypes = [ctypes.c_size_t]
lib.pgFree.argtypes = [ctypes.c_void_p]
lib.pgToHost.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.pgToDevice.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.pgCopy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def initialize():
    lib.pgInitialize()


def finalize():
    lib.pgFinalize()


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
        if array is None:
            return cls(None, 0, (ctypes.c_int * 5)())
        shape = array.shape
        return cls(
            array.ptr, len(shape), (ctypes.c_int * 5)(*shape, *[0] * (5 - len(shape)))
        )


class Dims(ctypes.Structure):
    """A block's shape: cells per direction and halo depth."""

    _fields_ = [(n, ctypes.c_int) for n in ("ni", "nj", "nk", "ng")]


class Range(ctypes.Structure):
    """The cells a kernel does: [i0, i1) x [j0, j1) x [k0, k1)."""

    _fields_ = [(n, ctypes.c_int) for n in ("i0", "i1", "j0", "j1", "k0", "k1")]


def cellRange(blk, nface):
    """The old nface convention as explicit bounds: -1 the whole block, 0 the
    interior, 1..6 a face's halo."""
    ng = blk.ng
    ni, nj, nk = blk.ni + 2 * ng - 1, blk.nj + 2 * ng - 1, blk.nk + 2 * ng - 1
    if nface == -1:
        return Range(0, ni, 0, nj, 0, nk)
    if nface == 0:
        return Range(ng, ni - ng, ng, nj - ng, ng, nk - ng)
    halo = {
        1: (0, ng, 0, nj, 0, nk),
        2: (ni - ng, ni, 0, nj, 0, nk),
        3: (0, ni, 0, ng, 0, nk),
        4: (0, ni, nj - ng, nj, 0, nk),
        5: (0, ni, 0, nj, 0, ng),
        6: (0, ni, 0, nj, nk - ng, nk),
    }
    return Range(*halo[nface])


class DeviceArray:
    """An array living where the kernels run. The device copy is the truth;
    get() is a snapshot for the host to use and let go of, set() is how the
    host writes one."""

    # the layout Kokkos uses on this device; numpy allocates to match
    order = "F" if lib.pgLayoutLeft() else "C"

    def __init__(self, shape, dtype=np.float64):
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = lib.pgAllocate(self.nbytes)

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
