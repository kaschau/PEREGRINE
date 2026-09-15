"""The backend an array is made on, and the array itself. A backend is the
compute unit the runtime was built for: a HostBackend (OpenMP) gives an
array numpy memory the kernels read in place,
a DeviceBackend (CUDA, HIP) gives it a runtime allocation and moves its
bytes over the bus. An array knows its backend and nothing else about where
it is: get() is always a fresh copy and set() always a copy in, on either,
so a snapshot taken for a writer survives the next step.

A backend makes arrays and moves their bytes, and nothing else: it knows no
block, table or kernel, and it is handed down from the multiBlock, never
looked up. An array holds its memory and its record; what an array is for,
a state, a flux, a buffer, is its owner's business, not this file's."""

import ctypes

import numpy as np

from .abi import lib, pgView, strides


class Array:
    """One array on a backend: its shape, order and strides, the record a
    kernel receives, and the memory its backend gave it."""

    def __init__(
        self, shape, backend, dtype=np.float64, *, name=None, kind=None, components=()
    ):
        self.shape = tuple(int(n) for n in shape)
        self.backend = backend
        self.dtype = np.dtype(dtype)
        self.order = backend.order
        self.name, self.kind, self.components = name, kind, components
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        # the record a kernel receives, fixed for the life of the array
        pad = [0] * (5 - len(self.shape))
        self.record = pgView(
            None,
            len(self.shape),
            (ctypes.c_int * 5)(*self.shape, *pad),
            (ctypes.c_long * 5)(*strides(self.shape, self.order), *pad),
        )
        # the memory, zeroed, as an address
        self.ptr = backend.memory(self)
        self.record.data = self.ptr

    def __del__(self):
        if getattr(self, "ptr", None):
            self.backend.release(self)

    def __repr__(self):
        return f"<Array {self.name or ''} {self.shape} on {self.backend!r}>"

    def get(self, component=None, wait=True):
        """A fresh host copy, of one trailing component when one is named;
        one asked for without waiting is only there after lib.pgFence()."""
        return self.backend.toHost(self, component, wait)

    def set(self, values, wait=True):
        """Copy host values in; without waiting, they have to outlive the copy."""
        values = np.asarray(values, self.dtype)
        assert values.shape == self.shape, (values.shape, self.shape)
        self.backend.fromHost(self, values, wait)


class BaseBackend:
    """The compute unit arrays are made for. It gives an array its memory,
    takes it back, and moves bytes in and out; its order is the layout the
    runtime keeps arrays in."""

    def __init__(self, order="C"):
        self.order = order

    def __repr__(self):
        return f"{type(self).__name__}({self.order!r})"

    def allocate(self, shape, dtype=np.float64, **info):
        """A zeroed array on this backend."""
        return Array(shape, self, dtype, **info)

    def memory(self, array):
        """Zeroed memory for :array:, as an address."""
        raise NotImplementedError

    def release(self, array):
        raise NotImplementedError

    def toHost(self, array, component, wait):
        raise NotImplementedError

    def fromHost(self, array, values, wait):
        raise NotImplementedError


class HostBackend(BaseBackend):
    """OpenMP: an array is numpy memory, which the kernels read in place."""

    def memory(self, array):
        array.data = np.zeros(array.shape, array.dtype, order=self.order)
        return array.data.ctypes.data

    def release(self, array):
        """numpy frees its own."""

    def toHost(self, array, component, wait):
        if component is None:
            return array.data.copy(order=self.order)
        return np.array(array.data[..., component], order=self.order)

    def fromHost(self, array, values, wait):
        array.data[...] = values


class DeviceBackend(BaseBackend):
    """CUDA or HIP: an array is a runtime allocation, and its bytes cross the
    bus through the runtime. In the left layout a trailing component is a
    contiguous run, so only it crosses; in the right layout the whole array
    is pulled and sliced."""

    def memory(self, array):
        return lib.pgAllocate(array.nbytes)

    def release(self, array):
        lib.pgFree(array.ptr)

    def toHost(self, array, component, wait):
        if component is None:
            host = np.empty(array.shape, array.dtype, order=self.order)
            lib.pgToHost(array.ptr, host.ctypes.data, array.nbytes, wait)
            return host
        if self.order != "F":
            whole = self.toHost(array, None, True)
            return np.array(whole[..., component], order=self.order)
        host = np.empty(array.shape[:-1], array.dtype, order="F")
        lib.pgToHost(
            array.ptr + component * host.nbytes, host.ctypes.data, host.nbytes, True
        )
        return host

    def fromHost(self, array, values, wait):
        host = np.require(values, array.dtype, self.order)
        lib.pgToDevice(host.ctypes.data, array.ptr, array.nbytes, wait)


def runtimeBackend():
    """The backend the runtime was built for, once it is up: the host when
    the kernels' memory is host memory, else the device, in its layout."""
    order = "F" if lib.pgLayoutLeft() else "C"
    return HostBackend(order) if lib.pgOnHost() else DeviceBackend(order)
