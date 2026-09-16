"""One array on a backend. An array knows its backend and nothing else
about where it is: get() is always a fresh copy and set() always a copy
in, on either, so a snapshot taken for a writer survives the next step. It
holds its memory and its record; what it is for, a state, a flux, a
buffer, is its owner's business."""

import ctypes

import numpy as np

from .abi import pgView, strides


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
