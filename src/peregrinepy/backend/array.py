"""One array on a backend. An array knows its backend and nothing else
about where it is: get() is always a fresh copy and set() always a copy
in, on either, so a snapshot taken for a writer survives the next step. It
holds its memory and its info; what it is for is its owner's business,
and a block array's shape and ranges are its kind's (multiBlock.arrays).
A pooled array is a kind of its own: a piece of a pool's memory."""

import ctypes

import numpy as np

from .abi import pgArrayInfo, strides


class BaseArray:
    """One array on a backend: its shape, order and strides, the info a
    kernel receives, and the memory its backend gave it."""

    def __init__(self, shape, backend, dtype=None, *, name=None):
        self.shape = tuple(int(n) for n in shape)
        self.backend = backend
        self.dtype = np.dtype(backend.fpdtype if dtype is None else dtype)
        self.order = backend.order
        self.name = name
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        # the memory, as an address, and whose it is
        self.ptr = self._memory()
        # what a kernel receives: where it is, its extents, and its strides
        # in elements, which is the only place the layout becomes an address
        pad = [0] * (5 - len(self.shape))
        self.info = pgArrayInfo(
            self.ptr,
            len(self.shape),
            (ctypes.c_int * 5)(*self.shape, *pad),
            (ctypes.c_long * 5)(*strides(self.shape, self.order), *pad),
        )

    def _memory(self):
        """Takes zeroed memory of this array's size from the backend; the
        array owns it."""
        self.owner = self
        return self.backend.memory(self)

    def __del__(self):
        if getattr(self, "ptr", None) and getattr(self, "owner", None) is self:
            self.backend.release(self)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name or ''} {self.shape} on {self.backend!r}>"

    def get(self, component=None, wait=True):
        """Copies the array to a fresh host array, one trailing component
        when one is named; one asked for without waiting is only there after
        lib.pgFence()."""
        return self.backend.toHost(self, component, wait)

    def pull(self, host, wait=True):
        """Copies the array into :host:, a host array of this shape and
        order kept for the purpose; without waiting, it is there after
        lib.pgFence()."""
        assert host.shape == self.shape and host.dtype == self.dtype, (
            host.shape,
            self.shape,
        )
        self.backend.pull(self, host, wait)

    def pullAside(self, host):
        """Copies the array into :host: beside the kernels, after what they
        have queued; there once lib.pgCopyWait() returns."""
        assert host.shape == self.shape and host.dtype == self.dtype
        self.backend.pullAside(self, host)

    def pushAside(self, host):
        """Copies :host: in beside the kernels; a kernel reading it is
        launched after lib.pgCopyWait()."""
        assert host.shape == self.shape and host.dtype == self.dtype
        self.backend.pushAside(self, host)

    def set(self, values, wait=True):
        """Copies host values in; without waiting, they have to outlive the
        copy."""
        values = np.asarray(values, self.dtype)
        assert values.shape == self.shape, (values.shape, self.shape)
        self.backend.fromHost(self, values, wait)


class PooledArray(BaseArray):
    """An array carved out of a pool's memory, from element :offset:: a
    kernel sees it as an array of its own, the pool moves it in bulk, and
    the pool owns the memory."""

    def __init__(self, pool, offset, shape, *, name=None):
        self.pool, self.offset = pool, int(offset)
        super().__init__(shape, pool.backend, pool.dtype, name=name)
        assert self.offset * self.dtype.itemsize + self.nbytes <= pool.nbytes

    def _memory(self):
        """Takes this array's place in the pool's memory."""
        self.owner = self.pool
        if hasattr(self.pool, "data"):
            # the host's memory is numpy's: the slice of it, in shape
            n = int(np.prod(self.shape))
            self.data = self.pool.data.reshape(-1)[self.offset : self.offset + n]
            self.data = self.data.reshape(self.shape, order=self.order)
        return self.pool.ptr + self.offset * self.dtype.itemsize
