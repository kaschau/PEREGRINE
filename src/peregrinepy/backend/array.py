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
        # an array within another's memory does not own it
        self.owner = self

    @classmethod
    def within(cls, pool, offset, shape, **info):
        """An array of :shape: over :pool:'s memory from element :offset:: a
        kernel sees it as an array of its own; the pool moves it in bulk."""
        array = cls.__new__(cls)
        array.shape = tuple(int(n) for n in shape)
        array.backend, array.dtype, array.order = pool.backend, pool.dtype, pool.order
        array.name, array.kind, array.components = (
            info.get("name"),
            info.get("kind"),
            info.get("components", ()),
        )
        array.nbytes = int(np.prod(array.shape)) * array.dtype.itemsize
        assert (
            offset + int(np.prod(array.shape))
        ) * array.dtype.itemsize <= pool.nbytes
        pad = [0] * (5 - len(array.shape))
        array.record = pgView(
            None,
            len(array.shape),
            (ctypes.c_int * 5)(*array.shape, *pad),
            (ctypes.c_long * 5)(*strides(array.shape, array.order), *pad),
        )
        array.ptr = pool.ptr + offset * array.dtype.itemsize
        array.record.data = array.ptr
        array.owner = pool
        if hasattr(pool, "data"):
            # the host's memory is numpy's: the slice of it, in shape
            array.data = pool.data.reshape(-1)[
                offset : offset + int(np.prod(array.shape))
            ].reshape(array.shape, order=array.order)
        return array

    def __del__(self):
        if getattr(self, "ptr", None) and getattr(self, "owner", None) is self:
            self.backend.release(self)

    def __repr__(self):
        return f"<Array {self.name or ''} {self.shape} on {self.backend!r}>"

    def get(self, component=None, wait=True):
        """A fresh host copy, of one trailing component when one is named;
        one asked for without waiting is only there after lib.pgFence()."""
        return self.backend.toHost(self, component, wait)

    def pull(self, host, wait=True):
        """Copy into :host:, a host array of this shape and order kept for
        the purpose; without waiting, it is there after lib.pgFence()."""
        assert host.shape == self.shape and host.dtype == self.dtype, (
            host.shape,
            self.shape,
        )
        self.backend.pull(self, host, wait)

    def pullAside(self, host):
        """Copy into :host: beside the kernels, after what they have queued;
        there once lib.pgCopyWait() returns."""
        assert host.shape == self.shape and host.dtype == self.dtype
        self.backend.pullAside(self, host)

    def pushAside(self, host):
        """Copy :host: in beside the kernels; a kernel reading it is launched
        after lib.pgCopyWait()."""
        assert host.shape == self.shape and host.dtype == self.dtype
        self.backend.pushAside(self, host)

    def set(self, values, wait=True):
        """Copy host values in; without waiting, they have to outlive the copy."""
        values = np.asarray(values, self.dtype)
        assert values.shape == self.shape, (values.shape, self.shape)
        self.backend.fromHost(self, values, wait)
