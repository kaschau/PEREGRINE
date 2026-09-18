"""What a backend is: the compute unit the runtime was built for. It makes
arrays and moves their bytes, tiles the tables the kernels run over, and
compiles the kernels for itself; it knows no block or kernel body, and it
is handed down from the multiBlock, never looked up. Its knobs come from
its own section of the config, backend-<name>: the items of one entry a
team does, and on a device the launch bound the kernels are compiled with,
which is the register budget the compiler works to."""

import numpy as np

from .abi import lib
from .array import BaseArray
from .jit import Jit
from .table import ArrayTable


class BaseBackend:
    """The compute unit arrays are made for. It gives an array its memory,
    takes it back, and moves bytes in and out; its order is the layout the
    runtime keeps arrays in."""

    # what the runtime calls it, lower case; its config section is backend-<name>
    name = None

    def __init__(self, order="C", launch=None, precision="double"):
        self.order = order
        # what every array it makes holds, the case's precision
        self.fpdtype = np.dtype({"double": np.float64, "single": np.float32}[precision])
        # how this backend launches, from its config section: the items of
        # one entry a team does, by what an item is, and on a device the
        # launch bound the kernels are compiled with, (threads, waves). A
        # backend that only makes arrays, a grid's, has no section
        self.tiles = self.launchBound = None
        # the array tables made here, one per list of entries, kept
        self.arrayTables = {}
        if launch is not None:
            self.tiles = {
                "cells": launch["tileSize"],
                "elements": launch["tileElements"],
            }
            if "launchThreads" in launch:
                self.launchBound = (launch["launchThreads"], launch["launchWaves"])

    def __repr__(self):
        return f"{type(self).__name__}({self.order!r})"

    @classmethod
    def fromRuntime(cls, config):
        """The backend the runtime was built for, once it is up, in the
        runtime's layout, with its section of the config."""
        from ..misc import subclassWhere

        order = "F" if lib.pgLayoutLeft() else "C"
        name = lib.pgBackend().decode().lower()
        return subclassWhere(cls, name=name)(
            order, config[f"backend-{name}"], config["simulation"]["precision"]
        )

    def arrayTable(self, entries):
        """Gives the array table of these entries -- the list itself, not a
        copy -- made the first time it is asked for and kept, tiled by this
        backend's tiles."""
        if id(entries) not in self.arrayTables:
            self.arrayTables[id(entries)] = ArrayTable(entries, self)
        return self.arrayTables[id(entries)]

    def jit(self, ng, mixture, simulation):
        """Makes the compiler for this backend's kernels, with its launch
        bound."""
        return Jit(ng, mixture, simulation, launch=self.launchBound)

    def allocate(self, shape, dtype=None, name=None):
        """Makes a zeroed array on this backend, of its precision unless
        another dtype is asked for."""
        return BaseArray(
            shape, self, self.fpdtype if dtype is None else dtype, name=name
        )

    def memory(self, array):
        """Zeroed memory for :array:, as an address."""
        raise NotImplementedError

    def release(self, array):
        raise NotImplementedError

    def toHost(self, array, component, wait):
        raise NotImplementedError

    def pull(self, array, host, wait):
        """Copy an array into host memory kept for it."""
        raise NotImplementedError

    def pullAside(self, array, host):
        raise NotImplementedError

    def pushAside(self, array, host):
        raise NotImplementedError

    def pinned(self, shape, dtype=None):
        """Host memory the device reaches at bus speed, as a numpy array in
        this backend's order, of its precision, kept for the run."""
        raise NotImplementedError

    def fromHost(self, array, values, wait):
        raise NotImplementedError
