"""What a backend is: the compute unit the runtime was built for. It makes
arrays and moves their bytes, tiles the tables the kernels run over, and
compiles the kernels for itself; it knows no block or kernel body, and it
is handed down from the multiBlock, never looked up. Its knobs come from
its own section of the config, backend-<name>: the items of one entry a
team does, and on a device the launch bound the kernels are compiled with,
which is the register budget the compiler works to."""

import numpy as np

from .abi import lib
from .array import Array
from .jit import Jit
from .table import Table


class BaseBackend:
    """The compute unit arrays are made for. It gives an array its memory,
    takes it back, and moves bytes in and out; its order is the layout the
    runtime keeps arrays in."""

    # what the runtime calls it, lower case; its config section is backend-<name>
    name = None

    def __init__(self, order="C", launch=None):
        self.order = order
        # how this backend launches, from its config section: the items of
        # one entry a team does, by what an item is, and on a device the
        # launch bound the kernels are compiled with, (threads, waves). A
        # backend that only makes arrays, a grid's, has no section
        self.tiles = self.launchBound = None
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
        return subclassWhere(cls, name=name)(order, config[f"backend-{name}"])

    def table(self, entries):
        """A table of entries kept here, tiled by this backend's tiles."""
        return Table(entries, self)

    def jit(self, ns, ng, tables, eos, diffusion=None, mixingRule="wilke"):
        """The compiler for this backend's kernels, with its launch bound."""
        return Jit(ns, ng, tables, eos, diffusion, mixingRule, launch=self.launchBound)

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
