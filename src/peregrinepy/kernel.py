"""One compiled C function, described by its own prototype, and the form a
solver calls it in."""

import ctypes
import re

import numpy as np

from .abi import View, lib


class Kernel:
    """Each parameter is filled by name: from a table for the views, dims,
    ranges and per-entry integers, from the species data for the case arrays
    and constants, and from keywords for the rest."""

    scalars = {"int": ctypes.c_int, "double": ctypes.c_double, "bool": ctypes.c_bool}
    prototype = re.compile(r"PG_ABI\s+(\w+)\s+(pg\w+)\s*\(([^)]*)\)")

    def __init__(self, name, restype, params):
        self.name = name
        # [(kind, name)] in call order
        self.params = params
        argtypes = []
        for kind, _ in params:
            if kind == "count":
                argtypes.append(ctypes.c_int)
            elif kind in ("view", "dims", "range", "ints", "doubles"):
                argtypes.append(ctypes.c_void_p)
            else:
                argtypes.append(self.scalars[kind])
        lib.declare(name, argtypes, {"int": ctypes.c_int, "void": None}[restype])

    @classmethod
    def parse(cls, source):
        """The one kernel a source declares, from its prototype."""
        found = cls.prototype.findall(source)
        if len(found) != 1:
            raise ValueError(
                f"a kernel source declares one PG_ABI function, this one {len(found)}"
            )
        restype, name, params = found[0]
        parsed = []
        for p in params.split(","):
            words = p.split()
            # a record by pointer is one per entry; by reference it is the case's one
            pname = words[-1].lstrip("*&")
            ptype = " ".join(words[:-1]) + ("*" if words[-1][0] in "*&" else "")
            ptype = ptype.replace(" *", "*").replace(" &", "*").replace("const ", "")
            if pname == "count":
                parsed.append(("count", pname))
            elif ptype == "pgView*":
                parsed.append(("view", pname.rstrip("_")))
            elif ptype == "pgDims*":
                parsed.append(("dims", pname))
            elif ptype == "pgRange*":
                parsed.append(("range", pname))
            elif ptype == "int*":
                parsed.append(("ints", pname))
            elif ptype == "double*":
                parsed.append(("doubles", pname))
            else:
                parsed.append((ptype, pname))
        return cls(name, restype, parsed)

    def __call__(self, table=None, th=None, nface=None, **given):
        """Run over a table's entries; :given: supplies the scalars, and any
        array by its parameter name."""
        args, keep = [], []
        for kind, name in self.params:
            if name in given:
                value = given.pop(name)
                if kind == "view" and not isinstance(value, ctypes.Array):
                    # one array, or a list of them, as a table column would be
                    records = value if isinstance(value, (list, tuple)) else [value]
                    value = (View * len(records))(*(View.of(v) for v in records))
                elif kind in ("ints", "doubles"):
                    value = np.ascontiguousarray(
                        value, dtype=np.int32 if kind == "ints" else np.float64
                    )
                    keep.append(value)
                    value = value.ctypes.data
                keep.append(value)
                args.append(
                    ctypes.addressof(value)
                    if isinstance(value, ctypes.Array)
                    else value
                )
            elif kind == "count":
                args.append(len(table))
            elif kind == "view":
                if th is not None and hasattr(th, name):
                    record = View.of(getattr(th, name))
                    keep.append(record)
                    args.append(ctypes.addressof(record))
                else:
                    args.append(ctypes.addressof(table.views(name)))
            elif kind == "dims":
                args.append(ctypes.addressof(table.dims))
            elif kind == "range":
                args.append(ctypes.addressof(table.ranges(nface)))
            elif kind == "ints":
                args.append(ctypes.addressof(table.ints(name)))
            elif kind == "int" and name == "nface":
                args.append(nface)
            elif kind in self.scalars and th is not None and hasattr(th, name):
                args.append(getattr(th, name))
            else:
                raise TypeError(f"{self.name} needs {name}")
        if given:
            raise TypeError(f"{self.name} takes no {', '.join(given)}")
        return getattr(lib, self.name)(*args)


class BoundKernel:
    """A kernel as a solver calls it: its source is compiled by the solver's
    jit, its table and species data are found at call time, and its fixed
    scalars are settled up front, so the call names only what varies."""

    def __init__(self, mb, source, table=None, defines=(), includes=(), **fixed):
        self.mb, self.source, self.fixed = mb, source, fixed
        self.defines, self.includes = tuple(defines), tuple(includes)
        # None is the block table; otherwise what to call for the table
        self.tableOf = table
        self.kernel = None
        # the name the config knows it by
        self.__name__ = source.rsplit("/", 1)[-1].removesuffix(".cpp")

    def compile(self):
        if self.kernel is None:
            self.kernel = self.mb.jit.kernel(self.source, self.defines, self.includes)
        return self

    def __call__(self, table=None, **given):
        self.compile()
        if table is None:
            table = self.mb.table if self.tableOf is None else self.tableOf()
        return self.kernel(table, self.mb.thtrdat, **self.fixed, **given)


class NullKernel:
    """What a solver holds where the config asks for nothing: takes any call
    and does nothing."""

    __name__ = "null"
    kernel = True
    source = None

    def __call__(self, *args, **kwargs):
        pass

    def compile(self):
        return self
