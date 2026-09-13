"""One compiled C function, described by its own prototype, and the form a
solver calls it in."""

import ctypes
import re

import numpy as np

from .abi import View, lib
from .jit import Jit


class Kernel:
    """Each parameter is filled by name: from a table for the views, dims,
    ranges and per-entry integers, from the species data for the case arrays
    and constants, and from keywords for the rest. What the prototype marks
    pgIn is read, pgOut written; that is what a step's graph orders by."""

    scalars = {"int": ctypes.c_int, "double": ctypes.c_double, "bool": ctypes.c_bool}
    prototype = re.compile(r"PG_ABI\s+(\w+)\s+(pg\w+)\s*\(([^)]*)\)")
    stencilDeclaration = re.compile(r"PG_STENCIL\((\d+)\)")

    def __init__(self, name, restype, params, reads, writes, stencil):
        self.name = name
        # [(kind, name)] in call order
        self.params = params
        self.reads, self.writes = reads, writes
        # how many halo layers it reaches into
        self.stencil = stencil
        argtypes = [
            ctypes.c_int if kind == "count" else self.scalars.get(kind, ctypes.c_void_p)
            for kind, _ in params
        ]
        lib.declare(name, argtypes, {"int": ctypes.c_int, "void": None}[restype])
        # how each parameter is found at a call
        self.resolvers = [self._resolver(kind, pname) for kind, pname in params]

    @classmethod
    def parse(cls, source):
        """The one kernel a source declares, from its prototype, and the
        stencil it declares: one layer unless it says more."""
        found = cls.prototype.findall(source)
        if len(found) != 1:
            raise ValueError(
                f"a kernel source declares one PG_ABI function, this one {len(found)}"
            )
        restype, name, params = found[0]
        parsed, reads, writes = [], [], []
        for p in params.split(","):
            words = p.split()
            # a record by pointer is one per entry; by reference it is the case's one
            pname = words[-1].lstrip("*&")
            ptype = " ".join(words[:-1]) + ("*" if words[-1][0] in "*&" else "")
            ptype = ptype.replace(" *", "*").replace(" &", "*").replace("const ", "")
            if pname == "count":
                parsed.append(("count", pname))
            elif ptype in ("pgView*", "pgIn*", "pgOut*"):
                array = pname.rstrip("_")
                parsed.append(("view", array))
                (writes if ptype == "pgOut*" else reads).append(array)
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
        stencil = cls.stencilDeclaration.search(source)
        return cls(
            name,
            restype,
            parsed,
            reads,
            writes,
            int(stencil.group(1)) if stencil else 1,
        )

    @classmethod
    def _resolver(cls, kind, name):
        """A function of (table, th, nface, given) giving one argument; what
        is given by keyword wins over what the table or species data hold."""

        def given(value, keep):
            if kind == "view":
                if not isinstance(value, ctypes.Array):
                    # one array, or a list of them, as a table column would be
                    records = value if isinstance(value, (list, tuple)) else [value]
                    value = (View * len(records))(*(View.of(v) for v in records))
                keep.append(value)
                return ctypes.addressof(value)
            if kind in ("ints", "doubles"):
                value = np.ascontiguousarray(
                    value, dtype=np.int32 if kind == "ints" else np.float64
                )
                keep.append(value)
                return value.ctypes.data
            return value

        def found(table, th, nface, keep):
            if kind == "count":
                return len(table)
            if kind == "view":
                if th is not None and hasattr(th, name):
                    record = View.of(getattr(th, name))
                    keep.append(record)
                    return ctypes.addressof(record)
                return ctypes.addressof(table.views(name))
            if kind == "dims":
                return ctypes.addressof(table.dims)
            if kind == "range":
                return ctypes.addressof(table.ranges(nface))
            if kind == "ints":
                return ctypes.addressof(table.ints(name))
            if kind == "int" and name == "nface":
                return nface
            if kind in cls.scalars and th is not None and hasattr(th, name):
                return getattr(th, name)
            raise TypeError(f"a call needs {name}")

        def resolve(table, th, nface, values, keep):
            return (
                given(values.pop(name), keep)
                if name in values
                else found(table, th, nface, keep)
            )

        return resolve

    def __call__(self, table=None, th=None, nface=None, **given):
        """Run over a table's entries; :given: supplies the scalars, and any
        array by its parameter name."""
        keep = []
        try:
            args = [r(table, th, nface, given, keep) for r in self.resolvers]
        except TypeError as e:
            raise TypeError(f"{self.name}: {e}") from None
        if given:
            raise TypeError(f"{self.name} takes no {', '.join(given)}")
        return getattr(lib, self.name)(*args)


class BoundKernel:
    """A kernel as a case calls it: over the case's block table, or a table
    of its own, with the case's species data, and with its fixed scalars
    settled up front so the call names only what varies -- or overrides one.
    What it reads, writes and reaches is known from its source the moment it
    is made; the case compiles it with the rest."""

    def __init__(
        self,
        table,
        thtrdat,
        source,
        tableOf=None,
        defines=(),
        includes=(),
        role=None,
        **fixed,
    ):
        self.table, self.thtrdat = table, thtrdat
        self.source, self.fixed = source, fixed
        self.defines, self.includes = tuple(defines), tuple(includes)
        # None is the case's block table; otherwise what to call for the table
        self.tableOf = tableOf
        self.kernel = Kernel.parse((Jit.compute / source).read_text())
        self.compiled = False
        # the name the config knows it by, and the one the case calls it by
        self.__name__ = source.rsplit("/", 1)[-1].removesuffix(".cpp")
        self.role = role or self.__name__

    @property
    def reads(self):
        return self.kernel.reads

    @property
    def writes(self):
        return self.kernel.writes

    @property
    def stencil(self):
        return self.kernel.stencil

    def __call__(self, table=None, **given):
        if not self.compiled:
            raise RuntimeError(f"{self.role} is not compiled")
        if table is None:
            table = self.table if self.tableOf is None else self.tableOf()
        return self.kernel(table, self.thtrdat, **{**self.fixed, **given})
