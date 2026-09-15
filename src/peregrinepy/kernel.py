"""One compiled C function, described by its own prototype, and the form a
solver calls it in."""

import ctypes
import re

import numpy as np

from .abi import View, lib
from .jit import Jit


class Column(ctypes.Structure):
    """A column member as the C++ declares it: the records python hands
    over, then the entry's record and the cell the launch shape pins."""

    _fields_ = [
        ("records", ctypes.c_void_p),
        ("at", ctypes.c_void_p),
        ("entry", ctypes.c_int),
        ("i", ctypes.c_int),
        ("j", ctypes.c_int),
        ("k", ctypes.c_int),
    ]


class FaceColumn(ctypes.Structure):
    """A face column: the records, then the face's record and the halo cell
    the launch shape pins."""

    _fields_ = [
        ("records", ctypes.c_void_p),
        ("at", ctypes.c_void_p),
        ("entry", ctypes.c_int),
        ("g", ctypes.c_int),
        ("i", ctypes.c_int),
        ("j", ctypes.c_int),
        ("nface", ctypes.c_int),
    ]


class PerEntryInt(ctypes.Structure):
    """An integer column read as a value: the column, then the entry's value
    the launch shape pins."""

    _fields_ = [("all", ctypes.c_void_p), ("value", ctypes.c_int)]


class Record(ctypes.Structure):
    """A case-wide record, by value: the device cannot follow a host pointer."""

    _fields_ = [("r", View)]


class Dims(ctypes.Structure):
    _fields_ = [("all", ctypes.c_void_p), ("at", ctypes.c_void_p)]


class Kernel:
    """Each argument is filled by name: from a table for the views, dims,
    ranges and per-entry integers, from the species data for the case arrays
    and constants, and from keywords for the rest. What the prototype marks
    pgIn is read, pgOut written; that is what a step's graph orders by. A
    kernel is a struct: the arguments are its members, the prototype takes
    it by reference, and python fills one record of it per call."""

    scalars = {"int": ctypes.c_int, "double": ctypes.c_double, "bool": ctypes.c_bool}
    prototype = re.compile(r"PG_ABI\s+(\w+)\s+(pg\w+)\s*\(([^)]*)\)")
    stencilDeclaration = re.compile(r"PG_STENCIL\((\d+)\)")
    rangeDeclaration = re.compile(r"PG_RANGE\(([^)]*)\)")
    # a struct's members come first, then its operator
    structHead = r"struct\s+{name}\s*(?::\s*(?:public\s+)?(\w+))?\s*\{{(.*?)(?=KOKKOS_INLINE_FUNCTION|operator\(|\}};)"
    known = ("pgView", "pgIn", "pgOut", "pgDims", "pgTiling", "int", "double", "bool")
    # a column member's head: what the kernel does with it, and for a flux
    # kernel which side of the face it is, which its name ends in
    columnHeads = {
        "in": ("r", ""),
        "out": ("w", ""),
        "inout": ("rw", ""),
        "inL": ("r", "L"),
        "inR": ("r", "R"),
        "inLL": ("r", "LL"),
        "inRR": ("r", "RR"),
        "faceIn": ("r", ""),
        "faceOut": ("w", ""),
        "faceInOut": ("rw", ""),
    }

    def __init__(
        self,
        name,
        restype,
        params,
        reads,
        writes,
        stencil,
        ranges,
        columns=None,
        structs=None,
    ):
        self.name = name
        # [(kind, name)] in call order
        self.params = params
        # a struct argument: its ctypes type and [(kind, member, field)]
        self.structs = structs or {}
        # what the prototype calls a column -> what the table calls it: a
        # direction's kernel names F and A, the table iF and iS
        self.columns = dict(columns or {})
        self.reads = [self.columns.get(r, r) for r in reads]
        self.writes = [self.columns.get(w, w) for w in writes]
        # how many halo layers it reaches into
        self.stencil = stencil
        # the ranges it declares, one per tiling it takes, in order
        self.ranges = ranges
        argtypes = [
            ctypes.c_int if kind == "count" else self.scalars.get(kind, ctypes.c_void_p)
            for kind, _ in params
        ]
        lib.declare(name, argtypes, {"int": ctypes.c_int, "void": None}[restype])
        # how each parameter is found at a call
        tilings = iter(ranges)
        self.resolvers = [
            self._resolver(kind, pname, next(tilings) if kind == "tiling" else None)
            for kind, pname in params
        ]

    @staticmethod
    def expand(text, defines):
        """The jit's defines applied to a prototype: a hook names its
        condition through PG_PASTE(PG_CONDITION, PG_HOOK)."""
        for define in defines:
            key, _, value = define.partition("=")
            text = re.sub(rf"\b{key}\b", value, text)
        return re.sub(r"PG_PASTE\((\w+),\s*(\w+)\)", r"\1_\2", text)

    @classmethod
    def members(cls, structName, texts):
        """[(kind, name, ctype)] of a struct's data members, a base's first,
        found among the texts a source reaches."""
        for text in texts:
            m = re.search(cls.structHead.format(name=structName), text, re.S)
            if m:
                break
            # the name may be an alias: a hook's is what the jit's defines say
            alias = re.search(rf"using\s+{structName}\s*=\s*([\w:]+)\s*;", text)
            if alias:
                return cls.members(alias.group(1).rsplit("::", 1)[-1], texts)
        else:
            raise ValueError(f"no struct {structName} in the source or its headers")
        base, body = m.groups()
        members = cls.members(base, texts) if base else []
        for statement in re.sub(r"//.*", "", body).split(";"):
            statement = " ".join(statement.split())
            if not statement or statement.startswith(("static", "//")):
                continue
            head, _, rest = (
                statement.replace("const ", "").replace("mutable ", "").partition(" ")
            )
            for declarator in rest.split(","):
                pointer = "*" in declarator
                mname = declarator.replace("*", "").strip()
                if "=" in declarator:
                    # a member with a default is the kernel's own: python
                    # leaves it, but the record has to hold its bytes
                    mname = mname.partition("=")[0].strip()
                    members.append(("own", mname, cls.scalars[head], head))
                elif head in cls.columnHeads:
                    access, side = cls.columnHeads[head]
                    if side and not mname.endswith(side):
                        raise ValueError(
                            f"struct {structName}: {mname} is {head}, so it ends in {side}"
                        )
                    ctype = FaceColumn if head.startswith("face") else Column
                    members.append(("column", mname.removesuffix(side), ctype, access))
                elif head == "record":
                    members.append(("record", mname, Record, head))
                elif head == "dims":
                    members.append(("dims", mname, Dims, head))
                elif head in ("pgIn", "pgOut", "pgView") and pointer:
                    members.append(("view", mname.rstrip("_"), ctypes.c_void_p, head))
                elif head in ("pgIn", "pgOut", "pgView"):
                    members.append(("record", mname.rstrip("_"), View, head))
                elif head == "perEntry<int>":
                    members.append(("ints", mname, PerEntryInt, head))
                elif head in ("int", "double") and pointer:
                    members.append((head + "s", mname, ctypes.c_void_p, head))
                elif head == "pgDims" and pointer:
                    members.append(("dims", mname, ctypes.c_void_p, head))
                elif head in cls.scalars and not pointer:
                    members.append((head, mname, cls.scalars[head], head))
                else:
                    raise ValueError(f"struct {structName}: what is {statement}?")
        return members

    @classmethod
    def parse(cls, source, includes=(), columns=None, defines=(), headers=()):
        """The one kernel a source declares, from its prototype; the stencil
        it declares, one layer unless it says more; and the ranges it and
        its forced includes declare, one per tiling it takes. An argument
        that is a struct by reference brings its members as arguments."""
        found = cls.prototype.findall(source)
        if len(found) != 1:
            raise ValueError(
                f"a kernel source declares one PG_ABI function, this one {len(found)}"
            )
        restype, name, params = found[0]
        params = cls.expand(params, defines)
        texts = [cls.expand(t, defines) for t in (source, *includes, *headers)]
        parsed, reads, writes, structs = [], [], [], {}
        for p in params.split(","):
            words = p.split()
            # a record by pointer is one per entry; by reference it is the case's one
            pname = words[-1].lstrip("*&")
            ptype = " ".join(words[:-1]) + ("*" if words[-1][0] in "*&" else "")
            ptype = ptype.replace(" *", "*").replace(" &", "*").replace("const ", "")
            if pname == "count":
                parsed.append(("count", pname))
            elif ptype.endswith("*") and ptype[:-1] not in cls.known:
                members = cls.members(ptype[:-1], texts)
                fields = [
                    (f"m{i}", ctype) for i, (_, _, ctype, _) in enumerate(members)
                ]
                record = type(ptype[:-1], (ctypes.Structure,), {"_fields_": fields})
                structs[pname] = (
                    record,
                    [(k, n, f, c) for (k, n, c, _), (f, _) in zip(members, fields)],
                )
                for kind, mname, _, head in members:
                    if kind == "view":
                        (writes if head == "pgOut" else reads).append(mname)
                    if kind == "column":
                        if "r" in head:
                            reads.append(mname)
                        if "w" in head:
                            writes.append(mname)
                parsed.append(("struct", pname))
            elif ptype in ("pgView*", "pgIn*", "pgOut*"):
                array = pname.rstrip("_")
                parsed.append(("view", array))
                (writes if ptype == "pgOut*" else reads).append(array)
            elif ptype == "pgDims*":
                parsed.append(("dims", pname))
            elif ptype == "pgTiling*":
                parsed.append(("tiling", pname))
            elif ptype == "int*":
                parsed.append(("ints", pname))
            elif ptype == "double*":
                parsed.append(("doubles", pname))
            else:
                parsed.append((ptype, pname))
        stencil = cls.stencilDeclaration.search(source)
        ranges = []
        for text in texts[: 1 + len(includes)]:
            for decl in cls.rangeDeclaration.findall(text):
                parts = [x.strip() for x in decl.split(",")]
                ranges.append((parts[0], *parts[1:]))
        tilings = sum(kind == "tiling" for kind, _ in parsed)
        if tilings != len(ranges):
            raise ValueError(
                f"{name} takes {tilings} tilings and declares {len(ranges)} ranges"
            )
        return cls(
            name,
            restype,
            parsed,
            reads,
            writes,
            int(stencil.group(1)) if stencil else 1,
            ranges,
            columns,
            structs,
        )

    def _resolver(self, kind, name, declared=None, ctype=None):
        """A function of (table, th, nface, given) giving one argument; what
        is given by keyword wins over what the table or species data hold.
        A column is handed over where the kernels run; a single record, the
        case's, on the host."""

        def given(value, keep, table):
            # a column is named: the kernel gets its device copy
            if kind == "view":
                return table.device(value)
            if kind == "column":
                return ctype(records=table.device(value))
            if kind in ("ints", "doubles"):
                value = np.ascontiguousarray(
                    value, dtype=np.int32 if kind == "ints" else np.float64
                )
                keep.append(value)
                return value.ctypes.data
            return value

        def tilingOf(table, nface):
            # facePlanes says how many layers before its components; faces
            # means the kernel's own direction
            rangeKind, *rest = declared
            if rangeKind == "faces":
                rangeKind = self.columns["F"][0] + "Faces"
            arg = None
            if rangeKind == "facePlanes":
                arg = table.ng if rest[0] == "ng" else int(rest[0])
                rest = rest[1:]
            components = rest[0] if rest else "1"
            return ctypes.addressof(table.tiling(rangeKind, arg, components, nface))

        def found(table, th, nface, keep, values):
            if kind == "count":
                return len(table)
            if kind == "struct":
                # one record of the struct, each member found as an argument is
                record, members = self.structs[name]
                filled = record()
                for mkind, mname, field, mtype in members:
                    if mkind == "own":
                        continue
                    resolve = self._resolver(mkind, mname, declared, mtype)
                    setattr(filled, field, resolve(table, th, nface, values, keep))
                keep.append(filled)
                return ctypes.addressof(filled)
            if kind == "column":
                # the table's column; the shape pins the rest
                return ctype(records=table.device(self.columns.get(name, name)))
            if kind == "record":
                return Record(r=View.of(getattr(th, name)))
            if kind == "dims":
                return Dims(all=table.device("dims"))
            if kind == "view":
                if th is not None and hasattr(th, name):
                    record = View.of(getattr(th, name))
                    keep.append(record)
                    return ctypes.addressof(record)
                return table.device(self.columns.get(name, name))
            if kind == "dims":
                return table.device("dims")
            if kind == "tiling":
                return tilingOf(table, nface)
            if kind == "ints":
                column = table.device(name)
                return ctype(all=column) if ctype is not None else column
            if kind == "int" and name == "nface":
                return nface
            if kind in self.scalars and th is not None and hasattr(th, name):
                return getattr(th, name)
            raise TypeError(f"a call needs {name}")

        def resolve(table, th, nface, values, keep):
            return (
                given(values.pop(name), keep, table)
                if name in values
                else found(table, th, nface, keep, values)
            )

        return resolve

    def __call__(self, table=None, th=None, nface=None, library=None, **given):
        """Run over a table's entries; :given: supplies the scalars, and any
        array by its parameter name. :library: is the one to call, when the
        name is not enough to say."""
        keep = []
        try:
            args = [r(table, th, nface, given, keep) for r in self.resolvers]
        except TypeError as e:
            raise TypeError(f"{self.name}: {e}") from None
        if given:
            raise TypeError(f"{self.name} takes no {', '.join(given)}")
        function = (
            getattr(lib, self.name)
            if library is None
            else lib.function(library, self.name)
        )
        return function(*args)


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
        columns=None,
        role=None,
        **fixed,
    ):
        self.table, self.thtrdat = table, thtrdat
        self.source, self.fixed = source, fixed
        self.defines, self.includes = tuple(defines), tuple(includes)
        # None is the case's block table; otherwise what to call for the table
        self.tableOf = tableOf
        path = Jit.compute / source
        # every header the source or a forced include reaches: a struct may
        # be declared in one
        headers = set()
        for f in (path, *(Jit.compute / i for i in self.includes)):
            Jit._headers(f, headers)
        self.kernel = Kernel.parse(
            path.read_text(),
            [(Jit.compute / i).read_text() for i in self.includes],
            columns,
            self.defines,
            [h.read_text() for h in headers],
        )
        # the library the jit compiled it into, once it has
        self.library = None
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
        return self.kernel(
            table, self.thtrdat, library=self.library, **{**self.fixed, **given}
        )


class KernelGroup:
    """A flux scheme's directions under the scheme's name, run one after the
    other."""

    def __init__(self, kernels):
        self.kernels = kernels
        self.__name__ = kernels[0].__name__

    def __call__(self, *args, **kwargs):
        for kernel in self.kernels:
            kernel(*args, **kwargs)


def byRole(kernels):
    """Every kernel under its role: a role whose kernels differ only by their
    defines (a scheme's directions) is the group of them, any other role is
    the last kernel bound to it."""
    roles = {}
    for k in kernels:
        roles.setdefault(k.role, []).append(k)
    return {
        role: KernelGroup(ks) if len({k.defines for k in ks}) > 1 else ks[-1]
        for role, ks in roles.items()
    }
