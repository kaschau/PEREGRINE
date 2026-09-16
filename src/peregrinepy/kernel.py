"""One compiled kernel, described by its own C++: the prototype names the
function, the struct it takes lists what it runs on, and the source declares
its stencil and its ranges. Python builds a ctypes twin of the struct from
the members, fills one record per call, and hands its address over with the
tilings. What a member is declared as says what the kernel does with it: an
`in` is read, an `out` written, and the step's graph orders kernels by that.
The jit hands a kernel its function; bind() settles what a call runs with.

A kernel is one launch and nothing else: it does not compile itself, owns no
table (it is bound to one), carries no tag (the solver's dict names it), and
knows nothing of the order it runs in, which is a flow's."""

import ctypes
import re

import numpy as np

from .abi import Column, DimsColumn, FaceColumn, PerEntryInt
from .bcs import getBc
from .jit import Jit
from .misc import subclassWhere
from .table import BaseRange


class Kernel:
    """A kernel over a table given at the call, or the bound one. Each
    argument is filled by name: from the table for the columns, dims,
    tilings and per-entry integers, and from keywords for the rest; the
    species data is baked in by the jit."""

    scalars = {"int": ctypes.c_int, "double": ctypes.c_double, "bool": ctypes.c_bool}
    prototype = re.compile(r"PG_ABI\s+(\w+)\s+(pg\w+)\s*\(([^)]*)\)")
    stencilDeclaration = re.compile(r"PG_STENCIL\((\d+)\)")
    rangeDeclaration = re.compile(r"PG_RANGE\(([^)]*)\)")
    # a struct's members come first, then its operator
    structHead = r"struct\s+{name}\s*(?::\s*(?:public\s+)?(\w+))?\s*\{{(.*?)(?=KOKKOS_INLINE_FUNCTION|operator\(|\}};)"
    known = ("pgView", "pgIn", "pgOut", "pgDims", "pgTiling", "int", "double", "bool")
    # a column member's head: where the thread looks and what the kernel does
    # with it; a cell-center column seen from a face ends in its side
    columnHeads = {
        "cellCenterIn": ("r", ""),
        "cellCenterOut": ("w", ""),
        "cellCenterInOut": ("rw", ""),
        "cellCenterL": ("r", "L"),
        "cellCenterR": ("r", "R"),
        "cellCenterLL": ("r", "LL"),
        "cellCenterRR": ("r", "RR"),
        "cellFaceIn": ("r", ""),
        "cellFaceOut": ("w", ""),
        "cellFaceInOut": ("rw", ""),
        "haloIn": ("r", ""),
        "haloOut": ("w", ""),
        "haloInOut": ("rw", ""),
        "blockFaceIn": ("r", ""),
        "recordIn": ("r", ""),
        "bufferIn": ("r", ""),
        "bufferOut": ("w", ""),
    }
    # the heads of a block face launch's columns, one twin for all
    blockFaceHeads = ("halo", "blockFace", "record", "buffer")

    def __init__(self, source, defines=(), includes=(), columns=None):
        self.source = source
        self.defines, self.includes = tuple(defines), tuple(includes)
        self.__name__ = source.rsplit("/", 1)[-1].removesuffix(".cpp")
        # what the source calls a column -> what the table calls it: a
        # direction's kernel names F and A, the table iF and iS
        self.columns = dict(columns or {})
        texts = [self.expand(t, self.defines) for t in Jit.texts(source, self.includes)]
        # the C function, its parameters [(kind, name)] in call order, and a
        # struct argument's ctypes type and [(kind, member, field, ctype)]
        self.name, self.restype, self.params, self.structs, reads, writes = self._parse(
            texts
        )
        self.reads = [self.columns.get(r, r) for r in reads]
        self.writes = [self.columns.get(w, w) for w in writes]
        # how many halo layers it reaches into
        stencil = self.stencilDeclaration.search(texts[0])
        self.stencil = int(stencil.group(1)) if stencil else 1
        # the ranges it and its forced includes declare, one per tiling it takes
        self.ranges = []
        for text in texts[: 1 + len(self.includes)]:
            for decl in self.rangeDeclaration.findall(text):
                self.ranges.append(self.rangeOf(decl))
        tilings = sum(kind == "tiling" for kind, _ in self.params)
        if tilings != len(self.ranges):
            raise ValueError(
                f"{self.name} takes {tilings} tilings and declares {len(self.ranges)} ranges"
            )
        self.argtypes = [
            self.scalars.get(kind, ctypes.c_void_p) for kind, _ in self.params
        ]
        # the compiled function, once the jit has handed it over
        self.function = None
        # what a call runs with unless it says otherwise
        self.table, self.fixed = None, {}
        # how each parameter is found at a call
        tilings = iter(self.ranges)
        self.resolvers = [
            self._resolver(kind, pname, next(tilings) if kind == "tiling" else None)
            for kind, pname in self.params
        ]

    def __repr__(self):
        return f"<{type(self).__name__} {self.__name__}>"

    ###########################################################################
    # Reading the C++
    ###########################################################################
    def rangeOf(self, declaration):
        """A PG_RANGE declaration as a range: its kind, then key = value
        parameters (ng and ne stand for the block's, and ne may be less a
        count, `ne - 1`); the cell faces are the kernel's own direction."""
        kind, *params = [x.strip() for x in declaration.split(",")]
        kwargs = {}
        for param in params:
            key, sep, value = (x.strip() for x in param.partition("="))
            if not sep:
                raise ValueError(f"PG_RANGE({declaration}): {param} is not key = value")
            kwargs[key] = value if value.startswith(("ng", "ne")) else int(value)
        if kind == "cellFaces":
            kwargs["axis"] = "ijk".index(self.columns["F"][0])
        return subclassWhere(BaseRange, kind=kind)(**kwargs)

    @staticmethod
    def expand(text, defines):
        """The jit's defines applied to a text: a bc names its struct through
        PG_BCTYPE and PG_BCHOOK."""
        for define in defines:
            key, _, value = define.partition("=")
            text = re.sub(rf"\b{key}\b", value, text)
        return text

    @classmethod
    def members(cls, structName, texts):
        """[(kind, name, ctype, head)] of a struct's data members, a base's
        first, found among the texts a source reaches."""
        for text in texts:
            m = re.search(cls.structHead.format(name=structName), text, re.S)
            if m:
                break
            # the name may be an alias: a bc's is what the jit's defines say
            alias = re.search(rf"using\s+{structName}\s*=\s*([\w:]+)\s*;", text)
            if alias:
                return cls.members(alias.group(1).rsplit("::", 1)[-1], texts)
        else:
            raise ValueError(f"no struct {structName} in the source or its headers")
        base, body = m.groups()
        members = cls.members(base, texts) if base else []
        for statement in re.sub(r"//.*", "", body).split(";"):
            statement = " ".join(statement.split())
            if not statement or statement.startswith("static"):
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
                    ctype = (
                        FaceColumn if head.startswith(cls.blockFaceHeads) else Column
                    )
                    members.append(("column", mname.removesuffix(side), ctype, access))
                elif head == "dims":
                    members.append(("dims", mname, DimsColumn, head))
                elif head == "perEntry<int>":
                    members.append(("ints", mname, PerEntryInt, head))
                elif head in ("int", "double") and pointer:
                    # an array given at the call
                    members.append((head + "s", mname, ctypes.c_void_p, head))
                elif head in cls.scalars and not pointer:
                    members.append((head, mname, cls.scalars[head], head))
                else:
                    raise ValueError(f"struct {structName}: what is {statement}?")
        return members

    def _parse(self, texts):
        """The one kernel a source declares, from its prototype: its C name
        and result, its parameters, and for a struct by reference its members
        and what they read and write. The parameter-list form of the three
        hand-unrolled schemes parses too; they are never called."""
        found = self.prototype.findall(texts[0])
        if len(found) != 1:
            raise ValueError(
                f"a kernel source declares one PG_ABI function, this one {len(found)}"
            )
        restype, name, params = found[0]
        parsed, reads, writes, structs = [], [], [], {}
        for p in params.split(","):
            words = p.split()
            # a record by pointer is one per entry; by reference it is the case's one
            pname = words[-1].lstrip("*&")
            ptype = " ".join(words[:-1]) + ("*" if words[-1][0] in "*&" else "")
            ptype = ptype.replace(" *", "*").replace(" &", "*").replace("const ", "")
            if ptype.endswith("*") and ptype[:-1] not in self.known:
                members = self.members(ptype[:-1], texts)
                fields = [
                    (f"m{i}", ctype) for i, (_, _, ctype, _) in enumerate(members)
                ]
                record = type(ptype[:-1], (ctypes.Structure,), {"_fields_": fields})
                structs[pname] = (
                    record,
                    [(k, n, f, c) for (k, n, c, _), (f, _) in zip(members, fields)],
                )
                for kind, mname, _, head in members:
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
        restype = {"int": ctypes.c_int, "void": None}[restype]
        return name, restype, parsed, structs, reads, writes

    ###########################################################################
    # Filling a call
    ###########################################################################
    def _resolver(self, kind, name, declared=None, ctype=None):
        """A function of (table, nface, given, keep) giving one argument;
        what is given by keyword wins over what the table holds. A column is
        the table's, where the kernels run."""

        def given(value, keep, table):
            # a column is named: the kernel gets the table's
            if kind == "view":
                return table.column(value)
            if kind == "column":
                return ctype(records=table.column(value))
            if kind in ("ints", "doubles"):
                value = np.ascontiguousarray(
                    value, dtype=np.int32 if kind == "ints" else np.float64
                )
                keep.append(value)
                return value.ctypes.data
            return value

        def tilingOf(table, nface):
            return ctypes.addressof(table.tiling(declared.at(nface)))

        def found(table, nface, keep, values):
            if kind == "struct":
                # one record of the struct, each member found as an argument is
                record, members = self.structs[name]
                filled = record()
                for mkind, mname, field, mtype in members:
                    if mkind == "own":
                        continue
                    resolve = self._resolver(mkind, mname, declared, mtype)
                    setattr(filled, field, resolve(table, nface, values, keep))
                keep.append(filled)
                return ctypes.addressof(filled)
            if kind == "column":
                # the table's column; the shape pins the rest
                return ctype(records=table.column(self.columns.get(name, name)))
            if kind == "dims":
                column = table.column("dims")
                return ctype(all=column) if ctype is DimsColumn else column
            if kind == "view":
                return table.column(self.columns.get(name, name))
            if kind == "tiling":
                return tilingOf(table, nface)
            if kind == "ints":
                column = table.column(name)
                return ctype(all=column) if ctype is PerEntryInt else column
            if kind == "int" and name == "nface":
                return nface
            raise TypeError(f"a call needs {name}")

        def resolve(table, nface, values, keep):
            return (
                given(values.pop(name), keep, table)
                if name in values
                else found(table, nface, keep, values)
            )

        return resolve

    def bind(self, table=None, **fixed):
        """What a call runs with unless it names its own: the table, and the
        scalars settled up front."""
        self.table, self.fixed = table, fixed
        return self

    def __call__(self, table=None, nface=None, **given):
        """Run over a table's entries; :given: supplies the scalars, and any
        array by its parameter name. :nface: narrows a cell-center range: 0
        the interior, 1..6 one block face's halo, -1 or none as declared."""
        if self.function is None:
            raise RuntimeError(f"{self.__name__} is not compiled")
        if table is None:
            table = self.table
        given = {**self.fixed, **given}
        bound = given.pop("nface", None)
        if nface is None:
            nface = bound
        keep = []
        try:
            args = [r(table, nface, given, keep) for r in self.resolvers]
        except TypeError as e:
            raise TypeError(f"{self.__name__}: {e}") from None
        if given:
            raise TypeError(f"{self.__name__} takes no {', '.join(given)}")
        return self.function(*args)


class CellCenterKernel(Kernel):
    """A kernel over the cell centers of the solver's blocks, bound to the
    block table."""


class CellFaceKernel(Kernel):
    """A flux kernel over the cell faces of one direction of the solver's
    blocks: the source names one direction's flux F, area vector A and
    faces, and the direction maps them to the table's; a scheme's three
    directions are a KernelGroup."""

    def __init__(self, source, direction):
        axis = "ijk"[direction]
        super().__init__(
            source,
            defines=(f"PG_DIRECTION={direction}",),
            columns={"F": f"{axis}F", "A": f"{axis}S", "Faces": f"{axis}Faces"},
        )
        self.direction = direction


class BlockFaceKernel(Kernel):
    """One bcType's body at one bcHook, over the halo cells of the block
    faces in a table given at the call: bc.cpp compiled with the bcType's
    header forced in."""

    def __init__(self, bcType, bcHook):
        super().__init__(
            "boundaryConditions/bc.cpp",
            defines=(f"PG_BCTYPE={bcType}", f"PG_BCHOOK={bcHook}"),
            includes=(getBc(bcType).header(),),
        )
        self.bcType, self.bcHook = bcType, bcHook


class HaloExchangeKernel(Kernel):
    """A pack or an unpack over the block faces of a halo exchange table
    given at the call: every trade of one variable, whose arrays are all of
    one ndim."""


class KernelGroup:
    """A flux scheme's directions, run one after the other: it reads and
    writes what any of them does and reaches as far as the widest."""

    def __init__(self, kernels):
        self.kernels = kernels
        self.__name__ = kernels[0].__name__
        self.reads = list(dict.fromkeys(r for k in kernels for r in k.reads))
        self.writes = list(dict.fromkeys(w for k in kernels for w in k.writes))
        self.stencil = max(k.stencil for k in kernels)

    def __call__(self, *args, **kwargs):
        for kernel in self.kernels:
            kernel(*args, **kwargs)
