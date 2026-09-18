"""One compiled kernel, described by its own C++: the prototype names the
function, the struct it takes lists what it runs on, and the source declares
its stencil and the kind of item it runs over. Python builds a ctypes twin
of the struct from the members, fills one argument per call, and hands its
address over with the tiling the call names. The jit hands a kernel its
function.

A kernel is one launch and nothing else: it does not compile itself, owns no
table and no range (a call names both), carries no tag (the solver's dict
names it), and knows nothing of the order it runs in, which is a graph's."""

import ctypes
import re
from functools import cache

import numpy as np

from .backend.abi import Column, Dims, BlockFaceColumn, PerEntryInt
from .backend.jit import Jit
from .ranges import CellCenterRange, CellFaceRange


class BaseKernel:
    """A kernel over the table and tiling a call gives. Each argument is
    filled by name: from the table for the columns, dims and per-entry
    integers, the tiling as given, and keywords for the rest; the species
    data is baked in by the jit."""

    scalars = {"int": ctypes.c_int, "double": ctypes.c_double, "bool": ctypes.c_bool}
    prototype = re.compile(r"PG_ABI\s+(\w+)\s+(pg\w+)\s*\(([^)]*)\)")
    stencilDeclaration = re.compile(r"PG_STENCIL\((\d+)\)")
    rangeDeclaration = re.compile(r"PG_RANGE\(([^)]*)\)")
    # a struct's members come first, then its operator
    structHead = r"struct\s+{name}\s*(?::\s*(?:public\s+)?(\w+))?\s*\{{(.*?)(?=KOKKOS_INLINE_FUNCTION|operator\(|\}};)"
    # the column aliases arrays.hpp and faces.hpp declare, read off the
    # headers: an alias may stand on another, and the column it ends on
    # says whether the member is pinned to a cell or to a halo cell of a
    # block face
    alias = re.compile(r"^using (\w+) = (\w+)[<;]", re.M)
    cellColumns = ("column", "cellFaceColumn")
    blockFaceColumns = ("haloColumn", "blockFaceColumn", "plainColumn")
    side = re.compile(r"cellCenter(LL|RR|L|R)$")

    @classmethod
    @cache
    def columnHeads(cls):
        """Reads the column aliases off the headers: head -> the twin python
        fills for a member of that head."""
        aliases = dict(
            cls.alias.findall(Jit.header("arrays.hpp") + Jit.header("faces.hpp"))
        )
        heads = {}
        for name in aliases:
            base = name
            while base in aliases:
                base = aliases[base]
            if base in cls.cellColumns:
                heads[name] = Column
            elif base in cls.blockFaceColumns:
                heads[name] = BlockFaceColumn
        return heads

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
        self.name, self.restype, self.params, self.structs = self._parse(texts)
        # how many halo layers it reaches into
        stencil = self.stencilDeclaration.search(texts[0])
        self.stencil = int(stencil.group(1)) if stencil else 1
        # the kind of item it runs over, and the components one is of
        declared = self.rangeDeclaration.findall(
            "".join(texts[: 1 + len(self.includes)])
        )
        tilings = sum(kind == "tiling" for kind, _ in self.params)
        if tilings != len(declared) or tilings > 1:
            raise ValueError(
                f"{self.name} takes {tilings} tilings and declares {len(declared)} ranges"
            )
        self.items, self.components = (
            self.itemsOf(declared[0]) if declared else (None, 1)
        )
        self.argtypes = [
            self.scalars.get(kind, ctypes.c_void_p) for kind, _ in self.params
        ]
        # the compiled function, once the jit has handed it over
        self.function = None
        # how each parameter is found at a call
        self.resolvers = [self._resolver(kind, pname) for kind, pname in self.params]

    def __repr__(self):
        return f"<{type(self).__name__} {self.__name__}>"

    def resolveComponents(self, ne):
        """Settles how many components an item is of for a case of :ne:
        equations: a declared count, or ne less a count."""
        if isinstance(self.components, str):
            _, _, less = self.components.partition("-")
            self.components = ne - (int(less) if less else 0)

    @property
    def tileKind(self):
        """Says which of the backend's tile knobs sizes a tile of this
        kernel's items: elements where an item is one element, else
        cells."""
        return (
            "elements" if self.items == "elements" or self.components != 1 else "cells"
        )

    @property
    def stages(self):
        """States what order a launch of this owes: one stage of one
        kernel."""
        return [[self]]

    ###########################################################################
    # Reading the C++
    ###########################################################################
    @staticmethod
    def itemsOf(declaration):
        """Reads a PG_RANGE declaration: the kind of item, and the
        components an item is of -- a count, or `ne` less a count, which
        the jit resolves for the case."""
        kind, *params = [x.strip() for x in declaration.split(",")]
        components = 1
        for param in params:
            key, sep, value = (x.strip() for x in param.partition("="))
            if key != "components" or not sep:
                raise ValueError(f"PG_RANGE({declaration}): what is {param}?")
            components = value if value.startswith("ne") else int(value)
        return kind, components

    @staticmethod
    def expand(text, defines):
        """Applies the jit's defines to a text: a bc names its struct through
        PG_BCTYPE and PG_BCHOOK."""
        for define in defines:
            key, _, value = define.partition("=")
            text = re.sub(rf"\b{key}\b", value, text)
        return text

    @classmethod
    def members(cls, structName, texts):
        """Lists a struct's data members as (kind, name, ctype, head), a
        base's first, found among the texts a source reaches."""
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
                    # leaves it, but the argument has to hold its bytes
                    mname = mname.partition("=")[0].strip()
                    members.append(("own", mname, cls.scalars[head], head))
                elif head in cls.columnHeads():
                    # a cell-center column seen from a cell face ends in its
                    # side, and so does the member's name: QL is the column Q
                    side = cls.side.match(head)
                    side = side.group(1) if side else ""
                    if not mname.endswith(side):
                        raise ValueError(
                            f"struct {structName}: {mname} is {head}, so it ends in {side}"
                        )
                    twin = cls.columnHeads()[head]
                    members.append(("column", mname.removesuffix(side), twin, head))
                elif head == "dims":
                    members.append(("dims", mname, Dims, head))
                elif head == "caseIn":
                    # one value the case holds where the kernels run, given
                    # at the call as the array holding it
                    members.append(("case", mname, ctypes.c_void_p, head))
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
        """Reads the one kernel a source declares, from its prototype: its C name
        and result, its parameters, and for a struct by reference its members
        and what they read and write. The parameter-list form of the three
        hand-unrolled schemes parses too; they are never called."""
        found = self.prototype.findall(texts[0])
        if len(found) != 1:
            raise ValueError(
                f"a kernel source declares one PG_ABI function, this one {len(found)}"
            )
        restype, name, params = found[0]
        parsed, structs = [], {}
        for p in params.split(","):
            words = p.split()
            # an arrayInfo by pointer is one per entry; by reference it is the case's one
            pname = words[-1].lstrip("*&")
            ptype = " ".join(words[:-1]) + ("*" if words[-1][0] in "*&" else "")
            ptype = ptype.replace(" *", "*").replace(" &", "*").replace("const ", "")
            if ptype in ("pgArrayInfo*", "pgArrayIn*", "pgArrayOut*"):
                parsed.append(("view", pname.rstrip("_")))
            elif ptype == "pgDims*":
                parsed.append(("dims", pname))
            elif ptype == "pgTiling*":
                parsed.append(("tiling", pname))
            elif ptype == "int*":
                parsed.append(("ints", pname))
            elif ptype == "double*":
                parsed.append(("doubles", pname))
            elif ptype.endswith("*"):
                # any other pointer is the kernel's own struct, by reference
                members = self.members(ptype[:-1], texts)
                fields = [
                    (f"m{i}", ctype) for i, (_, _, ctype, _) in enumerate(members)
                ]
                argument = type(ptype[:-1], (ctypes.Structure,), {"_fields_": fields})
                structs[pname] = (
                    argument,
                    [(k, n, f, c) for (k, n, c, _), (f, _) in zip(members, fields)],
                )
                parsed.append(("struct", pname))
            else:
                parsed.append((ptype, pname))
        restype = {"int": ctypes.c_int, "void": None}[restype]
        return name, restype, parsed, structs

    ###########################################################################
    # Filling a call
    ###########################################################################
    def _resolver(self, kind, name, ctype=None):
        """Makes the function of (table, tiling, given, keep) that finds one
        argument of a call: what is given by keyword wins over what the
        table holds, and a keyword for a column names another column of the
        table."""
        column = self.columns.get(name, name)
        fromTable = {
            "struct": lambda table, tiling, given, keep: self._argument(
                name, table, tiling, given, keep
            ),
            "column": lambda table, *_: ctype(arrayInfos=table.arrayInfos(column)),
            "view": lambda table, *_: table.arrayInfos(column),
            "dims": lambda table, *_: self._all(ctype, Dims, table.arrayInfos("dims")),
            "ints": lambda table, *_: self._all(
                ctype, PerEntryInt, table.arrayInfos(name)
            ),
            "tiling": lambda table, tiling, *_: ctypes.addressof(tiling),
        }
        fromKeyword = {
            "case": lambda value, table, keep: value.ptr,
            "column": lambda value, table, keep: ctype(
                arrayInfos=table.arrayInfos(value)
            ),
            "view": lambda value, table, keep: table.arrayInfos(value),
            "ints": lambda value, table, keep: self._hostArray(value, np.int32, keep),
            "doubles": lambda value, table, keep: self._hostArray(
                value, np.float64, keep
            ),
        }
        find = fromTable.get(kind, lambda *_: self._missing(name))
        take = fromKeyword.get(kind, lambda value, *_: value)

        def resolve(table, tiling, given, keep):
            if name in given:
                return take(given.pop(name), table, keep)
            return find(table, tiling, given, keep)

        return resolve

    def _argument(self, name, table, tiling, given, keep):
        """Fills the struct argument of this name for one call, each member
        found as an argument is."""
        argument, members = self.structs[name]
        filled = argument()
        for mkind, mname, field, mtype in members:
            if mkind != "own":
                resolve = self._resolver(mkind, mname, mtype)
                setattr(filled, field, resolve(table, tiling, given, keep))
        keep.append(filled)
        return ctypes.addressof(filled)

    @staticmethod
    def _all(ctype, twin, column):
        # a per-entry column is one twin for all entries, or the pointer itself
        return ctype(all=column) if ctype is twin else column

    @staticmethod
    def _hostArray(value, dtype, keep):
        value = np.ascontiguousarray(value, dtype=dtype)
        keep.append(value)
        return value.ctypes.data

    @staticmethod
    def _missing(name):
        raise TypeError(f"a call needs {name}")

    def __call__(self, table, tiling, **given):
        """Runs over the tiling's ranges of the table's entries; :given:
        supplies the scalars, and any array by its parameter name. A
        tiling of nothing launches nothing."""
        if self.function is None:
            raise RuntimeError(f"{self.__name__} is not compiled")
        if tiling.tileKind != self.tileKind:
            raise TypeError(
                f"{self.__name__} runs over {self.tileKind}, this tiling is of {tiling.tileKind}"
            )
        if not tiling.tiles:
            return None
        keep = []
        try:
            args = [r(table, tiling, given, keep) for r in self.resolvers]
        except TypeError as e:
            raise TypeError(f"{self.__name__}: {e}") from None
        if given:
            raise TypeError(f"{self.__name__} takes no {', '.join(given)}")
        return self.function(*args)


class CellCenterKernel(BaseKernel):
    """A kernel whose item is a cell center, or a cell center and one of
    its components: the flow kernels, and a boundary condition, which runs
    over the halo cells behind block faces."""

    def rangeOf(self, blk):
        """Gives the range object of a block this kernel's items are over."""
        return CellCenterRange(blk.extents, blk.ng)

    def concerns(self, face):
        """Says whether a block face's halo is this kernel's to do again
        once a message has filled it: every one."""
        return True

    def behind(self, face):
        """Gives the ranges of a block face this kernel does again once a
        message has filled its halo: the halo cells."""
        return CellCenterRange(face.blk.extents, face.blk.ng).halo(face.nface)


class BCKernel(CellCenterKernel):
    """One boundary condition's body at one hook: bc.cpp compiled with the
    bcType's header forced in, over the halo cells behind the block faces
    carrying that bcType."""

    def __init__(self, bc, bcHook):
        self.bcType, self.bcHook = bc.bcType, bcHook
        super().__init__(
            "boundaryConditions/bc.cpp",
            defines=(f"PG_BCTYPE={bc.bcType}", f"PG_BCHOOK={bcHook}"),
            includes=(bc.header(),),
        )
        self.__name__ = f"{bc.bcType}@{bcHook}"


class CellFaceKernel(BaseKernel):
    """A flux kernel over the cell faces of one direction: the source names
    one direction's flux F, area vector A and faces, and the direction maps
    them to the table's; a scheme's three directions are a kernel group."""

    def __init__(self, source, direction):
        axis = "ijk"[direction]
        super().__init__(
            source,
            defines=(f"PG_DIRECTION={direction}",),
            columns={"F": f"{axis}F", "A": f"{axis}S", "Faces": f"{axis}Faces"},
        )
        self.direction = direction

    def rangeOf(self, blk):
        """Gives the range object of a block this kernel's items are over:
        the cell faces of its direction."""
        return CellFaceRange(blk.extents, blk.ng, self.direction)

    def concerns(self, face):
        """Says whether a block face's plane is this kernel's to do again:
        only the flux normal to a block face reads across it."""
        return face.myAxis == self.direction

    def behind(self, face):
        """Gives the ranges of a block face this kernel does again once a
        message has filled its halo: the plane of cell faces on it."""
        return CellFaceRange(
            face.blk.extents, face.blk.ng, self.direction
        ).blockFacePlane(face.nface)


class HaloExchangeKernel(BaseKernel):
    """A pack or an unpack over the buffer planes of trading block faces:
    the variable's array and buffer are named at the call."""


class BaseKernelGroup:
    """Several kernels under one tag, reaching as far as the widest. What
    order they owe each other is stated by :stages:: kernels independent
    within a stage, each stage after the last."""

    def __init__(self, kernels, name=None):
        self.kernels = kernels
        self.__name__ = name or kernels[0].__name__
        self.stencil = max(k.stencil for k in kernels)


class OrderedKernelGroup(BaseKernelGroup):
    """A group whose kernels run each after the last."""

    @property
    def stages(self):
        return [[kernel] for kernel in self.kernels]


class UnorderedKernelGroup(BaseKernelGroup):
    """A group whose kernels read and write nothing of each other's, so no
    order is owed between them."""

    @property
    def stages(self):
        return [self.kernels]
