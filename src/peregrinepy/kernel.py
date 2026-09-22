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
from .backend import getSources
from .ranges import CellCenterRange, CellFaceRange


class BaseKernel:
    """A kernel over the table and tiling a call gives. Each argument is
    filled by name: from the table for the columns, dims and per-entry
    integers, the tiling as given, and keywords for the rest; the species
    data is baked in by the jit."""

    # a floating point value's ctype is the case's, settled at compile
    scalars = {"int": ctypes.c_int, "fpdtype": None, "bool": ctypes.c_bool}
    fpdtypes = {ctypes.c_double: np.float64, ctypes.c_float: np.float32}
    prototype = re.compile(r"PG_ABI\s+(\w+)\s+(pg\w+)\s*\(([^)]*)\)")
    stencilDeclaration = re.compile(r"PG_STENCIL\((\d+)\)")
    rangeDeclaration = re.compile(r"PG_RANGE\(([^)]*)\)")
    # a struct's members come first, then its operator
    structHead = r"struct\s+{name}\s*(?::\s*(?:public\s+)?(\w+))?\s*\{{(.*?)(?=KOKKOS_INLINE_FUNCTION|operator\(|\}};)"
    # the column aliases arrays.hpp and faces.hpp declare, read off the
    # headers: an alias may stand on another, and the kind it ends on says
    # whether the member is pinned to a cell or to a halo cell of a block
    # face
    alias = re.compile(r"^using (\w+) = (\w+)[<;]", re.M)
    cellColumns = (
        "column",
        "cellRank",
        "cellScal",
        "faceRank",
        "faceStrad",
        "cellStrad",
        "cellStradScal",
    )
    blockFaceColumns = ("haloColumn", "blockFaceColumn", "plainColumn")

    @classmethod
    @cache
    def columnHeads(cls):
        """Reads the column aliases off the headers: head -> the twin python
        fills for a member of that head."""
        aliases = dict(
            cls.alias.findall(
                getSources().header("arrays.hpp") + getSources().header("faces.hpp")
            )
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
        texts = [
            self.expand(t, self.defines)
            for t in getSources().texts(source, self.includes)
        ]
        # the C function, its parameters [(kind, name)] in call order, and a
        # struct argument's ctypes type and [(kind, member, field, ctype)]
        self.name, self.restype, self.params, self.structs = self._parse(texts)
        # how many halo layers it reaches into, said by the source or a
        # header forced into it
        declared = self.stencilDeclaration.findall(
            "".join(texts[: 1 + len(self.includes)])
        )
        self.stencil = max((int(n) for n in declared), default=1)
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
        # the compiled function, once the jit has handed it over, with the
        # argument types and struct layout settled for the case's precision
        self.function = None
        self.argtypes = None
        self.fpctype = None
        # how each parameter is found at a call
        self.resolvers = [self._resolver(kind, pname) for kind, pname in self.params]

    def __repr__(self):
        return f"<{type(self).__name__} {self.__name__}>"

    @staticmethod
    def bakedValues(owner, values):
        """Gives the defines that bake the config's values of one piece into
        it: PG_<OWNER>_<NAME>."""
        return [
            f"PG_{owner.upper()}_{name.upper()}={float(value)!r}"
            for name, value in dict(values).items()
        ]

    def resolveComponents(self, ne):
        """Settles how many components an item is of for a case of :ne:
        equations: a declared count, or ne less a count."""
        if isinstance(self.components, str):
            _, _, less = self.components.partition("-")
            self.components = ne - (int(less) if less else 0)

    def resolveFpdtype(self, fpctype):
        """Settles every floating point value of the call for the case's
        precision, :fpctype: its ctype: the scalar arguments' types, and
        each struct argument's layout."""
        self.fpctype = fpctype
        scalars = {**self.scalars, "fpdtype": fpctype}
        self.argtypes = [scalars.get(kind, ctypes.c_void_p) for kind, _ in self.params]
        for pname, (name, members) in self.structs.items():
            fields = [
                (f"m{i}", scalars.get(head, ctype))
                for i, (_, _, ctype, head) in enumerate(members)
            ]
            argument = type(name, (ctypes.Structure,), {"_fields_": fields})
            self.structs[pname] = (
                argument,
                [(k, n, f, c) for (k, n, c, _), (f, _) in zip(members, fields)],
            )

    @staticmethod
    def connOffRankFaces(blk):
        """Gives the numbers of a block's faces connected off the rank,
        whose halos a message brings."""
        return {f.nface for f in blk.faces if f.connOffRank}

    def reads(self, name):
        """Says whether this kernel names a column of the table's array
        :name:, from any side."""
        return any(
            kind == "column" and self.columns.get(mname, mname) == name
            for _, members in self.structs.values()
            for kind, mname, *_ in members
        )

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
            # a static member, a template head ahead of a method, or the
            # base's name is not data
            if not statement or statement.startswith(("static", "template", "using")):
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
                    members.append(("column", mname, cls.columnHeads()[head], head))
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
            elif ptype == "fpdtype*":
                parsed.append(("fpdtypes", pname))
            elif ptype.endswith("*"):
                # any other pointer is the kernel's own struct, by reference:
                # its members, laid out once the precision is settled
                structs[pname] = (ptype[:-1], self.members(ptype[:-1], texts))
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
                ctype, PerEntryInt, table.arrayInfos(column)
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
            "fpdtypes": lambda value, table, keep: self._ownArray(
                value, self.fpdtypes[self.fpctype], keep
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
    def _ownArray(value, dtype, keep):
        # the kernel writes into it, so it has to be the caller's own
        # array, in the case's precision
        value = np.asarray(value)
        if value.dtype != dtype or not value.flags.c_contiguous:
            raise TypeError(
                f"an array a kernel fills is the caller's, {np.dtype(dtype).name} and contiguous"
            )
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
        if self.fpctype is None:
            raise RuntimeError(f"{self.__name__} has no precision settled")
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
        return CellCenterRange(blk.extents, blk.ng, self.connOffRankFaces(blk))

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

    def __init__(self, source, direction, defines=(), includes=()):
        axis = "ijk"[direction]
        super().__init__(
            source,
            defines=(f"PG_DIRECTION={direction}", *defines),
            includes=includes,
            columns={"F": f"{axis}F", "A": f"{axis}S", "Faces": f"{axis}Faces"},
        )
        self.direction = direction

    def rangeOf(self, blk):
        """Gives the range object of a block this kernel's items are over:
        the cell faces of its direction."""
        return CellFaceRange(
            blk.extents, blk.ng, self.direction, self.connOffRankFaces(blk)
        )

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


class FluxKernel(CellFaceKernel):
    """An advective flux composed by the jit from the scheme the config
    names, a formula or reconstruct-limiter-formula, with a :secondary:
    formula blended in by a :switch: whose :switchValues: are baked in."""

    reconstructions = ("piecewiseConstant", "muscl")

    @classmethod
    def composed(cls, scheme):
        """Says whether a scheme is a composition of advFlux/flux.cpp, or a
        flux source of its own."""
        return (
            getSources().compute
            / "advFlux"
            / "formula"
            / f"{scheme.split('-')[-1]}.hpp"
        ).is_file()

    def __init__(self, scheme, direction, secondary=None, switch=None, switchValues=()):
        parts = scheme.split("-")
        formula = parts[-1]
        limiter = parts[1] if len(parts) == 3 else None
        # a formula alone takes the cells as they are, as far as it declares
        # it reaches
        reconstruct = "piecewiseConstant" if len(parts) == 1 else parts[0]
        if len(parts) not in (1, 3) or reconstruct not in self.reconstructions:
            raise ValueError(
                f"{scheme!r} is not formula or reconstruct-limiter-formula"
            )
        if (secondary is None) != (switch is None):
            raise ValueError("a secondary flux and a switch go together")
        defines = [
            f"PG_PRIMARY={formula}",
            f"PG_RECONSTRUCT={reconstruct}",
            f"PG_BASE={switch or reconstruct}",
        ]
        # the formulas ahead of the reconstruction, the switch on it after,
        # the limiter ahead of all
        includes = [
            f"advFlux/formula/{formula}.hpp",
            f"advFlux/reconstruct/{reconstruct}.hpp",
        ]
        if limiter:
            defines.append(f"PG_LIMITER={limiter}")
            includes.insert(0, f"advFlux/limiter/{limiter}.hpp")
        if secondary:
            defines.append(f"PG_SECONDARY={secondary}")
            includes.insert(1, f"advFlux/formula/{secondary}.hpp")
            includes.append(f"advFlux/switch/{switch}.hpp")
            defines += self.bakedValues(switch, switchValues)
        super().__init__("advFlux/flux.cpp", direction, defines, includes)
        self.__name__ = scheme + (f" with {secondary} by {switch}" if switch else "")


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

    def reads(self, name):
        """Says whether any kernel of the group reads the array :name:."""
        return any(k.reads(name) for stage in self.stages for k in stage)


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
