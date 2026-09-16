"""PEREGRINE's C ABI from the python side: the runtime's functions, one
ctypes twin of every struct the C++ side declares in abi.hpp, and one of
every member a kernel struct is built from in arrays.hpp, laid out the same
(tests/abi pins the sizes both headers assert). The backends and
arrays are in backend.py, the tables in table.py, the kernels in kernel.py.

Nothing here knows what an array, a table or a kernel is for. A new C
function is one entry in Library.runtime; a new struct is one twin; nothing
else belongs in this file."""

import ctypes
from pathlib import Path


class Library:
    """The runtime and every kernel library the process has loaded. The
    runtime's functions are bound onto this when it is started; a kernel's
    is handed to the kernel by the jit out of the kernel's own library, so
    two libraries exporting one name never meet."""

    # the runtime's functions: argument types and result
    runtime = {
        "pgInitialize": ([], None),
        "pgFinalize": ([], None),
        "pgLayoutLeft": ([], ctypes.c_int),
        "pgOnHost": ([], ctypes.c_int),
        "pgBackend": ([], ctypes.c_char_p),
        "pgAllocate": ([ctypes.c_size_t], ctypes.c_void_p),
        "pgFree": ([ctypes.c_void_p], None),
        "pgToHost": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            None,
        ),
        "pgToDevice": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            None,
        ),
        "pgFence": ([], None),
    }

    def __init__(self):
        self._libs = {}

    def load(self, path, shared=False):
        """A library, loaded once; the runtime is shared so the kernels
        resolve Kokkos out of it, a kernel's is its own."""
        if path not in self._libs:
            mode = ctypes.RTLD_GLOBAL if shared else ctypes.RTLD_LOCAL
            self._libs[path] = ctypes.CDLL(str(path), mode=mode)
        return self._libs[path]

    def function(self, path, name, argtypes, restype=None):
        """:name: out of the library at :path:, typed."""
        function = getattr(self.load(path), name)
        function.argtypes, function.restype = argtypes, restype
        return function

    def initialize(self):
        """Load the runtime and start Kokkos. Nothing before this touches a
        device, so a pre or post processing run never needs one."""
        path = next(Path(__file__).parent.parent.glob("libpgruntime.*"))
        self.load(path, shared=True)
        for name, (argtypes, restype) in self.runtime.items():
            setattr(self, name, self.function(path, name, argtypes, restype))
        self.pgInitialize()

    def finalize(self):
        self.pgFinalize()

    def __getattr__(self, name):
        if name in self.runtime:
            raise AttributeError(f"{name}: the runtime is not initialized")
        raise AttributeError(name)


lib = Library()


###############################################################################
# The structs of abi.hpp
###############################################################################
class pgView(ctypes.Structure):
    """One array as a kernel receives it (pgView in abi.hpp): where it is,
    its extents, and its strides in elements, which is the only place the
    device layout is turned into an address."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("extent", ctypes.c_int * 5),
        ("stride", ctypes.c_long * 5),
    ]


# the record of an array an entry does not have: a kernel never reads it
null = pgView(None, 0, (ctypes.c_int * 5)(), (ctypes.c_long * 5)())


def strides(shape, order):
    """Element strides of a contiguous array of this shape and order."""
    out, step = [0] * len(shape), 1
    axes = range(len(shape)) if order == "F" else reversed(range(len(shape)))
    for axis in axes:
        out[axis] = step
        step *= shape[axis]
    return out


class pgCells(ctypes.Structure):
    """The cells of one entry a launch does (pgCells in abi.hpp): a start
    and an extent per axis, the components, and the item count."""

    _fields_ = [
        ("start", ctypes.c_int * 3),
        ("extent", ctypes.c_int * 4),
        ("n", ctypes.c_int),
    ]


class pgDims(ctypes.Structure):
    """A block's shape (pgDims in abi.hpp): cells per direction; the halo
    depth is compiled in."""

    _fields_ = [(n, ctypes.c_int) for n in ("ni", "nj", "nk")]

    @classmethod
    def of(cls, blk):
        return cls(blk.ni, blk.nj, blk.nk)


class pgTiling(ctypes.Structure):
    """One launch over every entry of a table (pgTiling in abi.hpp): the
    entry of each tile, the first tile and the items of each entry, each
    entry's cells, all where the kernels run, and the items a tile is."""

    _fields_ = [
        ("entry", ctypes.c_void_p),
        ("first", ctypes.c_void_p),
        ("items", ctypes.c_void_p),
        ("cells", ctypes.c_void_p),
        ("count", ctypes.c_int),
        ("tiles", ctypes.c_int),
        ("tile", ctypes.c_int),
    ]


###############################################################################
# The members of a kernel struct, as arrays.hpp declares them: python fills
# the first field of each, the launch shape pins the rest
###############################################################################
class Column(ctypes.Structure):
    """A block column (column<T, O>): the records, then what the launch
    pins, the entry's data and strides and the cell."""

    _fields_ = [
        ("records", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("stride", ctypes.c_int * 5),
        ("entry", ctypes.c_int),
        ("i", ctypes.c_int),
        ("j", ctypes.c_int),
        ("k", ctypes.c_int),
        ("pad", ctypes.c_int),
    ]


class FaceColumn(ctypes.Structure):
    """A face column (faceColumn<T>): the records, then the face's record
    and the halo cell."""

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
    """An integer column read as a value (perEntry<int>): the column, then
    the entry's value."""

    _fields_ = [("all", ctypes.c_void_p), ("value", ctypes.c_int)]


class DimsColumn(ctypes.Structure):
    """The dims column (dims): every entry's shape, then the entry's."""

    _fields_ = [("all", ctypes.c_void_p), ("at", ctypes.c_void_p)]
