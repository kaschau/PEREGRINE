"""Compiling the kernels a case needs, one library each, when it needs them.

The runtime is built once by CMake and records how it was compiled in
toolchain.json; every kernel is compiled the same way, into a store keyed by
its source, the headers it includes, the toolchain and its defines. A case
loads only the libraries it will call, and each kernel is handed its own
function out of its own library.

The jit compiles and hands back callables; it knows nothing of tags, tables,
arrays, or the order kernels run in."""

import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

from .abi import lib
from .toolchain import Toolchain


class Jit:
    """Compiling kernels for one case, into the store they are kept in."""

    package = Path(__file__).parent
    compute = package.parent / "compute"
    includeLine = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
    cacheDir = Path(
        os.environ.get("PEREGRINE_CACHE", Path.home() / ".cache" / "peregrinepy")
    )

    def __init__(self, ns, ng, tileSize):
        # a kernel is compiled for one species count, halo depth and tile size
        self.defines = (
            f"NS={ns}",
            f"NE={5 + ns - 1}",
            f"NG={ng}",
            f"PG_TILE={tileSize}",
        )
        self.toolchain = Toolchain.read(self.package / "toolchain.json")

    ###########################################################################
    # The compute tree, read once
    ###########################################################################
    @classmethod
    def header(cls, relpath):
        """The text of one file of the compute tree."""
        return (cls.compute / relpath).read_text()

    @classmethod
    def _headers(cls, path, seen):
        """Every header :path: reaches through its quoted includes that lives
        in the compute tree, transitively; the rest are the toolchain's."""
        for name in cls.includeLine.findall(path.read_text()):
            for header in (path.parent / name, cls.compute / name):
                if header.is_file():
                    if header not in seen:
                        seen.add(header)
                        cls._headers(header, seen)
                    break
        return seen

    @classmethod
    @cache
    def files(cls, source, includes=()):
        """What a kernel is compiled from: its source, the forced includes,
        and every header they reach, in that order."""
        path = cls.compute / source
        forced = tuple(cls.compute / i for i in includes)
        headers = set()
        for f in (path, *forced):
            cls._headers(f, headers)
        return (path, *forced, *sorted(headers - {path, *forced}))

    @classmethod
    def texts(cls, source, includes=()):
        """Those files' texts, for a kernel to read its struct out of."""
        return [f.read_text() for f in cls.files(source, includes)]

    ###########################################################################
    # The store
    ###########################################################################
    def library(self, source, defines=(), includes=()):
        """Where the store keeps the library for one kernel source: keyed on
        everything it is compiled from, the toolchain, and the case's defines
        and includes."""
        defines = self.defines + tuple(defines)
        forced = tuple(self.compute / i for i in includes)
        key = hashlib.sha256()
        for f in self.files(source, tuple(includes)):
            key.update(f.read_bytes())
        key.update(repr(vars(self.toolchain)).encode())
        key.update(" ".join(sorted(defines)).encode())
        key.update(" ".join(str(i) for i in forced).encode())
        stem = Path(source).stem
        return self.cacheDir / f"{stem}-{key.hexdigest()[:16]}{self.toolchain.suffix}"

    def build(self, source, defines=(), includes=()):
        """The library for one kernel source, compiled if the store has no
        current one. Returns its path."""
        out = self.library(source, defines, includes)
        if out.exists():
            return out

        path = self.compute / source
        defines = self.defines + tuple(defines)
        includes = tuple(self.compute / i for i in includes)
        out.parent.mkdir(parents=True, exist_ok=True)
        # ranks on one node race to the same file; the first to the lock builds it
        with open(out.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not out.exists():
                # built aside and moved in whole, so a reader never sees a partial file
                with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                    built = Path(tmp) / out.name
                    # a sanitized run interposes its runtime into every child; not the compiler
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if k != "DYLD_INSERT_LIBRARIES"
                    }
                    result = subprocess.run(
                        self.toolchain.command(path, built, defines, includes),
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    if result.returncode:
                        raise RuntimeError(
                            f"{source} did not compile:\n{result.stderr}"
                        )
                    shutil.move(built, out)
        # whoever built it removes the lock; a waiter finds it already gone
        with contextlib.suppress(FileNotFoundError):
            os.remove(out.with_suffix(".lock"))
        return out

    def compile(self, kernels):
        """Every kernel's function: the distinct requests among :kernels: are
        built at once, since they are independent, then each kernel is handed
        its function out of its own library."""
        requests = {(k.source, k.defines, k.includes) for k in kernels}
        with ThreadPoolExecutor() as pool:
            paths = dict(zip(requests, pool.map(lambda r: self.build(*r), requests)))
        for k in kernels:
            path = paths[(k.source, k.defines, k.includes)]
            k.function = lib.function(path, k.name, k.argtypes, k.restype)
