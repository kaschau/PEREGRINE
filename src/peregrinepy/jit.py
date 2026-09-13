"""Compiling the kernels a case needs, one library each, when it needs them.

The runtime is built once by CMake and records how it was compiled in
toolchain.json; every kernel is compiled the same way, into a cache keyed by
its source, the headers it includes, the toolchain and its defines. A case
loads only the libraries it will call."""

import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .abi import lib
from .toolchain import Toolchain


class Jit:
    """Compiling kernels for one case, into the cache they are kept in."""

    package = Path(__file__).parent
    compute = package.parent / "compute"
    includeLine = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
    cacheDir = Path(
        os.environ.get("PEREGRINE_CACHE", Path.home() / ".cache" / "peregrinepy")
    )

    def __init__(self, ns):
        # a kernel is compiled for one species count and halo depth; the depth
        # is known once every kernel of the case is
        self.ns, self.ng = ns, None
        self.toolchain = Toolchain.read(self.package / "toolchain.json")

    @property
    def defines(self):
        assert self.ng is not None, "the halo depth is not known yet"
        return (f"NS={self.ns}", f"NE={5 + self.ns - 1}", f"NG={self.ng}")

    def _headers(self, path, seen):
        """Every header :path: reaches through its quoted includes that lives
        in the compute tree, transitively; the rest are the toolchain's."""
        for name in self.includeLine.findall(path.read_text()):
            for header in (path.parent / name, self.compute / name):
                if header.is_file():
                    if header not in seen:
                        seen.add(header)
                        self._headers(header, seen)
                    break
        return seen

    def library(self, source, defines=(), includes=()):
        """Where the cache keeps the library for one kernel source: keyed on
        the source and every header it or a forced include reaches, the
        toolchain, and the case's defines and includes."""
        path = self.compute / source
        defines = self.defines + tuple(defines)
        includes = tuple(self.compute / i for i in includes)
        headers = set()
        for f in (path, *includes):
            self._headers(f, headers)
        key = hashlib.sha256()
        for f in (path, *includes, *sorted(headers - {path, *includes})):
            key.update(f.read_bytes())
        key.update(repr(vars(self.toolchain)).encode())
        key.update(" ".join(sorted(defines)).encode())
        key.update(" ".join(str(i) for i in includes).encode())
        return (
            self.cacheDir / f"{path.stem}-{key.hexdigest()[:16]}{self.toolchain.suffix}"
        )

    def build(self, source, defines=(), includes=()):
        """The library for one kernel source, compiled if the cache has no
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
        """Every one of :kernels: not yet compiled: built at once, since they
        are independent, then loaded."""
        pending = [k for k in kernels if not k.compiled]
        # two kernels bound from one source build one library
        requests = {(k.source, k.defines, k.includes) for k in pending}
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda r: self.build(*r), requests))
        for k in pending:
            lib.load(self.build(k.source, k.defines, k.includes))
            k.compiled = True
