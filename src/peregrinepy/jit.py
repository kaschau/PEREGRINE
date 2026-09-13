"""Compiling the kernels a case needs, one library each, when it needs them.

The runtime is built once by CMake and records how it was compiled in
toolchain.json; every kernel is compiled the same way, into a cache keyed by
its source, the headers it includes, the toolchain and its defines. A case
loads only the libraries it will call."""

import fcntl
import hashlib
import os
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
    # what every kernel includes; a change to these is a change to every kernel
    headers = ("abi.hpp", "kokkosTypes.hpp", "kernelUtils.hpp")
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

    def build(self, source, defines=(), includes=()):
        """The library for one kernel source, compiled if the cache has no
        current one. Returns its path. A source's own headers, in and under
        its directory, are part of its key; :defines: and :includes: are its
        own on top of the case's."""
        path = self.compute / source
        defines = self.defines + tuple(defines)
        includes = tuple(self.compute / i for i in includes)
        key = hashlib.sha256()
        headers = [self.compute / h for h in self.headers] + sorted(
            path.parent.rglob("*.hpp")
        )
        for f in (path, *headers, *includes):
            key.update(f.read_bytes())
        key.update(repr(vars(self.toolchain)).encode())
        key.update(" ".join(sorted(defines)).encode())
        key.update(" ".join(str(i) for i in includes).encode())
        out = (
            self.cacheDir / f"{path.stem}-{key.hexdigest()[:16]}{self.toolchain.suffix}"
        )
        if out.exists():
            return out

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
        os.remove(out.with_suffix(".lock"))
        return out

    def compile(self, kernels):
        """Every one of :kernels: not yet compiled: built at once, since they
        are independent, then loaded."""
        pending = [k for k in kernels if not k.compiled]
        with ThreadPoolExecutor() as pool:
            list(
                pool.map(lambda k: self.build(k.source, k.defines, k.includes), pending)
            )
        for k in pending:
            lib.load(self.build(k.source, k.defines, k.includes))
            k.compiled = True
