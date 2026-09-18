"""Where the compiled libraries and the baked headers live: a store keyed
by content, shared by every case and every rank on the machine, filled by
whoever gets to a missing entry first."""

import contextlib
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class Store:
    """A directory of libraries and headers named by what they are made
    from."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def library(self, source, key, toolchain):
        """Where the library of :source: keyed by :key: goes."""
        return (
            self.directory
            / f"{Path(source).stem}-{key.hexdigest()[:16]}{toolchain.suffix}"
        )

    def header(self, prefix, text):
        """Writes a header once per distinct text, and gives its path: named
        by its text, so one written another way is another file."""
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        path = self.directory / f"{prefix}-{key}.hpp"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # written aside and moved in whole, so a reader never sees a partial file
            with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
                f.write(text)
            shutil.move(f.name, path)
        return path

    def build(self, source, out, command):
        """Runs a library's build unless the store has it; ranks on one
        node race to the same file, and the first to the lock builds it."""
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
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
                    command[command.index(str(out))] = str(built)
                    result = subprocess.run(
                        command, capture_output=True, text=True, env=env
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

    def runtime(self, sources, toolchain):
        """The runtime library -- Kokkos and the memory the kernels run on
        -- built like a kernel, against the whole of Kokkos, keyed on its
        source, the headers it reaches and the toolchain. Returns its path."""
        source = "runtime.cpp"
        key = hashlib.sha256()
        for f in sources.files(source):
            key.update(f.read_bytes())
        key.update(toolchain.key.encode())
        out = self.library(source, key, toolchain)
        command = toolchain.command(
            sources.compute / source, out, link=toolchain.runtimeLink
        )
        return self.build(source, out, command)
