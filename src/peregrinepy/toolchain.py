"""How a kernel is compiled: the compiler and flags CMake settled on for the
runtime, read back out of its compile_commands.json. Run by the install
step; imports nothing from the package so it can run before the runtime is
in place."""

import json
import shlex
import sys
from pathlib import Path


class Toolchain:
    """A compiler, its flags, and how it links one source into one library."""

    def __init__(self, compiler, flags, link, suffix):
        self.compiler, self.flags, self.link, self.suffix = (
            compiler,
            flags,
            link,
            suffix,
        )

    @classmethod
    def fromCompileCommands(cls, commandsPath, suffix):
        """runtime.cpp's compile line with its own input and output taken out."""
        entries = json.loads(Path(commandsPath).read_text())
        entry = next(e for e in entries if e["file"].endswith("runtime.cpp"))
        words = (
            shlex.split(entry["command"]) if "command" in entry else entry["arguments"]
        )
        compiler, flags = words[0], []
        skip = False
        for w in words[1:]:
            if skip:
                skip = False
            elif w in ("-o", "-c"):
                skip = w == "-o"
            elif w != entry["file"] and not w.endswith("_EXPORTS"):
                flags.append(w)
        # a kernel library leaves Kokkos to the runtime it is loaded after
        link = ["-shared"] + (
            ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
        )
        return cls(compiler, flags, link, suffix)

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def write(self, path):
        Path(path).write_text(json.dumps(vars(self), indent=1))

    def command(self, source, out, defines=()):
        """The one command that compiles and links a source into a library."""
        return [
            self.compiler,
            *self.flags,
            *(f"-D{d}" for d in defines),
            *self.link,
            str(source),
            "-o",
            str(out),
        ]


if __name__ == "__main__":
    commands, suffix, out = sys.argv[1:4]
    Toolchain.fromCompileCommands(commands, suffix).write(out)
