"""The compute tree, read once: what a kernel source is compiled from, and
the texts a kernel reads its struct out of."""

import re
from pathlib import Path


class Sources:
    """The C++ under one directory, src/compute: a source, the headers it
    reaches through its quoted includes, and their texts."""

    includeLine = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)

    def __init__(self, compute):
        self.compute = Path(compute)
        # every walk done, kept: a source's files never change under a run
        self.walked = {}

    def header(self, relpath):
        """The text of one file of the tree."""
        return (self.compute / relpath).read_text()

    def headers(self, path, seen):
        """Every header :path: reaches through its quoted includes that lives
        in the tree, transitively; the rest are the toolchain's."""
        for name in self.includeLine.findall(path.read_text()):
            for header in (path.parent / name, self.compute / name):
                if header.is_file():
                    if header not in seen:
                        seen.add(header)
                        self.headers(header, seen)
                    break
        return seen

    def files(self, source, includes=()):
        """What a kernel is compiled from: its source, the forced includes,
        and every header they reach, in that order."""
        includes = tuple(includes)
        if (source, includes) not in self.walked:
            path = self.compute / source
            forced = tuple(self.compute / i for i in includes)
            headers = set()
            for f in (path, *forced):
                self.headers(f, headers)
            self.walked[(source, includes)] = (
                path,
                *forced,
                *sorted(headers - {path, *forced}),
            )
        return self.walked[(source, includes)]

    def texts(self, source, includes=()):
        """Those files' texts, for a kernel to read its struct out of."""
        return [f.read_text() for f in self.files(source, includes)]
