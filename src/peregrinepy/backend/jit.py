"""Compiling the kernels a case needs, one library each, when it needs them.

The runtime and every kernel are compiled the same way, with the toolchain
of the Kokkos install, into the store, keyed by the source, the headers it
reaches, the toolchain and the defines. A case loads only the libraries it
will call, and each kernel is handed its own function out of its own
library.

The jit compiles and hands back callables; it knows nothing of tags, tables,
arrays, or the order kernels run in."""

import ctypes
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .abi import lib


class Jit:
    """Compiling kernels for one case: what the case bakes into every
    kernel, and what it forces into the ones that reach a model's header."""

    # what a source that reaches a data header is built with, by the table
    tableHeaders = (("species.hpp", "species"), ("reactions.hpp", "reactions"))

    def __init__(self, ng, mixture, simulation, launch=None):
        """Makes the compiler for one case: its halo depth, its mixture --
        the species count, species data and reactions -- and its simulation
        section, which names the equation of state, the species diffusion
        model and the mixing rule; on a device the backend's launch bound
        too."""
        # a kernel is compiled for one species count and halo depth, and on
        # a device for one launch bound, (threads, waves) from the backend's
        # config section; none is the host's unbounded launch
        self.ne = 5 + mixture.ns - 1
        self.fpdtype = {"double": "double", "single": "float"}[simulation["precision"]]
        self.fpctype = {"double": ctypes.c_double, "float": ctypes.c_float}[
            self.fpdtype
        ]
        self.defines = (
            f"NS={mixture.ns}",
            f"NE={self.ne}",
            f"NG={ng}",
            f"PG_FPDTYPE={self.fpdtype}",
            *(["PG_SINGLE=1"] if self.fpdtype == "float" else []),
        )
        if launch is not None:
            threads, waves = launch
            self.defines += (f"PG_LAUNCH_THREADS={threads}", f"PG_LAUNCH_WAVES={waves}")
        from . import getSources, getStore, getToolchain

        self.toolchain, self.sources, self.store = (
            getToolchain(),
            getSources(),
            getStore(),
        )
        # the mixture's tables, baked into a header each: the species data
        # the species kernels are built with, the reactions the chemistry
        # kernels are; the case's equation of state, forced in ahead of any
        # source that reaches thermo/eos.hpp; and its species diffusion
        # model, ahead of any that reaches transport/diffusion.hpp
        self.baked = {
            name: self.store.header(name, self.tablesText(name, tables))
            for name, tables in mixture.tables().items()
        }
        self.eos = simulation["eos"]
        self.diffusion = simulation["diffusion"]
        self.mixingRule = simulation["mixingRule"]

    ###########################################################################
    # What a case bakes and forces in
    ###########################################################################
    @classmethod
    def tablesText(cls, prefix, tables):
        """The tables of one case -- its species data, its reactions -- as
        one header of initializer lists, hexfloat so every value is exact,
        integers as they are; species.hpp and chemistry/reactions.hpp
        declare the accessors over them. A (rows, terms) array is written
        flat with its term count."""
        lines = [
            f"// the {prefix} data of one case, written by the jit",
            f"#define PG_{prefix.upper()}_DATA",
        ]
        for name, value in tables.items():
            macro = "PG_" + re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).upper()
            each = cls._ints if np.issubdtype(value.dtype, np.integer) else cls._doubles
            if value.ndim == 2:
                lines.append(f"#define {macro}_TERMS {value.shape[1]}")
                lines.append(f"#define {macro} {each(value.ravel())}")
            elif value.ndim == 0:
                lines.append(f"#define {macro} {each([value])[1:-1]}")
            else:
                lines.append(f"#define {macro} {each(value)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _doubles(a):
        # each in the case's precision: a double literal would narrow in the braces
        return "{" + ", ".join(f"fpdtype({float(x).hex()})" for x in a) + "}"

    @staticmethod
    def _ints(a):
        return "{" + ", ".join(str(int(x)) for x in a) + "}"

    def _case(self, source, includes):
        """What the case adds to a source's build: its defines and forced
        includes. A source that reaches species.hpp is built with the
        species data, one that reaches chemistry/reactions.hpp with the
        reactions; one that reaches thermo/eos.hpp with the case's eos header
        and PG_EOS naming it; one that reaches transport/diffusion.hpp or
        transport/mixingRule.hpp with the case's model's header and
        PG_DIFFUSION or PG_MIXING_RULE naming it."""
        compute = self.sources.compute
        names = {f.name for f in self.sources.files(source, includes)}
        defines, forced = (), tuple(compute / i for i in includes)
        if "diffusion.hpp" in names:
            if self.diffusion is None:
                raise ValueError(f"{source} needs a species diffusion model")
            defines += (f"PG_DIFFUSION={self.diffusion}",)
            forced = (
                compute / "transport" / "diffusion" / f"{self.diffusion}.hpp",
                *forced,
            )
        if "mixingRule.hpp" in names:
            defines += (f"PG_MIXING_RULE={self.mixingRule}",)
            forced = (
                compute / "transport" / "mixingRule" / f"{self.mixingRule}.hpp",
                *forced,
            )
        if "eos.hpp" in names:
            defines += (f"PG_EOS={self.eos}",)
            forced = (compute / "thermo" / f"{self.eos}.hpp", *forced)
        for header, name in self.tableHeaders:
            if header in names:
                if name not in self.baked:
                    raise ValueError(
                        f"{source} reads {name} the case's mixture does not have"
                    )
                forced = (self.baked[name], *forced)
        return defines, forced

    def _reached(self, source, includes, forced):
        """Every file a build reads: the source's walk, the forced
        includes' walks, and the baked headers among them."""
        files = set(self.sources.files(source, includes))
        for f in forced:
            files.add(f)
            if f.is_relative_to(self.sources.compute):
                self.sources.headers(f, files)
        return sorted(files)

    ###########################################################################
    # The kernels' libraries
    ###########################################################################
    def library(self, source, defines=(), includes=()):
        """Where the store keeps the library for one kernel source: keyed on
        everything it is compiled from, the toolchain, and the case's defines
        and includes."""
        caseDefines, forced = self._case(source, includes)
        defines = self.defines + caseDefines + tuple(defines)
        key = hashlib.sha256()
        for f in self._reached(source, includes, forced):
            key.update(f.read_bytes())
        key.update(self.toolchain.key.encode())
        key.update(" ".join(sorted(defines)).encode())
        key.update(" ".join(i.name for i in forced).encode())
        return self.store.library(source, key, self.toolchain)

    def build(self, source, defines=(), includes=()):
        """The library for one kernel source, compiled if the store has no
        current one. Returns its path."""
        out = self.library(source, defines, includes)
        caseDefines, includes = self._case(source, includes)
        defines = self.defines + caseDefines + tuple(defines)
        command = self.toolchain.command(
            self.sources.compute / source, out, defines, includes
        )
        return self.store.build(source, out, command)

    def compile(self, kernels):
        """Every kernel's function: the distinct requests among :kernels: are
        built at once, since they are independent, then each kernel is handed
        its function out of its own library."""
        requests = {(k.source, k.defines, k.includes) for k in kernels}
        with ThreadPoolExecutor() as pool:
            paths = dict(zip(requests, pool.map(lambda r: self.build(*r), requests)))
        for k in kernels:
            path = paths[(k.source, k.defines, k.includes)]
            k.resolveComponents(self.ne)
            k.resolveFpdtype(self.fpctype)
            k.function = lib.function(path, k.name, k.argtypes, k.restype)
