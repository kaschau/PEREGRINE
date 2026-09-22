"""How the runtime and every kernel are compiled: with the compiler and
flags the Kokkos install at $Kokkos_ROOT was built with, which its cmake
files record, against its headers and libraries. Nothing is configured or
installed ahead of a run. A toolchain per Kokkos device adds what that
device's compiler does not do by itself."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class BaseToolchain:
    """A compiler, its flags, and how it links one source into one
    library: a kernel's, which leaves Kokkos to the runtime it is loaded
    after, or the runtime's, which carries the whole of Kokkos."""

    # the Kokkos device this toolchain compiles for
    device = None
    # a cmake setting Kokkos exports, quoted or not
    setting = r"^set\({name} \"?([^\")]*)\"?\)$"
    # the words of an interface property, inside cmake's generator expressions
    words = re.compile(r"\\\$<\\\$<(?:COMPILE|LINK)_LANGUAGE:CXX>:([^>]*)>")
    # kokkosalgorithms is headers; these three are what the runtime carries
    libraries = ("kokkoscore", "kokkoscontainers", "kokkossimd")

    @staticmethod
    def cmakeDir(root):
        """The directory of the install's cmake files: the install, or
        that directory itself as cmake's own Kokkos_DIR names it."""
        root = Path(root)
        cmake = root if (root / "KokkosConfig.cmake").is_file() else None
        cmake = cmake or next(root.glob("lib*/cmake/Kokkos"), None)
        if cmake is None:
            raise FileNotFoundError(
                f"no Kokkos install at {root}: Kokkos_ROOT names the prefix "
                "holding lib*/cmake/Kokkos"
            )
        return cmake

    @classmethod
    def devicesOf(cls, root):
        """The devices the install at :root: was built for."""
        common = (cls.cmakeDir(root) / "KokkosConfigCommon.cmake").read_text()
        return cls._setting(common, "Kokkos_DEVICES").split(";")

    def __init__(self, root):
        cmake = self.cmakeDir(root)
        self.root = cmake.parent.parent.parent
        common = (cmake / "KokkosConfigCommon.cmake").read_text()
        targets = (cmake / "KokkosTargets.cmake").read_text()
        self.compiler = self._setting(common, "Kokkos_CXX_COMPILER")
        self.arch = self._setting(common, "Kokkos_ARCH")
        standard = self._setting(common, "Kokkos_CXX_STANDARD")
        self.suffix = ".dylib" if sys.platform == "darwin" else ".so"
        # debug info and the asserts in a kernel are only wanted when looking
        # for a problem
        debug = bool(os.environ.get("PEREGRINE_JIT_DEBUG"))
        self.flags = [
            *(
                f"-D{d}"
                for d in self._property(targets, "INTERFACE_COMPILE_DEFINITIONS")
            ),
            f"-I{Path(__file__).parent.parent.parent / 'compute'}",
            "-isystem",
            str(self.root / "include"),
            "-O3",
            *(["-g"] if debug else ["-DNDEBUG"]),
            f"-std=gnu++{standard}",
            "-fPIC",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            *self._property(targets, "INTERFACE_COMPILE_OPTIONS"),
            *self.deviceFlags(),
        ]
        self.link = ["-shared"] + (
            ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
        )
        linked = self._property(targets, "INTERFACE_LINK_LIBRARIES")
        self.runtimeLink = [
            "-shared",
            *self._property(targets, "INTERFACE_LINK_OPTIONS"),
            *self.kokkosLink(cmake.parent.parent),
            *(["-ldl"] if "dl" in linked else []),
            *self.deviceLink(),
        ]

    def kokkosLink(self, libdir):
        """The link words that put Kokkos into the runtime: every object of
        a static build's archives, or a shared build's libraries, found where
        they are when the runtime loads."""
        archives = [libdir / f"lib{name}.a" for name in self.libraries]
        if all(a.is_file() for a in archives):
            return self.wholeArchive([str(a) for a in archives])
        return [
            f"-L{libdir}",
            *(f"-l{name}" for name in self.libraries),
            f"-Wl,-rpath,{libdir}",
        ]

    def wholeArchive(self, archives):
        """The link words that take every object of the archives."""
        if sys.platform == "darwin":
            return [f"-Wl,-force_load,{a}" for a in archives]
        return ["-Wl,--whole-archive", *archives, "-Wl,--no-whole-archive"]

    def deviceFlags(self):
        """What the device adds to a compile beyond what Kokkos records."""
        return []

    def deviceLink(self):
        """What the device adds to the runtime's link beyond what its
        compiler links by itself."""
        return []

    @classmethod
    def _setting(cls, text, name):
        m = re.search(cls.setting.format(name=name), text, re.M)
        if not m:
            raise ValueError(f"the Kokkos install does not record {name}")
        return m.group(1)

    @classmethod
    def _property(cls, text, name):
        """The words of every target's interface property, in order, once
        each: a plain list, or one wrapped in generator expressions."""
        found = []
        for value in re.findall(rf'{name} "([^"]*)"', text):
            for group in cls.words.findall(value) or [value]:
                found += [w for w in group.split(";") if w and w not in found]
        return found

    @property
    def key(self):
        """What a library built with this toolchain is keyed on."""
        return repr((self.compiler, self.flags, self.link, self.runtimeLink))

    # a sanitized build for the tests, from the environment
    sanitize = (
        ["-fsanitize=address", "-fno-omit-frame-pointer"]
        if os.environ.get("PEREGRINE_ASAN")
        else []
    )

    def command(self, source, out, defines=(), includes=(), link=None):
        """The one command that compiles and links a source into a library;
        :includes: are headers forced in ahead of it."""
        return [
            self.compiler,
            *self.flags,
            *(f"-D{d}" for d in defines),
            # two words: nvcc_wrapper only recognizes the flag on its own
            *(w for i in includes for w in ("-include", str(i))),
            *self.sanitize,
            str(source),
            *(self.link if link is None else link),
            "-o",
            str(out),
        ]


class SerialToolchain(BaseToolchain):
    device = "SERIAL"


class OpenMPToolchain(BaseToolchain):
    """The host compiler's OpenMP, which Kokkos records only as a cmake
    target: the flags are found by compiling a probe -- -fopenmp, or Apple's
    clang's front-end form with the libomp homebrew keeps."""

    device = "OPENMP"

    def __init__(self, root):
        self.openmp = None
        super().__init__(root)

    def _probe(self):
        if self.openmp:
            return
        candidates = [(["-fopenmp"], ["-fopenmp"])]
        brew = shutil.which("brew")
        if sys.platform == "darwin" and brew:
            prefix = subprocess.run(
                [brew, "--prefix", "libomp"], capture_output=True, text=True
            ).stdout.strip()
            candidates.append(
                (
                    ["-Xclang", "-fopenmp", "-isystem", f"{prefix}/include"],
                    [f"-L{prefix}/lib", "-lomp"],
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "openmp.cpp"
            probe.write_text(
                "#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n"
            )
            for compile, link in candidates:
                result = subprocess.run(
                    [self.compiler, *compile, *link, str(probe), "-o", f"{tmp}/openmp"],
                    capture_output=True,
                )
                if result.returncode == 0:
                    self.openmp = (compile, link)
                    return
        raise EnvironmentError(f"{self.compiler} does not compile OpenMP")

    def deviceFlags(self):
        self._probe()
        return self.openmp[0]

    def deviceLink(self):
        self._probe()
        return self.openmp[1]


class CudaToolchain(BaseToolchain):
    """nvcc through Kokkos's wrapper, which links the CUDA runtime by
    itself; the driver library Kokkos also calls is linked through the
    stub beside nvcc, and a run finds the real one."""

    device = "CUDA"

    def deviceFlags(self):
        # a host function called from device code is an error, not a body
        # nvcc silently drops
        return ["--Werror", "cross-execution-space-call"]

    def deviceLink(self):
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise EnvironmentError("the CUDA toolchain needs nvcc on the path")
        cuda = Path(nvcc).resolve().parent.parent
        stubs = next(cuda.glob("lib*/stubs"), None) or next(
            cuda.glob("targets/*/lib/stubs")
        )
        return [f"-L{stubs}", "-lcuda"]


class HipToolchain(BaseToolchain):
    """hipcc, which compiles for the device and links its runtime by
    itself."""

    device = "HIP"

    def wholeArchive(self, archives):
        # -xhip covers every input after it, so the archives are put back
        return ["-x", "none", *super().wholeArchive(archives)]
