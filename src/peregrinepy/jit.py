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
    """The kernels one config calls for, and the cache they are built into."""

    package = Path(__file__).parent
    compute = package.parent / "compute"
    # what every kernel includes; a change to these is a change to every kernel
    headers = ("abi.hpp", "kokkosTypes.hpp", "kernelUtils.hpp")
    cacheDir = Path(
        os.environ.get("PEREGRINE_CACHE", Path.home() / ".cache" / "peregrinepy")
    )

    def __init__(self, config, defines=()):
        self.config = config
        self.defines = tuple(defines)
        self.toolchain = Toolchain.read(self.package / "toolchain.json")

    @property
    def sources(self):
        """The kernel sources the config calls for, relative to src/compute."""
        rhs, mc = self.config["RHS"], self.config["mcPhysics"]
        picked = [f"thermo/{mc['eos']}.cpp"]
        if rhs["diffusion"]:
            picked.append(
                {
                    ("kineticTheory", "binary"): "transport/kineticTheory.cpp",
                    ("kineticTheory", "lewis"): "transport/kineticTheoryUnityLewis.cpp",
                    ("chungDenseGas", "lewis"): "transport/chungDenseGasUnityLewis.cpp",
                    ("constantProps", "lewis"): "transport/constantProps.cpp",
                }[(mc["trans"], mc["diffusion"])]
            )
            picked.append("diffFlux/alphaDampingFlux.cpp")
            if rhs["subgrid"] is not None:
                picked.append(f"subgrid/{rhs['subgrid']}.cpp")
        for flux in (rhs["primaryAdvFlux"], rhs["secondaryAdvFlux"]):
            if flux is not None:
                picked.append(f"advFlux/{flux}.cpp")
        if rhs["switchAdvFlux"] is not None:
            picked.append(
                f"switches/{rhs['switchAdvFlux'].removesuffix('Pressure')}.cpp"
            )
        integrator = self.config["timeIntegration"]["integrator"]
        picked.append(
            "timeIntegration/dualTime.cpp"
            if integrator == "dualTime"
            else "timeIntegration/rk4Stages.cpp"
        )
        # the utilities and boundary conditions every case may reach
        for folder in ("utils", "boundaryConditions"):
            picked += sorted(
                str(f.relative_to(self.compute))
                for f in (self.compute / folder).glob("*.cpp")
            )
        return picked

    def build(self, source):
        """The library for one kernel source, compiled if the cache has no
        current one. Returns its path."""
        path = self.compute / source
        key = hashlib.sha256()
        for f in (path, *(self.compute / h for h in self.headers)):
            key.update(f.read_bytes())
        key.update(repr(vars(self.toolchain)).encode())
        key.update(" ".join(sorted(self.defines)).encode())
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
                    result = subprocess.run(
                        self.toolchain.command(path, built, self.defines),
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode:
                        raise RuntimeError(
                            f"{source} did not compile:\n{result.stderr}"
                        )
                    shutil.move(built, out)
        os.remove(out.with_suffix(".lock"))
        return out

    def compile(self):
        """Build every kernel the config calls for, at once since they are
        independent, and needing no device: a login node can fill the cache."""
        with ThreadPoolExecutor() as pool:
            return list(pool.map(self.build, self.sources))

    def load(self):
        """Build what the config calls for and load it, so its kernels can be
        called."""
        for library in self.compile():
            lib.load(library)
