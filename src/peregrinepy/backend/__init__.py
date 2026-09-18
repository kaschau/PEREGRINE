"""Where the kernels run, and how they get there: the runtime and its ABI,
arrays on it, the tables the kernels run over, and the compiler that
builds them. Backend.fromRuntime(config) is the one entry; the rest of
the package stands on the backend it returns."""

import os
from functools import cache
from pathlib import Path

from . import abi
from .array import BaseArray, PooledArray
from .base import BaseBackend as Backend
from .device import CudaBackend, DeviceBackend, HipBackend
from .host import HostBackend, OpenMPBackend, SerialBackend
from .jit import Jit
from .sources import Sources
from .store import Store
from .table import ArrayTable
from .toolchain import (
    BaseToolchain,
    CudaToolchain,
    HipToolchain,
    OpenMPToolchain,
    SerialToolchain,
)


@cache
def getSources():
    """Gives the compute tree, src/compute, read once a process."""
    return Sources(Path(__file__).parent.parent.parent / "compute")


@cache
def getStore():
    """Gives the store, $PEREGRINE_CACHE or ~/.cache/peregrinepy, once a
    process."""
    return Store(
        os.environ.get("PEREGRINE_CACHE", Path.home() / ".cache" / "peregrinepy")
    )


@cache
def getToolchain():
    """Gives the toolchain of the Kokkos install $Kokkos_ROOT or $Kokkos_DIR
    names, once a process: the one for the device it was built for."""
    root = os.environ.get("Kokkos_ROOT") or os.environ.get("Kokkos_DIR")
    if not root:
        raise EnvironmentError(
            "Kokkos_ROOT names the Kokkos install the runtime and kernels are built against"
        )
    devices = BaseToolchain.devicesOf(root)
    for toolchain in (CudaToolchain, HipToolchain, OpenMPToolchain, SerialToolchain):
        if toolchain.device in devices:
            return toolchain(root)
    raise EnvironmentError(f"no toolchain for a Kokkos built for {devices}")


__all__ = [
    "abi",
    "BaseArray",
    "PooledArray",
    "Backend",
    "CudaBackend",
    "DeviceBackend",
    "HipBackend",
    "HostBackend",
    "OpenMPBackend",
    "SerialBackend",
    "Jit",
    "Sources",
    "Store",
    "ArrayTable",
    "BaseToolchain",
    "CudaToolchain",
    "HipToolchain",
    "OpenMPToolchain",
    "SerialToolchain",
    "getToolchain",
    "getSources",
    "getStore",
]
