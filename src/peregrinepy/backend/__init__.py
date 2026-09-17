"""Where the kernels run, and how they get there: the runtime and its ABI,
arrays on it, the tables the kernels run over, and the compiler that
builds them. Backend.fromRuntime(config) is the one entry; the rest of
the package stands on the backend it returns."""

from . import abi
from .array import BaseArray, PooledArray
from .base import BaseBackend as Backend
from .device import CudaBackend, DeviceBackend, HipBackend
from .host import HostBackend, OpenMPBackend, SerialBackend
from .jit import Jit
from .table import ArrayTable

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
    "ArrayTable",
]
