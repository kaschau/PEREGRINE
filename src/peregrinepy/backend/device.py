"""The device backends: an array is a runtime allocation, and its bytes
cross the bus through the runtime. A device section carries the launch
bound its kernels are compiled with; the measured defaults differ by
card, which is why each is its own class and section."""

import numpy as np

from .abi import lib
from .base import BaseBackend


class DeviceBackend(BaseBackend):
    """CUDA or HIP: an array is a runtime allocation, and its bytes cross the
    bus through the runtime. In the left layout a trailing component is a
    contiguous run, so only it crosses; in the right layout the whole array
    is pulled and sliced."""

    def memory(self, array):
        return lib.pgAllocate(array.nbytes)

    def release(self, array):
        lib.pgFree(array.ptr)

    def toHost(self, array, component, wait):
        if component is None:
            host = np.empty(array.shape, array.dtype, order=self.order)
            lib.pgToHost(array.ptr, host.ctypes.data, array.nbytes, wait)
            return host
        if self.order != "F":
            whole = self.toHost(array, None, True)
            return np.array(whole[..., component], order=self.order)
        host = np.empty(array.shape[:-1], array.dtype, order="F")
        lib.pgToHost(
            array.ptr + component * host.nbytes, host.ctypes.data, host.nbytes, True
        )
        return host

    def fromHost(self, array, values, wait):
        host = np.require(values, array.dtype, self.order)
        lib.pgToDevice(host.ctypes.data, array.ptr, array.nbytes, wait)


class CudaBackend(DeviceBackend):
    name = "cuda"


class HipBackend(DeviceBackend):
    name = "hip"
