"""The host backends: an array is numpy memory, which the kernels read in
place. A host team is one thread, so a host section carries no launch
bound."""

import numpy as np

from .base import BaseBackend


class HostBackend(BaseBackend):
    """An array is numpy memory, which the kernels read in place."""

    def memory(self, array):
        array.data = np.zeros(array.shape, array.dtype, order=self.order)
        return array.data.ctypes.data

    def release(self, array):
        """numpy frees its own."""

    def toHost(self, array, component, wait):
        if component is None:
            return array.data.copy(order=self.order)
        return np.array(array.data[..., component], order=self.order)

    def fromHost(self, array, values, wait):
        array.data[...] = values


class SerialBackend(HostBackend):
    name = "serial"


class OpenMPBackend(HostBackend):
    name = "openmp"
