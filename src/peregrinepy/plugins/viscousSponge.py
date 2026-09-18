import numpy as np

from ..graph import LaunchNode
from ..kernel import CellCenterKernel
from .base import BasePlugin


class ViscousSponge(BasePlugin):
    """The viscosity raised along a line from origin to ending, to
    multiplier times itself, once the transport properties are made:
    after consistify, over every cell."""

    name = "viscousSponge"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        origin, ending = np.array(cfgsect["origin"]), np.array(cfgsect["ending"])
        length = np.linalg.norm(ending - origin)
        normal = (ending - origin) / length
        # what the kernel takes: the unit normal, where it starts along it,
        # its length, and the multiplier
        self.values = dict(
            nx=normal[0],
            ny=normal[1],
            nz=normal[2],
            start=float(origin @ normal),
            length=float(length),
            mult=cfgsect["multiplier"],
        )
        self.kernel = CellCenterKernel("utils/viscousSponge.cpp")

    def declKernels(self):
        return {"viscousSponge": self.kernel}

    def after(self):
        return {"consistify": [LaunchNode(self.kernel, "all", **self.values)]}
