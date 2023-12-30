import numpy as np

from ..misc import frozenDict
from .topologyFace import topologyFace


class gridFace(topologyFace):
    faceType = "grid"

    def __init__(self, nface):
        super().__init__(nface)

        self.array = frozenDict()
        self.array["periodicRotMatrixUp"] = None
        self.array["periodicRotMatrixDown"] = None

        if self.faceType == "grid":
            self.array._freeze()

    @topologyFace.bcType.setter
    def bcType(self, value):
        topologyFace.bcType.fset(self, value)

    @topologyFace.periodicAxis.setter
    def periodicAxis(self, axis):
        topologyFace.periodicAxis.fset(self, axis)

        # Do we need to compute the rotational matrix?
        if not self.bcType.startswith("periodicRot"):
            return
        elif self.periodicSpan is None:
            # To compute the rot matrix we need the span now.
            raise AttributeError("Must set periodicSpan before setting periodicAxis")

        # Compute rotation matrix for positive and negative rotation
        rotUp = np.zeros((3, 3))
        th = self.periodicSpan * np.pi / 180.0
        ct = np.cos(th)
        st = np.sin(th)
        ux, uy, uz = tuple(axis)
        rotUp[0, 0] = ct + ux**2 * (1 - ct)
        rotUp[0, 1] = ux * uy * (1 - ct) - uz * st
        rotUp[0, 2] = ux * uz * (1 - ct) + uy * st

        rotUp[1, 0] = uy * ux * (1 - ct) + uz * st
        rotUp[1, 1] = ct + uy**2 * (1 - ct)
        rotUp[1, 2] = uy * uz * (1 - ct) - ux * st

        rotUp[2, 0] = uz * ux * (1 - ct) - uy * st
        rotUp[2, 1] = uz * uy * (1 - ct) + ux * st
        rotUp[2, 2] = ct + uz**2 * (1 - ct)

        rotDown = np.zeros((3, 3))
        ct = np.cos(-th)
        st = np.sin(-th)
        rotDown[0, 0] = ct + ux**2 * (1 - ct)
        rotDown[0, 1] = ux * uy * (1 - ct) - uz * st
        rotDown[0, 2] = ux * uz * (1 - ct) + uy * st

        rotDown[1, 0] = uy * ux * (1 - ct) + uz * st
        rotDown[1, 1] = ct + uy**2 * (1 - ct)
        rotDown[1, 2] = uy * uz * (1 - ct) - ux * st

        rotDown[2, 0] = uz * ux * (1 - ct) - uy * st
        rotDown[2, 1] = uz * uy * (1 - ct) + ux * st
        rotDown[2, 2] = ct + uz**2 * (1 - ct)

        # if we are a solver face, these will be turned into
        # views/mirrors after this
        self.array["periodicRotMatrixUp"] = rotUp
        self.array["periodicRotMatrixDown"] = rotDown
