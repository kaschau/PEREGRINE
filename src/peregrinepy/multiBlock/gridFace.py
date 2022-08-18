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
        rotUp[0, 0] = ct + ux ** 2 * (1 - ct)
        rotUp[0, 1] = ux * uy * (1 - ct) - uz * st
        rotUp[0, 2] = ux * uz * (1 - ct) + uy * st

        rotUp[1, 0] = uy * ux * (1 - ct) + uz * st
        rotUp[1, 1] = ct + uy ** 2 * (1 - ct)
        rotUp[1, 2] = uy * uz * (1 - ct) - ux * st

        rotUp[2, 0] = uz * ux * (1 - ct) - uy * st
        rotUp[2, 1] = uz * uy * (1 - ct) + ux * st
        rotUp[2, 2] = ct + uz ** 2 * (1 - ct)

        rotDown = np.zeros((3, 3))
        ct = np.cos(-th)
        st = np.sin(-th)
        rotDown[0, 0] = ct + ux ** 2 * (1 - ct)
        rotDown[0, 1] = ux * uy * (1 - ct) - uz * st
        rotDown[0, 2] = ux * uz * (1 - ct) + uy * st

        rotDown[1, 0] = uy * ux * (1 - ct) + uz * st
        rotDown[1, 1] = ct + uy ** 2 * (1 - ct)
        rotDown[1, 2] = uy * uz * (1 - ct) - ux * st

        rotDown[2, 0] = uz * ux * (1 - ct) - uy * st
        rotDown[2, 1] = uz * uy * (1 - ct) + ux * st
        rotDown[2, 2] = ct + uz ** 2 * (1 - ct)

        self.array["periodicRotMatrixUp"] = rotUp
        self.array["periodicRotMatrixDown"] = rotDown

    @property
    def neighborNface(self):

        nface = self.nface
        orientation = self.orientation

        # Which character in the orientation string do we look at to
        #  determine the orientation of a face's normal axis
        faceToOrientIndexMapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        # If we are a small face (1,3,5) and we have a direction (key)
        #  from above, what is our neighbor face (value)
        orientToSmallFaceMapping = {1: 2, 2: 4, 3: 6, 4: 1, 5: 3, 6: 5}
        # If we are a large face (2,4,6) and we have a direction (key)
        #  from above, what is our neighbor face (value)
        orientToLargeFaceMapping = {1: 1, 2: 3, 3: 5, 4: 2, 5: 4, 6: 6}

        direction = int(orientation[faceToOrientIndexMapping[nface]])

        if nface in [1, 3, 5]:
            nface2 = orientToSmallFaceMapping[direction]
        elif nface in [2, 4, 6]:
            nface2 = orientToLargeFaceMapping[direction]

        return nface2

    @property
    def neighborOrientation(self):

        orientation = self.orientation

        dirToOrientIndexMapping = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}
        neighborOrientation = [None, None, None]
        for i in range(3):
            n = int(orientation[i])
            indx = dirToOrientIndexMapping[n]

            if indx == i:
                neighborOrientation[indx] = str(n)
            elif n in (1, 2, 3):
                neighborOrientation[indx] = str(i + 1)
            elif n in (4, 5, 6):
                neighborOrientation[indx] = str(i + 4)

        assert (
            None not in neighborOrientation
        ), "Something wrong in getNeighborOrientation routine. Check connectivity"

        return "".join(neighborOrientation)
