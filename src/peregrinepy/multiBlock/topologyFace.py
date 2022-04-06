class topologyFace:

    faceType = "topology"

    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self._nface = nface
        self._bcFam = None
        self._bcType = "adiabaticNoSlipWall"
        self._neighbor = None
        self._orientation = None

        self.periodicSpan = None
        self._periodicAxis = None

    @property
    def nface(self):
        return self._nface

    @nface.setter
    def nface(self, value):
        assert type(value) is int, f"nface must be an integer not {type(value)}"
        assert 1 <= value <= 6, "nface must be between (1,6)"
        self._nface = value

    @property
    def bcFam(self):
        return self._bcFam

    @bcFam.setter
    def bcFam(self, value):
        tV = type(value)
        if tV not in (type(None), str):
            raise TypeError(f"bcFam must be a string not {type(value)}.")
        self._bcFam = value

    @property
    def bcType(self):
        return self._bcType

    @bcType.setter
    def bcType(self, value):
        tV = type(value)
        if tV not in (type(None), str):
            raise TypeError(f"bcType must be a string not {tV}")
        validBcTypes = (
            # Interior, periodic
            "b0",
            "periodicTransLow",
            "periodicTransHigh",
            "periodicRotLow",
            "periodicRotHigh",
            # Inlets
            "constantVelocitySubsonicInlet",
            "supersonicInlet",
            "constantMassFluxSubsonicInlet",
            "cubicSplineSubsonicInlet",
            # Exits
            "constantPressureSubsonicExit",
            "supersonicExit",
            # Walls
            "adiabaticNoSlipWall",
            "adiabaticSlipWall",
            "adiabaticMovingWall",
            "isoTNoSlipWall",
            "isoTSlipWall",
            "isoTMovingWall",
        )
        assert (
            value in validBcTypes
        ), f"{value} is not a valid bcType. Must be one of {validBcTypes}"
        self._bcType = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        tV = type(value)
        if tV not in (type(None), int):
            raise TypeError(f"neighbor must be a int not {type(value)}.")
        self._neighbor = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        tV = type(value)
        if tV not in (type(None), str):
            raise TypeError(f"orientation must be a str not {type(value)}.")
        self._orientation = value

    # Periodic stuff
    @property
    def periodicAxis(self):
        return self._periodicAxis

    @periodicAxis.setter
    def periodicAxis(self, axis):
        import numpy as np

        axis = axis / np.linalg.norm(np.array(axis))
        self._periodicAxis = axis

        if not self.bcType.startswith("periodicRot"):
            return
        elif self.periodicSpan is None:
            raise AttributeError("Please set periodicSpan before setting periodicAxis")

        # Compute rotation matrix for positive and negative rotatoin
        rotUp = np.zeros((3, 3))
        th = self.periodicSpan * np.pi / 180.0
        ct = np.cos(th)
        st = np.sin(th)
        ux, uy, uz = tuple(axis)
        rotUp[0, 0] = ct + ux ** 2 * (1 - ct)
        rotUp[0, 1] = ux * uy * (1 - ct) * uz * st
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
        ux, uy, uz = tuple(axis)
        rotDown[0, 0] = ct + ux ** 2 * (1 - ct)
        rotDown[0, 1] = ux * uy * (1 - ct) * uz * st
        rotDown[0, 2] = ux * uz * (1 - ct) + uy * st

        rotDown[1, 0] = uy * ux * (1 - ct) + uz * st
        rotDown[1, 1] = ct + uy ** 2 * (1 - ct)
        rotDown[1, 2] = uy * uz * (1 - ct) - ux * st

        rotDown[2, 0] = uz * ux * (1 - ct) - uy * st
        rotDown[2, 1] = uz * uy * (1 - ct) + ux * st
        rotDown[2, 2] = ct + uz ** 2 * (1 - ct)

        if self.faceType == "topology":
            self.periodicRotMatrixUp = rotUp
            self.periodicRotMatrixDown = rotDown
        elif self.faceType == "solver":
            from ..misc import createViewMirrorArray

            createViewMirrorArray(self, "periodicRotMatrixUp", (3, 3))
            createViewMirrorArray(self, "periodicRotMatrixDown", (3, 3))

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
