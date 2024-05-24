import numpy as np


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
        assert isinstance(value, int), f"nface must be an integer not {type(value)}"
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
            "stagnationSubsonicInlet",
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
        a = axis / np.linalg.norm(axis)
        self._periodicAxis = a

    @property
    def neighborNface(self):
        """Get the face number or our neighbor's face we share."""
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
