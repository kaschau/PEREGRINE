from ..misc import frozenDict


class connectivityDict(frozenDict):
    def __setitem__(self, key, value):
        if key == "bcType":
            if value not in [
                "s1",
                "b0",
                "b1",
                "constantVelocitySubsonicInlet",
                "constantPressureSubsonicExit",
                "adiabaticNoSlipWall",
                "adiabaticSlipWall",
                "adiabaticMovingWall",
                "isoTMovingWall",
            ]:
                raise KeyError(f"{value} is not a valid input for bcType.")
        elif key == "neighbor":
            if type(value) not in [type(None), int]:
                raise KeyError(f"{value} is not a valid input for neighbor.")
        elif key == "orientation":
            if type(value) not in [type(None), str]:
                raise KeyError(f"{value} is not a valid input for orientation.")
            if value is str and len(value) != 3:
                raise KeyError(f"{value} is not a valid input for orientation.")

        super().__setitem__(key, value)


class topologyFace:

    __slots__ = (
        "nface",
        "connectivity"
    )

    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        self.connectivity = connectivityDict(
            {
                "bcFam": None,
                "bcType": "adiabaticSlipWall",
                "neighbor": None,
                "orientation": None,
            }
        )

        self.connectivity._freeze()

    @property
    def neighborFace(self):

        nface = self.nface
        orientation = self.connectivity["orientation"]

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

        orientation = self.connectivity["orientation"]

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
