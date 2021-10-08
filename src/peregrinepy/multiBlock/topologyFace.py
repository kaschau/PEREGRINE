

class topologyFace:

    __slots__ = (
        "nface",
        "bcFam",
        "bcType",
        "neighbor",
        "orientation"
    )

    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        self.bcFam = None
        self.bcType = "adiabaticSlipWall"
        self.neighbor = None
        self.orientation = None

    @property
    def neighborFace(self):

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
