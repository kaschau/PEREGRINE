from .topologyFace import topologyFace


class topologyBlock:
    """topologyBlock object is the most basic object a peregrinepy.multiBlock
    (or one of its descendants) can be.
    """

    blockType = "topology"

    def __init__(self, nblki):
        self.nblki = nblki

        self.faces = []
        if self.blockType in ["topology", "grid", "restart"]:
            for fn in [1, 2, 3, 4, 5, 6]:
                self.faces.append(topologyFace(fn))

    def getFace(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.faces[int(nface) - 1]

    def neighborNfaceOf(self, nface, *, across):
        """Our face `nface` in the numbering of the neighbor across our face
        `across`. The two blocks lie side by side along `nface`, so it keeps
        its end. Not the connection itself, where the blocks face each other
        and ends swap; for that ask the face for its neighborNface."""
        face, connection = self.getFace(nface), self.getFace(across)
        assert (
            face.myAxis != connection.myAxis
        ), f"face {nface} is not adjacent to the connection on face {across}"
        axis, counterAligned = face.signedAxis(connection.orientation[face.myAxis])
        return 2 * axis + (1 if face.amILow != counterAligned else 2)
