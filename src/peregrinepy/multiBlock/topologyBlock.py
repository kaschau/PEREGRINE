from .topologyFace import topologyFace


class topologyBlock:
    """topologyBlock object is the most basic object a peregrinepy.multiBlock
    (or one of its descendants) can be.
    """

    blockType = "topology"

    def __init__(self, nblki):
        self.nblki = nblki

        # base block in uncut grid
        self.baseNblki = nblki
        # slice of base block in uncut grid (None = entire block)
        self.baseSlice = None

        self.ni = 0
        self.nj = 0
        self.nk = 0

        self.faces = [self._newFace(nface) for nface in range(1, 7)]

    def setExtents(self, ni, nj, nk):
        """This block is this big. A block is built before anyone knows how
        big it is -- a mesher works it out, a reader finds it in the file --
        so this is the moment it can be filled in, and whatever a kind of
        block derives from its extents is derived here."""
        self.ni, self.nj, self.nk = int(ni), int(nj), int(nk)

    def splitAlong(self, axis, cutIndex):
        """The two halves of whatever this block holds along its extents, as
        {name: (low, high)}, taken before either half is resized. A topology
        block holds nothing but the extents themselves."""
        return {}

    def _newFace(self, nface):
        # the kind of face this kind of block is bounded by
        return topologyFace(nface)

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
