import numpy as np

from ..misc import getCommRankSize


class topologyFace:
    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        # what kind of boundary it is, and the name a grid gives it for a
        # case to say what it is: None for an interface or an unnamed boundary
        self.bcType = "adiabaticSlipWall"
        self.bcName = None
        self.periodicRotation = None
        self.neighbor = None
        # which rank holds the neighbor
        self.commRank = None
        self.orientation = None
        self.periodicTranslation = None

    ###########################################################################
    # How a halo arriving through this face is moved onto it
    ###########################################################################
    def setPeriodic(self, rotation=None, translation=None):
        """A halo coming through here lands at R @ p + t. A translational
        periodic is not turned and a rotational one is not moved, so each
        names only its own and the other falls out as doing nothing."""
        self.periodicTranslation = (
            np.zeros(3) if translation is None else np.array(translation, np.float64)
        )
        self.periodicRotation = (
            np.eye(3) if rotation is None else np.array(rotation, np.float64)
        )

    @property
    def neighborPlaneAlignment(self):
        """How our neighbor's face plane lies against ours: whether its two
        axes cross ours, and which of them run backwards. The neighbor names
        one of our axes for each of its own, and the one naming the normal of
        the face we share is not in the plane, so it is dropped."""
        theirNormal = (self.neighborNface - 1) // 2
        theirPlane = [
            self.signedAxis(code)
            for m, code in enumerate(self.neighborOrientation)
            if m != theirNormal
        ]
        ourPlane = [m for m in range(3) if m != self.myAxis]
        transposed = theirPlane[0][0] == ourPlane[1]
        flipped = tuple(
            m for m, (_, counterAligned) in enumerate(theirPlane) if counterAligned
        )
        return transposed, flipped

    @property
    def transposed(self):
        """Whether the neighbor's face axes cross ours; not on a boundary."""
        return self.neighbor is not None and self.neighborPlaneAlignment[0]

    @property
    def flipped(self):
        """Which of the neighbor's face axes run backwards against ours;
        none on a boundary."""
        return self.neighborPlaneAlignment[1] if self.neighbor is not None else ()

    def alignToMe(self, plane):
        """Our neighbor's face plane, laid out so it matches ours element for
        element. The inverse of laying one of ours out the way they read it,
        so the flips come off before the transpose does."""
        transposed, flipped = self.neighborPlaneAlignment
        if flipped:
            plane = np.flip(plane, flipped)
        if transposed:
            plane = np.moveaxis(plane, (0, 1), (1, 0))
        return plane

    def plane(self, index):
        """The index-plane of a block array normal to this face. Which plane a
        face number picks out is topology; whether there is an array to pick it
        out of is the block's business."""
        return (slice(None),) * self.myAxis + (index,)

    @property
    def firstPlane(self):
        """The outermost plane of a block array on this face."""
        return self.plane(0 if self.amILow else -1)

    def copyFrom(self, other):
        """Take on another face's description: what kind of boundary it is,
        what it is called, how its neighbor's axes run against ours, and how a
        halo through it is moved. Not which block it meets -- only the caller
        knows that."""
        self.bcType = other.bcType
        self.bcName = other.bcName
        self.orientation = other.orientation
        if other.periodicRotation is None:
            self.periodicRotation = None
            self.periodicTranslation = None
        else:
            self.setPeriodic(
                rotation=other.periodicRotation,
                translation=other.periodicTranslation,
            )

    def setInterior(self, neighbor, orientation="123"):
        """This face is shared with a block now rather than bounding anything,
        so it is nameless, reads nothing from a case, and nothing moves a halo
        through it."""
        self.bcType = "interior"
        self.bcName = None
        self.neighbor = neighbor
        self.orientation = orientation
        self.periodicRotation = None
        self.periodicTranslation = None

    @staticmethod
    def rotationAbout(axis, degrees):
        """Turning about the unit vector :axis: by :degrees:, negative the
        other way. See http://paulbourke.net/geometry/rotate/"""
        u = np.array(axis, np.float64)
        u = u / np.linalg.norm(u)
        th = degrees * np.pi / 180.0
        ct, st = np.cos(th), np.sin(th)
        cross = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
        return ct * np.eye(3) + st * cross + (1.0 - ct) * np.outer(u, u)

    @staticmethod
    def signedAxis(code):
        """The signed axis an orientation character names: which axis, and
        whether it runs against ours."""
        n = int(code)
        return (n - 1) % 3, n > 3

    @staticmethod
    def orientationCode(axis, counterAligned):
        """The orientation character naming a signed axis."""
        return str(axis + 1 + (3 if counterAligned else 0))

    @property
    def myAxis(self):
        """the index axis this face bounds, 0, 1 or 2"""
        return (self.nface - 1) // 2

    @property
    def amILow(self):
        """whether this face is at the low end of its axis"""
        return self.nface % 2 == 1

    @property
    def direction(self):
        """the letter of the axis this face bounds, i, j or k"""
        return "ijk"[self.myAxis]

    @property
    def connOffRank(self):
        """Whether the block across this face is on another rank."""
        return self.neighbor is not None and self.commRank != getCommRankSize()[1]

    @property
    def neighborNface(self):
        """The face of our neighbor that we share. The orientation character
        of our axis names the neighbor's axis; an aligned neighbor meets our
        low face with its high one."""
        axis, counterAligned = self.signedAxis(self.orientation[self.myAxis])
        return 2 * axis + (1 if self.amILow == counterAligned else 2)

    @property
    def neighborOrientation(self):
        """The orientation string our neighbor holds for this connection: ours
        says where its axes run, so the inverse says where ours do."""
        out = [None, None, None]
        for ours, code in enumerate(self.orientation):
            theirs, counterAligned = self.signedAxis(code)
            out[theirs] = self.orientationCode(ours, counterAligned)
        assert None not in out, "orientation does not name three distinct axes"
        return "".join(out)

    @property
    def alignsNeighborBy(self):
        """The relabelling, as (perm, flips), that puts our neighbor's axes in
        our frame. Applied to the neighbor it makes this connection "123", so
        the two blocks lie inline and can simply be concatenated."""
        perm, flips = [], []
        for code in self.orientation:
            axis, counterAligned = self.signedAxis(code)
            perm.append(axis)
            flips.append(counterAligned)
        return perm, flips
