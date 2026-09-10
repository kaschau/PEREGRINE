import numpy as np

from ..bcs import validBcTypes


class topologyFace:
    faceType = "topology"

    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self._nface = nface
        self._bcFam = None
        self._bcType = "adiabaticSlipWall"
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
        assert (
            value in validBcTypes()
        ), f"{value} is not a valid bcType. Must be one of {validBcTypes()}"
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
        # a periodic face carries no axis until its bcFams entry is read, so a
        # grid straight off disk has periodics that do not know which way yet
        self._periodicAxis = None if axis is None else axis / np.linalg.norm(axis)

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
