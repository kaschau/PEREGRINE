from functools import cache

import numpy as np

from ..compute import bcs as computeBcs
from ..misc import null, subclasses, subclassWhere


class BaseBC:
    """One boundary condition.

    Everything the rest of the code needs to know about a bc is declared here:
    the name it goes by in the input files, which compute submodule holds the
    kernel that applies it, what it reads out of its bcFams entry, and whether
    it sits on a block interface. Adding a bc means adding a subclass, and
    nothing else has a list to keep in step.

    The gradient rules a bc applies are deliberately absent -- they live in the
    C++ body alone. A second declaration of them here is what put v3's
    isoTSlipWall out of step with its own kernel.
    """

    # the name in the grid file's connectivity and in bcFams.yaml
    bcType = None
    # the compute.bcs submodule holding the kernel, or None for no kernel
    family = None
    # input key -> index into the face's qBcVals
    values = {}
    # a bc with nothing to read can be left out of bcFams.yaml
    needsBcFam = True
    # whether the face is shared with another block rather than standing alone
    hasNeighbor = False

    @classmethod
    def kernel(cls):
        if cls.family is None:
            return null
        return getattr(getattr(computeBcs, cls.family), cls.bcType)

    @classmethod
    def prep(cls, blk, face, valueDict):
        """Read this face's bcFams entry onto it."""
        if valueDict.get("profile", False):
            cls._profile(blk, face)
        else:
            cls._constants(blk, face, valueDict)

    @classmethod
    def _profile(cls, blk, face):
        ng = blk.ng
        with open(
            f"./Input/profiles/{face.bcFam}_{blk.nblki}_{face.nface}.npy", "rb"
        ) as f:
            face.array["qBcVals"][ng:-ng, ng:-ng, :] = np.load(f)
            face.array["QBcVals"][ng:-ng, ng:-ng, :] = np.load(f)
        # extend the profile out into the face's own halo
        for array in (face.array["qBcVals"], face.array["QBcVals"]):
            array[0:ng, :, :] = array[[ng], :, :]
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]

    @classmethod
    def _constants(cls, blk, face, valueDict):
        for key, index in cls.values.items():
            face.array["qBcVals"][:, :, index] = valueDict[key]


# called once per face of every block; the registry is fixed after import
@cache
def getBc(bcType):
    """The class for a bcType, which is also the check that it is one."""
    return subclassWhere(BaseBC, bcType=bcType)


@cache
def validBcTypes():
    return tuple(sorted(c.bcType for c in subclasses(BaseBC) if c.bcType))


def prep(blk, face, valueDict):
    getBc(face.bcType).prep(blk, face, valueDict)
