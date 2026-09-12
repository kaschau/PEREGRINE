from functools import cache

import numpy as np

from ..kernels import boundaryConditions
from ..misc import null, subclasses, subclassWhere


class BaseBC:
    """One boundary condition.

    Everything the rest of the code needs to know about a bc is declared here:
    the name it goes by in the input files, which compute submodule holds the
    kernel that applies it, what it reads out of its config entry, and whether
    it sits on a block interface. Adding a bc means adding a subclass, and
    nothing else has a list to keep in step.

    The gradient rules a bc applies are deliberately absent -- they live in the
    C++ body alone. A second declaration of them here is what put v3's
    isoTSlipWall out of step with its own kernel.
    """

    # what this bc is called, which is what the grid's connectivity stores
    bcType = None
    # the family of kernel it is (walls, inlets, exits, periodics), or None for no kernel
    family = None
    # input key -> index into the face's qBcVals
    values = {}
    # whether the face is shared with another block rather than standing alone
    hasNeighbor = False

    @classmethod
    def kernel(cls):
        if cls.family is None:
            return null
        return getattr(boundaryConditions, cls.family)[cls.bcType]

    @classmethod
    def prep(cls, blk, face, valueDict, qBcVals, QBcVals):
        """Read this face's config entry into the host arrays it will be
        given."""
        if valueDict.get("profile", False):
            cls._profile(blk, face, qBcVals, QBcVals)
        else:
            cls._constants(blk, face, valueDict, qBcVals, QBcVals)

    @classmethod
    def _profile(cls, blk, face, qBcVals, QBcVals):
        ng = blk.ng
        with open(
            f"./Input/profiles/{face.bcName}_{blk.nblki}_{face.nface}.npy", "rb"
        ) as f:
            qBcVals[ng:-ng, ng:-ng, :] = np.load(f)
            QBcVals[ng:-ng, ng:-ng, :] = np.load(f)
        # extend the profile out into the face's own halo
        for array in (qBcVals, QBcVals):
            array[0:ng, :, :] = array[[ng], :, :]
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]

    @classmethod
    def _constants(cls, blk, face, valueDict, qBcVals, QBcVals):
        for key, index in cls.values.items():
            qBcVals[:, :, index] = valueDict[key]


# called once per face of every block; the registry is fixed after import
@cache
def getBc(bcType):
    """The class for a bcType, which is also the check that it is one."""
    return subclassWhere(BaseBC, bcType=bcType)


@cache
def validBcTypes():
    return tuple(sorted(c.bcType for c in subclasses(BaseBC) if c.bcType))


def prep(blk, face, valueDict):
    """Give a face the values its entry sets, where the kernels run."""
    qBcVals = np.zeros(face.shapeOf("qBcVals"))
    QBcVals = np.zeros(face.shapeOf("QBcVals"))
    getBc(face.bcType).prep(blk, face, valueDict, qBcVals, QBcVals)
    face.allocate("qBcVals", "QBcVals")
    face.qBcVals.set(qBcVals)
    face.QBcVals.set(QBcVals)
