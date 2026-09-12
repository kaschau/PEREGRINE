import numpy as np


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
    # which folder of boundaryConditions/ holds it
    family = None
    # the hooks in the step it has a kernel for; an interior face has none
    hooks = ()
    # input key -> index into the face's qBcVals
    values = {}
    # whether the face is shared with another block rather than standing alone
    hasNeighbor = False

    @classmethod
    def header(cls, hook):
        """The header applying this bc at one hook, relative to src/compute."""
        return f"boundaryConditions/{cls.family}/{cls.bcType}/{hook}.hpp"

    @classmethod
    def setValues(cls, face, valueDict):
        """Give a face the values its config entry sets, where the kernels
        run. Some conditions have prep work of their own, a constant mass flux
        or a profile read off disk, so the entry is read rather than copied."""
        blk = face.blk
        qBcVals = np.zeros(face.shapeOf("qBcVals"))
        QBcVals = np.zeros(face.shapeOf("QBcVals"))
        if valueDict.get("profile", False):
            cls._profile(blk, face, qBcVals, QBcVals)
        else:
            cls._constants(blk, face, valueDict, qBcVals, QBcVals)
        face.allocate(qBcVals=qBcVals, QBcVals=QBcVals)

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
