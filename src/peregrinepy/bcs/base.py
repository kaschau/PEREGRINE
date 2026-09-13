import numpy as np


class BaseBC:
    """One boundary condition, on one face.

    The class declares everything the rest of the code needs to know about a
    kind of condition: the name it goes by in the input files, which folder of
    boundaryConditions/ holds it, the hooks of the step it has a kernel for,
    what it reads out of its config entry, and whether it sits on a block
    interface. Adding a bc means adding a subclass, and nothing else has a
    list to keep in step. An instance is a face's own: it reads that face's
    config entry.

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
    # hook -> which case of that hook's kernel this condition is; set with the registry
    kind = {}

    def __init__(self, face):
        self.face = face

    @classmethod
    def header(cls, hook):
        """The header applying this bc at one hook, relative to src/compute."""
        return f"boundaryConditions/{cls.family}/{cls.bcType}/{hook}.hpp"

    def setValues(self, valueDict):
        """Give the face the values its config entry sets, where the kernels
        run. Some conditions have prep work of their own, a constant mass flux
        or a profile read off disk, so the entry is read rather than copied."""
        face = self.face
        qBcVals = np.zeros(face.shapeOf("qBcVals"))
        QBcVals = np.zeros(face.shapeOf("QBcVals"))
        if valueDict.get("profile"):
            self._profile(valueDict["profile"], qBcVals, QBcVals)
        else:
            self._constants(valueDict, qBcVals, QBcVals)
        face.allocate(qBcVals=qBcVals, QBcVals=QBcVals)

    def _profile(self, directory, qBcVals, QBcVals):
        """The face's values from :directory:/<bcName>_<nblki>_<nface>.npy."""
        face, blk = self.face, self.face.blk
        ng = blk.ng
        with open(f"{directory}/{face.bcName}_{blk.nblki}_{face.nface}.npy", "rb") as f:
            qBcVals[ng:-ng, ng:-ng, :] = np.load(f)
            QBcVals[ng:-ng, ng:-ng, :] = np.load(f)
        # extend the profile out into the face's own halo
        for array in (qBcVals, QBcVals):
            array[0:ng, :, :] = array[[ng], :, :]
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]

    def _constants(self, valueDict, qBcVals, QBcVals):
        for key, index in self.values.items():
            qBcVals[:, :, index] = valueDict[key]
