import re
from functools import cache

import numpy as np

from ..backend.jit import Jit


class BaseBC:
    """One boundary condition, on one face.

    The class declares everything the rest of the code needs to know about a
    kind of condition: the name it goes by in the input files, which folder of
    boundaryConditions/ holds it, the bcHooks it has a kernel for,
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
    # input key -> index into the face's qBcVals
    values = {}
    # whether the face is shared with another block rather than standing alone
    hasNeighbor = False

    def __init__(self, face):
        self.face = face

    @classmethod
    def header(cls):
        """The header holding this bc's bcHooks, relative to src/compute."""
        return f"boundaryConditions/{cls.family}/{cls.bcType}.hpp"

    @classmethod
    @cache
    def bcHooks(cls):
        """The bcHooks this bc has a kernel for -- euler, preDqDxyz,
        postDqDxyz -- as its header declares them; an interior face has
        none."""
        if cls.family is None:
            return ()
        text = Jit.header(cls.header())
        return tuple(re.findall(r"^struct (\w+) \{", text, re.M))

    def setValues(self, valueDict):
        """Give the face the values its config entry sets, where the kernels
        run. Some conditions have prep work of their own, a constant mass flux
        or a profile read off disk, so the entry is read rather than copied."""
        face = self.face
        shape = face.bcValuesShape
        qBcVals, QBcVals = np.zeros(shape), np.zeros(shape)
        if valueDict.get("profile"):
            self._profile(valueDict["profile"], qBcVals, QBcVals)
        else:
            self._constants(valueDict, qBcVals, QBcVals)
        face.allocate("qBcVals", shape)
        face.allocate("QBcVals", shape)
        face.qBcVals.set(qBcVals)
        face.QBcVals.set(QBcVals)

    def _profile(self, directory, qBcVals, QBcVals):
        """Reads the face's values from
        :directory:/<bcName>_<nblki>_<nface>.npy, one plane over the block
        face proper."""
        face, blk = self.face, self.face.blk
        with open(f"{directory}/{face.bcName}_{blk.nblki}_{face.nface}.npy", "rb") as f:
            qBcVals[...] = np.load(f)
            QBcVals[...] = np.load(f)

    def _constants(self, valueDict, qBcVals, QBcVals):
        for key, index in self.values.items():
            qBcVals[:, :, index] = valueDict[key]
