"""What a boundary of a block is, whatever the physics: the class that
declares a kind of boundary, which each physics puts its own under its
own base, and the inlet behaviors any physics' inlets share."""

import re
from functools import cache

import numpy as np

from ..backend.jit import Jit
from ..misc import subclasses, subclassWhere


class BaseBC:
    """One boundary condition, on one face.

    The class declares everything the rest of the code needs to know about a
    kind of condition: the name it goes by in the input files, which folder of
    boundaryConditions/ holds it, the bcHooks it has a kernel for,
    what it reads out of its config entry, and whether it sits on a block
    interface. Adding a bc means adding a subclass, under the physics it
    serves, and nothing else has a list to keep in step: a physics finds
    its conditions by name under its own base. An instance is a face's own:
    it reads that face's config entry.

    The gradient rules a bc applies are deliberately absent -- they live in the
    C++ body alone. A second declaration of them here is what put v3's
    isoTSlipWall out of step with its own kernel.
    """

    # what this bc is called, which is what the grid's connectivity stores
    bcType = None
    # input key -> index into the face's qBcVals
    values = {}

    def __init__(self, face):
        self.face = face

    @classmethod
    @cache
    def named(cls, bcType):
        """Gives the class for a bcType, which is also the check that it is
        one."""
        return subclassWhere(cls, bcType=bcType)

    @classmethod
    @cache
    def bcTypes(cls):
        """Names every boundary condition under this base, sorted."""
        return tuple(sorted(c.bcType for c in subclasses(cls) if c.bcType))

    @classmethod
    @cache
    def withHook(cls, bcHook):
        """Names every bcType with a kernel at :bcHook:, sorted, which is the
        order they launch in."""
        return tuple(t for t in cls.bcTypes() if bcHook in cls.named(t).bcHooks())

    @classmethod
    def header(cls):
        """The header holding this bc's bcHooks, relative to src/compute."""
        return f"boundaryConditions/{cls.bcType}.hpp"

    @classmethod
    @cache
    def bcHooks(cls):
        """The bcHooks this bc has a kernel for -- euler, preDqDxyz,
        postDqDxyz -- as its header declares them."""
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


class InletBC(BaseBC):
    """An inlet also carries the composition."""

    def _constants(self, valueDict, qBcVals, QBcVals):
        super()._constants(valueDict, qBcVals, QBcVals)
        # the species are the variables past p, u, v, w, T
        for i, name in enumerate(self.face.blk.primVars[5:]):
            if name in valueDict:
                qBcVals[:, :, 5 + i] = valueDict[name]


class MassFluxInletBC(InletBC):
    """An inlet that sets a target momentum from the face normal rather
    than a velocity."""

    def _constants(self, valueDict, qBcVals, QBcVals):
        super()._constants(valueDict, qBcVals, QBcVals)
        face, blk = self.face, self.face.blk
        # the normal has to point into the block
        sign = 1.0 if face.amILow else -1.0
        mDot = valueDict["mDotPerUnitArea"]
        _, n = blk.faceNormals(face.direction)
        for m in range(3):
            QBcVals[:, :, m + 1] = sign * face.blockFacePlane(n[m]) * mDot
