from .base import BaseBC


class InletBC(BaseBC):
    """an inlet also carries the composition"""

    family = "inlets"

    def _constants(self, valueDict, qBcVals, QBcVals):
        super()._constants(valueDict, qBcVals, QBcVals)
        for i, spn in enumerate(self.face.blk.speciesNames[0:-1]):
            if spn in valueDict:
                qBcVals[:, :, 5 + i] = valueDict[spn]


class ConstantVelocitySubsonicInlet(InletBC):
    bcType = "constantVelocitySubsonicInlet"
    hooks = ("euler", "postDqDxyz")
    values = {"u": 1, "v": 2, "w": 3, "T": 4}


class SupersonicInlet(InletBC):
    bcType = "supersonicInlet"
    hooks = ("euler", "postDqDxyz")
    values = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}


class StagnationSubsonicInlet(InletBC):
    bcType = "stagnationSubsonicInlet"
    hooks = ("euler", "postDqDxyz")
    values = {"pt": 0, "Tt": 4}


class ConstantMassFluxSubsonicInlet(InletBC):
    """sets a target momentum from the face normal rather than a velocity"""

    bcType = "constantMassFluxSubsonicInlet"
    hooks = ("euler", "postEos", "postDqDxyz")
    values = {"T": 4}

    def _constants(self, valueDict, qBcVals, QBcVals):
        super()._constants(valueDict, qBcVals, QBcVals)
        face, blk = self.face, self.face.blk
        d = {1: "i", 2: "i", 3: "j", 4: "j", 5: "k", 6: "k"}[face.nface]
        # the normal has to point into the block
        sign = 1.0 if face.nface in (1, 3, 5) else -1.0
        mDot = valueDict["mDotPerUnitArea"]
        _, n = blk.faceNormals(d)
        for m in range(3):
            QBcVals[:, :, m + 1] = sign * face.boundary(n[m]) * mDot
