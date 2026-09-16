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
    values = {"u": 1, "v": 2, "w": 3, "T": 4}


class SupersonicInlet(InletBC):
    bcType = "supersonicInlet"
    values = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}


class StagnationSubsonicInlet(InletBC):
    bcType = "stagnationSubsonicInlet"
    values = {"pt": 0, "Tt": 4}


class ConstantMassFluxSubsonicInlet(InletBC):
    """sets a target momentum from the face normal rather than a velocity"""

    bcType = "constantMassFluxSubsonicInlet"
    values = {"T": 4}

    def _constants(self, valueDict, qBcVals, QBcVals):
        super()._constants(valueDict, qBcVals, QBcVals)
        face, blk = self.face, self.face.blk
        # the normal has to point into the block
        sign = 1.0 if face.amILow else -1.0
        mDot = valueDict["mDotPerUnitArea"]
        _, n = blk.faceNormals(face.direction)
        for m in range(3):
            QBcVals[:, :, m + 1] = sign * face.boundary(n[m]) * mDot
