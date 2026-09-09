import numpy as np


class BaseBcPrep:
    """Reads a boundary condition's bcFam entry onto its face.

    The values come either from a profile file, the same way for every bc, or
    from constants named in the input; the keywords say which input key lands
    in which qBcVals index.
    """

    def __init__(self, **values):
        self.values = values

    def __call__(self, blk, face, valueDict):
        if valueDict.get("profile", False):
            self._profile(blk, face)
        else:
            self._constants(blk, face, valueDict)

    def _profile(self, blk, face):
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

    def _constants(self, blk, face, valueDict):
        for key, index in self.values.items():
            face.array["qBcVals"][:, :, index] = valueDict[key]


class InletPrep(BaseBcPrep):
    """an inlet also carries the composition"""

    def _constants(self, blk, face, valueDict):
        super()._constants(blk, face, valueDict)
        for i, spn in enumerate(blk.speciesNames[0:-1]):
            if spn in valueDict:
                face.array["qBcVals"][:, :, 5 + i] = valueDict[spn]


class MassFluxPrep(InletPrep):
    """target momentum from the face normal, rather than a velocity"""

    def _constants(self, blk, face, valueDict):
        super()._constants(blk, face, valueDict)
        d = {1: "i", 2: "i", 3: "j", 4: "j", 5: "k", 6: "k"}[face.nface]
        # the normal has to point into the block
        sign = 1.0 if face.nface in (1, 3, 5) else -1.0
        mDot = valueDict["mDotPerUnitArea"]
        for m, c in enumerate("xyz", start=1):
            face.array["QBcVals"][:, :, m] = (
                sign * blk.array[f"{d}n{c}"][face.s1_] * mDot
            )


_preps = {
    # walls
    "adiabaticNoSlipWall": BaseBcPrep(),
    "adiabaticSlipWall": BaseBcPrep(),
    "adiabaticMovingWall": BaseBcPrep(u=1, v=2, w=3),
    "isoTNoSlipWall": BaseBcPrep(T=4),
    "isoTSlipWall": BaseBcPrep(T=4),
    "isoTMovingWall": BaseBcPrep(u=1, v=2, w=3, T=4),
    # exits
    "constantPressureSubsonicExit": BaseBcPrep(p=0),
    "supersonicExit": BaseBcPrep(),
    # inlets
    "constantVelocitySubsonicInlet": InletPrep(u=1, v=2, w=3, T=4),
    "supersonicInlet": InletPrep(p=0, u=1, v=2, w=3, T=4),
    "stagnationSubsonicInlet": InletPrep(pt=0, Tt=4),
    "constantMassFluxSubsonicInlet": MassFluxPrep(T=4),
}


def prep(blk, face, valueDict):
    """Apply a face's bcFam values, however its bc wants them."""
    try:
        bcPrep = _preps[face.bcType]
    except KeyError:
        raise ValueError(f"No bc prep for {face.bcType}.")
    bcPrep(blk, face, valueDict)
