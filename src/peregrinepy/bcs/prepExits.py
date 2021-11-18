import numpy as np


def prep_constantPressureSubsonicExit(blk, face, valueDict):
    ng = blk.ng
    if "profile" in valueDict and valueDict["profile"]:
        with open(
            f"./Input/profiles/{face.bcFam}_{blk.nblki}_{face.nface}.npy", "rb"
        ) as f:
            face.array["qBcVals"][ng:-ng, ng:-ng, :] = np.load(f)
            face.array["QBcVals"][ng:-ng, ng:-ng, :] = np.load(f)
        # We will fill out the whole face just for kicks
        for array in [face.array["qBcVals"], face.array["QBcVals"]]:
            array[0:ng, :, :] = array[[ng], :, :]
            array[-ng::, :, :] = array[[-ng], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng], :]
        return

    # Otherwise set the constant value inputs
    face.array["qBcVals"][:, :, 0] = valueDict["p"]


def prep_supersonicExit(blk, face, valueDict):
    pass
