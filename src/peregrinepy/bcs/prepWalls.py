import numpy as np


def prep_adiabaticNoSlipWall(blk, face, valueDict):
    pass


def prep_adiabaticSlipWall(blk, face, valueDict):
    pass


def prep_adiabaticMovingWall(blk, face, valueDict):
    ng = blk.ng
    try:
        profile = valueDict["profile"]
    except KeyError:
        profile = False

    if profile:
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
    face.array["qBcVals"][:, :, 1] = valueDict["u"]
    face.array["qBcVals"][:, :, 2] = valueDict["v"]
    face.array["qBcVals"][:, :, 3] = valueDict["w"]


def prep_isoTNoSlipWall(blk, face, valueDict):
    ng = blk.ng
    try:
        profile = valueDict["profile"]
    except KeyError:
        profile = False

    if profile:
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
    face.array["qBcVals"][:, :, 4] = valueDict["T"]


def prep_isoTSlipWall(blk, face, valueDict):
    ng = blk.ng
    try:
        profile = valueDict["profile"]
    except KeyError:
        profile = False

    if profile:
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
    face.array["qBcVals"][:, :, 4] = valueDict["T"]


def prep_isoTMovingWall(blk, face, valueDict):
    ng = blk.ng
    try:
        profile = valueDict["profile"]
    except KeyError:
        profile = False

    if profile:
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
    face.array["qBcVals"][:, :, 1] = valueDict["u"]
    face.array["qBcVals"][:, :, 2] = valueDict["v"]
    face.array["qBcVals"][:, :, 3] = valueDict["w"]
    face.array["qBcVals"][:, :, 4] = valueDict["T"]
