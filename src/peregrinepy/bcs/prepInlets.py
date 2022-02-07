import numpy as np


def prep_constantVelocitySubsonicInlet(blk, face, valueDict):
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
    for i, spn in enumerate(blk.speciesNames[0:-1]):
        try:
            face.array["qBcVals"][:, :, 5 + i] = valueDict[spn]
        except KeyError:
            pass


def prep_supersonicInlet(blk, face, valueDict):
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
    face.array["qBcVals"][:, :, 0] = valueDict["p"]
    face.array["qBcVals"][:, :, 1] = valueDict["u"]
    face.array["qBcVals"][:, :, 2] = valueDict["v"]
    face.array["qBcVals"][:, :, 3] = valueDict["w"]
    face.array["qBcVals"][:, :, 4] = valueDict["T"]
    for i, spn in enumerate(blk.speciesNames[0:-1]):
        try:
            face.array["qBcVals"][:, :, 5 + i] = valueDict[spn]
        except KeyError:
            pass


def prep_constantMassFluxSubsonicInlet(blk, face, valueDict):
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
    nface = face.nface
    s1_ = face.s1_
    # Get the inlet face normal
    if nface == 1:
        nx = blk.array["inx"][s1_]
        ny = blk.array["iny"][s1_]
        nz = blk.array["inz"][s1_]
    elif nface == 2:
        nx = -blk.array["inx"][s1_]
        ny = -blk.array["iny"][s1_]
        nz = -blk.array["inz"][s1_]
    elif nface == 3:
        nx = blk.array["jnx"][s1_]
        ny = blk.array["jny"][s1_]
        nz = blk.array["jnz"][s1_]
    elif nface == 4:
        nx = -blk.array["jnx"][s1_]
        ny = -blk.array["jny"][s1_]
        nz = -blk.array["jnz"][s1_]
    elif nface == 5:
        nx = blk.array["knx"][s1_]
        ny = blk.array["kny"][s1_]
        nz = blk.array["knz"][s1_]
    elif nface == 6:
        nx = -blk.array["knx"][s1_]
        ny = -blk.array["kny"][s1_]
        nz = -blk.array["knz"][s1_]

    # Get the total mass flow per unit area
    mDotPerUnitArea = valueDict["mDotPerUnitArea"]
    # Set target values of rhou, rhov, rhow
    face.array["QBcVals"][:, :, 1] = nx * mDotPerUnitArea
    face.array["QBcVals"][:, :, 2] = ny * mDotPerUnitArea
    face.array["QBcVals"][:, :, 3] = nz * mDotPerUnitArea

    face.array["qBcVals"][:, :, 4] = valueDict["T"]
    for i, spn in enumerate(blk.speciesNames[0:-1]):
        try:
            face.array["qBcVals"][:, :, 5 + i] = valueDict[spn]
        except KeyError:
            pass
