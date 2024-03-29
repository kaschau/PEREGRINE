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
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]
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


def prep_cubicSplineSubsonicInlet(blk, face, valueDict):
    from ..compute.pgkokkos import hostView5, view4

    # set the temperature and species values
    face.array["qBcVals"][:, :, 4] = valueDict["T"]
    face.intervalDt = valueDict["intervalDt"]
    for i, spn in enumerate(blk.speciesNames[0:-1]):
        try:
            face.array["qBcVals"][:, :, 5 + i] = valueDict[spn]
        except KeyError:
            pass
    # read in the alpha file
    with open(
        f"./Input/{face.bcFam}Alphas/alphas_{blk.nblki}_{face.nface}.npy", "rb"
    ) as f:
        au = np.load(f)
        av = np.load(f)
        aw = np.load(f)

    # ALWAYS ON THE HOST
    shape = (
        tuple([4])
        + tuple([au.shape[1]])
        + face.array["qBcVals"].shape[0:2]
        + tuple([3])
    )
    face.cubicSplineAlphas = hostView5(
        "cubicSplineAlphas",
        *shape,
    )

    ng = blk.ng
    alphas = np.array(face.cubicSplineAlphas, copy=False)
    alphas[:, :, ng:-ng, ng:-ng, 0] = au[:]
    alphas[:, :, ng:-ng, ng:-ng, 1] = av[:]
    alphas[:, :, ng:-ng, ng:-ng, 2] = aw[:]
    for g in range(ng):
        for m in range(3):
            alphas[:, :, g, :, m] = alphas[:, :, ng, :, m]
            alphas[:, :, -g - 1, :, m] = alphas[:, :, -ng - 1, :, m]
            alphas[:, :, :, g, m] = alphas[:, :, :, ng, m]
            alphas[:, :, :, -g - 1, m] = alphas[:, :, :, -ng - 1, m]

    # create the device space interval alphas (ones we are
    # actually using at any given time) (dont need the mirror or
    # array as all that is done on c++ side)
    shape = tuple([4]) + face.array["qBcVals"].shape[0:2] + tuple([3])
    face.intervalAlphas = view4("intervalAlphas", *shape)


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
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]
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
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]
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


def prep_stagnationSubsonicInlet(blk, face, valueDict):
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
            array[-ng::, :, :] = array[[-ng - 1], :, :]
            array[:, 0:ng, :] = array[:, [ng], :]
            array[:, -ng::, :] = array[:, [-ng - 1], :]
        return

    # We only need total temperature and pressure for this b
    face.array["qBcVals"][:, :, 0] = valueDict["pt"]
    face.array["qBcVals"][:, :, 4] = valueDict["Tt"]
    for i, spn in enumerate(blk.speciesNames[0:-1]):
        try:
            face.array["qBcVals"][:, :, 5 + i] = valueDict[spn]
        except KeyError:
            pass
