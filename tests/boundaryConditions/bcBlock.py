import peregrinepy as pg
import numpy as np


def create(bc, adv, spdata):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = adv
    config["RHS"]["diffusion"] = True
    config["thermochem"]["spdata"] = spdata

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[8, 6, 4],
        lengths=[1, 1, 1],
    )

    mb.initSolverArrays(config)

    mb.generateHalo()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    blk = mb[0]
    for face in blk.faces:
        face.bcType = bc
        face.array["qBcVals"] = np.random.random(blk.ne)
        face.array["QBcVals"] = np.random.random(blk.ne)

    qshape = blk.array["q"][:, :, :, 0].shape
    p = np.random.uniform(low=101325.0 * 0.1, high=101325 * 10)
    u = np.random.uniform(low=-200, high=200, size=qshape)
    v = np.random.uniform(low=-200, high=200, size=qshape)
    w = np.random.uniform(low=-200, high=200, size=qshape)
    T = np.random.uniform(low=200, high=3000)
    if blk.ns > 1:
        Y = np.random.uniform(low=0.0, high=1.0, size=(blk.ns - 1))
        Y = Y / np.sum(Y)

    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 1] = u
    blk.array["q"][:, :, :, 2] = v
    blk.array["q"][:, :, :, 3] = w
    blk.array["q"][:, :, :, 4] = T
    if blk.ns > 1:
        blk.array["q"][:, :, :, 5::] = Y
    blk.updateDeviceView("q")

    mb.eos(blk, mb.thtrdat, 0, "prims")

    if blk.ns > 1:
        Ybc = np.random.uniform(low=0.0, high=1.0, size=blk.ns)
        Ybc = Ybc / np.sum(Ybc)

    for face in blk.faces:
        pbc = np.random.uniform(low=101325 * 0.9, high=101325 * 1.1)
        ubc = np.random.uniform(low=1, high=1000)
        vbc = np.random.uniform(low=1, high=1000)
        wbc = np.random.uniform(low=1, high=1000)
        Tbc = np.random.uniform(low=300 * 0.9, high=300 * 1.1)
        mDotPerAbc = np.random.uniform(low=1, high=1000)

        face.bcType = bc
        # Primative bcs
        face.array["qBcVals"] = np.zeros((blk.array["q"][face.s1_].shape))
        inputBcValues = {}
        inputBcValues["p"] = pbc
        inputBcValues["u"] = ubc
        inputBcValues["v"] = vbc
        inputBcValues["w"] = wbc
        inputBcValues["T"] = Tbc
        if blk.ns > 1:
            for n, spn in enumerate(blk.speciesNames[0:-1]):
                inputBcValues[spn] = Ybc[n]

        # Conservative like bcs
        face.array["QBcVals"] = np.zeros((blk.array["Q"][face.s1_].shape))
        inputBcValues["mDotPerUnitArea"] = mDotPerAbc
        # Just so we can check we set the target mdot to the zeroth (unused)
        # index of the QBcVals
        face.array["QBcVals"][:, :, 0] = mDotPerAbc

        for bcmodule in [pg.bcs.prepInlets, pg.bcs.prepExits, pg.bcs.prepWalls]:
            try:
                func = getattr(bcmodule, "prep_" + face.bcType)
                func(blk, face, inputBcValues)
                break
            except AttributeError:
                pass

        pg.misc.createViewMirrorArray(
            face, ["qBcVals", "QBcVals"], face.array["qBcVals"].shape
        )

    return mb
