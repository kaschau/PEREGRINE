import kokkos
import peregrinepy as pg
import numpy as np


def create(bc="adiabaticSlipWall"):
    config = pg.files.configFile()
    if np.random.random() > 0.5:
        config["RHS"]["primaryAdvFlux"] = "secondOrderKEEP"
    else:
        config["RHS"]["primaryAdvFlux"] = "fourthOrderKEEP"
    if np.random.random() > 0.5:
        config["thermochem"]["spdata"] = ["Air"]
    else:
        config["thermochem"]["spdata"] = "thtr_CH4_O2_Stanford_Skeletal.yaml"

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[8, 6, 4],
        lengths=[1, 1, 1],
    )

    mb.generateHalo()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces:
        face.bcType = bc
        face.array["qBcVals"] = np.random.random(blk.ne)
        face.array["QBcVals"] = np.random.random(blk.ne)

    qshape = blk.array["q"][:, :, :, 0].shape
    p = np.random.uniform(low=10000, high=100000, size=qshape)
    u = np.random.uniform(low=1, high=1000, size=qshape)
    v = np.random.uniform(low=1, high=1000, size=qshape)
    w = np.random.uniform(low=1, high=1000, size=qshape)
    T = np.random.uniform(low=100, high=1000, size=qshape)
    if blk.ns > 1:
        Y = np.random.uniform(
            low=0.0, high=1.0, size=tuple(list(qshape) + [blk.ns - 1])
        )
        Y = Y / np.sum(Y, axis=-1)[:, :, :, np.newaxis]

    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 1] = u
    blk.array["q"][:, :, :, 2] = v
    blk.array["q"][:, :, :, 3] = w
    blk.array["q"][:, :, :, 4] = T
    if blk.ns > 1:
        blk.array["q"][:, :, :, 5::] = Y

    mb.eos(blk, mb.thtrdat, 0, "prims")

    dqdxshape = blk.array["dqdx"].shape
    blk.array["dqdx"][:] = np.random.random((dqdxshape))
    blk.array["dqdy"][:] = np.random.random((dqdxshape))
    blk.array["dqdz"][:] = np.random.random((dqdxshape))

    pbc = np.random.uniform(low=10000, high=100000)
    ubc = np.random.uniform(low=1, high=1000)
    vbc = np.random.uniform(low=1, high=1000)
    wbc = np.random.uniform(low=1, high=1000)
    Tbc = np.random.uniform(low=100, high=1000)
    mDotPerAbc = np.random.uniform(low=1, high=1000)
    if blk.ns > 1:
        Ybc = np.random.uniform(low=0.0, high=1.0, size=blk.ns)
        Ybc = Ybc / np.sum(Ybc)
    for face in blk.faces:
        face.bcType = bc
        # Primative bcs
        face.array["qBcVals"] = np.zeros((blk.ne))
        face.array["qBcVals"][0] = pbc
        face.array["qBcVals"][1] = ubc
        face.array["qBcVals"][2] = vbc
        face.array["qBcVals"][3] = wbc
        face.array["qBcVals"][4] = Tbc
        if blk.ns > 1:
            for n in range(blk.ns - 1):
                face.array["qBcVals"][5 + n] = Ybc[n]

        # Conservative like bcs
        face.array["QBcVals"] = np.zeros((blk.ne))
        face.array["QBcVals"][0] = mDotPerAbc

        for bcmodule in [pg.bcs.inlets, pg.bcs.exits, pg.bcs.walls]:
            try:
                func = getattr(bcmodule, "prep_" + face.bcType)
                func(blk, face)
                break
            except AttributeError:
                pass

        pg.misc.createViewMirrorArray(
            face, ["qBcVals", "QBcVals"], (blk.ne,), kokkos.HostSpace
        )

    return mb
