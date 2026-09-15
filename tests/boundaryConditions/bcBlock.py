import peregrinepy as pg
import numpy as np
from ..gases import configure


def create(bc, adv, gas):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = adv
    config["RHS"]["diffusion"] = True
    configure(config, gas)

    mb = pg.integrators.getSolver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[8, 6, 4], lengths=[1, 1, 1]
        ),
    )

    # perturb the ineterio points a bit
    for blk in mb.blocks:
        i = blk.ng + 1
        nodes = blk.nodes.get()
        size = nodes[i:-i, i:-i, i:-i].shape
        nodes[i:-i, i:-i, i:-i] += np.random.uniform(-1, 1, size) * 0.02
        blk.nodes.set(nodes)

    mb.generateHalo()
    mb.computeMetrics()

    blk = mb.blocks[0]
    for face in blk.faces:
        face.bcType = bc

    q = blk.q.get()
    qshape = q.shape[:3]
    p = np.random.uniform(low=101325.0 * 0.1, high=101325 * 10, size=qshape)
    u = np.random.uniform(low=-200, high=200, size=qshape)
    v = np.random.uniform(low=-200, high=200, size=qshape)
    w = np.random.uniform(low=-200, high=200, size=qshape)
    T = np.random.uniform(low=200, high=3000, size=qshape)
    if blk.ns > 1:
        Y = np.random.uniform(low=0.0, high=1.0, size=qshape + (blk.ns - 1,))
        Y = Y / np.sum(Y, axis=-1)[:, :, :, np.newaxis]

    q[:, :, :, 0] = p
    q[:, :, :, 1] = u
    q[:, :, :, 2] = v
    q[:, :, :, 3] = w
    q[:, :, :, 4] = T
    if blk.ns > 1:
        q[:, :, :, 5::] = Y
    blk.q.set(q)

    mb.stateFromPrims(nface=-1)

    mb.dqdxyz()

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
        inputBcValues = {}
        inputBcValues["p"] = pbc
        inputBcValues["u"] = ubc
        inputBcValues["v"] = vbc
        inputBcValues["w"] = wbc
        inputBcValues["T"] = Tbc
        inputBcValues["pt"] = pbc
        inputBcValues["Tt"] = Tbc
        if blk.ns > 1:
            for n, spn in enumerate(blk.speciesNames[0:-1]):
                inputBcValues[spn] = Ybc[n]

        # Conservative like bcs
        inputBcValues["mDotPerUnitArea"] = mDotPerAbc
        face.bc.setValues(inputBcValues)
        # the target mdot goes in the zeroth (unused) index of QBcVals, for the check
        QBcVals = face.QBcVals.get()
        QBcVals[:, :, 0] = mDotPerAbc
        face.QBcVals.set(QBcVals)

    return mb
