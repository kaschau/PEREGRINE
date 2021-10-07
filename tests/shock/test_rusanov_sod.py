import peregrinepy as pg
import numpy as np
from pathlib import Path


def test_rusanov_sod():
    import kokkos

    kokkos.initialize()

    # Left State
    pL = 1.0
    TL = 0.0035529237601776465
    uL = 0.0

    # Right State
    pR = 0.1
    TR = 0.0028423390081421173
    uR = 0.0

    x0 = 0.5
    t = 0.2
    dt = 1e-4

    nx = 201
    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    config["RHS"]["primaryAdvFlux"] = "rusanov"
    config["solver"]["timeIntegration"] = "rk1"
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[nx, 2, 2],
        lengths=[1, 0.1, 0.1],
    )

    mb.initSolverArrays(config)

    blk = mb[0]

    for face in blk.faces:
        face.connectivity["bcType"] = "adiabaticSlipWall"

    pg.mpiComm.blockComm.setBlockCommunication(mb)
    mb.unifyGrid()
    mb.computeMetrics()
    indx = np.where(blk.array["xc"][:, 1, 1] > x0)

    # Initialize domain to Left properties
    blk.array["q"][:, :, :, 0] = pL
    blk.array["q"][:, :, :, 1] = uL
    blk.array["q"][:, :, :, 2] = 0.0
    blk.array["q"][:, :, :, 3] = 0.0
    blk.array["q"][:, :, :, 4] = TL

    # Initialize Right properties
    blk.array["q"][indx, :, :, 0] = pR
    blk.array["q"][indx, :, :, 1] = uR
    blk.array["q"][indx, :, :, 4] = TR

    # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    while mb.tme < t:
        mb.step(dt)

    relpath = str(Path(__file__).parent)
    with open(relpath + "/rusanov_sod_results.npy", "rb") as f:
        q = np.load(f)
        Q = np.load(f)

    passfail = []
    passfail.append(np.allclose(blk.array["q"], q))
    passfail.append(np.allclose(blk.array["Q"], Q))

    kokkos.finalize()

    assert passfail
