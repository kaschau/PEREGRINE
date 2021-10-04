#!/usr/bin/env python
"""

Couette Flow with top wall moving at 5m/s

"""
import kokkos
import peregrinepy as pg
import numpy as np

np.seterr(all="raise")


def simulate():

    config = pg.files.configFile()
    config["simulation"]["dt"] = 1e-5
    config["simulation"]["niter"] = 500000
    config["simulation"]["niterout"] = 10000
    config["simulation"]["niterprint"] = 1000
    config["RHS"]["diffusion"] = True

    mb = pg.multiblock.generateMultiblockSolver(1, config)
    pg.grid.create.multiblockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[5, 40, 2],
        lengths=[0.01, 0.1, 0.01],
    )

    mb.initSolverArrays(config)

    blk = mb[0]

    # face 1
    blk.getFaceConn(1)["bcFam"] = None
    blk.getFaceConn(1)["bcType"] = "b1"
    blk.getFaceConn(1)["neighbor"] = 0
    blk.getFaceConn(1)["orientation"] = "123"
    blk.getFace(1).commRank = 0
    # face 2
    blk.getFaceConn(2)["bcFam"] = None
    blk.getFaceConn(2)["bcType"] = "b1"
    blk.getFaceConn(2)["neighbor"] = 0
    blk.getFaceConn(2)["orientation"] = "123"
    blk.getFace(2).commRank = 0
    # face 3
    blk.getFaceConn(3)["bcFam"] = None
    blk.getFaceConn(3)["bcType"] = "adiabatic_noslip_wall"
    blk.getFaceConn(3)["neighbor"] = None
    blk.getFaceConn(3)["orientation"] = None
    blk.getFace(3).commRank = 0
    # face 4 isoT moving wall
    blk.getFaceConn(4)["bcFam"] = "whoosh"
    blk.getFaceConn(4)["bcType"] = "adiabatic_moving_wall"
    blk.getFaceConn(4)["neighbor"] = None
    blk.getFaceConn(4)["orientation"] = None
    blk.getFace(4).commRank = 0

    for face in [5, 6]:
        blk.getFaceConn(face)["bcFam"] = None
        blk.getFaceConn(face)["bcType"] = "adiabatic_slip_wall"
        blk.getFace(face).commRank = 0

    blk.getFace(4).bcVals = {"u": 5.0, "v": 0.0, "w": 0.0}

    mb.generateHalo()

    pg.mpicomm.blockComm.setBlockCommunication(mb)

    mb.unifyGrid()

    mb.computeMetrics()

    pg.writers.writeGrid(mb, config["io"]["griddir"])

    blk.array["q"][1:-1, 1:-1, 1, 0] = 101325.0
    blk.array["q"][:, :, :, 1:4] = 0.0
    blk.array["q"][1:-1, 1:-1, 1, 4] = 300.0

    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    pg.writers.writeRestart(mb, config["io"]["outputdir"], gridPath="../Grid")

    for niter in range(config["simulation"]["niter"]):

        if mb.nrt % config["simulation"]["niterprint"] == 0:
            print(mb.nrt, mb.tme)

        mb.step(config["simulation"]["dt"])

        if mb.nrt % config["simulation"]["niterout"] == 0:
            pg.writers.writeRestart(mb, config["io"]["outputdir"], gridPath="../Grid")


if __name__ == "__main__":
    try:
        from os import mkdir

        mkdir("./Grid")
        mkdir("./Input")
        mkdir("./Output")
    except FileExistsError:
        pass
    try:
        kokkos.initialize()
        simulate()
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
