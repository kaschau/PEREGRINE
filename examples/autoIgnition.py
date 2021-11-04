#!/usr/bin/env python
"""

Auto ignition flame calculation.

"""

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import cantera as ct
import matplotlib.pyplot as plt
from pathlib import Path


def simulate():

    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../src/peregrinepy/thermo_transport/database/source")
    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    config["solver"]["timeIntegration"] = "strang"
    config["thermochem"]["chemistry"] = True
    config["thermochem"]["mechanism"] = "chem_CH4_O2_Stanford_Skeletal"
    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["spdata"] = "thtr_CH4_O2_Stanford_Skeletal.yaml"
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[2, 2, 2],
        lengths=[0.01, 0.01, 0.01],
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    ng = blk.ng
    for face in blk.faces:
        face.bcType = "adiabaticNoSlipWall"

    pg.mpiComm.blockComm.setBlockCommunication(mb)

    mb.unifyGrid()

    mb.computeMetrics()

    T, p = 1100.0, 101325
    gas = ct.Solution("CH4_O2_Stanford_Skeletal.yaml")
    # set the gas state
    gas.TP = T, p
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")
    r1 = ct.Reactor(gas)
    sim = ct.ReactorNet([r1])

    blk.array["q"][:, :, :, 0] = gas.P
    blk.array["q"][:, :, :, 4] = gas.T
    blk.array["q"][:, :, :, 5::] = gas.Y[0:-1]

    # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    dt = 1e-6
    niterout = 1000
    pgT = []
    pgO2 = []
    ctT = []
    ctO2 = []
    t = []
    print("Time   PEREGRINE  CANTERA")
    while mb.tme < 0.05:

        if mb.nrt % niterout == 0:
            pgT.append(blk.array["q"][ng, ng, ng, 4])
            pgO2.append(blk.array["q"][ng, ng, ng, 7])
            ctT.append(gas.T)
            ctO2.append(gas.Y[2])
            t.append(mb.tme)

            print(f"{mb.tme:.2e} {blk.array['q'][ng,ng,ng,4]:.2f} {gas.T:.2f}")

        mb.step(dt)
        sim.advance(mb.tme)

    plt.plot(t, pgT, label="PEREGRINE")
    plt.plot(t, ctT, label="CANTERA")
    plt.title("T [K]")
    plt.legend()
    plt.show()
    plt.plot(t, pgO2, label="PEREGRINE")
    plt.plot(t, ctO2, label="CANTERA")
    plt.title("O2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
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
