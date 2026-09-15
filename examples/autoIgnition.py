#!/usr/bin/env python
"""

Auto ignition flame calculation.

"""

from mpi4py import MPI  # noqa: F401
import peregrinepy as pg
import cantera as ct
import matplotlib.pyplot as plt
from pathlib import Path


def simulate():
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../src/peregrinepy/mixture/database/source")

    # Cantera stuff
    T, p = 1100.0, 101325
    gas = ct.Solution("CH4_O2_FFCMY.yaml")
    # set the gas state
    gas.TP = T, p
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")
    r1 = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r1])

    # PEREGRINE stuff
    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    config["timeIntegration"]["integrator"] = "rk4"
    config["mcPhysics"]["chemistry"] = True
    config["mcPhysics"]["mixture"] = "chem_CH4_O2_FFCMY"
    config["mcPhysics"]["nChemSubSteps"] = 10
    config["mcPhysics"]["eos"] = "tpg"
    config["mcPhysics"]["mixture"] = "thtr_CH4_O2_FFCMY.yaml"
    # the reactor's state, uniform over the block
    config["initialConditions"]["p"] = gas.P
    config["initialConditions"]["T"] = gas.T
    config["initialConditions"]["Y"] = {
        s: y for s, y in zip(gas.species_names, gas.Y) if y > 0.0
    }
    config.validateConfig()
    mb = pg.integrators.getSolver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[0.01, 0.01, 0.01]
        ),
    )

    blk = mb.blocks[0]
    ng = blk.ng
    for face in blk.faces:
        face.bcType = "adiabaticNoSlipWall"

    dt = 1e-9
    config["timeIntegration"]["dt"] = dt
    niterout = 1000
    pgT = []
    pgO2 = []
    ctT = []
    ctO2 = []
    t = []
    print(mb)
    print("Time   PEREGRINE  CANTERA")
    while mb.tme < 0.05:
        if mb.nrt % niterout == 0:
            q = blk.q.get()
            pgT.append(q[ng, ng, ng, 4])
            pgO2.append(q[ng, ng, ng, 7])
            ctT.append(gas.T)
            ctO2.append(gas.Y[2])
            t.append(mb.tme)

            print(f"{mb.tme:.2e} {q[ng,ng,ng,4]:.2f} {gas.T:.2f}")

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
        pg.abi.lib.initialize()
        simulate()
        pg.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
