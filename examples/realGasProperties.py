#!/usr/bin/env python
"""

Real gas properties.

"""

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate():
    config = pg.files.configFile()
    config["RHS"]["diffusion"] = True
    config["thermochem"]["eos"] = "cubic"
    config["thermochem"]["spdata"] = ["CO2"]
    config["thermochem"]["trans"] = "chungDenseGasUnityLewis"
    config.validateConfig()
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

    mb.setBlockCommunication()

    mb.unifyGrid()

    mb.computeMetrics(config["RHS"]["diffOrder"])

    ps = np.linspace(25, 35, 3)
    Ts = np.linspace(600, 1600, 100)

    rhos = np.zeros((len(ps), len(Ts)))
    cps = np.zeros((len(ps), len(Ts)))
    hs = np.zeros((len(ps), len(Ts)))
    cs = np.zeros((len(ps), len(Ts)))

    mus = np.zeros((len(ps), len(Ts)))
    kappas = np.zeros((len(ps), len(Ts)))

    for j, p in enumerate(ps):
        for i, T in enumerate(Ts):
            # set the gas state
            blk.array["q"][:, :, :, 0] = p * 1e6
            blk.array["q"][:, :, :, 4] = T

            # Update cons
            blk.updateDeviceView(["q"])
            mb.eos(blk, mb.thtrdat, 0, "prims")
            # Update transport
            mb.trans(blk, mb.thtrdat, 0)

            blk.updateHostView(["Q", "qh", "qt"])
            rhos[j, i] = blk.array["Q"][ng, ng, ng, 0]
            cps[j, i] = blk.array["qh"][ng, ng, ng, 1]
            hs[j, i] = blk.array["qh"][ng, ng, ng, 2]
            cs[j, i] = blk.array["qh"][ng, ng, ng, 3]

            mus[j, i] = blk.array["qt"][ng, ng, ng, 0]
            kappas[j, i] = blk.array["qt"][ng, ng, ng, 1]

    fig, axs = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f"Thermo Properties of {config['thermochem']['spdata'][0]}")
    axs[0, 0].set_ylabel("rho [kg/m^3]")
    axs[0, 1].set_ylabel("Cp [J/kg.K]")
    axs[1, 0].set_ylabel("h [J/kg]")
    axs[1, 1].set_ylabel("c [m/s]")
    for i, p in enumerate(ps):
        axs[0, 0].plot(Ts, rhos[i, :], label=f"p={p} MPa")
        axs[0, 1].plot(Ts, cps[i, :], label=f"p={p} MPa")
        axs[1, 0].plot(Ts, hs[i, :], label=f"p={p} MPa")
        axs[1, 1].plot(Ts, cs[i, :], label=f"p={p} MPa")
    axs[1, 0].set_xlabel("Temperature [K]")
    axs[1, 1].set_xlabel("Temperature [K]")
    for ax in axs.ravel():
        ax.legend()
        ax.grid()
    plt.show()

    fig.suptitle(f"Transport Properties of {config['thermochem']['spdata'][0]}")
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax2.set_xlabel("Temperature [K]")
    ax1.set_ylabel("Thermal Cond. [W/m^2.K]")
    ax2.set_ylabel("Viscosity [Pa.s]")
    for i, p in enumerate(ps):
        ax1.plot(Ts, kappas[i, :], label=f"p={p} MPa")
        ax2.plot(Ts, mus[i, :], label=f"p={p} MPa")
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
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
