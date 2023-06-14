#!/usr/bin/env python
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 1 for the KEEP scheme (blue line)


"""

from mpi4py import MPI

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("~/.config/matplotlib/stylelib/whitePresentation.mplstyle")

save = False


def simulate(index="i"):
    config = pg.files.configFile()
    config["timeIntegration"]["integrator"] = "rk3"
    config["RHS"]["primaryAdvFlux"] = "myKEEP"
    CFL = 0.5
    gammaRatio = 1.0  # gammaA/gammaB
    config["thermochem"]["spdata"] = ["A", "B"]
    # config["RHS"]["shockHandling"] = "artificialDissipation"
    # config["RHS"]["secondaryAdvFlux"] = "scalarDissipation"
    # config["RHS"]["switchAdvFlux"] = "jamesonPressure"
    config["RHS"]["diffusion"] = False
    config.validateConfig()
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    print(mb)
    print(f"{CFL = }")
    print(f"{gammaRatio = }")

    # modify cpB to match gamma ratio
    mb.thtrdat.array["cp0"][1] = mb.thtrdat.array["cp0"][0] / gammaRatio
    mb.thtrdat.updateDeviceView(["cp0"])

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    nx = 41
    dimsPerBlock = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)
    periodic = rotate([True, False, False], index)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=dimsPerBlock,
        lengths=lengths,
        periodic=periodic,
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces:
        if face.bcType.endswith("Wall"):
            face.bcType = "adiabaticSlipWall"
    if index == "i":
        blk.getFace(1).commRank = 0
        blk.getFace(2).commRank = 0
    elif index == "j":
        blk.getFace(3).commRank = 0
        blk.getFace(4).commRank = 0
    elif index == "k":
        blk.getFace(5).commRank = 0
        blk.getFace(6).commRank = 0

    mb.setBlockCommunication()
    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    ng = blk.ng
    RA = mb.thtrdat.Ru / mb.thtrdat.array["MW"][0]
    RB = mb.thtrdat.Ru / mb.thtrdat.array["MW"][1]
    cpA = mb.thtrdat.array["cp0"][0]
    cpB = mb.thtrdat.array["cp0"][1]
    ccArray = {"i": "xc", "j": "yc", "k": "zc"}
    uIndex = {"i": 1, "j": 2, "k": 3}
    xc = blk.array[ccArray[index]]
    # pressure
    blk.array["q"][:, :, :, 0] = 1.0
    # u
    blk.array["q"][:, :, :, uIndex[index]] = 1.0
    # species A
    blk.array["q"][:, :, :, 5] = 0.5 + 0.5 * np.sin(2 * np.pi * xc)
    YA = blk.array["q"][:, :, :, 5]
    YB = 1.0 - YA
    cp = YA * cpA + YB * cpB
    R = RA * YA + RB * YB
    gamma = cp / (cp - R)
    initial_rho = 2.0 + np.sin(2 * np.pi * xc)
    initial_T = 1.0 / (R * initial_rho)
    blk.array["q"][:, :, :, 4] = initial_T

    # Update cons
    blk.updateDeviceView(["q", "Q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    # entropy stuff
    s = cp / gamma * np.log(blk.array["q"][:, :, :, 4]) - R * np.log(
        blk.array["Q"][:, :, :, 0]
    )
    blk.array["s"][:] = blk.array["Q"][:, :, :, 0] * s
    blk.updateDeviceView(["s"])
    pg.consistify(mb)

    sDerived = []
    sEvolved = []
    t = []
    dx = 1.0 / (nx - 1.0)
    lam = np.sqrt(np.max(gamma * R * blk.array["q"][:, :, :, 4])) + np.max(
        blk.array["q"][:, :, :, uIndex[index]]
    )
    dt = CFL * dx / lam
    tEnd = 11.0
    while mb.tme < tEnd:
        abort = pg.mpiComm.mpiUtils.checkForNan(mb)
        if abort > 0:
            print("Nan")
            break

        if mb.nrt % 20 == 0:
            pg.misc.progressBar(mb.tme, tEnd)

        if mb.nrt % 10 == 0:
            blk.updateHostView(["q", "Q"])
            YA = blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5]
            YB = 1.0 - blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5]
            cp = YA * cpA + YB * cpB
            R = RA * YA + RB * YB
            gamma = cp / (cp - R)
            dS = np.sum(
                blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0]
                * (
                    cp / gamma * np.log(blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4])
                    - R * np.log(blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0])
                )
            )
            sDerived.append(dS)
            eS = np.sum(blk.array["s"][ng:-ng, ng:-ng, ng:-ng])
            sEvolved.append(eS)
            t.append(mb.tme)

        mb.step(dt)

    blk.updateHostView(["q", "Q"])

    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    YA = blk.array["q"][s_][:, 5]
    YB = 1.0 - YA
    cp = YA * cpA + YB * cpB
    R = RA * YA + RB * YB
    gamma = cp / (cp - R)
    x = blk.array[ccArray[index]][s_]
    rho = blk.array["Q"][s_][:, 0]
    p = blk.array["q"][s_][:, 0]
    u = blk.array["q"][s_][:, uIndex[index]]
    T = blk.array["q"][s_][:, 4]
    sD = rho * (cp / gamma * np.log(T) - R * np.log(rho))
    se = blk.array["s"][s_]

    # with open("data.npy", "wb") as f:
    #     np.save(f, t)
    #     np.save(f, sDerived)
    #     np.save(f, sEvolved)

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    # ax1.set_title("1D Advection Results")
    ax1.set_xlabel(r"$x$")
    ax1.plot(x, rho, color="g", label=r"$\rho$")
    ax1.plot(x, p, color="r", label="$p$")
    ax1.plot(x, u, color="k", label="$u$")
    ax1.plot(x, YA, color="orange", label="$Y_{A}$")
    ax1.scatter(
        x,
        initial_rho[ng:-ng, ng:-ng, ng:-ng],
        marker="o",
        facecolor="w",
        edgecolor="b",
        label=r"$\rho_{\text{exact}}$",
    )
    ax1.legend()
    # fig.savefig("oneDAdvKEEPpe.png")
    plt.show()
    plt.clf()


#     # entropy space
#     ax1.plot(x, sd, color="k", label="Recon")
#     ax1.plot(x, se, color="r", label=r"$\partial{\rho s}/\partial{t}$")
#     ax1.set_ylabel(r"$\rho s \quad [ c_{v}\ln (T) - R \ln (\rho) ]$")
#     ax1.set_xlabel(r"$x$")
#     ax1.legend()
#     #plt.show()
#     plt.clf()
#
#     # entropy total
#     fig, ax1 = plt.subplots(figsize=(5, 3.5))
#     ax1.plot(
#         t,
#         (sEvolved - sEvolved[0]) / sEvolved[0],
#         c="orange",
#         marker="o",
#         label=r"$\partial{\rho s}/\partial{t}$",
#         markevery=0.05,
#     )
#     ax1.plot(
#         t, (sDerived - sDerived[0]) / sDerived[0], label=r"$s=c_{v}\log(T)-R\log(\rho)$"
#     )
#     ax1.set_ylabel(r"$\Delta(\rho s) / {(\rho s)}_0$")
#     ax1.set_xlabel(r"$t$")
#     ax1.legend()
#     ax1.set_xlim([0, 11])
#     # plt.savefig("CentralsEvo.png")
#     plt.show()
#     plt.close()


if __name__ == "__main__":
    try:
        pg.compute.pgkokkos.initialize()
        simulate("i")
        pg.compute.pgkokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
