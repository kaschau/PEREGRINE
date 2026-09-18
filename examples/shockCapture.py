#!/usr/bin/env python3

"""
Solves the shock tube problem defined with arbutrary left and right states states separated by a membrane
at some x location between zero and one.

Solves the problem numerically with peregrine, and exactly using an exact Riemann solver.

See

Riemann Solvers and Numerical Methods for Fluid Dynamic 3rd Ed.
Eleuterio F. Toro
Spring

for more.
"""

from mpi4py import MPI  # noqa: F401

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

# the debug gas of the Toro cases: R = 281.4, gamma = 1.4
db = {"DB": {"MW": 29.54065178914549, "cp0": 1000.0}}


def guessP(test):
    pL, rhoL, uL = test.pL, test.rhoL, test.uL
    pR, rhoR, uR = test.pR, test.rhoR, test.uR
    cL, cR = test.cL, test.cR
    gamma = test.gamma

    # Gamma constants
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g3 = 2.0 * gamma / (gamma - 1.0)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)
    g7 = (gamma - 1.0) / 2.0

    qUser = 2.0
    cup = 0.25 * (rhoL + rhoR) * (cL + cR)
    ppv = 0.5 * (pL + pR) + 0.5 * (uL - uR) * cup
    ppv = max(0.0, ppv)
    pmin = min(pL, pR)
    pmax = max(pL, pR)
    qmax = pmax / pmin

    if (qmax < qUser) and ((pmin < ppv) and (ppv < pmax)):
        pM = ppv
    else:
        if ppv < pmin:
            pQ = (pL / pR) ** g1
            uM = (pQ * uL / cL + uR / cR + g4 * (pQ - 1.0)) / (pQ / cL + 1.0 / cR)
            pTL = 1.0 + g7 * (uL - uM) / cL
            pTR = 1.0 + g7 * (uM - uR) / cR
            pM = 0.5 * (pL * pTL**g3 + pR * pTR**g3)
        else:
            gEL = np.sqrt((g5 / rhoL) / (g6 * pL + ppv))
            gER = np.sqrt((g5 / rhoR) / (g6 * pR + ppv))
            pM = (gEL * pL + gER * pR - (uR - uL)) / (gEL + gER)

    return pM


def prefun(p, test, side):
    if side == "L":
        pK, rhoK = test.pL, test.rhoL
        cK = test.cL
    else:
        pK, rhoK = test.pR, test.rhoR
        cK = test.cR
    gamma = test.gamma
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)

    if p < pK:
        pRatio = p / pK
        F = g4 * cK * (pRatio**g1 - 1.0)
        FD = (1.0 / (rhoK * cK)) * pRatio ** (-g2)
    else:
        AK = g5 / rhoK
        BK = g6 * pK
        qrt = np.sqrt(AK / (BK + p))
        F = (p - pK) * qrt
        FD = (1.0 - 0.5 * (p - pK) / (BK + p)) * qrt

    return F, FD


def pStar(test):
    uL = test.uL
    uR = test.uR

    maxIter = 100
    tol = 1e-6

    n = 0
    deltaP = 1e10

    pOld = guessP(test)
    uDiff = uR - uL
    for n in range(maxIter):
        fL, fDL = prefun(pOld, test, "L")
        fR, fDR = prefun(pOld, test, "R")
        p = pOld - (fL + fR + uDiff) / (fDL + fDR)
        deltaP = 2.0 * abs((p - pOld) / (p + pOld))
        pOld = p
        if deltaP < tol:
            break
    else:
        raise ValueError("Did not converge.")

    u = 0.5 * (uL + uR + fR - fL)
    return p, u


def sample(test, pM, uM, s):
    pL, rhoL, uL = test.pL, test.rhoL, test.uL
    pR, rhoR, uR = test.pR, test.rhoR, test.uR
    cL, cR = test.cL, test.cR
    gamma = test.gamma
    # Gamma constants
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g3 = 2.0 * gamma / (gamma - 1.0)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)
    g7 = (gamma - 1.0) / 2.0
    g8 = gamma - 1.0

    if s < uM:
        if pM < pL:
            shL = uL - cL
            if s < shL:
                rho = rhoL
                u = uL
                p = pL
            else:
                cmL = cL * (pM / pL) ** g1
                stL = uM - cmL
                if s > stL:
                    rho = rhoL * (pM / pL) ** (1.0 / gamma)
                    u = uM
                    p = pM
                else:
                    u = g5 * (cL + g7 * uL + s)
                    c = g5 * (cL + g7 * (uL - s))
                    rho = rhoL * (c / cL) ** g4
                    p = pL * (c / cL) ** g3
        else:
            pmL = pM / pL
            sL = uL - cL * np.sqrt(g2 * pmL + g1)
            if s < sL:
                rho = rhoL
                u = uL
                p = pL
            else:
                rho = rhoL * (pmL + g6) / (pmL * g6 + 1.0)
                u = uM
                p = pM
    else:
        if pM > pR:
            pmR = pM / pR
            sR = uR + cR * np.sqrt(g2 * pmR + g1)
            if s > sR:
                rho = rhoR
                u = uR
                p = pR
            else:
                rho = rhoR * (pmR + g6) / (pmR * g6 + 1.0)
                u = uM
                p = pM
        else:
            shR = uR + cR
            if s > shR:
                rho = rhoR
                u = uR
                p = pR
            else:
                cmR = cR * (pM / pR) ** g1
                stR = uM + cmR
                if s < stR:
                    rho = rhoR * (pM / pR) ** (1.0 / gamma)
                    u = uM
                    p = pM
                else:
                    u = g5 * (-cR + g7 * uR + s)
                    c = g5 * (cR - g7 * (uR - s))
                    rho = rhoR * (c / cR) ** g4
                    p = pR * (c / cR) ** g3

    e = p / rho / g8
    return p, u, rho, e


def solve(test, npts=250):
    uL = test.uL
    uR = test.uR
    cL, cR = test.cL, test.cR
    gamma = test.gamma
    x0 = test.x0

    g4 = 2.0 / (gamma - 1.0)
    assert g4 * (cL + cR) > (uR - uL)

    pM, uM = pStar(test)

    pts = np.linspace(0, 1, npts)
    res = {
        "x": np.empty(npts),
        "p": np.empty(npts),
        "u": np.empty(npts),
        "rho": np.empty(npts),
        "energy": np.empty(npts),
    }
    for i, x in enumerate(pts):
        s = (x - x0) / test.t
        p, u, rho, e = sample(test, pM, uM, s)
        res["x"][i] = x
        res["p"][i] = p
        res["u"][i] = u
        res["rho"][i] = rho
        res["energy"][i] = e

    return res


class state:
    def __init__(self, test, gamma, R):
        self.name = str(test)

        if test == 0:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1

            self.x0 = 0.5
            self.t = 0.2
            self.dt = 1e-4

        if test == 1:
            self.rhoL = 1.0
            self.uL = 0.75
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1

            self.x0 = 0.3
            self.t = 0.2
            self.dt = 1e-4

        if test == 10:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1

            self.x0 = 0.3
            self.t = 0.2
            self.dt = 1e-4

        if test == 2:
            self.rhoL = 1.0
            self.uL = -2.0
            self.pL = 0.4

            self.rhoR = 1.0
            self.uR = 2.0
            self.pR = 0.4

            self.x0 = 0.5
            self.t = 0.15
            self.dt = 1e-4

        if test == 3:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1000.0

            self.rhoR = 1.0
            self.uR = 0.0
            self.pR = 0.01

            self.x0 = 0.5
            self.t = 0.012
            self.dt = 5e-6

        if test == 4:
            self.rhoL = 5.99924
            self.uL = 19.5975
            self.pL = 460.894

            self.rhoR = 5.99242
            self.uR = -6.19633
            self.pR = 46.0950

            self.x0 = 0.4
            self.t = 0.035
            self.dt = 1e-5

        if test == 5:
            self.rhoL = 1.0
            self.uL = -19.5975
            self.pL = 1000.0
            self.x0 = 0.8

            self.rhoR = 1.0
            self.uR = -19.5975
            self.pR = 0.01

            self.t = 0.012
            self.dt = 1e-5

        self.TL = self.pL / (self.rhoL * R)
        self.TR = self.pR / (self.rhoR * R)
        self.cL = np.sqrt(gamma * self.pL / self.rhoL)
        self.cR = np.sqrt(gamma * self.pR / self.rhoR)
        self.gamma = gamma


def simulate(testnum, index="i"):
    nx = 201
    config = pg.files.configFile()
    config["simulation"]["physics"] = "euler"
    config["simulation"]["mixture"] = db
    config["RHS"]["shockHandling"] = "artificialDissipation"
    config["RHS"]["primaryAdvFlux"] = "KEEPpe"
    config["RHS"]["secondaryAdvFlux"] = "scalarDissipation"
    config["RHS"]["switchAdvFlux"] = "vanLeer"
    config["timeIntegration"]["integrator"] = "rk3"

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    dimsPerBlock = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)

    # the gas, from its constants
    species = db[next(iter(db))]
    R = 8314.46261815324 / species["MW"]
    cp = species["cp0"]
    gamma = cp / (cp - R)

    test = state(testnum, gamma, R)
    print("State {}".format(testnum))
    print("--------------------------")
    print("Left State")
    print("PL = {}".format(test.pL))
    print("TL = {}".format(test.TL))
    print("rhoL = {}".format(test.rhoL))
    print("uL = {}".format(test.uL))
    print("--------------------------")
    print("Right State")
    print("PR = {}".format(test.pR))
    print("TR = {}".format(test.TR))
    print("rhoR = {}".format(test.rhoR))
    print("uR = {}".format(test.uR))
    print("--------------------------")

    # the ends of the tube: fed where the flow comes in, let out where it
    # leaves, walls where it is still; the sides slip
    ccAxis = {"i": 0, "j": 1, "k": 2}
    lowFace, highFace = 2 * ccAxis[index] + 1, 2 * ccAxis[index] + 2
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    boundaryNames = dict.fromkeys(range(1, 7), "walls")

    def end(name, u, p, T, nface):
        if u == 0.0:
            return
        boundaryNames[nface] = name
        # in through the low end, or in through the high end, is an inlet
        if (u > 0) == (nface == lowFace):
            velo = rotate([u, 0.0, 0.0], index)
            config["bcValues"][name] = {
                "bcType": "constantVelocitySubsonicInlet",
                "u": velo[0],
                "v": velo[1],
                "w": velo[2],
                "T": T,
            }
        else:
            config["bcValues"][name] = {
                "bcType": "constantPressureSubsonicExit",
                "p": p,
            }

    end("left", test.uL, test.pL, test.TL, lowFace)
    end("right", test.uR, test.pR, test.TR, highFace)
    config.validateConfig()

    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=dimsPerBlock,
        lengths=lengths,
        boundaryNames=boundaryNames,
    )
    mb = pg.multiBlock.solver(config, mesh)
    print(mb)

    blk = mb.blocks[0]
    ng = blk.ng

    uIndex = {"i": 1, "j": 2, "k": 3}
    xc = blk.cells.get()[..., ccAxis[index]]
    # the primitive vector, p, u, v, w, T: the left and right states
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[:, :, :, 0] = np.where(xc <= test.x0, test.pL, test.pR)
    q[:, :, :, uIndex[index]] = np.where(xc <= test.x0, test.uL, test.uR)
    q[:, :, :, 4] = np.where(xc <= test.x0, test.TL, test.TR)
    mb.setPrimitives([q])

    bar = pg.misc.Progress(test.t)
    while mb.tme < test.t:
        bar.at(mb.tme)
        mb.integrator.step(test.dt)

    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    data = mb.exportData(blk, ["rho", "p", "u", "v", "w"])
    x = blk.cells.get()[..., ccAxis[index]][s_]
    rho = data["rho"][s_]
    p = data["p"][s_]
    phi = blk.phi.get()[s_][:, uIndex[index] - 1]
    u = data["uvw"[ccAxis[index]]][s_]
    e = blk.qh.get()[s_][:, 4]

    res = solve(test)
    rx = res["x"]
    rrho = res["rho"]
    ru = res["u"]
    rp = res["p"]
    re = res["energy"]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]

    ax1.set_title(f"{mb.tme:.2f}")

    ms = 1.5
    lw = 0.5
    # rho
    ax1.set_xlabel(r"x")
    ax1.plot(x, phi, "--", color="gold", label="phi", linewidth=lw)
    ax1.plot(x, rho, color="k", label="rho", linewidth=lw)
    ax1.scatter(rx, rrho, color="k", label="Analyticsl", marker="o", s=ms)
    ax1.legend()

    # velocity
    ax2.set_xlabel(r"x")
    ax2.plot(x, u, color="k", label="u", linewidth=lw)
    ax2.scatter(rx, ru, color="k", label="Analytical", marker="o", s=ms)
    ax2.legend()

    # pressure
    ax3.set_xlabel(r"x")
    ax3.plot(x, p, color="k", label="p", linewidth=lw)
    ax3.scatter(rx, rp, color="k", label="Analytical", marker="o", s=ms)
    ax3.legend()

    # energy
    ax4.set_xlabel(r"x")
    ax4.plot(x, e / rho, color="k", label="e", linewidth=lw)
    ax4.scatter(rx, re, color="k", label="Analyticsl", marker="o", s=ms)
    ax4.legend()

    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        pg.backend.abi.lib.initialize()
        testnum = 5
        index = "i"
        simulate(testnum, index)
        pg.backend.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
