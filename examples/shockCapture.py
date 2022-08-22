#!/usr/bin/env python

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize


def soundSpeed(gamma, pressure, density):
    return np.sqrt(gamma * pressure / density)


def shockTubeFunction(p4, p1, p5, rho1, rho5, gamma):
    """
    Shock tube equation
    """
    z = p4 / p5 - 1.0
    c1 = soundSpeed(gamma, p1, rho1)
    c5 = soundSpeed(gamma, p5, rho5)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    g2 = 2.0 * gamma

    fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1.0 + gp1 / g2 * z)
    fact = (1.0 - fact) ** (g2 / gm1)

    return p1 * fact - p4


def calculateRegions(pl, ul, rhol, pr, ur, rhor, gamma):
    """
    Compute regions
    :rtype : tuple
    :return: returns p, rho and u for regions 1,3,4,5 as well as the shock speed
    """
    # if pl > pr...
    rho1 = rhol
    p1 = pl
    u1 = ul
    rho5 = rhor
    p5 = pr
    u5 = ur

    # unless...
    if pl < pr:
        rho1 = rhor
        p1 = pr
        u1 = ur
        rho5 = rhol
        p5 = pl
        u5 = ul

    # solve for post-shock pressure
    p4 = scipy.optimize.fsolve(shockTubeFunction, p1, (p1, p5, rho1, rho5, gamma))[0]

    # compute post-shock density and velocity
    z = p4 / p5 - 1.0
    c5 = soundSpeed(gamma, p5, rho5)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gmfac1 = 0.5 * gm1 / gamma
    gmfac2 = 0.5 * gp1 / gamma

    fact = np.sqrt(1.0 + gmfac2 * z)

    u4 = c5 * z / (gamma * fact)
    rho4 = rho5 * (1.0 + gmfac2 * z) / (1.0 + gmfac1 * z)

    # shock speed
    w = c5 * fact

    # compute values at foot of rarefaction
    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1) ** (1.0 / gamma)
    return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w


def calcPositions(
    pl,
    pr,
    region1,
    region3,
    w,
    xi,
    t,
    gamma,
):
    """
    :return: tuple of positions in the following order ->
            Head of Rarefaction: xhd,  Foot of Rarefaction: xft,
            Contact Discontinuity: xcd, Shock: xsh
    """
    p1, rho1 = region1[:2]  # don't need velocity
    p3, rho3, u3 = region3
    c1 = soundSpeed(gamma, p1, rho1)
    c3 = soundSpeed(gamma, p3, rho3)

    if pl > pr:
        xsh = xi + w * t
        xcd = xi + u3 * t
        xft = xi + (u3 - c3) * t
        xhd = xi - c1 * t
    else:
        # pr > pl
        xsh = xi - w * t
        xcd = xi - u3 * t
        xft = xi - (u3 - c3) * t
        xhd = xi + c1 * t

    return xhd, xft, xcd, xsh


def regionStates(pl, pr, region1, region3, region4, region5):
    """
    :return: dictionary (region no.: p, rho, u), except for rarefaction region
    where the value is a string, obviously
    """
    if pl > pr:
        return {
            "Region 1": region1,
            "Region 2": "RAREFACTION",
            "Region 3": region3,
            "Region 4": region4,
            "Region 5": region5,
        }
    else:
        return {
            "Region 1": region5,
            "Region 2": region4,
            "Region 3": region3,
            "Region 4": "RAREFACTION",
            "Region 5": region1,
        }


def create_arrays(
    pl,
    pr,
    xl,
    xr,
    positions,
    state1,
    state3,
    state4,
    state5,
    npts,
    gamma,
    t,
    xi,
):
    """
    :return: tuple of x, p, rho and u values across the domain of interest
    """
    xhd, xft, xcd, xsh = positions
    p1, rho1, u1 = state1
    p3, rho3, u3 = state3
    p4, rho4, u4 = state4
    p5, rho5, u5 = state5
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    x_arr = np.linspace(xl, xr, npts)
    rho = np.zeros(npts, dtype=float)
    p = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    c1 = soundSpeed(gamma, p1, rho1)
    if pl > pr:
        for i, x in enumerate(x_arr):
            if x < xhd:
                rho[i] = rho1
                p[i] = p1
                u[i] = u1
            elif x < xft:
                u[i] = 2.0 / gp1 * (c1 + (x - xi) / t)
                fact = 1.0 - 0.5 * gm1 * u[i] / c1
                rho[i] = rho1 * fact ** (2.0 / gm1)
                p[i] = p1 * fact ** (2.0 * gamma / gm1)
            elif x < xcd:
                rho[i] = rho3
                p[i] = p3
                u[i] = u3
            elif x < xsh:
                rho[i] = rho4
                p[i] = p4
                u[i] = u4
            else:
                rho[i] = rho5
                p[i] = p5
                u[i] = u5
    else:
        for i, x in enumerate(x_arr):
            if x < xsh:
                rho[i] = rho5
                p[i] = p5
                u[i] = -u1
            elif x < xcd:
                rho[i] = rho4
                p[i] = p4
                u[i] = -u4
            elif x < xft:
                rho[i] = rho3
                p[i] = p3
                u[i] = -u3
            elif x < xhd:
                u[i] = -2.0 / gp1 * (c1 + (xi - x) / t)
                fact = 1.0 + 0.5 * gm1 * u[i] / c1
                rho[i] = rho1 * fact ** (2.0 / gm1)
                p[i] = p1 * fact ** (2.0 * gamma / gm1)
            else:
                rho[i] = rho1
                p[i] = p1
                u[i] = -u1

    return x_arr, p, rho, u


def solve(left_state, right_state, geometry, t, gamma, npts=500):
    """
    Solves the Sod shock tube problem (i.e. riemann problem) of discontinuity
    across an interface.

    Parameters
    ----------
    left_state, right_state: tuple
        A tuple of the state (pressure, density, velocity) on each side of the
        shocktube barrier for the ICs.
    geometry: tuple
        A tuple of positions for (left boundary, right boundary, barrier)
    t: float
        Time to calculate the solution at
    gamma: float
        Adiabatic index for the gas.
    npts: int
        number of points for array of pressure, density and velocity
    Returns
    -------
    positions: dict
        Locations of the important places (rarefaction wave, shock, etc...)
    regions: dict
        constant pressure, density and velocity states in distinct regions
    values: dict
        Arrays of pressure, density, and velocity as a function of position.
        The density ('rho') is the gas density.
        Also calculates the specific internal energy
    """

    pl, rhol, ul = left_state
    pr, rhor, ur = right_state
    xl, xr, xi = geometry

    # basic checking
    if xl >= xr:
        print("xl has to be less than xr!")
        exit()
    if xi >= xr or xi <= xl:
        print("xi has in between xl and xr!")
        exit()

    # calculate regions
    region1, region3, region4, region5, w = calculateRegions(
        pl, ul, rhol, pr, ur, rhor, gamma
    )

    regions = regionStates(pl, pr, region1, region3, region4, region5)

    # calculate positions
    x_positions = calcPositions(pl, pr, region1, region3, w, xi, t, gamma)

    pos_description = (
        "Head of Rarefaction",
        "Foot of Rarefaction",
        "Contact Discontinuity",
        "Shock",
    )
    positions = dict(zip(pos_description, x_positions))

    # create arrays
    x, p, rho, u = create_arrays(
        pl,
        pr,
        xl,
        xr,
        x_positions,
        region1,
        region3,
        region4,
        region5,
        npts,
        gamma,
        t,
        xi,
    )

    energy = p / (rho * (gamma - 1.0))
    val_dict = {
        "x": x,
        "p": p,
        "rho": rho,
        "u": u,
        "energy": energy,
    }

    return positions, regions, val_dict


class state:
    def __init__(self, test, R):

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


def simulate(testnum, index="i"):

    nx = 201
    config = pg.files.configFile()
    config["thermochem"]["spdata"] = ["DB"]
    config["RHS"]["shockHandling"] = "artificialDissipation"
    config["RHS"]["primaryAdvFlux"] = "secondOrderKEEP"
    config["RHS"]["secondaryAdvFlux"] = "scalarDissipation"
    config["RHS"]["switchAdvFlux"] = "vanLeer"
    config["timeIntegration"]["integrator"] = "rk3"
    config.validateConfig()
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    Ru = mb.thtrdat.Ru
    MW = mb.thtrdat.array["MW"][0]
    R = Ru / MW
    cp = mb.thtrdat.array["cp0"][0]
    gamma = cp / (cp - R)

    print(mb)

    test = state(testnum, R)
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

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    dimsPerBlock = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=dimsPerBlock,
        lengths=lengths,
    )

    mb.initSolverArrays(config)

    blk = mb[0]
    ng = blk.ng

    for face in blk.faces:
        face.bcType = "adiabaticSlipWall"

    mb.setBlockCommunication()
    mb.unifyGrid()
    mb.computeMetrics(fdOrder=2)

    ccArray = {"i": "xc", "j": "yc", "k": "zc"}
    uIndex = {"i": 1, "j": 2, "k": 3}
    xc = blk.array[ccArray[index]]
    # Initialize Left/Right properties
    blk.array["q"][:, :, :, 0] = np.where(xc <= test.x0, test.pL, test.pR)
    blk.array["q"][:, :, :, uIndex[index]] = np.where(xc <= test.x0, test.uL, test.uR)
    blk.array["q"][:, :, :, 4] = np.where(xc <= test.x0, test.TL, test.TR)

    # Update boundary conditions
    if index == "i":
        lowFace = 1
        highFace = 2
    elif index == "j":
        lowFace = 3
        highFace = 4
    elif index == "k":
        lowFace = 5
        highFace = 6

    face = blk.getFace(lowFace)
    if test.uL == 0.0:
        pass
    else:
        face.array["qBcVals"] = np.zeros((blk.array["q"][face.s1_].shape))
        inputBcValues = {}
        if test.uL > 0:
            face.bcType = "constantVelocitySubsonicInlet"
            bcVelo = rotate([test.uL, 0.0, 0.0], index)
            inputBcValues["u"] = bcVelo[0]
            inputBcValues["v"] = bcVelo[1]
            inputBcValues["w"] = bcVelo[2]
            inputBcValues["T"] = test.TL
            pg.bcs.prepInlets.prep_constantVelocitySubsonicInlet(
                blk, face, inputBcValues
            )
        elif test.uL < 0:
            face.bcType = "constantPressureSubsonicExit"
            inputBcValues["p"] = test.pL
            pg.bcs.prepExits.prep_constantPressureSubsonicExit(blk, face, inputBcValues)
        shape = blk.array["q"][face.s1_].shape
        pg.misc.createViewMirrorArray(face, "qBcVals", shape)

    face = blk.getFace(highFace)
    if test.uR == 0.0:
        pass
    else:
        face.array["qBcVals"] = np.zeros((blk.array["q"][face.s1_].shape))
        inputBcValues = {}
        if test.uR < 0:
            face.bcType = "constantVelocitySubsonicInlet"
            bcVelo = rotate([test.uR, 0.0, 0.0], index)
            inputBcValues["u"] = bcVelo[0]
            inputBcValues["v"] = bcVelo[1]
            inputBcValues["w"] = bcVelo[2]
            inputBcValues["T"] = test.TR
            pg.bcs.prepInlets.prep_constantVelocitySubsonicInlet(
                blk, face, inputBcValues
            )
        elif test.uR > 0:
            face.bcType = "constantPressureSubsonicExit"
            inputBcValues["p"] = test.pR
            pg.bcs.prepExits.prep_constantPressureSubsonicExit(blk, face, inputBcValues)
        shape = blk.array["q"][face.s1_].shape
        pg.misc.createViewMirrorArray(face, "qBcVals", shape)

    # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)
    while mb.tme < test.t:
        pg.misc.progressBar(mb.tme, test.t)
        mb.step(test.dt)

    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    x = blk.array[ccArray[index]][s_]
    rho = blk.array["Q"][s_][:, 0]
    p = blk.array["q"][s_][:, 0]
    phi = blk.array["phi"][s_][:, uIndex[index] - 1]
    u = blk.array["q"][s_][:, uIndex[index]]
    e = blk.array["qh"][s_][:, 4]

    _, _, res = solve(
        (test.pL, test.rhoL, test.uL),
        (test.pR, test.rhoR, test.uR),
        (0.0, 1.0, test.x0),
        test.t,
        gamma=gamma,
        npts=250,
    )
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
        kokkos.initialize()
        testnum = 0
        index = "i"
        simulate(testnum, index)
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
