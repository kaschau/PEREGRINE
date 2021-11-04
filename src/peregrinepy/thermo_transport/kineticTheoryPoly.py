from .completeSpecies import completeSpecies
from .MM_Tables import delta, tstar22, omega22_table, tstar, astar_table
import numpy as np
from scipy import interpolate as intrp


def kineticTheoryPoly(usersp, refsp, eos):
    kb = refsp["kb"]
    eps0 = refsp["epsilon0"]
    avogadro = refsp["avogadro"]
    Ru = refsp["Ru"]
    ns = len(usersp.keys())

    if eos == "cpg":
        cp0 = completeSpecies("cp0", usersp, refsp)
        NASA7 = [None for i in range(ns)]

        def cp_R(cp0, poly, T, MW):
            return cp0 * T / (Ru * MW)

    elif eos == "tpg":
        NASA7 = completeSpecies("NASA7", usersp, refsp)
        cp0 = [None for i in range(ns)]

        def cp_R(cp0, poly, T, MW):
            if T <= poly[0]:
                return sum([poly[i + 1 + 7] * T ** (i) for i in range(5)])
            else:
                return sum([poly[i + 1] * T ** (i) for i in range(5)])

    deg = 4
    # Maximum and minumum temperatures to generate poly'l
    Tmin = 200
    Tmax = 3500
    # Generate range of temperatures
    npts = 50
    Ts = np.linspace(Tmin, Tmax, npts)

    # Collision integral interpolations
    intrp_o22 = intrp.RectBivariateSpline(tstar22, delta, omega22_table, kx=5, ky=5)
    intrp_Astar = intrp.RectBivariateSpline(tstar, delta, astar_table, kx=5, ky=5)

    # Get molecular mass
    MW = completeSpecies("MW", usersp, refsp)
    mass = np.array([M / avogadro for M in MW])

    # epsilon
    well = completeSpecies("well", usersp, refsp)
    # sigma
    diam = completeSpecies("diam", usersp, refsp)
    # mu
    dipole = completeSpecies("dipole", usersp, refsp)

    # see if molecule is polar
    polar = dipole > 0.0

    # alpha
    polarize = completeSpecies("polarize", usersp, refsp)

    # z_rot
    zrot = completeSpecies("zrot", usersp, refsp)
    # W_ac
    acentric = completeSpecies("acentric", usersp, refsp)

    # determine rotational DOF
    geom = completeSpecies("geometry", usersp, refsp)
    rotDOF = []
    for g in geom:
        if g == "atom":
            rotDOF.append(0.0)
        elif g == "linear":
            rotDOF.append(1.0)
        elif g == "nonlinear":
            rotDOF.append(1.5)

    ##########################################
    # Collision Parameters (reduced stuff)
    ##########################################
    r_mass = np.zeros((ns, ns))
    r_well = np.zeros((ns, ns))
    r_diam = np.zeros((ns, ns))
    r_dipole = np.zeros((ns, ns))

    r_deltastar = np.zeros((ns, ns))

    for i in range(ns):
        for j in range(i, ns):
            # reduced mass
            r_mass[i, j] = mass[i] * mass[j] / (mass[i] + mass[j])
            # spheriacl collision diameter
            r_diam[i, j] = 0.5 * (diam[i] + diam[j])
            # effective well depth
            r_well[i, j] = np.sqrt(well[i] * well[j])
            # effective dipole moment
            r_dipole[i, j] = np.sqrt(dipole[i] * dipole[j])

            # reduced dipole delta*
            r_deltastar[i, j] = (
                0.5
                * r_dipole[i, j] ** 2
                / (4 * np.pi * eps0 * r_well[i, j] * r_diam[i, j] ** 3)
            )

            # Correct for polarity
            if polar[i] == polar[j]:
                f_well = 1.0
                f_diam = 1.0
            else:
                kp, knp = (i, j) if polar[i] else (j, i)
                d3np = diam[knp] ** 3
                d3p = diam[kp] ** 3
                alphastar = polarize[knp] / d3np
                dipolestar = r_dipole[kp, kp] / np.sqrt(
                    4 * np.pi * eps0 * d3p * well[kp]
                )
                xi = 1.0 + 0.25 * alphastar * dipolestar ** 2 * np.sqrt(
                    well[kp] / well[knp]
                )
                f_well = xi ** 2
                f_diam = xi ** (-1 / 6)

            r_well[i, j] *= f_well
            r_diam[i, j] *= f_diam

            # properties are symmetric
            r_mass[j, i] = r_mass[i, j]
            r_diam[j, i] = r_diam[i, j]
            r_well[j, i] = r_well[i, j]
            r_dipole[j, i] = r_dipole[i, j]
            r_deltastar[j, i] = r_deltastar[i, j]

    ##########################################
    # Viscosities
    ##########################################
    visc = np.zeros((npts, ns))
    for i, T in enumerate(Ts):
        Tstar = T * kb / well
        omga22 = intrp_o22(Tstar, r_deltastar.diagonal(), grid=False)
        visc[i, :] = (
            (5.0 / 16.0) * np.sqrt(np.pi * mass * kb * T) / (np.pi * diam ** 2 * omga22)
        )

    ##########################################
    # Thermal Conductivity
    ##########################################
    # NOTE The EOS has an effect on the transport properties via the calculation of cp
    #     so if you use tpg you will use NASA7 to help compute thermal conductivities,
    #     if you use cpg you will use constant cp to compute kappa.
    cond = np.zeros((npts, ns))
    for i, T in enumerate(Ts):
        for k in range(ns):
            Tstar = kb * 298.0 / well[k]
            fz_298 = (
                1.0
                + np.pi ** 1.5 / np.sqrt(Tstar) * (0.5 + 1.0 / Tstar)
                + (0.25 * np.pi ** 2 + 2) / Tstar
            )

            Tstar = T * kb / well[k]

            omga22 = intrp_o22(Tstar, r_deltastar[k, k], grid=False)
            Astar = intrp_Astar(Tstar, r_deltastar[k, k], grid=False)
            omga11 = omga22 / Astar

            # self diffusion coeff
            diffcoeff = (
                3.0
                / 16.0
                * np.sqrt(2.0 * np.pi / r_mass[k, k])
                * (kb * T) ** 1.5
                / (np.pi * diam[k] ** 2 * omga11)
            )

            f_int = MW[k] / (Ru * T) * diffcoeff / visc[i, k]
            cv_rot = rotDOF[k]
            A_factor = 2.5 - f_int
            fz_tstar = (
                1.0
                + np.pi ** 1.5 / np.sqrt(Tstar) * (0.5 + 1.0 / Tstar)
                + (0.25 * np.pi ** 2 + 2) / Tstar
            )
            B_factor = zrot[k] * fz_298 / fz_tstar + 2.0 / np.pi * (
                5 / 3 * rotDOF[k] + f_int
            )
            c1 = 2.0 / np.pi * A_factor / B_factor
            cv_int = cp_R(cp0[k], NASA7[k], T, MW[k]) - 2.5 - cv_rot
            f_rot = f_int * (1.0 + c1)
            f_trans = 2.5 * (1.0 - c1 * cv_rot / 1.5)
            cond[i, k] = (
                visc[i, k]
                / MW[k]
                * Ru
                * (f_trans * 1.5 + f_rot * cv_rot + f_int * cv_int)
            )

    ##########################################
    # Binary Diffusion
    ##########################################
    diff = np.zeros((npts, ns, ns))
    for k in range(ns):
        for j in range(k, ns):
            for i, T in enumerate(Ts):
                Tstar = T * kb / r_well[j, k]

                omga22 = intrp_o22(Tstar, r_deltastar[j, k], grid=False)
                Astar = intrp_Astar(Tstar, r_deltastar[j, k], grid=False)
                omga11 = omga22 / Astar

                # To get pressure dependence, we evaluate the coeff at unit pressure
                # then when we actually NEED the coeff, we use divide by the real pressure
                diffcoeff = (
                    3.0
                    / 16.0
                    * np.sqrt(2.0 * np.pi / r_mass[k, j])
                    * (kb * T) ** 1.5
                    / (np.pi * r_diam[j, k] ** 2 * omga11)
                )

                diff[i, k, j] = diffcoeff
                diff[i, j, k] = diff[i, k, j]

    # Create and set the polynoial coefficients
    logTs = np.log(Ts)
    sqrtTs = np.sqrt(Ts)

    # We fit the visc pol'y to the sqrtT as visc is proportional to sqrtT
    visc = visc / sqrtTs[:, None]
    w = 1.0 / (visc ** 2)
    muPoly = np.array(
        [list(np.polyfit(logTs, visc[:, k], deg=deg, w=w[:, k])) for k in range(ns)]
    )

    # We fit the cond pol'y to the sqrtT as cond is proportional to sqrtT
    cond = cond / np.sqrt(Ts[:, None])
    w = 1.0 / (cond ** 2)
    kappaPoly = np.array(
        [list(np.polyfit(logTs, cond[:, k], deg=deg, w=w[:, k])) for k in range(ns)]
    )

    Dij = []
    diff = diff / Ts[:, None, None] ** 1.5
    w = 1.0 / (diff ** 2)
    for k in range(ns):
        for j in range(k, ns):
            Dij.append(list(np.polyfit(logTs, diff[:, k, j], deg=deg, w=w[:, k, j])))

    DijPoly = np.array(Dij)

    return muPoly, kappaPoly, DijPoly
