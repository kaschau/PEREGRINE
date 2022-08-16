from .completeSpecies import completeSpecies
from .chungConstants import viscConsts, conductConsts
import numpy as np


def chungDenseGas(usersp, refsp):
    ns = len(usersp.keys())

    # MW
    MW = completeSpecies("MW", usersp, refsp)
    # mu
    dipole = completeSpecies("dipole", usersp, refsp)
    # W_ac
    acentric = completeSpecies("acentric", usersp, refsp)
    # Tcrit
    Tcrit = completeSpecies("Tcrit", usersp, refsp)
    # Vctir
    Vcrit = completeSpecies("Vcrit", usersp, refsp)

    chungA = np.zeros((ns, 10))
    chungB = np.zeros((ns, 7))
    redDipole = np.zeros(ns)
    for n in range(ns):
        # Chung wants reduced dipole moments in units of Debye???
        dp = dipole[n] / 3.33564e-30
        # And Vcrit in cm^3/mol
        Vc = Vcrit[n] * MW[n] * 1e3
        redDipole[n] = 131.3 * dp / np.sqrt(Vc * Tcrit[n])
        for i in range(10):
            chungA[n, i] = (
                viscConsts[i, 0]
                + viscConsts[i, 1] * acentric[n]
                + viscConsts[i, 2] * redDipole[n] ** 4
            )  # + viscConsts[i,3] <- I guess were ignoring k for now?

        for i in range(7):
            chungB[n, i] = (
                conductConsts[i, 0]
                + conductConsts[i, 1] * acentric[n]
                + conductConsts[i, 2] * redDipole[n] ** 4
            )  # + conductConsts[i,3] <- I guess were ignoring k for now?

    return chungA, chungB, redDipole
