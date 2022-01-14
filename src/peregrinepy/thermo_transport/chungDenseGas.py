from .completeSpecies import completeSpecies
from .chungConstants import viscConsts, conductConsts
import numpy as np


def chungDenseGas(usersp, refsp):
    ns = len(usersp.keys())

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
        redDipole[n] = 131.3 * dipole[n] / np.sqrt(Vcrit[n] * Tcrit[n])
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
