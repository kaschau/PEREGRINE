"""The exact solution of a Riemann problem of a calorically perfect gas:
Toro's iterative solver (Riemann Solvers and Numerical Methods for Fluid
Dynamics, 3rd ed., chapter 4), with his test cases, which a shock tube
run is measured against."""

import numpy as np


class RiemannProblem:
    """Two states at rest against each other at x0 on the unit line, and
    the exact solution at time t."""

    # Toro's tests: rhoL, uL, pL, rhoR, uR, pR, x0, t, and the step a run takes
    cases = {
        0: (1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 0.5, 0.2, 1e-4),
        1: (1.0, 0.75, 1.0, 0.125, 0.0, 0.1, 0.3, 0.2, 1e-4),
        2: (1.0, -2.0, 0.4, 1.0, 2.0, 0.4, 0.5, 0.15, 1e-4),
        3: (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01, 0.5, 0.012, 5e-6),
        4: (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950, 0.4, 0.035, 1e-5),
        5: (1.0, -19.5975, 1000.0, 1.0, -19.5975, 0.01, 0.8, 0.012, 1e-5),
    }

    def __init__(self, rhoL, uL, pL, rhoR, uR, pR, x0, t, dt, gamma=1.4, R=281.4):
        self.rhoL, self.uL, self.pL = rhoL, uL, pL
        self.rhoR, self.uR, self.pR = rhoR, uR, pR
        self.x0, self.t, self.dt = x0, t, dt
        self.gamma, self.R = gamma, R
        self.TL = pL / (rhoL * R)
        self.TR = pR / (rhoR * R)
        self.cL = np.sqrt(gamma * pL / rhoL)
        self.cR = np.sqrt(gamma * pR / rhoR)

    @classmethod
    def toro(cls, test, gamma=1.4, R=281.4):
        """Makes one of Toro's tests: 0 is Sod's tube."""
        return cls(*cls.cases[test], gamma=gamma, R=R)

    def _g(self):
        gamma = self.gamma
        return dict(
            g1=(gamma - 1.0) / (2.0 * gamma),
            g2=(gamma + 1.0) / (2.0 * gamma),
            g3=2.0 * gamma / (gamma - 1.0),
            g4=2.0 / (gamma - 1.0),
            g5=2.0 / (gamma + 1.0),
            g6=(gamma - 1.0) / (gamma + 1.0),
            g7=(gamma - 1.0) / 2.0,
            g8=gamma - 1.0,
        )

    def guessP(self):
        """Guesses the star pressure: the primitive variable solution where
        it applies, else the two rarefaction or two shock approximation."""
        pL, rhoL, uL, cL = self.pL, self.rhoL, self.uL, self.cL
        pR, rhoR, uR, cR = self.pR, self.rhoR, self.uR, self.cR
        g = self._g()
        cup = 0.25 * (rhoL + rhoR) * (cL + cR)
        ppv = max(0.0, 0.5 * (pL + pR) + 0.5 * (uL - uR) * cup)
        pmin, pmax = min(pL, pR), max(pL, pR)
        if pmax / pmin < 2.0 and pmin < ppv < pmax:
            return ppv
        if ppv < pmin:
            pQ = (pL / pR) ** g["g1"]
            uM = (pQ * uL / cL + uR / cR + g["g4"] * (pQ - 1.0)) / (pQ / cL + 1.0 / cR)
            pTL = 1.0 + g["g7"] * (uL - uM) / cL
            pTR = 1.0 + g["g7"] * (uM - uR) / cR
            return 0.5 * (pL * pTL ** g["g3"] + pR * pTR ** g["g3"])
        gEL = np.sqrt((g["g5"] / rhoL) / (g["g6"] * pL + ppv))
        gER = np.sqrt((g["g5"] / rhoR) / (g["g6"] * pR + ppv))
        return (gEL * pL + gER * pR - (uR - uL)) / (gEL + gER)

    def _prefun(self, p, pK, rhoK, cK):
        """Gives one side's pressure function and its derivative."""
        g = self._g()
        if p < pK:
            pRatio = p / pK
            return g["g4"] * cK * (pRatio ** g["g1"] - 1.0), (
                1.0 / (rhoK * cK)
            ) * pRatio ** (-g["g2"])
        AK, BK = g["g5"] / rhoK, g["g6"] * pK
        qrt = np.sqrt(AK / (BK + p))
        return (p - pK) * qrt, (1.0 - 0.5 * (p - pK) / (BK + p)) * qrt

    def star(self, tol=1e-6, maxIter=100):
        """Gives the star region's pressure and velocity, by Newton."""
        pOld = self.guessP()
        uDiff = self.uR - self.uL
        for _ in range(maxIter):
            fL, fDL = self._prefun(pOld, self.pL, self.rhoL, self.cL)
            fR, fDR = self._prefun(pOld, self.pR, self.rhoR, self.cR)
            p = pOld - (fL + fR + uDiff) / (fDL + fDR)
            deltaP = 2.0 * abs((p - pOld) / (p + pOld))
            pOld = p
            if deltaP < tol:
                break
        else:
            raise ValueError("the star pressure did not converge")
        return p, 0.5 * (self.uL + self.uR + fR - fL)

    def sample(self, pM, uM, s):
        """Gives p, u, rho, e on the ray x / t = s of the solution."""
        pL, rhoL, uL, cL = self.pL, self.rhoL, self.uL, self.cL
        pR, rhoR, uR, cR = self.pR, self.rhoR, self.uR, self.cR
        gamma, g = self.gamma, self._g()
        if s < uM:
            if pM < pL:
                if s < uL - cL:
                    rho, u, p = rhoL, uL, pL
                else:
                    cmL = cL * (pM / pL) ** g["g1"]
                    if s > uM - cmL:
                        rho, u, p = rhoL * (pM / pL) ** (1.0 / gamma), uM, pM
                    else:
                        u = g["g5"] * (cL + g["g7"] * uL + s)
                        c = g["g5"] * (cL + g["g7"] * (uL - s))
                        rho, p = rhoL * (c / cL) ** g["g4"], pL * (c / cL) ** g["g3"]
            else:
                pmL = pM / pL
                if s < uL - cL * np.sqrt(g["g2"] * pmL + g["g1"]):
                    rho, u, p = rhoL, uL, pL
                else:
                    rho, u, p = rhoL * (pmL + g["g6"]) / (pmL * g["g6"] + 1.0), uM, pM
        else:
            if pM > pR:
                pmR = pM / pR
                if s > uR + cR * np.sqrt(g["g2"] * pmR + g["g1"]):
                    rho, u, p = rhoR, uR, pR
                else:
                    rho, u, p = rhoR * (pmR + g["g6"]) / (pmR * g["g6"] + 1.0), uM, pM
            else:
                if s > uR + cR:
                    rho, u, p = rhoR, uR, pR
                else:
                    cmR = cR * (pM / pR) ** g["g1"]
                    if s < uM + cmR:
                        rho, u, p = rhoR * (pM / pR) ** (1.0 / gamma), uM, pM
                    else:
                        u = g["g5"] * (-cR + g["g7"] * uR + s)
                        c = g["g5"] * (cR - g["g7"] * (uR - s))
                        rho, p = rhoR * (c / cR) ** g["g4"], pR * (c / cR) ** g["g3"]
        return p, u, rho, p / rho / g["g8"]

    def solve(self, x):
        """Gives the exact p, u, rho and e at time t at the points :x:."""
        g4 = 2.0 / (self.gamma - 1.0)
        if g4 * (self.cL + self.cR) <= self.uR - self.uL:
            raise ValueError("the states separate: a vacuum forms")
        pM, uM = self.star()
        res = {name: np.empty(len(x)) for name in ("p", "u", "rho", "energy")}
        for i, xi in enumerate(x):
            res["p"][i], res["u"][i], res["rho"][i], res["energy"][i] = self.sample(
                pM, uM, (xi - self.x0) / self.t
            )
        return res
