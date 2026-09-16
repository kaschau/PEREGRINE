"""Where an eos gets cp against temperature, and h and s from it: the same
four polynomials in ln T on each species whatever the source."""

import numpy as np

from . import Ru, Tref, anyOf
from ..misc import subclasses
from .thermoData import BaseThermoData
from .polyFitMixin import PolyFitMixin
from .species import Species


class BaseCaloricModel:
    """Where an eos gets cp against temperature, and h and s from it. Not
    selectable: each eos names its own. One method per quantity, each for one
    species, in the order the eos stores them: cpPoly, hPoly, hRef, sPoly."""

    def __init__(self, configSect):
        self.configSect = configSect
        # what it reads off each species
        self.needs = ()


class ConstantCpModel(BaseCaloricModel):
    """One cp per species, stated by the case in J/kg/K: a degree-0 polynomial,
    so h and s are closed form."""

    def __init__(self, configSect):
        super().__init__(configSect)
        self.needs += ("cp0",)

    def cpPoly(self, sp):
        # a constant is its own fit, exactly
        return [sp["cp0"] * sp["MW"] / Ru], 0.0

    def hPoly(self, sp):
        return sp["cpPoly"]

    def hRef(self, sp):
        # J/kmol; a non-reacting case need not state it
        return sp.get("dHf298", 0.0) / Ru - Tref * sp["cpPoly"][0]

    def sPoly(self, sp):
        # J/kmol/K; a non-reacting case need not state it
        c = sp["cpPoly"][0]
        return [sp.get("s298", 0.0) / Ru - c * np.log(Tref), c]


class TempDepCpModel(BaseCaloricModel, PolyFitMixin):
    """cp against temperature refit from whatever data the species carries."""

    def __init__(self, configSect):
        super().__init__(configSect)
        self.needs += (anyOf(f.name for f in subclasses(BaseThermoData)),)

    def cpPoly(self, sp):
        """The fit and the relative error it achieved: within the tolerance,
        or the best the degree cap can do."""
        T, cpR = self._sample(sp)
        tol, deg = self.configSect["reFitTol"], self.configSect["reFitMaxDegree"]
        return self.fitLowestDegree(np.log(T), cpR, tol, deg)

    def hPoly(self, sp):
        return self.integratePolyExp(sp["cpPoly"])

    def hRef(self, sp):
        hRefR, _ = BaseThermoData.findReferenceFormat(sp).hs(sp)
        return hRefR - Tref * self.evaluate(sp["hPoly"], np.log(Tref))

    def sPoly(self, sp):
        _, sRefR = BaseThermoData.findReferenceFormat(sp).hs(sp)
        S = self.integratePoly(sp["cpPoly"])
        S[0] = sRefR - self.evaluate(S, np.log(Tref))
        return S

    def _sample(self, sp):
        """(T, cp/R) over the case's range, from the species' data extended
        past its ends where the case goes further."""
        Tlow, Thigh = self.configSect["Trange"]
        T, cpR = BaseThermoData.findReferenceFormat(sp).cp(sp)
        if Tlow < T[0]:
            T, cpR = self._extendDown(sp, T, cpR, Tlow)
        if Thigh > T[-1]:
            T, cpR = self._extendUp(T, cpR, Thigh)
        inside = (T >= Tlow) & (T <= Thigh)
        return T[inside], cpR[inside]

    def _extendDown(self, sp, T, cpR, Tlow):
        """Linearly from cp at the data's floor to the frozen-mode value at 0 K,
        at the data's spacing."""
        # cp/R at 0 K: translation, and rotation if the shape is known
        frozen = 2.5 + Species.rotDOF[sp["geometry"]] if "geometry" in sp else cpR[0]
        dT = T[1] - T[0]
        Tx = np.arange(T[0] - dT, Tlow, -dT)
        if Tx.size == 0 or Tx[-1] > Tlow:
            Tx = np.append(Tx, Tlow)
        Tx = Tx[::-1]
        return np.concatenate([Tx, T]), np.concatenate(
            [frozen + (cpR[0] - frozen) * Tx / T[0], cpR]
        )

    @staticmethod
    def _extendUp(T, cpR, Thigh):
        """cp held at its last value, at the data's spacing."""
        dT = T[-1] - T[-2]
        Tx = np.arange(T[-1] + dT, Thigh, dT)
        if Tx.size == 0 or Tx[-1] < Thigh:
            Tx = np.append(Tx, Thigh)
        return np.concatenate([T, Tx]), np.concatenate([cpR, np.full_like(Tx, cpR[-1])])
