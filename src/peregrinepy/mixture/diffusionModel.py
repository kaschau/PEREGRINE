"""How species diffuse, a separate choice from how momentum and heat do."""

import numpy as np

from . import kB
from .baseModel import BaseModel
from .polyFitMixin import PolyFitMixin


class BaseSpeciesDiffusionModel(BaseModel):
    """One way of getting the species diffusion coefficients."""


class BinaryModel(BaseSpeciesDiffusionModel, PolyFitMixin):
    """Every pair's diffusion coefficient from the collision integrals."""

    name = "binary"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.fromSpecies += ("well", "diam", "dipole", "polarize")
        # the pair coefficients come off kinetic theory's machinery
        self.dependsOn = ("kineticTheory",)
        self.provides["dij"] = "dij"

    def dij(self, species):
        """Each species' row of the pair matrix: every pair's diffusion
        coefficient at unit pressure, D / T^1.5 fitted in ln T."""
        from .transportModel import KineticTheoryModel

        m = KineticTheoryModel(self.cfgsect).collisionParameters(species)
        Ts, rMass, rWell, rDiam = m["Ts"], m["rMass"], m["rWell"], m["rDiam"]
        ns = len(rMass)
        k, j = np.triu_indices(ns)

        # (T, pair): reduced temperature, and the collision integrals there
        Tstar = np.outer(Ts, kB / rWell[k, j])
        delta = np.broadcast_to(m["rDeltaStar"][k, j], Tstar.shape)
        omega11 = m["omega22"](Tstar, delta, grid=False) / m["astar"](
            Tstar, delta, grid=False
        )

        # at unit pressure; the kernel divides by the real one
        diff = (
            (3.0 / 16.0)
            * np.sqrt(2.0 * np.pi / rMass[k, j])
            * (kB * Ts[:, None]) ** 1.5
            / (np.pi * rDiam[k, j] ** 2 * omega11)
        )
        diff = diff / Ts[:, None] ** 1.5
        tol, deg = self.cfgsect["reFitTol"], self.cfgsect["reFitMaxDegree"]
        rows = [[None] * ns for _ in range(ns)]
        for a, b, D in zip(k, j, diff.T):
            rows[a][b] = rows[b][a] = self.fitLowestDegree(np.log(Ts), D, tol, deg)[0]
        return rows


class LewisModel(BaseSpeciesDiffusionModel):
    """D = kappa / (rho cp Le), with a LewisModel number of one unless the case says."""

    name = "lewis"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.provides["lewis"] = "lewis"

    def lewis(self, species):
        return [sp.get("lewis", 1.0) for sp in species.values()]
