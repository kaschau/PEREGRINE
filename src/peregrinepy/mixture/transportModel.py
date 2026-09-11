"""How momentum and heat diffuse: each species' viscosity and conductivity."""

import numpy as np
from scipy import interpolate as intrp

from . import Ru, Tref, anyOf, avogadro, database, debye
from . import epsilon0 as eps0, kB as kb
from .baseModel import BaseModel
from .polyFitMixin import PolyFitMixin
from .eosModel import RealGasModel
from .species import Species


class BaseTransportModel(BaseModel):
    """One way of getting the pure-species viscosity and conductivity."""


class KineticTheoryModel(BaseTransportModel, PolyFitMixin):
    """Chapman-Enskog from each species' Lennard-Jones parameters. The binary
    diffusion model reads its collision parameters too."""

    name = "kineticTheory"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.fromSpecies += ("well", "diam", "dipole", "polarize", "zrot", "geometry")
        # the conductivity needs cp against temperature, which these put on the species
        self.dependsOn = (anyOf(("tpg", "realGas")),)
        # ordered: the conductivity divides by the viscosity
        self.provides.update(muPoly="muPoly", kappaPoly="kappaPoly")

    def collisionParameters(self, species):
        """The temperatures every fit is made over, the collision integral
        interpolants, and each species' and each pair's reduced quantities."""
        Ts = np.linspace(*self.cfgsect["Trange"], 50)

        ci = database("collisionIntegrals")
        omega22 = intrp.RectBivariateSpline(
            ci["tstar22"], ci["delta"], ci["omega22"], kx=5, ky=5
        )
        astar = intrp.RectBivariateSpline(
            ci["tstar"], ci["delta"], ci["astar"], kx=5, ky=5
        )

        MW = self.collect("MW", species)
        well = self.collect("well", species)
        diam = self.collect("diam", species)
        dipole = self.collect("dipole", species)
        polarize = self.collect("polarize", species)
        rotDOF = np.array(
            [Species.rotDOF[g] for g in self.collect("geometry", species)]
        )
        mass = MW / avogadro

        # every pair's reduced mass, diameter, well and dipole
        rMass = np.outer(mass, mass) / np.add.outer(mass, mass)
        rDiam = 0.5 * np.add.outer(diam, diam)
        rWell = np.sqrt(np.outer(well, well))
        rDipole = np.sqrt(np.outer(dipole, dipole))
        rDeltaStar = 0.5 * rDipole**2 / (4 * np.pi * eps0 * rWell * rDiam**3)

        # a polar-nonpolar pair: induction deepens the well and shrinks the diameter
        polar = dipole > 0.0
        mixed = polar[:, None] != polar[None, :]
        # for each pair, the polar member's and the nonpolar member's properties
        polarOf = np.where(
            polar[:, None], np.arange(len(MW))[:, None], np.arange(len(MW))[None, :]
        )
        nonpolarOf = np.where(
            polar[:, None], np.arange(len(MW))[None, :], np.arange(len(MW))[:, None]
        )
        alphaStar = polarize[nonpolarOf] / diam[nonpolarOf] ** 3
        dipoleStar = dipole[polarOf] / np.sqrt(
            4 * np.pi * eps0 * diam[polarOf] ** 3 * well[polarOf]
        )
        xi = 1.0 + 0.25 * alphaStar * dipoleStar**2 * np.sqrt(
            well[polarOf] / well[nonpolarOf]
        )
        rWell = np.where(mixed, rWell * xi**2, rWell)
        rDiam = np.where(mixed, rDiam * xi ** (-1 / 6), rDiam)

        return dict(
            Ts=Ts,
            omega22=omega22,
            astar=astar,
            MW=MW,
            well=well,
            diam=diam,
            mass=mass,
            zrot=self.collect("zrot", species),
            rotDOF=rotDOF,
            rMass=rMass,
            rDiam=rDiam,
            rWell=rWell,
            rDeltaStar=rDeltaStar,
        )

    def muPoly(self, species):
        """Each species' viscosity: sqrt(visc / sqrt(T)) fitted in ln T."""
        m = self.collisionParameters(species)
        Ts, well, diam, mass = m["Ts"], m["well"], m["diam"], m["mass"]

        # (T, species): reduced temperature, and the collision integral there
        Tstar = np.outer(Ts, kb / well)
        delta = np.broadcast_to(m["rDeltaStar"].diagonal(), Tstar.shape)
        omega22 = m["omega22"](Tstar, delta, grid=False)
        visc = (
            (5.0 / 16.0)
            * np.sqrt(np.pi * mass * kb * Ts[:, None])
            / (np.pi * diam**2 * omega22)
        )

        visc = np.sqrt(visc / np.sqrt(Ts)[:, None])
        tol, deg = self.cfgsect["reFitTol"], self.cfgsect["reFitMaxDegree"]
        return [self.fitLowestDegree(np.log(Ts), v, tol, deg)[0] for v in visc.T]

    def kappaPoly(self, species):
        """Each species' conductivity: cond / sqrt(T) fitted in ln T."""
        m = self.collisionParameters(species)
        Ts, well, diam, MW = m["Ts"], m["well"], m["diam"], m["MW"]
        zrot, rotDOF, rMass = m["zrot"], m["rotDOF"], m["rMass"].diagonal()
        logTs = np.log(Ts)

        # the species already carry muPoly and cpPoly; undo their fitted forms
        visc = np.array(
            [
                self.evaluate(sp["muPoly"], logTs) ** 2 * np.sqrt(Ts)
                for sp in species.values()
            ]
        ).T
        cpR = np.array(
            [self.evaluate(sp["cpPoly"], logTs) for sp in species.values()]
        ).T

        # (T, species) throughout
        T = Ts[:, None]
        Tstar = np.outer(Ts, kb / well)
        delta = np.broadcast_to(m["rDeltaStar"].diagonal(), Tstar.shape)
        omega22 = m["omega22"](Tstar, delta, grid=False)
        omega11 = omega22 / m["astar"](Tstar, delta, grid=False)

        # self diffusion, and the Parker rotational relaxation at T and at Tref
        selfDiff = (
            (3.0 / 16.0)
            * np.sqrt(2.0 * np.pi / rMass)
            * (kb * T) ** 1.5
            / (np.pi * diam**2 * omega11)
        )

        def fz(Tstar):
            return (
                1.0
                + np.pi**1.5 / np.sqrt(Tstar) * (0.5 + 1.0 / Tstar)
                + (0.25 * np.pi**2 + 2) / Tstar
            )

        fInt = MW / (Ru * T) * selfDiff / visc
        cvRot = rotDOF
        A = 2.5 - fInt
        B = zrot * fz(kb * Tref / well) / fz(Tstar) + 2.0 / np.pi * (
            5 / 3 * rotDOF + fInt
        )
        c1 = 2.0 / np.pi * A / B
        cvInt = cpR - 2.5 - cvRot
        fRot = fInt * (1.0 + c1)
        fTrans = 2.5 * (1.0 - c1 * cvRot / 1.5)
        cond = visc / MW * Ru * (fTrans * 1.5 + fRot * cvRot + fInt * cvInt)

        cond = cond / np.sqrt(Ts)[:, None]
        tol, deg = self.cfgsect["reFitTol"], self.cfgsect["reFitMaxDegree"]
        return [self.fitLowestDegree(logTs, c, tol, deg)[0] for c in cond.T]


class ChungDenseGasModel(BaseTransportModel):
    """Chung's high-pressure correlation, written around a critical point."""

    name = "chungDenseGas"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.fromSpecies += ("dipole",)
        # the same need as the real gas eos, which will have put the point on the species
        self.derivable[RealGasModel.critical] = ("well", "diam")
        self.dependsOn = ("realGas",)
        self.provides["chungCoefficients"] = ("chungA", "chungB", "redDipole")

    def chungCoefficients(self, species):
        """The viscosity and conductivity coefficient tables and the reduced
        dipole they are built from."""
        chung = database("chung")
        visc, cond = np.array(chung["viscosity"]), np.array(chung["conductivity"])
        MW = self.collect("MW", species)
        dipole = self.collect("dipole", species)
        Tcrit = self.collect("Tcrit", species)
        Vcrit = self.collect("Vcrit", species)
        acentric = self.collect("acentric", species)

        # Chung's reduced dipole wants debye and cm^3/mol
        Vc = Vcrit * MW * 1e3
        redDipole = 131.3 * (dipole / debye) / np.sqrt(Vc * Tcrit)

        # (species, coefficient): a0 + a1 w + a2 mu_r^4; the association factor a3 is not carried
        terms = np.stack([np.ones_like(acentric), acentric, redDipole**4], axis=1)
        chungA = terms @ visc[:, :3].T
        chungB = terms @ cond[:, :3].T
        return chungA, chungB, redDipole


class ConstantPropsModel(BaseTransportModel):
    """One viscosity and one conductivity per species, stated by the case."""

    name = "constantProps"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.requiredInput += ("mu0", "kappa0")
