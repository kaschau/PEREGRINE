"""Compressible flow of a mixture without diffusion: the equation of state,
the advective flux and its apply, the euler boundary conditions."""

import numpy as np

from ...files.configFile import pgConfigError
from ...graph import BCNode, ExchangeGraphs, Graph, LaunchNode, RedoNode
from ...kernel import (
    BCKernel,
    CellCenterKernel,
    CellFaceKernel,
    FluxKernel,
    UnorderedKernelGroup,
)
from ...mixture import Mixture
from ...multiBlock.arrays import CellCenterArray, CellFaceArray
from .boundaries import BaseEulerBC
from ..base import BaseSimulation


class EulerSimulation(BaseSimulation):
    """The Euler equations of a mixture: the conserved state Q, its
    derivative dQ, the thermodynamic state -- p and T in q, what the eos
    keeps in qh -- and the cell-face fluxes of one advective scheme."""

    physics = "euler"
    viscous = False
    metrics = (
        "Jinv",
        "dIJK",
        "dENCdxyz",
        "iFaces",
        "jFaces",
        "kFaces",
        "iS",
        "jS",
        "kS",
    )
    # the base its boundaries are found under, and the hooks they have
    # bodies at
    bcBase = BaseEulerBC
    bcHooks = ("euler",)

    def __init__(self, config):
        super().__init__(config)
        self.mixture = Mixture(config["simulation"])
        self.ne = 5 + self.mixture.ns - 1

    def validate(self, config):
        rhs, sim, ti = config["RHS"], config["simulation"], config["timeIntegration"]
        if rhs["subgrid"] is not None:
            raise pgConfigError("subgrid", rhs["subgrid"], "not until its round")
        if (rhs["secondaryAdvFlux"] is None) != (rhs["switchAdvFlux"] is None):
            raise pgConfigError(
                "secondaryAdvFlux",
                rhs["secondaryAdvFlux"],
                "a secondary flux and a switch go together",
            )
        if config["viscousSponge"]["spongeON"]:
            raise pgConfigError(
                "viscousSponge", True, "not until the composition round"
            )
        if sim["chemistry"]:
            raise pgConfigError("chemistry", True, "not until the composition round")
        if rhs["primaryAdvFlux"] is None:
            raise pgConfigError("primaryAdvFlux", None, "a case has a primary flux")
        if rhs["primaryAdvFlux"] == "fourthOrderKEEP":
            raise pgConfigError(
                "primaryAdvFlux",
                "fourthOrderKEEP",
                "not until it is written as a kernel",
            )
        if sim["eos"] not in ("cpg", "tpg", "realGas"):
            raise pgConfigError("eos", sim["eos"])
        if ti["integrator"] == "dualTime":
            if sim["eos"] not in ("cpg", "tpg"):
                raise pgConfigError(
                    "dualTime", sim["eos"], "only cpg and tpg are supported"
                )
            if ti["controller"] != "fixed":
                raise pgConfigError(
                    "dualTime", ti["controller"], "only a fixed time step is supported"
                )
            if ti["pseudoIntegrator"] == "dualTime":
                raise pgConfigError(
                    "pseudoIntegrator",
                    "dualTime",
                    "the pseudo time scheme is Runge-Kutta",
                )

    @property
    def primVars(self):
        return ["p", "u", "v", "w", "T"] + self.mixture.speciesNames[:-1]

    @property
    def exportVars(self):
        """Adds the density ahead, and the last species -- what the others
        leave -- behind."""
        return ["rho"] + self.primVars + self.mixture.speciesNames[-1:]

    def exportData(self, blk, names):
        """Derives the variables from Q and q: p and T are q's, the
        velocity and the mass fractions Q's over the density."""
        Q, q = blk.Q.get(), blk.q.get()
        rho = Q[..., 0]
        # the halo's edges and corners are never written, so hold no density
        with np.errstate(divide="ignore", invalid="ignore"):
            rhoinv = 1.0 / rho
            species = self.mixture.speciesNames
            values = {"rho": rho, "p": q[..., 0], "T": q[..., 1]}
            for n, name in enumerate("uvw"):
                values[name] = Q[..., 1 + n] * rhoinv
            for n, name in enumerate(species[:-1]):
                values[name] = Q[..., 5 + n] * rhoinv
            # the last species is what the others leave: all of it, for one
            values[species[-1]] = np.ones_like(rho) - sum(
                values[name] for name in species[:-1]
            )
        return {name: values[name] for name in names}

    def arrays(self):
        ne, ns = self.ne, self.mixture.ns
        arrays = {}
        # cell face fluxes
        for axis, d in enumerate("ijk"):
            arrays[f"{d}F"] = dict(kind=CellFaceArray, components=ne, axis=axis)
        # the conserved state
        arrays["Q"] = dict(kind=CellCenterArray, components=ne, exchanged=True)
        # time derivative (RHS)
        arrays["dQ"] = dict(kind=CellCenterArray, components=ne)
        # thermodynamic state, p and T
        arrays["q"] = dict(kind=CellCenterArray, components=2)
        # what the eos keeps
        arrays["qh"] = dict(
            kind=CellCenterArray, components=self.mixture.eos.qhComponents(ns)
        )
        return arrays

    def declKernels(self):
        rhs = self.config["RHS"]
        k = super().declKernels()
        # the jit bakes the eos into these
        k["stateFromCons"] = CellCenterKernel("thermo/stateFromCons.cpp")
        k["stateFromPrims"] = CellCenterKernel("thermo/stateFromPrims.cpp")
        # one flux per direction, unordered: composed from a formula and a
        # reconstruction, with the secondary blended in by the switch, or a
        # scheme of its own
        scheme = rhs["primaryAdvFlux"]
        k["advFlux"] = UnorderedKernelGroup(
            [
                (
                    FluxKernel(
                        scheme,
                        d,
                        rhs["secondaryAdvFlux"],
                        rhs["switchAdvFlux"],
                        rhs["switchValues"],
                    )
                    if FluxKernel.composed(scheme)
                    else CellFaceKernel(f"advFlux/{scheme}.cpp", d)
                )
                for d in range(3)
            ],
            name=scheme,
        )
        k["applyFlux"] = CellCenterKernel("utils/applyFlux.cpp")
        # boundary conditions by hook
        for bcHook in self.bcHooks:
            k[f"bcs {bcHook}"] = UnorderedKernelGroup(
                [
                    BCKernel(self.bcBase.named(t), bcHook)
                    for t in self.bcBase.withHook(bcHook)
                ],
                name=f"bcs {bcHook}",
            )
        return k

    def bakes(self):
        return self.mixture, self.config["simulation"]

    def graphs(self):
        """Gives consistify -- the Q exchange around the equation of state
        and the euler boundary conditions, the halos a message brought done
        again after it lands -- and the right-hand side, the advective flux
        and its apply."""
        k = self.kernels
        consistify = ExchangeGraphs(
            "consistify",
            "Q",
            during=[
                LaunchNode(k["stateFromCons"], "all"),
                BCNode(k["bcs euler"], "here"),
            ],
            after=[BCNode(k["bcs euler"], "remote"), RedoNode(k["stateFromCons"])],
        )
        rhs = Graph(
            "rhs",
            [
                LaunchNode(k["advFlux"], "interior"),
                LaunchNode(k["applyFlux"], "interior"),
            ],
        )
        return {"consistify": [consistify], "rhs": [rhs]}

    def initialState(self):
        """Gives the uniform primitive vector of the config's initial
        conditions: p, u, v, w, T and the mass fractions by species name,
        the last species taking the remainder."""
        ic = self.config["initialConditions"]
        Y = ic["Y"]
        names = self.mixture.speciesNames
        unknown = sorted(set(Y) - set(names))
        if unknown:
            raise pgConfigError(
                "initialConditions",
                "Y",
                f"names {unknown}, which are not species of this mixture: {names}.",
            )
        if sum(Y.values()) > 1.0 + 1e-12:
            raise pgConfigError(
                "initialConditions", "Y", f"sums to {sum(Y.values())}, more than one."
            )
        values = [ic[key] for key in ("p", "u", "v", "w", "T")]
        return values + [Y.get(name, 0.0) for name in names[:-1]]

    def report(self):
        sim = self.config["simulation"]
        return (
            f"  Physics: {self.physics}\n"
            f"  Species: {self.mixture.speciesNames}\n"
            f"  Equation of State: {sim['eos']}\n"
        )
