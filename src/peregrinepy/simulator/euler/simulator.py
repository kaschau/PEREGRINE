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
from ...mixture import getMixture
from ...multiBlock.arrays import CellCenterArray, CellFaceArray
from .boundaries import BaseEulerBC
from ..base import BaseSimulator


class EulerSimulator(BaseSimulator):
    """The Euler equations of a mixture: the conserved state Q, its
    derivative dQ, the thermodynamic state -- p and T in q, what the eos
    keeps in qh -- and the cell-face fluxes of one advective scheme."""

    name = "euler"
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
        self.mixture = getMixture(self.config)
        self.ne = 5 + self.mixture.ns - 1

    def validate(self):
        rhs, ti = self.config["RHS"], self.config["timeIntegration"]
        mixture, chemistry = self.config["mixture"], self.config["chemistry"]
        if (rhs["secondaryAdvFlux"] is None) != (rhs["switchAdvFlux"] is None):
            raise pgConfigError(
                "secondaryAdvFlux",
                rhs["secondaryAdvFlux"],
                "a secondary flux and a switch go together",
            )
        if chemistry["source"] not in (None, "explicit", "substepped"):
            raise pgConfigError("chemistry", chemistry["source"])
        bisections = chemistry["entropyBisections"]
        if not isinstance(bisections, int) or bisections < 0:
            raise pgConfigError(
                "entropyBisections", bisections, "is a count, none for no cap"
            )
        if rhs["primaryAdvFlux"] is None:
            raise pgConfigError("primaryAdvFlux", None, "a case has a primary flux")
        if mixture["eos"] not in ("cpg", "tpg", "realGas"):
            raise pgConfigError("eos", mixture["eos"])
        if ti["chemistryJacobian"] not in (None, "diagonal"):
            raise pgConfigError("chemistryJacobian", ti["chemistryJacobian"])
        if ti["chemistryJacobian"] and not chemistry["source"]:
            raise pgConfigError(
                "chemistryJacobian",
                ti["chemistryJacobian"],
                "a chemistry Jacobian needs a chemistry",
            )
        if ti["chemistryJacobian"] and ti["integrator"] != "dualTime":
            raise pgConfigError(
                "chemistryJacobian",
                ti["chemistryJacobian"],
                "the chemistry Jacobian is dual time's pseudo system's",
            )
        if not isinstance(ti["lowMach"], bool):
            raise pgConfigError("lowMach", ti["lowMach"], "is on or off")
        for key in ("pseudoCFL", "pseudoVNN"):
            if not isinstance(ti[key], (int, float)) or ti[key] <= 0:
                raise pgConfigError(key, ti[key], "is a positive number")
        if ti["integrator"] == "dualTime":
            controller = self.config["simulation"]["controller"]
            if controller != "fixed":
                raise pgConfigError(
                    "dualTime", controller, "only a fixed time step is supported"
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
        arrays["Q"] = dict(
            kind=CellCenterArray, components=ne, exchanged=True, vectors=1
        )
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
        # finite-rate chemistry begins dQ with the source -- the production
        # rates of the state, or the source substepped over the step, the
        # reactions baked in by the jit -- and the fluxes are appended to
        # it; without, they begin it
        source = self.config["chemistry"]["source"]
        if source == "explicit":
            k["productionRateSource"] = CellCenterKernel(
                "chemistry/productionRateSource.cpp"
            )
        if source == "substepped":
            chemistry = self.config["chemistry"]
            k["finiteRateSubstep"] = CellCenterKernel(
                "chemistry/finiteRateSubstep.cpp",
                defines=[
                    f"PG_CHEMISTRY_MAX_SUBSTEPS={int(chemistry['maxSubSteps'])}",
                    f"PG_CHEMISTRY_ENTROPY_BISECTIONS={int(chemistry['entropyBisections'])}",
                ],
            )
        k["applyFlux"] = CellCenterKernel(
            "utils/applyFlux.cpp", defines=["PG_FLUXES_APPEND=1"] if source else []
        )
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

    # The chemistry source is the cell's own, over the interior, and wants
    # nothing of the fluxes: it begins dQ first in the right-hand side,
    # which in a viscous case puts it under the gradient exchange with the
    # fluxes, and their apply appends to it
    def sourceNodes(self, dt):
        """Gives what begins dQ with the chemistry source, ahead of the
        fluxes: the production rates of the state, or the source
        substepped over the step :dt:."""
        k = self.kernels
        if "productionRateSource" in k:
            return [LaunchNode(k["productionRateSource"], "interior")]
        if "finiteRateSubstep" in k:
            return [LaunchNode(k["finiteRateSubstep"], "interior", dt=dt)]
        return []

    def graphs(self, dt):
        """Gives consistify -- the Q exchange around the equation of state
        and the euler boundary conditions, the halos a message brings left
        out while it flies and done after it lands -- and the right-hand
        side: the chemistry source, the advective flux and its apply."""
        k = self.kernels
        consistify = ExchangeGraphs(
            "consistify",
            "Q",
            during=[
                LaunchNode(k["stateFromCons"], "allLocal"),
                BCNode(k["bcs euler"]),
            ],
            after=[RedoNode(k["stateFromCons"])],
        )
        rhs = Graph(
            "rhs",
            [
                *self.sourceNodes(dt),
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
        return (
            f"  Simulator: {self.name}\n"
            f"  Species: {self.mixture.speciesNames}\n"
            f"  Equation of State: {self.config['mixture']['eos']}\n"
        )
