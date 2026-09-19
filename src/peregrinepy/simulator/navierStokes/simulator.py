"""Compressible flow of a mixture with diffusion: the Euler equations, the
transport properties, the gradients and their exchange, and the diffusive
flux, with the boundary conditions' gradient hooks."""

from .boundaries import BaseNSBC
from ...files.configFile import pgConfigError
from ...kernel import CellCenterKernel, CellFaceKernel, UnorderedKernelGroup
from ...multiBlock.arrays import CellCenterArray
from ...graph import BCNode, ExchangeGraphs, LaunchNode, RedoNode
from ..euler import EulerSimulator


class NavierStokesSimulator(EulerSimulator):
    """The Navier-Stokes equations of a mixture: Euler's, the gradients of
    velocity, temperature and mass fractions, the transport properties,
    and the diffusive flux."""

    name = "navierStokes"
    viscous = True
    bcBase = BaseNSBC
    bcHooks = ("euler", "preDqDxyz", "postDqDxyz")
    # the cell length, which the ducros switch reads
    metrics = EulerSimulator.metrics + ("cellLength",)

    def validate(self, config):
        super().validate(config)
        sim = config["simulation"]
        if sim["trans"] is None:
            raise pgConfigError("trans", None, "a viscous case has a transport model")

    def arrays(self):
        ne, ns = self.ne, self.mixture.ns
        arrays = super().arrays()
        # gradients of u, v, w, T, Y; one halo plane exchanged
        arrays["grads"] = dict(
            kind=CellCenterArray, components=(ne - 1, 3), exchanged=1
        )
        # transport properties
        arrays["qt"] = dict(kind=CellCenterArray, components=2 + ns)
        return arrays

    def declKernels(self):
        sim = self.config["simulation"]
        k = super().declKernels()
        # the jit bakes the diffusion model into this
        k["trans"] = CellCenterKernel(f"transport/{sim['trans']}.cpp")
        k["dqdxyz"] = CellCenterKernel("utils/dq2FD.cpp")
        k["diffFlux"] = UnorderedKernelGroup(
            [CellFaceKernel("diffFlux/alphaDampingFlux.cpp", d) for d in range(3)]
        )
        return k

    def graphs(self, dt):
        """Gives consistify with the transport after the state, and the
        right-hand side as the gradient exchange: the gradient boundary
        conditions and the gradients ahead of it, the chemistry source and
        the fluxes while it flies, then the apply. What a message brings is
        left out while it flies and done once it lands: the halos, and the
        planes of the fluxes that read the gradients -- the diffusive flux,
        and the advective one only when its switch does."""
        k = self.kernels
        consistify = ExchangeGraphs(
            "consistify",
            "Q",
            during=[
                LaunchNode(k["stateFromCons"], "allLocal"),
                BCNode(k["bcs euler"], "onRank"),
                LaunchNode(k["trans"], "allLocal"),
            ],
            after=[
                BCNode(k["bcs euler"], "offRank"),
                RedoNode(k["stateFromCons"], k["trans"]),
            ],
        )
        stale = [g for g in (k["advFlux"], k["diffFlux"]) if g.reads("grads")]
        flight = lambda g: LaunchNode(g, "interiorLocal" if g in stale else "interior")
        rhs = ExchangeGraphs(
            "rhs",
            "grads",
            ahead=[BCNode(k["bcs preDqDxyz"]), LaunchNode(k["dqdxyz"], "interior")],
            during=[
                *self.sourceNodes(dt),
                flight(k["advFlux"]),
                BCNode(k["bcs postDqDxyz"], "onRank"),
                flight(k["diffFlux"]),
            ],
            after=[
                BCNode(k["bcs postDqDxyz"], "offRank"),
                RedoNode(*stale),
                LaunchNode(k["applyFlux"], "interior"),
            ],
        )
        return {"consistify": [consistify], "rhs": [rhs]}
