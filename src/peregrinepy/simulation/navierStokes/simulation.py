"""Compressible flow of a mixture with diffusion: the Euler equations, the
transport properties, the gradients and their exchange, and the diffusive
flux, with the boundary conditions' gradient hooks."""

from .boundaries import BaseNSBC
from ...files.configFile import pgConfigError
from ...kernel import CellCenterKernel, CellFaceKernel, UnorderedKernelGroup
from ...multiBlock.arrays import CellCenterArray
from ...graph import BCNode, ExchangeGraphs, LaunchNode, RedoNode
from ..euler import EulerSimulation


class NavierStokesSimulation(EulerSimulation):
    """The Navier-Stokes equations of a mixture: Euler's, the gradients of
    velocity, temperature and mass fractions, the transport properties,
    and the diffusive flux."""

    physics = "navierStokes"
    viscous = True
    bcBase = BaseNSBC
    bcHooks = ("euler", "preDqDxyz", "postDqDxyz")
    # the cell length, which the ducros switch reads
    metrics = EulerSimulation.metrics + ("cellLength",)

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

    def graphs(self):
        """Gives consistify with the transport after the state, and the
        right-hand side as the gradient exchange: the gradient boundary
        conditions and the gradients ahead of it, the fluxes while it
        flies, then the remote faces done again and the apply."""
        k = self.kernels
        consistify = ExchangeGraphs(
            "consistify",
            "Q",
            during=[
                LaunchNode(k["stateFromCons"], "all"),
                BCNode(k["bcs euler"], "here"),
                LaunchNode(k["trans"], "all"),
            ],
            after=[
                BCNode(k["bcs euler"], "remote"),
                RedoNode(k["stateFromCons"], k["trans"]),
            ],
        )
        rhs = ExchangeGraphs(
            "rhs",
            "grads",
            ahead=[BCNode(k["bcs preDqDxyz"]), LaunchNode(k["dqdxyz"], "interior")],
            during=[
                LaunchNode(k["advFlux"], "interior"),
                BCNode(k["bcs postDqDxyz"], "here"),
                LaunchNode(k["diffFlux"], "interior"),
            ],
            after=[
                BCNode(k["bcs postDqDxyz"], "remote"),
                RedoNode(k["advFlux"], k["diffFlux"]),
                LaunchNode(k["applyFlux"], "interior"),
            ],
        )
        return {"consistify": [consistify], "rhs": [rhs]}
