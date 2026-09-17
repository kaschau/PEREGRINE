"""The bcHooks are fixed points of the flows: a bcHook the case's flow does
not have runs nothing, a flow lists its slots in order, and a slot the
solver did not fill is a bug, not a null."""

import pytest

import peregrinepy as pg
from peregrinepy.graph import Graph

from ..gases import configure


def _solver(diffusion):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["RHS"]["diffusion"] = diffusion
    configure(config, "air")
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1]
    )
    return pg.integrators.getSolver(config, mesh=mesh)


def test_bcHookOutsideTheFlowRunsNothing(my_setup):
    mb = _solver(diffusion=False)
    assert mb.graphs["rhs"].bcs("preDqDxyz") is None
    assert not [tag for tag in mb.kernels if tag.endswith("@preDqDxyz")]
    mb.applyBcs("preDqDxyz")


def test_flowListsItsSlots(my_setup):
    mb = _solver(diffusion=True)
    # everything runs while the halos are in flight; what a message brings
    # is done again after it lands
    assert [n.name for n in mb.graphs["rhs"].nodes] == [
        "bcs preDqDxyz",
        "dqdxyz",
        "haloExchange grads start",
        "primaryAdvFlux",
        "haloExchange grads send",
        "bcs postDqDxyz",
        "diffFlux",
        "haloExchange grads finish",
        "primaryAdvFlux diffFlux remote",
        "applyFlux",
    ]
    assert [n.name for n in mb.graphs["consistify"].nodes] == [
        "haloExchange Q start",
        "stateFromCons",
        "haloExchange Q send",
        "bcs euler",
        "trans",
        "haloExchange Q finish",
        "stateFromCons remote",
        "trans remote",
    ]
    assert [n.name for n in mb.graphs["consistifyFromPrims"].nodes][:2] == [
        "stateFromPrims",
        "haloExchange Q start",
    ]
    del mb.kernels["applyFlux"]
    with pytest.raises(KeyError):
        Graph.rhs(mb)
