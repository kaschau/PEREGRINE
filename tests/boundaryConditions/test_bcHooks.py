"""The bcHooks are fixed points of the physics: a hook the case's physics
does not have has no kernels and no node, and each physics lays its graphs
out in one order, what a message brings done again after it lands."""

import pytest

import peregrinepy as pg

from ..gases import configure


def _solver(physics):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    configure(config, "air", physics)
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1]
    )
    return pg.multiBlock.solver(config, mesh=mesh)


def _nodes(mb, stage):
    return [[n.name for n in g.nodes] for g in mb.graphs[stage]]


def test_bcHookOutsideThePhysicsHasNothing(my_setup):
    mb = _solver("euler")
    assert "bcs preDqDxyz" not in mb.kernels
    assert all("preDqDxyz" not in name for names in _nodes(mb, "rhs") for name in names)
    with pytest.raises(KeyError):
        mb.applyBcs("preDqDxyz")


def test_eulerLaysOutItsGraphs(my_setup):
    mb = _solver("euler")
    assert _nodes(mb, "consistify") == [
        ["pack Q", "directHaloFill Q"],
        ["stateFromCons", "bcs euler onRank"],
        ["unpack Q", "bcs euler offRank", "redo stateFromCons"],
    ]
    assert _nodes(mb, "rhs") == [["KEPaEC", "applyFlux"]]


def test_navierStokesLaysOutItsGraphs(my_setup):
    mb = _solver("navierStokes")
    assert _nodes(mb, "consistify") == [
        ["pack Q", "directHaloFill Q"],
        ["stateFromCons", "bcs euler onRank", "constantProps"],
        ["unpack Q", "bcs euler offRank", "redo stateFromCons constantProps"],
    ]
    # the gradients go out; everything runs while they fly; what a message
    # brings is done again after it lands
    assert _nodes(mb, "rhs") == [
        ["bcs preDqDxyz all", "dq2FD", "pack grads", "directHaloFill grads"],
        ["KEPaEC", "bcs postDqDxyz onRank", "alphaDampingFlux"],
        [
            "unpack grads",
            "bcs postDqDxyz offRank",
            "redo alphaDampingFlux",
            "applyFlux",
        ],
    ]
