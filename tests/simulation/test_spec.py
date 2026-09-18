"""A simulation is a spec: what each physics declares, which boundaries it
has, and the graphs it lays out, with no solver in hand."""

import pytest

import peregrinepy as pg
from peregrinepy.files.configFile import pgConfigError
from peregrinepy.graph import BCNode, ExchangeGraphs, Graph, LaunchNode, RedoNode
from peregrinepy.simulation import BaseEulerBC, BaseNSBC

from ..gases import configure


def simulation(physics, gas="air", **rhs):
    config = pg.files.configFile()
    configure(config, gas, physics)
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["RHS"].update(rhs)
    return pg.simulation.getSimulation(config)


def test_eulerDeclaresTheInviscidCase():
    sim = simulation("euler")
    assert not sim.viscous
    assert sim.primVars == ["p", "u", "v", "w", "T"]
    assert sim.exportVars == ["rho", "p", "u", "v", "w", "T", "Air"]
    assert set(sim.arrays()) == {"iF", "jF", "kF", "Q", "dQ", "q", "qh"}
    assert sim.arrays()["Q"]["exchanged"] is True
    assert set(sim.metrics) == {
        "Jinv",
        "dIJK",
        "dENCdxyz",
        "iFaces",
        "jFaces",
        "kFaces",
        "iS",
        "jS",
        "kS",
    }
    assert sim.bcBase is BaseEulerBC and sim.bcHooks == ("euler",)


def test_navierStokesAddsDiffusion():
    sim = simulation("navierStokes")
    euler = simulation("euler")
    assert sim.viscous
    assert set(sim.arrays()) == set(euler.arrays()) | {"grads", "qt"}
    # only the plane past the block face of the gradients is ever wanted
    assert sim.arrays()["grads"]["exchanged"] == 1
    assert sim.bcBase is BaseNSBC and sim.bcHooks == (
        "euler",
        "preDqDxyz",
        "postDqDxyz",
    )


def test_speciesWidenThePrimitiveVector():
    sim = simulation("navierStokes", "CH4_O2")
    ns = sim.mixture.ns
    assert ns > 1
    assert sim.primVars == ["p", "u", "v", "w", "T"] + sim.mixture.speciesNames[:-1]
    assert sim.ne == 5 + ns - 1
    assert sim.arrays()["Q"]["components"] == sim.ne


def test_eulerCannotStick():
    assert "adiabaticNoSlipWall" in BaseNSBC.bcTypes()
    assert "adiabaticNoSlipWall" not in BaseEulerBC.bcTypes()
    assert set(BaseEulerBC.bcTypes()) < set(BaseNSBC.bcTypes())
    with pytest.raises(KeyError):
        BaseEulerBC.named("isoTNoSlipWall")


def test_kernelsAreByTagWithTheBoundariesByHook():
    sim = simulation("navierStokes")
    k = sim.declKernels()
    assert k is sim.kernels
    for hook in sim.bcHooks:
        group = k[f"bcs {hook}"]
        assert [x.bcType for x in group.kernels] == list(sim.bcBase.withHook(hook))
    assert "trans" in k and "diffFlux" in k
    assert "trans" not in simulation("euler").declKernels()


def test_graphsAreSaidInNamesWithoutASolver():
    sim = simulation("navierStokes")
    sim.declKernels()
    graphs = sim.graphs()
    assert set(graphs) == {"consistify", "rhs"}
    (consistify,), (rhs,) = graphs["consistify"], graphs["rhs"]
    assert isinstance(consistify, ExchangeGraphs) and consistify.array == "Q"
    assert isinstance(rhs, ExchangeGraphs) and rhs.array == "grads"
    # the redo after a message covers what ran while it flew
    kinds = [type(n) for n in rhs.after]
    assert kinds == [BCNode, RedoNode, LaunchNode]
    euler = simulation("euler")
    euler.declKernels()
    (rhs,) = euler.graphs()["rhs"]
    assert isinstance(rhs, Graph) and [type(n) for n in rhs.nodes] == [
        LaunchNode,
        LaunchNode,
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("secondaryAdvFlux", "rusanov"),
        ("switchAdvFlux", "jamesonPressure"),
        ("primaryAdvFlux", None),
    ],
)
def test_whatIsNotDescribedYetIsRefused(key, value):
    with pytest.raises(pgConfigError):
        simulation("euler", **{key: value})


def test_aViscousCaseNeedsATransportModel():
    config = pg.files.configFile()
    configure(config, "air", "navierStokes")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["simulation"]["trans"] = None
    with pytest.raises(pgConfigError):
        pg.simulation.getSimulation(config)
    config["simulation"]["physics"] = "euler"
    pg.simulation.getSimulation(config)


def test_theInitialStateNamesSpeciesByName():
    config = pg.files.configFile()
    configure(config, "CH4_O2", "euler")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["initialConditions"]["Y"] = {"CH4": 0.2, "O2": 0.8}
    sim = pg.simulation.getSimulation(config)
    state = sim.initialState()
    names = sim.mixture.speciesNames
    assert len(state) == sim.ne
    assert state[5 + names.index("CH4")] == 0.2 and state[5 + names.index("O2")] == 0.8
    config["initialConditions"]["Y"] = {"Xe": 1.0}
    with pytest.raises(pgConfigError):
        sim.initialState()
