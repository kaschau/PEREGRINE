"""A node is said in names and bound to the solver's means; a graph is
captured once and runs from then on; a boundary on no face launches
nothing."""

import numpy as np

import peregrinepy as pg
from peregrinepy.graph import BCNode, Graph, LaunchNode, RedoNode

from ..gases import configure


def solver():
    config = pg.files.configFile()
    configure(config, "air", "navierStokes")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["bcValues"]["walls"] = {"bcType": "adiabaticNoSlipWall"}
    mesh = pg.mesher.CubeMesher(
        mbDims=[2, 1, 1],
        dimsPerBlock=[5, 4, 3],
        lengths=[2, 1, 1],
        boundaryNames={1: "walls"},
    )
    return pg.multiBlock.solver(config, mesh)


def test_launchNodeBindsEveryBlockOverTheRange(my_setup):
    mb = solver()
    node = LaunchNode(mb.kernels["stateFromCons"], "interior").bind(*mb.means)
    ((kernel, table, tiling),) = node.stages[0]
    assert table is mb.blockArrayTable and tiling.count == len(mb.blocks)
    # a group's kernels are one stage: siblings
    group = LaunchNode(mb.kernels["advFlux"], "interior").bind(*mb.means)
    assert len(group.stages) == 1 and len(group.stages[0]) == 3


def test_bcNodeTilesOnlyTheFacesCarryingEachType(my_setup):
    mb = solver()
    node = BCNode(mb.kernels["bcs euler"], "all").bind(*mb.means)
    (stage,) = node.stages
    byType = {kernel.bcType: tiling.count for kernel, _, tiling in stage}
    # every outer face is a no-slip wall; the two interior faces are none
    assert byType["adiabaticNoSlipWall"] == 10
    assert all(count == 0 for t, count in byType.items() if t != "adiabaticNoSlipWall")
    assert node.name == "bcs euler all"


def test_redoNodeCoversRemoteFacesOnly(my_setup):
    mb = solver()
    node = RedoNode(mb.kernels["stateFromCons"], mb.kernels["advFlux"]).bind(
        *mb.means
    )
    # one rank: nothing is remote, so every tiling is empty
    assert [len(stage) for stage in node.stages] == [1, 3]
    assert all(tiling.count == 0 for stage in node.stages for _, _, tiling in stage)


def test_aGraphIsCapturedOnceAndRunsAgain(my_setup):
    mb = solver()
    g = Graph("test", [LaunchNode(mb.kernels["copy"], "full", A="dQ", B="Q")])
    g.bind(*mb.means)
    assert g.captured is None
    g.run()
    first = g.captured
    assert first is not None
    Q = mb.blocks[0].Q.get()
    assert np.array_equal(mb.blocks[0].dQ.get(), Q)
    mb.blocks[0].Q.set(Q * 2)
    g.run()
    assert g.captured is first
    assert np.array_equal(mb.blocks[0].dQ.get(), Q * 2)
    g.drop()
    assert g.captured is None


def test_launchOutsideAGraphTakesScalarsNow(my_setup):
    mb = solver()
    blk = mb.blocks[0]
    mb.launch("copy", "full", A="dQ", B="Q")
    assert np.array_equal(blk.dQ.get(), blk.Q.get())
    cfl = np.zeros(3)
    mb.kernels["CFLmax"] = pg.kernel.CellCenterKernel("utils/CFLmax.cpp")
    mb.jit.compile([mb.kernels["CFLmax"]])
    mb.launch("CFLmax", "interior", cfl=cfl)
    # at rest: acoustic and combined, nothing convective
    assert cfl[0] > 0 and cfl[1] == 0 and cfl[2] == cfl[0]
