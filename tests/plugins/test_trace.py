"""The trace plugin: points, lines and planes named in the config land on
the cells nearest them when the case starts, and the primitives there are
appended to one csv per trace each time it acts."""

import numpy as np
import pytest

import peregrinepy as pg

from ..gases import configure


def box(tmp_path, **simulation):
    config = pg.files.configFile()
    config["simulation"].update(simulation)
    configure(config, "air", "navierStokes")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["simulation"]["dt"] = 1e-6
    config["simulation"]["niter"] = 2
    config["initialConditions"]["u"] = 10.0
    config["plugins"]["trace"] = dict(
        niterOut=1,
        traces=dict(
            probe=dict(
                type="point",
                p0=[0.31, 0.52, 0.5],
                file=str(tmp_path / "p.csv"),
                niterOut=2,
            ),
            line=dict(
                type="line",
                p0=[0.0, 0.5, 0.5],
                p1=[1.0, 0.5, 0.5],
                n=6,
                file=str(tmp_path / "line.csv"),
            ),
        ),
    )
    mesh = pg.mesher.CubeMesher(
        mbDims=[2, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1], periodic=[True] * 3
    )
    return config, pg.multiBlock.solver(config, mesh)


def test_pointsLandOnTheirNearestCells(my_setup, tmp_path):
    config, mb = box(tmp_path)
    plugin = mb.plugins["trace"]
    fileName, points, mine = plugin.traces["probe"]
    ((n, nblki, i, j, k),) = mine
    center = mb.getBlock(nblki).cells.get()[i, j, k]
    nearest = min(
        np.linalg.norm(
            b.cells.get()[b.interior].reshape(-1, 3) - points[0], axis=1
        ).min()
        for b in mb.blocks
    )
    assert np.linalg.norm(center - points[0]) == pytest.approx(nearest)
    # the line's six points span both blocks, every one placed once
    _, points, mine = plugin.traces["line"]
    assert sorted(m[0] for m in mine) == list(range(6))
    assert {m[1] for m in mine} == {0, 1}


def test_aTraceAppendsARowPerPointPerAct(my_setup, tmp_path):
    config, mb = box(tmp_path)
    mb.integrator.run()
    lines = (tmp_path / "line.csv").read_text().splitlines()
    assert lines[0] == "time, x, y, z, " + ", ".join(mb.primVars)
    rows = np.loadtxt(tmp_path / "line.csv", delimiter=",", skiprows=1)
    assert rows.shape == (2 * 6, 4 + mb.ne)
    # the probe names its own cadence: once, after the second step
    probe = np.loadtxt(tmp_path / "p.csv", delimiter=",", skiprows=1, ndmin=2)
    assert probe.shape == (1, 4 + mb.ne) and probe[0, 0] == pytest.approx(mb.tme)
    # the last act's rows are the state now, at the cells the points landed on
    last = rows[6:]
    assert np.allclose(last[:, 0], mb.tme)
    _, points, mine = mb.plugins["trace"].traces["line"]
    for n, nblki, i, j, k in mine:
        blk = mb.getBlock(nblki)
        values = mb.exportData(blk, mb.primVars)
        assert np.allclose(last[n, 1:4], blk.cells.get()[i, j, k])
        assert np.allclose(last[n, 4:], [values[v][i, j, k] for v in mb.primVars])


def test_theCflControllerLandsOnDtOut(my_setup, tmp_path):
    # steps sized by the CFL, none a multiple of the probe's dtOut, are
    # shortened to land on it: every row is at a multiple, none twice
    config, mb = box(tmp_path, controller="cfl", maxCFL=0.5, maxDt=1e-3, niter=8)
    dtOut = 1e-4
    config["plugins"]["trace"]["traces"]["probe"].update(dtOut=dtOut, niterOut=None)
    mb = pg.multiBlock.solver(config, mb.mesh)
    mb.integrator.run()
    probe = np.loadtxt(tmp_path / "p.csv", delimiter=",", skiprows=1, ndmin=2)
    times = probe[:, 0]
    assert len(times) == int(mb.tme / dtOut + 1e-9) >= 2
    assert np.allclose(times, np.arange(1, len(times) + 1) * dtOut, rtol=1e-12)
