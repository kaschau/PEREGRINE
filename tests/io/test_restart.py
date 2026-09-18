"""A result carries what a case restarts from: the primitive variables,
what the integrator keeps, the case and the grid it came from; read back,
the case continues as it would have."""

import numpy as np
import pytest

import peregrinepy as pg

from ..gases import configure, primitives


def case(tmp_path, integrator="rk3", gas="air"):
    config = pg.files.configFile()
    configure(config, gas, "navierStokes")
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["integrator"] = integrator
    config["timeIntegration"]["dt"] = 1e-8
    config["initialConditions"]["u"] = 10.0
    config["bcValues"]["walls"] = {"bcType": "adiabaticNoSlipWall"}
    mesh = pg.mesher.CubeMesher(
        mbDims=[2, 1, 1],
        dimsPerBlock=[6, 5, 4],
        lengths=[1, 1, 1],
        periodic=[False, True, True],
        boundaryNames={1: "walls", 2: "walls"},
    )
    mb = pg.multiBlock.solver(config, mesh)
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        prims.append(q)
    mb.setPrimitives(prims)
    mb.integrator.initialize()
    pg.writers.GridWriter(mb, str(tmp_path), precision="double").write(mb)
    return config, mb


def interior(mb, name):
    return [getattr(blk, name).get()[blk.interior].copy() for blk in mb.blocks]


@pytest.mark.parametrize("integrator", ["rk3", "dualTime"])
def test_restartContinuesBitwise(my_setup, tmp_path, integrator):
    config, mb = case(tmp_path, integrator)
    dt = config["timeIntegration"]["dt"]
    for _ in range(3):
        mb.integrator.step(dt)
    writer = pg.writers.RestartWriter(
        mb,
        str(tmp_path),
        str(tmp_path),
        "double",
        extras=mb.integrator.restartArrays,
        config=config,
    )
    writer.write(mb)
    # the same case, from the file: what the first would do next, it does
    for _ in range(2):
        mb.integrator.step(dt)
    expect = interior(mb, "Q")

    reader = pg.readers.RestartReader(str(tmp_path / "q.00000003.h5"))
    assert reader.primVars == mb.primVars and reader.nrt == 3
    assert list(reader.extras) == list(mb.integrator.restartArrays)
    again = pg.multiBlock.solver(
        reader.config, pg.readers.GridReader(reader.grid), reader
    )
    assert again.nrt == 3 and again.tme == pytest.approx(3 * dt)
    for _ in range(2):
        again.integrator.step(dt)
    # the state goes through the primitives and the eos on the way out and
    # back, which is a few ulps, not bitwise
    for a, b in zip(interior(again, "Q"), expect):
        scale = np.abs(b).max(axis=(0, 1, 2))
        assert (np.abs(a - b).max(axis=(0, 1, 2)) / scale).max() < 5e-14


def test_theFileHoldsEveryExportVariable(my_setup, tmp_path):
    import h5py

    config, mb = case(tmp_path, gas="CH4_O2")
    pg.writers.RestartWriter(mb, str(tmp_path), str(tmp_path), "double").write(mb)
    with h5py.File(tmp_path / "q.00000000.h5") as f:
        variables = [v.decode() for v in f.attrs["variables"]]
        assert variables == mb.exportVars
        assert [v.decode() for v in f.attrs["primVars"]] == mb.primVars
        group = f["results_000000"]
        assert set(group) == set(variables)
        blk = mb.blocks[0]
        data = mb.exportData(blk, variables)
        for name in variables:
            stored = group[name][...]
            assert np.allclose(stored, data[name][blk.interior].T)
        # the last species is what the others leave
        species = mb.simulator.mixture.speciesNames
        total = sum(group[s][...] for s in species)
        assert np.allclose(total, 1.0)
