"""A case runs in the precision its config names: every array and kernel
value double, or single. Single is checked against double on the gate's
box, and an array a kernel fills has to be the caller's, in the case's
precision."""

import ctypes

import numpy as np
import pytest

import peregrinepy as pg

from ..gate import cases
from ..gases import configure


def box(precision, gas="air", integrator="rk3"):
    config = pg.files.configFile()
    configure(config, gas, "navierStokes")
    config["simulation"]["precision"] = precision
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["integrator"] = integrator
    config["timeIntegration"]["dt"] = 1e-9
    config["bcValues"]["walls"] = {"bcType": "adiabaticNoSlipWall"}
    mb = pg.multiBlock.solver(config, cases.mesh())
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = cases.primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        if mb.ne > 5:
            base = rng.random(mb.ne - 5) / (mb.ne - 5)
            q[..., 5:] = base * (1 + 0.01 * rng.random(q.shape[:3] + (mb.ne - 5,)))
        prims.append(q)
    mb.setPrimitives(prims)
    mb.integrator.initialize()
    cases.step(mb, 12)
    return mb


def interior(mb):
    ng = mb.ng
    return [blk.Q.get()[ng:-ng, ng:-ng, ng:-ng].astype(np.float64) for blk in mb.blocks]


@pytest.mark.parametrize(
    "gas,integrator,within",
    # measured: 2.2e-6, 3.4e-6, and 2.4e-5 with the eos Newton in single
    [("air", "rk3", 1e-5), ("air", "dualTime", 1e-5), ("CH4_O2", "rk3", 1e-4)],
)
def test_singleFollowsDouble(my_setup, gas, integrator, within):
    double = box("double", gas, integrator)
    single = box("single", gas, integrator)
    assert double.blocks[0].Q.dtype == np.float64
    assert single.blocks[0].Q.dtype == np.float32
    assert single.backend.fpdtype == np.float32
    assert single.kernels["applyFlux"].fpctype is ctypes.c_float
    assert "PG_FPDTYPE=float" in single.jit.defines
    for d, s in zip(interior(double), interior(single)):
        assert np.isfinite(s).all()
        assert np.abs(d - s).max() < within * np.abs(d).max()


def test_anArrayAKernelFillsIsTheCallers(my_setup):
    # the cfl controller's reduction: its result lands in the array given
    mb = box("single")
    kernel = pg.kernel.CellCenterKernel("utils/CFLmax.cpp")
    mb.jit.compile([kernel])
    tiling = mb.blockArrayTable.tiling(kernel, "interior")
    with pytest.raises(TypeError):
        kernel(mb.blockArrayTable, tiling, cfl=np.zeros(3))
    cfl = np.zeros(3, np.float32)
    kernel(mb.blockArrayTable, tiling, cfl=cfl)
    assert cfl[0] > 0
