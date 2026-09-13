"""Every integrator steps a uniform flow through a periodic box and leaves
it as it was."""

import numpy as np
import pytest

import peregrinepy as pg

from ..gases import configure


@pytest.mark.parametrize(
    "integrator", ["rk1", "rk2", "rk3", "rk34", "rk4", "maccormack", "dualTime"]
)
@pytest.mark.parametrize("diffusion", [False, True])
def test_step(my_setup, integrator, diffusion):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEEPpe"
    config["RHS"]["diffusion"] = diffusion
    config["timeIntegration"]["integrator"] = integrator
    config["timeIntegration"]["dt"] = 1e-6
    config["initialConditions"]["u"] = 10.0
    configure(config, "air")

    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1],
            dimsPerBlock=[6, 6, 6],
            lengths=[1, 1, 1],
            periodic=[True, True, True],
        ),
    )

    blk = mb.blocks[0]
    Q0 = blk.Q.get()
    for _ in range(3):
        mb.step(config["timeIntegration"]["dt"])
    Q = blk.Q.get()
    ng = blk.ng
    assert np.isfinite(Q).all()
    # against each component's own size; the transverse momenta are zero
    scale = np.abs(Q0[ng, ng, ng]).max()
    assert (
        np.abs(Q[ng:-ng, ng:-ng, ng:-ng] - Q0[ng:-ng, ng:-ng, ng:-ng]).max()
        < 1e-10 * scale
    )
    assert mb.nrt == 3
