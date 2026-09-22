"""The viscosity mixing rules: Herning and Zipperer's against the same rule
in numpy over Cantera's species viscosities (within the fits' tolerance),
with its distance from Cantera's Wilke printed for the record."""

from pathlib import Path

import cantera as ct
import numpy as np
import peregrinepy as pg

from ..gases import primitives
import pytest

pytestmark = pytest.mark.parametrize("ctfile", ["CH4_O2_FFCMY.yaml", "GRI30.yaml"])


def test_herning(my_setup, ctfile):
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../../src/peregrinepy/mixture/database/mechanisms")
    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=1000000)
    T = np.random.uniform(low=300, high=3500)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)
    gas.TPY = T, p, Y

    config = pg.files.configFile()
    mc = config["mixture"]
    mc["species"] = ctfile
    mc["eos"] = "tpg"
    mc["Trange"] = (300.0, 3500.0)
    mc["trans"] = "kineticTheory"
    mc["diffusion"] = "lewis"
    mc["mixingRule"] = "herning"
    config["simulation"]["simulator"] = "navierStokes"
    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
        ),
    )
    assert mb.jit.mixingRule == "herning"
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[:, :, :, 0] = p
    q[:, :, :, 4] = T
    q[:, :, :, 5::] = Y[0:-1]
    mb.setPrimitives([q])
    ng = blk.ng
    mu = blk.qt.get()[ng, ng, ng, 0]

    X, sqrtMW = gas.X, np.sqrt(gas.molecular_weights)
    herning = np.sum(X * gas.species_viscosities * sqrtMW) / np.sum(X * sqrtMW)
    print(
        f"mu: herning {mu:.6e} | numpy herning {herning:.6e} "
        f"({abs(mu - herning) / herning * 100:.2e} %) | cantera wilke {gas.viscosity:.6e} "
        f"({abs(mu - gas.viscosity) / gas.viscosity * 100:.2f} %)"
    )
    assert abs(mu - herning) / herning < mc["reFitTol"] * 10
