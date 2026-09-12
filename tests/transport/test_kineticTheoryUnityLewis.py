from pathlib import Path

import cantera as ct
import numpy as np

import peregrinepy as pg
import pytest

##############################################
# Test kineticTheory + Unity Lewis
##############################################

pytestmark = pytest.mark.parametrize(
    "ctfile",
    ["CH4_O2_FFCMY.yaml", "GRI30.yaml", "C2H4_Air_Skeletal.yaml"],
)


def test_kineticTheoryUnityLewis(my_setup, ctfile):
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../../src/peregrinepy/mixture/database/mechanisms")

    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=1000000)
    T = np.random.uniform(low=200, high=3500)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    config = pg.files.configFile()
    config["mcPhysics"]["mixture"] = ctfile
    config["mcPhysics"]["eos"] = "tpg"
    config["mcPhysics"]["Trange"] = (300.0, 3500.0)
    config["mcPhysics"]["trans"] = "kineticTheory"
    config["mcPhysics"]["diffusion"] = "lewis"
    config["RHS"]["diffusion"] = True

    mb = pg.multiBlock.buildSolver(config, 1)
    pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    ).mesh(mb)

    blk = mb[0]

    mb.generateHalo()
    mb.computeMetrics()

    gas.TPY = T, p, Y
    q = blk.q.get()
    q[:, :, :, 0] = p
    q[:, :, :, 4] = T
    q[:, :, :, 5::] = Y[0:-1]
    blk.q.set(q)

    # Update transport
    assert mb.trans.__name__ == "kineticTheoryUnityLewis"
    mb.eos(blk, mb.thtrdat, 0, "prims")
    mb.trans(blk, mb.thtrdat, 0)
    q, qt = blk.q.get(), blk.qt.get()
    ng = blk.ng

    # test the properties
    pgprim = q[ng, ng, ng]
    pgtrns = qt[ng, ng, ng]

    def print_diff(name, c, p):
        diff = np.abs(c - p) / p * 100
        print(f"{name:<6s}: {c:16.8e} | {p:16.8e} | {diff:16.15e}")

        return diff

    pd = []
    print("******** Transport Properties *********")
    print(f'       {"Cantera":<16}  | {"PEREGRINE":<16} | {"%Error":<6}')
    print("Primatives")
    pd.append(print_diff("p", gas.P, pgprim[0]))
    pd.append(print_diff("T", gas.T, pgprim[4]))
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff(n, gas.Y[i], pgprim[5 + i]))
    pd.append(print_diff(gas.species_names[-1], gas.Y[-1], 1.0 - np.sum(pgprim[5::])))
    print("Mixture Properties")
    pd.append(print_diff("mu", gas.viscosity, pgtrns[0]))
    pd.append(print_diff("kappa", gas.thermal_conductivity, pgtrns[1]))
    for i, n in enumerate(gas.species_names):
        Dct = gas.thermal_conductivity / (gas.density * gas.cp_mass)
        pd.append(print_diff(f"D_{n}", Dct, pgtrns[2 + i]))

    passfail = np.all(np.array(pd) < 1.0)
    assert passfail
