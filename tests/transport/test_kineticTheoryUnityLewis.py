from pathlib import Path

import cantera as ct
import numpy as np

import peregrinepy as pg
import pytest

##############################################
# Test kineticTheory + Unity Lewis
##############################################

pytestmark = pytest.mark.parametrize(
    "ctfile,thfile",
    [
        (
            "CH4_O2_Stanford_Skeletal.yaml",
            "thtr_CH4_O2_Stanford_Skeletal.yaml",
        ),
        (
            "GRI30.yaml",
            "thtr_GRI30.yaml",
        ),
    ],
)


def test_kineticTheoryUnityLewis(my_setup, ctfile, thfile):

    relpath = str(Path(__file__).parent)
    ct.add_directory(
        relpath + "/../../src/peregrinepy/thermo_transport/database/source"
    )

    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=1000000)
    T = np.random.uniform(low=200, high=3500)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    config = pg.files.configFile()
    config["thermochem"]["spdata"] = thfile
    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["trans"] = "kineticTheoryUnityLewis"
    config["RHS"]["diffusion"] = True

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb, mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    )
    mb.initSolverArrays(config)

    blk = mb[0]

    mb.generateHalo()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    gas.TPY = T, p, Y
    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 4] = T
    blk.array["q"][:, :, :, 5::] = Y[0:-1]
    blk.updateDeviceView("q")

    # Update transport
    assert mb.trans.__name__ == "kineticTheoryUnityLewis"
    mb.eos(blk, mb.thtrdat, 0, "prims")
    mb.trans(blk, mb.thtrdat, 0)
    blk.updateHostView(["q", "qt"])

    # test the properties
    pgprim = blk.array["q"][1, 1, 1]
    pgtrns = blk.array["qt"][1, 1, 1]

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
