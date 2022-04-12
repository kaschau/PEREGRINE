from pathlib import Path

import cantera as ct
import numpy as np
import peregrinepy as pg
import pytest

##############################################
# Test chemistry
##############################################

pytestmark = pytest.mark.parametrize(
    "ctfile,thfile,chmfile",
    [
        (
            "CH4_O2_Stanford_Skeletal.yaml",
            "thtr_CH4_O2_Stanford_Skeletal.yaml",
            "chem_CH4_O2_Stanford_Skeletal",
        ),
        (
            "GRI30.yaml",
            "thtr_GRI30.yaml",
            "chem_GRI30",
        ),
    ],
)


def test_chemistry(my_setup, thfile, ctfile, chmfile):

    config = pg.files.configFile()

    relpath = str(Path(__file__).parent)
    ct.add_directory(
        relpath + "/../../src/peregrinepy/thermo_transport/database/source"
    )

    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=1e6, high=30e6)
    T = np.random.uniform(low=300, high=3000)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    gas.TPY = T, p, Y

    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["spdata"] = thfile
    config["thermochem"]["chemistry"] = True
    config["thermochem"]["mechanism"] = chmfile
    config["RHS"]["diffusion"] = False

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb, mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    )
    mb.initSolverArrays(config)

    blk = mb[0]

    mb.generateHalo()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 4] = T
    blk.array["q"][:, :, :, 5::] = Y[0:-1]
    blk.updateDeviceView(["q"])

    # Update cons
    pg.compute.thermo.tpg(blk, mb.thtrdat, 0, "prims")
    # zero out dQ
    pg.compute.utils.dQzero(blk)
    mb.expChem(blk, mb.thtrdat)

    blk.updateHostView(["q", "dQ"])
    # test the properties
    pgprim = blk.array["q"][1, 1, 1]
    pgchem = blk.array["dQ"][1, 1, 1]

    def print_diff(name, c, p):
        if np.abs(c - p) == 0.0:
            diff = 0.0
        else:
            diff = np.abs(c - p) / p * 100
        print(f"{name:<6s}: {c:16.8e} | {p:16.8e} | {diff:16.15e}")

        return diff

    pd = []
    print("******** Primatives to Conservatives ***************")
    print(f'       {"Cantera":<16}  | {"PEREGRINE":<16} | {"%Error":<6}')
    print("Primatives")
    pd.append(print_diff("p", gas.P, pgprim[0]))
    pd.append(print_diff("T", gas.T, pgprim[4]))
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff(n, gas.Y[i], pgprim[5 + i]))
    pd.append(print_diff(gas.species_names[-1], gas.Y[-1], 1.0 - np.sum(pgprim[5::])))
    print("Chemical Source Terms")
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(
            print_diff(
                f"omega_{n:<4}",
                gas.net_production_rates[i] * gas.molecular_weights[i],
                pgchem[5 + i],
            )
        )

    passfail = np.all(np.array(pd) < 1e-10)
    assert passfail
