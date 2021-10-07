import peregrinepy as pg
import numpy as np
import cantera as ct

import sys
from pathlib import Path

# np.random.seed(111)

##############################################
# Test for all positive i aligned orientations
##############################################


def test_chemistry():
    import kokkos

    kokkos.initialize()
    config = pg.files.configFile()

    relpath = str(Path(__file__).parent)
    ct.add_directory(
        relpath + "/../../src/peregrinepy/thermo_transport/database/source"
    )
    if np.random.random() > 0.5:
        ctfile = "CH4_O2_Stanford_Skeletal.yaml"
        thfile = "thtr_CH4_O2_Stanford_Skeletal.yaml"
        config["thermochem"]["mechanism"] = "chem_CH4_O2_Stanford_Skeletal"
    else:
        ctfile = "GRI30.yaml"
        thfile = "thtr_GRI30.yaml"
        config["thermochem"]["mechanism"] = "chem_GRI30"

    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=1000, high=2000)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    gas.TPY = T, p, Y

    config["thermochem"]["spdata"] = thfile
    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["chemistry"] = True
    config["RHS"]["diffusion"] = False

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb, mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    )
    mb.initSolverArrays(config)

    blk = mb[0]

    mb.generateHalo()
    mb.computeMetrics()

    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 4] = T
    blk.array["q"][:, :, :, 5::] = Y[0:-1]

    # Update cons
    pg.compute.thermo.tpg(blk, mb.thtrdat, 0, "prims")
    # zero out dQ
    pg.compute.utils.dQzero(blk)
    mb.expChem(blk, mb.thtrdat)

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

    kokkos.finalize()

    passfail = np.all(np.array(pd) < 1e-10)
    assert passfail


if __name__ == "__main__":
    test_chemistry()
