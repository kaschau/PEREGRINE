import peregrinepy as pg
import numpy as np
import cantera as ct

import sys
from pathlib import Path

# np.random.seed(111)

##############################################
# Test for all positive i aligned orientations
##############################################


def test_kineticTheory():
    import kokkos

    kokkos.initialize()

    relpath = str(Path(__file__).parent)
    ct.add_directory(
        relpath + "/../../src/peregrinepy/thermo_transport/database/source"
    )
    if np.random.random() > 0.5:
        ctfile = "CH4_O2_Stanford_Skeletal.yaml"
        thfile = "thtr_CH4_O2_Stanford_Skeletal.yaml"
    else:
        ctfile = "GRI30.yaml"
        thfile = "thtr_GRI30.yaml"
    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=1000000)
    T = np.random.uniform(low=200, high=3500)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    config = pg.files.configFile()
    config["thermochem"]["spdata"] = thfile
    config["thermochem"]["eos"] = "tpg"
    config["RHS"]["diffusion"] = True

    mb = pg.multiblock.generateMultiblockSolver(1, config)
    pg.grid.create.multiblockCube(
        mb, mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    )
    mb.initSolverArrays(config)

    blk = mb[0]

    mb.generateHalo()
    mb.computeMetrics()

    gas.TPY = T, p, Y
    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 4] = T
    blk.array["q"][:, :, :, 5::] = Y[0:-1]

    # Update transport
    pg.compute.transport.kineticTheory(blk, mb.thtrdat, 0)

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
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff(f"D_{n}", gas.mix_diff_coeffs_mass[i], pgtrns[2 + i]))

    kokkos.finalize()

    passfail = np.all(np.array(pd) < 1.0)
    assert passfail