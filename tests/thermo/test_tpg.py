import peregrinepy as pg
import numpy as np
import cantera as ct
from pathlib import Path

# np.random.seed(111)

###############################################################################
# Test for all positive i aligned orientations
###############################################################################


def test_tpg():
    import kokkos

    kokkos.initialize()

    relpath = str(Path(__file__).parent)
    ctfile = relpath + "/ct_test_tpg.yaml"
    thfile = relpath + "/thtr_ct_test_tpg.yaml"
    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=100, high=1000)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    gas.TPY = T, p, Y

    config = pg.files.config_file()
    config["thermochem"]["spdata"] = thfile
    config["thermochem"]["eos"] = "tpg"
    config["RHS"]["diffusion"] = False

    mb = pg.multiblock.generate_multiblock_solver(1, config)
    pg.grid.create.multiblock_cube(
        mb, mb_dimensions=[1, 1, 1],
        dimensions_perblock=[2, 2, 2],
        lengths=[1, 1, 1]
    )
    mb.init_solver_arrays(config)

    blk = mb[0]

    mb.generate_halo()
    mb.compute_metrics()

    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 4] = T
    blk.array["q"][:, :, :, 5::] = Y[0:-1]

    # Update cons
    pg.compute.thermo.tpg(blk, mb.thtrdat, 0, "prims")

    # test the properties
    pgcons = blk.array["Q"][1, 1, 1]
    pgprim = blk.array["q"][1, 1, 1]
    pgthrm = blk.array["qh"][1, 1, 1]

    def print_diff(name, c, p):
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
    print("Conservatives")
    pd.append(print_diff("rho", gas.density, pgcons[0]))
    pd.append(print_diff("e", gas.int_energy_mass, pgcons[4] / pgcons[0]))
    pd.append(print_diff("e(qh)", gas.int_energy_mass, pgthrm[4] / pgcons[0]))
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff("rho" + n, gas.Y[i] * gas.density, pgcons[5 + i]))
    pd.append(
        print_diff(
            "rho" + gas.species_names[-1],
            gas.Y[-1] * gas.density,
            pgcons[0] - np.sum(pgcons[5::]),
        )
    )
    print("Mixture Properties")
    pd.append(print_diff("gamma", gas.cp / gas.cv, pgthrm[0]))
    pd.append(print_diff("cp", gas.cp, pgthrm[1]))
    pd.append(print_diff("h", gas.enthalpy_mass, pgthrm[2] / pgcons[0]))
    for i, n in enumerate(gas.species_names):
        pd.append(print_diff("h_" + n,
                             (gas.standard_enthalpies_RT[i]
                              * ct.gas_constant*gas.T
                              / gas.molecular_weights[i]), pgthrm[5 + i]))

    # Go the other way
    # Scramble the primatives
    blk.array["q"][:, :, :, 0] = 0.0
    blk.array["q"][:, :, :, 4] = 0.0
    blk.array["q"][:, :, :, 5::] = np.zeros(len(Y[0:-1]))
    pg.consistify(mb)

    print("********  Conservatives to Primatives ***************")
    print(f'       {"Cantera":<15}  | {"PEREGRINE":<15} | {"%Error":<5}')
    print("Conservatives")
    pd.append(print_diff("rho", gas.density, pgcons[0]))
    pd.append(print_diff("e", gas.int_energy_mass, pgcons[4] / pgcons[0]))
    pd.append(print_diff("e(qh)", gas.int_energy_mass, pgthrm[4] / pgcons[0]))
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff("rho" + n, gas.Y[i] * gas.density, pgcons[5 + i]))
    pd.append(
        print_diff(
            "rho" + gas.species_names[-1],
            gas.Y[-1] * gas.density,
            pgcons[0] - np.sum(pgcons[5::]),
        )
    )
    print("Primatives")
    pd.append(print_diff("p", gas.P, pgprim[0]))
    pd.append(print_diff("T", gas.T, pgprim[4]))
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff(n, gas.Y[i], pgprim[5 + i]))
    pd.append(print_diff(gas.species_names[-1], gas.Y[-1], 1.0 - np.sum(pgprim[5::])))
    print("Mixture Properties")
    pd.append(print_diff("gamma", gas.cp / gas.cv, pgthrm[0]))
    pd.append(print_diff("cp", gas.cp, pgthrm[1]))
    pd.append(print_diff("h", gas.enthalpy_mass, pgthrm[2] / pgcons[0]))
    for i, n in enumerate(gas.species_names[0:-1]):
        pd.append(print_diff("h_" + n,
                             (gas.standard_enthalpies_RT[i]
                              * ct.gas_constant*gas.T
                              / gas.molecular_weights[i]), pgthrm[5 + i]))

    kokkos.finalize()

    passfail = np.all(np.array(pd) < 0.0001)
    assert passfail


if __name__ == "__main__":
    test_tpg()
