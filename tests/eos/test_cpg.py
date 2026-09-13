import peregrinepy as pg
import numpy as np
import cantera as ct
from pathlib import Path

#############################################
# Test cpg eos
#############################################


def test_cpg(my_setup):
    relpath = str(Path(__file__).parent)
    ctfile = relpath + "/ct_test_cpg.yaml"
    gas = ct.Solution(ctfile)
    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=100, high=1000)
    Y = np.random.uniform(low=0.0, high=1.0, size=gas.n_species)
    Y = Y / np.sum(Y)

    gas.TPY = T, p, Y

    config = pg.files.configFile()
    config["mcPhysics"]["mixture"] = ctfile
    config["mcPhysics"]["eos"] = "cpg"
    config["RHS"]["diffusion"] = False

    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
        ),
    )

    blk = mb.blocks[0]
    ng = blk.ng

    q = blk.q.get()
    q[:, :, :, 0] = p
    q[:, :, :, 1:4] = 0.0
    q[:, :, :, 4] = T
    q[:, :, :, 5::] = Y[0:-1]

    # Update cons
    assert mb.stateFromPrims.__name__ == "cpgFromPrims"
    blk.q.set(q)
    mb.stateFromPrims(nface=0)
    q, Q, qh = blk.q.get(), blk.Q.get(), blk.qh.get()

    # test the properties
    pgcons = Q[ng, ng, ng]
    pgprim = q[ng, ng, ng]
    pgthrm = qh[ng, ng, ng]

    def print_diff(name, c, p):
        diff = np.abs(c - p) / c * 100
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
        pd.append(
            print_diff(
                "h_" + n,
                (
                    gas.standard_enthalpies_RT[i]
                    * ct.gas_constant
                    * gas.T
                    / gas.molecular_weights[i]
                ),
                pgthrm[5 + i],
            )
        )

    # Go the other way
    # Scramble the primatives
    q[:, :, :, 0] = 0.0
    q[:, :, :, 4] = 0.0
    q[:, :, :, 5::] = np.zeros(len(Y[0:-1]))
    blk.q.set(q)
    mb.stateFromCons(nface=0)
    q, Q, qh = blk.q.get(), blk.Q.get(), blk.qh.get()
    pgcons, pgprim, pgthrm = Q[ng, ng, ng], q[ng, ng, ng], qh[ng, ng, ng]

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
    for i, n in enumerate(gas.species_names):
        pd.append(
            print_diff(
                "h_" + n,
                (
                    gas.standard_enthalpies_RT[i]
                    * ct.gas_constant
                    * gas.T
                    / gas.molecular_weights[i]
                ),
                pgthrm[5 + i],
            )
        )

    passfail = np.all(np.array(pd) < 0.0001)
    assert passfail
