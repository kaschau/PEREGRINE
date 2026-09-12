import numpy as np
import peregrinepy as pg


def print_diff(name, c, p, scale=None):
    """Percent difference, against c or against a scale for a quantity that
    can pass through zero."""
    diff = np.abs(c - p) / (abs(c) if scale is None else scale) * 100
    print(f"{name:<6s}: {c:16.8e} | {p:16.8e} | {diff:16.15e}")

    return diff


def test_cubic(my_setup):
    config = pg.files.configFile()
    config["mcPhysics"]["mixture"] = ["O2", "N2", "CO2", "CH4"]
    config["mcPhysics"]["eos"] = "realGas"
    config["mcPhysics"]["Trange"] = (300.0, 3500.0)
    config["RHS"]["diffusion"] = False

    mb = pg.multiBlock.solver(config, 1)
    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=300, high=1000)
    Y = np.random.uniform(low=0.0, high=1.0, size=mb[0].ns)
    Y = Y / np.sum(Y)
    pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    ).mesh(mb)

    blk = mb[0]
    ng = blk.ng

    mb.generateHalo()
    mb.computeMetrics()

    q = blk.q.get()
    q[:, :, :, 0] = p
    q[:, :, :, 1:4] = 0.0
    q[:, :, :, 4] = T
    q[:, :, :, 5::] = Y[0:-1]

    # Update cons
    assert mb.stateFromPrims.__name__ == "realGasFromPrims"
    blk.q.set(q)
    mb.stateFromPrims(nface=0)
    # Go the other way
    mb.stateFromCons(nface=0)
    q = blk.q.get()

    # test the properties
    pgprim = q[ng, ng, ng]

    print("******** Prim -> Cons -> Prim *********")
    print(f'       {"Input":<15}  | {"Output":<15} | {"%Error":<5}')
    pd = []
    pd.append(print_diff("p", p, pgprim[0]))
    pd.append(print_diff("T", T, pgprim[4]))
    for i, n in enumerate(mb[0].speciesNames[0:-1]):
        pd.append(print_diff(n, Y[i], pgprim[5 + i]))

    # every property is a refit to the case's tolerance, in percent here
    assert np.all(np.array(pd) < config["mcPhysics"]["reFitTol"] * 100)
