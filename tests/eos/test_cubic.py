import numpy as np
import peregrinepy as pg


def print_diff(name, c, p):
    diff = np.abs(c - p) / c * 100
    print(f"{name:<6s}: {c:16.8e} | {p:16.8e} | {diff:16.15e}")

    return diff


def test_cubic(my_setup):
    config = pg.files.configFile()
    config["thermochem"]["spdata"] = ["O2", "N2", "CO2", "CH4"]
    config["thermochem"]["eos"] = "cubic"
    config["RHS"]["diffusion"] = False

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=100, high=1000)
    Y = np.random.uniform(low=0.0, high=1.0, size=mb[0].ns)
    Y = Y / np.sum(Y)
    pg.grid.create.multiBlockCube(
        mb, mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    ng = blk.ng

    mb.generateHalo()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    blk.array["q"][:, :, :, 0] = p
    blk.array["q"][:, :, :, 1:4] = 0.0
    blk.array["q"][:, :, :, 4] = T
    blk.array["q"][:, :, :, 5::] = Y[0:-1]

    # Update cons
    assert mb.eos.__name__ == "cubic"
    blk.updateDeviceView(["q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    # Go the other way
    mb.eos(blk, mb.thtrdat, 0, "cons")
    blk.updateHostView(["q", "Q", "qh"])

    # test the properties
    pgprim = blk.array["q"][ng, ng, ng]

    print("******** Prim -> Cons -> Prim *********")
    print(f'       {"Input":<15}  | {"Output":<15} | {"%Error":<5}')
    pd = []
    pd.append(print_diff("p", p, pgprim[0]))
    pd.append(print_diff("T", T, pgprim[4]))
    for i, n in enumerate(mb[0].speciesNames[0:-1]):
        pd.append(print_diff(n, Y[i], pgprim[5 + i]))

    passfail = np.all(np.array(pd) < 0.0001)
    assert passfail
