import numpy as np
import peregrinepy as pg

from ..gases import primitives


def print_diff(name, c, p, scale=None):
    """Percent difference, against c or against a scale for a quantity that
    can pass through zero."""
    diff = np.abs(c - p) / (abs(c) if scale is None else scale) * 100
    print(f"{name:<6s}: {c:16.8e} | {p:16.8e} | {diff:16.15e}")

    return diff


def test_cubic(my_setup):
    config = pg.files.configFile()
    config["simulation"]["mixture"] = ["O2", "N2", "CO2", "CH4"]
    config["simulation"]["eos"] = "realGas"
    config["simulation"]["Trange"] = (300.0, 3500.0)
    config["simulation"]["physics"] = "euler"

    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
        ),
    )
    blk = mb.blocks[0]
    ng = blk.ng

    p = np.random.uniform(low=10000, high=100000)
    T = np.random.uniform(low=300, high=1000)
    Y = np.random.uniform(low=0.0, high=1.0, size=mb.simulation.mixture.ns)
    Y = Y / np.sum(Y)

    q = primitives(mb, blk)
    q[:, :, :, 0] = p
    q[:, :, :, 1:4] = 0.0
    q[:, :, :, 4] = T
    q[:, :, :, 5::] = Y[0:-1]

    # Update cons
    assert mb.jit.eos == "realGas"
    mb.setPrimitives([q])
    # Go the other way
    mb.consistify()
    q = primitives(mb, blk)

    # test the properties
    pgprim = q[ng, ng, ng]

    print("******** Prim -> Cons -> Prim *********")
    print(f'       {"Input":<15}  | {"Output":<15} | {"%Error":<5}')
    pd = []
    pd.append(print_diff("p", p, pgprim[0]))
    pd.append(print_diff("T", T, pgprim[4]))
    for i, n in enumerate(mb.simulation.mixture.speciesNames[0:-1]):
        pd.append(print_diff(n, Y[i], pgprim[5 + i]))

    # every property rests on the refit, and the check asks exactly what the
    # fit achieved for these species (within the tolerance, or the best its
    # cap could do), in percent, plus the rounding the kernel adds
    achieved = max(sp["cpFitError"] for sp in mb.simulation.mixture.species.values())
    assert np.all(np.array(pd) < achieved * 100 + 1e-6)
