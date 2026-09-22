import peregrinepy as pg

from ..gases import primitives
import numpy as np

##############################################
# Test constant properties transport
##############################################


def test_constantProps(my_setup):
    config = pg.files.configFile()
    config["mixture"]["species"] = {
        "Air": {
            "MW": 28.96,
            "cp0": 1005.0,
            "mu0": 1.8591191080521142e-05,
            "kappa0": 0.02625394405190068,
        }
    }
    config["mixture"]["eos"] = "cpg"
    config["mixture"]["trans"] = "constantProps"
    config["simulation"]["simulator"] = "navierStokes"

    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
        ),
    )

    blk = mb.blocks[0]

    p = np.random.uniform(low=10000, high=1000000)
    T = np.random.uniform(low=200, high=3500)
    q = primitives(mb, blk)
    q[:, :, :, 0] = p
    q[:, :, :, 4] = T
    mb.setPrimitives([q])

    # Update transport
    assert mb.kernels["trans"].__name__ == "constantProps"
    q, qt = primitives(mb, blk), blk.qt.get()
    ng = blk.ng

    # test the properties
    pgprim = q[ng, ng, ng]
    pgtrns = qt[ng, ng, ng]

    def print_diff(name, c, p):
        diff = np.abs(c - p) / c * 100
        print(f"{name:<9s}: {c:16.8e} | {p:16.8e} | {diff:16.15e}")

        return diff

    pd = []
    print("******** Transport Properties *********")
    print(f'{"":<13s}{"Reference":<13}  | {"PEREGRINE":<16} | {"%Error":<6}')
    print("Primatives")
    pd.append(print_diff("p", p, pgprim[0]))
    pd.append(print_diff("T", T, pgprim[4]))
    print("Transport Properties")
    pd.append(print_diff("mu", 1.8591191080521142e-05, pgtrns[0]))
    pd.append(print_diff("kappa", 0.02625394405190068, pgtrns[1]))

    passfail = np.all(np.array(pd) < 1e-9)
    assert passfail
