import peregrinepy as pg
import numpy as np

##############################################
# Test constant properties transport
##############################################


def test_constantProps(my_setup):
    config = pg.files.configFile()
    config["mcPhysics"]["mixture"] = {
        "Air": {
            "MW": 28.96,
            "cp0": 1005.0,
            "mu0": 1.8591191080521142e-05,
            "kappa0": 0.02625394405190068,
        }
    }
    config["mcPhysics"]["eos"] = "cpg"
    config["mcPhysics"]["trans"] = "constantProps"
    config["RHS"]["diffusion"] = True

    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
        ),
    )

    blk = mb.blocks[0]

    p = np.random.uniform(low=10000, high=1000000)
    T = np.random.uniform(low=200, high=3500)
    q = blk.q.get()
    q[:, :, :, 0] = p
    q[:, :, :, 4] = T
    blk.q.set(q)

    # Update transport
    assert mb.trans.__name__ == "constantProps"
    mb.trans(nface=0)
    q, qt = blk.q.get(), blk.qt.get()
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
