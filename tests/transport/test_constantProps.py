import kokkos
import peregrinepy as pg
import numpy as np
from pathlib import Path

# np.random.seed(111)

##############################################
# Test for all positive i aligned orientations
##############################################


class TestConstantPropsTrans:
    def setup_method(self):
        kokkos.initialize()

    def teardown_method(self):
        kokkos.finalize()

    def test_constantProps(self):

        relpath = str(Path(__file__).parent)
        config = pg.files.configFile()
        config["thermochem"]["spdata"] = relpath + "/AIR.yaml"
        config["thermochem"]["eos"] = "cpg"
        config["thermochem"]["trans"] = "constantProps"
        config["RHS"]["diffusion"] = True

        mb = pg.multiBlock.generateMultiBlockSolver(1, config)
        pg.grid.create.multiBlockCube(
            mb, mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
        )
        mb.initSolverArrays(config)

        blk = mb[0]

        mb.generateHalo()
        mb.computeMetrics(config["RHS"]["diffOrder"])

        p = np.random.uniform(low=10000, high=1000000)
        T = np.random.uniform(low=200, high=3500)
        blk.array["q"][:, :, :, 0] = p
        blk.array["q"][:, :, :, 4] = T

        # Update transport
        assert mb.trans.__name__ == "constantProps"
        mb.trans(blk, mb.thtrdat, 0)

        # test the properties
        pgprim = blk.array["q"][1, 1, 1]
        pgtrns = blk.array["qt"][1, 1, 1]

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
