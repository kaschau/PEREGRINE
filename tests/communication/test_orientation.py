import peregrinepy as pg
import numpy as np
import mpi4py.rc

mpi4py.rc.finalize = False
mpi4py.rc.initialize = False
from mpi4py import MPI


class twoblock123:
    def __init__(self):
        self.config = pg.files.configFile()
        ng = 1 + int(np.random.random() / 0.5)
        self.mb = pg.multiBlock.solver(2, ["Air"], ng=ng)

        pg.grid.create.multiBlockCube(
            self.mb,
            mbDims=[2, 1, 1],
            dimsPerBlock=[6, 3, 2],
            lengths=[2, 1, 1],
        )

        blk0 = self.mb[0]
        blk1 = self.mb[1]
        blk0.getFace(2).commRank = 0
        blk1.getFace(1).commRank = 0

        self.xshape = self.mb[0].array["x"].shape
        self.qshape = self.mb[0].array["q"].shape

        for blk in self.mb:
            blk.array["x"][:] = np.random.random((self.xshape))
            blk.array["y"][:] = np.random.random((self.xshape)) + 1
            blk.array["z"][:] = np.random.random((self.xshape)) + 10
            blk.array["q"][:] = np.random.random((self.qshape)) + 100


class TestOrientation:
    @classmethod
    def setup_class(cls):
        MPI.Init()

    @classmethod
    def teardown_class(cls):
        MPI.Finalize()

    ##############################################
    # Test for all positive i aligned orientations
    ##############################################
    def test_123(self):

        tb = twoblock123()
        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        # Reorient and update communication info
        tb.mb.setBlockCommunication()
        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        passfail = []
        for var, shape, off in zip(
            ["x", "y", "z", "q"],
            [tb.xshape, tb.xshape, tb.xshape, tb.qshape],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            for k in range(shape[2]):
                for j in range(shape[1]):
                    for i in range(ng):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][i, j, k]
                        )
                        check1 = np.all(
                            blk0.array[var][-ng + i, j, k]
                            == blk1.array[var][ng + 1 - off + i, j, k]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            passfail.append(check0)
            passfail.append(check1)

        assert False not in passfail

    def test_135(self):

        tb = twoblock123()
        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(
                np.flip(blk1.array[var], axis=2), (0, 1, 2), (0, 2, 1)
            )
        blk1.nj = tb.xshape[2] - 2 * ng
        blk1.nk = tb.xshape[1] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "135"
        blk1.getFace(1).orientation = "162"

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][i, -(k + 1), j]
                        )
                        check1 = np.all(
                            blk0.array[var][-ng + i, j, k]
                            == blk1.array[var][ng + 1 - off + i, -(k + 1), j]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0

    def test_162(self):
        tb = twoblock123()
        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(
                np.flip(blk1.array[var], axis=1), (0, 1, 2), (0, 2, 1)
            )
        blk1.nj = tb.xshape[2] - 2 * ng
        blk1.nk = tb.xshape[1] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "162"
        blk1.getFace(1).orientation = "135"

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][i, k, -(j + 1)]
                        )
                        check1 = np.all(
                            blk0.array[var][-ng + i, j, k]
                            == blk1.array[var][ng + 1 - off + i, k, -(j + 1)]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0

    ##############################################
    # Test for all positive j aligned orientations
    ##############################################
    def test_231(self):
        tb = twoblock123()

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (1, 2, 0))
        blk1.ni = tb.xshape[2] - 2 * ng
        blk1.nj = tb.xshape[0] - 2 * ng
        blk1.nk = tb.xshape[1] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "231"

        blk1.getFace(1).neighbor = None
        blk1.getFace(1).bcType = "adiabaticSlipWall"
        blk1.getFace(1).orientation = None
        blk1.getFace(1).commRank = None

        blk1.getFace(3).neighbor = 0
        blk1.getFace(3).bcType = "b0"
        blk1.getFace(3).orientation = "312"
        blk1.getFace(3).commRank = 0

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][k, i, j]
                        )
                        check1 = np.all(
                            blk0.array[var][-ng + i, j, k]
                            == blk1.array[var][k, ng + 1 - off + i, j]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0

    ##############################################
    # Test for all positive k aligned orientations
    ##############################################
    def test_312(self):
        tb = twoblock123()

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (2, 0, 1))
        blk1.ni = tb.xshape[1] - 2 * ng
        blk1.nj = tb.xshape[2] - 2 * ng
        blk1.nk = tb.xshape[0] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "312"

        blk1.getFace(1).neighbor = None
        blk1.getFace(1).bcType = "adiabaticSlipWall"
        blk1.getFace(1).orientation = None
        blk1.getFace(1).commRank = None

        blk1.getFace(5).neighbor = 0
        blk1.getFace(5).bcType = "b0"
        blk1.getFace(5).orientation = "231"
        blk1.getFace(5).commRank = 0

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][j, k, i]
                        )
                        check1 = np.all(
                            blk0.array[var][-ng + i, j, k]
                            == blk1.array[var][j, k, ng + 1 - off + i]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0

    ##############################################
    # Test for all negative i aligned orientations
    ##############################################
    def test_432(self):
        tb = twoblock123()

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (0, 2, 1))
        blk1.ni = tb.xshape[0] - 2 * ng
        blk1.nj = tb.xshape[2] - 2 * ng
        blk1.nk = tb.xshape[1] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "432"

        blk1.getFace(1).neighbor = None
        blk1.getFace(1).bcType = "adiabaticSlipWall"
        blk1.getFace(1).orientation = None
        blk1.getFace(1).commRank = None

        blk1.getFace(2).neighbor = 0
        blk1.getFace(2).bcType = "b0"
        blk1.getFace(2).orientation = "432"
        blk1.getFace(2).commRank = 0

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][-(i + 1), k, j]
                        )
                        check1 = np.all(
                            blk0.array[var][-(i + 1), j, k]
                            == blk1.array[var][-(2 * ng + 1) + off + i, k, j]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0

    ##############################################
    # Test for all negative j aligned orientations
    ##############################################
    def test_513(self):
        tb = twoblock123()

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (1, 0, 2))
        blk1.ni = tb.xshape[1] - 2 * ng
        blk1.nj = tb.xshape[0] - 2 * ng
        blk1.nk = tb.xshape[2] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "513"

        blk1.getFace(1).neighbor = None
        blk1.getFace(1).bcType = "adiabaticSlipWall"
        blk1.getFace(1).orientation = None
        blk1.getFace(1).commRank = None

        blk1.getFace(4).neighbor = 0
        blk1.getFace(4).bcType = "b0"
        blk1.getFace(4).orientation = "243"
        blk1.getFace(4).commRank = 0

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][j, -(i + 1), k]
                        )
                        check1 = np.all(
                            blk0.array[var][-(i + 1), j, k]
                            == blk1.array[var][j, -(2 * ng + 1) + off + i, k]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0

    ##############################################
    # Test for all negative k aligned orientations
    ##############################################
    def test_621(self):
        tb = twoblock123()

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in ["x", "y", "z", "q"]:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (2, 1, 0))
        blk1.ni = tb.xshape[2] - 2 * ng
        blk1.nj = tb.xshape[1] - 2 * ng
        blk1.nk = tb.xshape[0] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "621"

        blk1.getFace(1).neighbor = None
        blk1.getFace(1).bcType = "adiabaticSlipWall"
        blk1.getFace(1).orientation = None
        blk1.getFace(1).commRank = None

        blk1.getFace(6).neighbor = 0
        blk1.getFace(6).bcType = "b0"
        blk1.getFace(6).orientation = "324"
        blk1.getFace(6).commRank = 0

        tb.mb.setBlockCommunication()

        # Execute communication
        pg.mpiComm.communicate(tb.mb, ["x", "y", "z", "q"])

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            ["x", "y", "z", "q"],
            [0, 0, 0, 1],
        ):
            check0 = True
            check1 = True
            shape = blk0.array[var].shape
            for i in range(ng):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        check0 = np.all(
                            blk0.array[var][-(2 * ng + 1) + off + i, j, k]
                            == blk1.array[var][k, j, -(i + 1)]
                        )
                        check1 = np.all(
                            blk0.array[var][-(i + 1), j, k]
                            == blk1.array[var][k, j, -(2 * ng + 1) + off + i]
                        )
                        if not check0 or not check1:
                            break
                    if not check0 or not check1:
                        break
                if not check0 or not check1:
                    break
            b02b1.append(check0)
            b12b0.append(check1)

        assert False not in b02b1
        assert False not in b12b0
