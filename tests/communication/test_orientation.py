import mpi4py.rc
import itertools
import peregrinepy as pg
import numpy as np
import pytest

mpi4py.rc.finalize = False
mpi4py.rc.initialize = False
from mpi4py import MPI


class twoblock123:
    def __init__(self, adv, spdata):
        self.config = pg.files.configFile()
        self.config["RHS"]["primaryAdvFlux"] = adv
        if adv == "secondOrderKEEP":
            self.config["RHS"]["diffOrder"] = 2
        elif adv == "fourthOrderKEEP":
            self.config["RHS"]["diffOrder"] = 4
        self.config["RHS"]["shockHandling"] = "hybrid"
        self.config["RHS"]["secondaryAdvFlux"] = "rusanov"
        self.config["thermochem"]["spdata"] = spdata
        self.config["RHS"]["diffusion"] = True
        self.mb = pg.multiBlock.generateMultiBlockSolver(2, self.config)

        pg.grid.create.multiBlockCube(
            self.mb,
            mbDims=[2, 1, 1],
            dimsPerBlock=[6, 3, 2],
            lengths=[2, 1, 1],
        )

        self.mb.initSolverArrays(self.config)
        self.mb.generateHalo()
        self.mb.computeMetrics(self.config["RHS"]["diffOrder"])

        blk0 = self.mb[0]
        blk1 = self.mb[1]
        blk0.getFace(2).commRank = 0
        blk1.getFace(1).commRank = 0
        ng = blk0.ng

        self.xshape = self.mb[0].array["x"].shape
        self.qshape = self.mb[0].array["q"].shape
        self.phishape = self.mb[0].array["phi"].shape

        self.varList = ["x", "y", "z", "q", "Q", "dqdx", "dqdy", "dqdz", "phi"]
        self.varShapes = [
            self.xshape,
            self.xshape,
            self.xshape,
            self.qshape,
            self.qshape,
            self.qshape,
            self.qshape,
            self.qshape,
            self.phishape,
        ]
        self.offsets = [0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.nLayers = [ng, ng, ng, ng, ng, 1, 1, 1, 1]

        for blk in self.mb:
            blk.array["x"][:] = np.random.random((self.xshape))
            blk.array["y"][:] = np.random.random((self.xshape))
            blk.array["z"][:] = np.random.random((self.xshape))
            blk.array["q"][:] = np.random.random((self.qshape))
            blk.array["Q"][:] = np.random.random((self.qshape))
            blk.array["dqdx"][:] = np.random.random((self.qshape))
            blk.array["dqdy"][:] = np.random.random((self.qshape))
            blk.array["dqdz"][:] = np.random.random((self.qshape))
            blk.array["phi"][:] = np.random.random((self.phishape))
            blk.updateDeviceView(self.varList)


pytestmark = pytest.mark.parametrize(
    "adv,spdata",
    list(
        itertools.product(
            ("secondOrderKEEP",),  # "fourthOrderKEEP"),
            (["Air"],),  # "thtr_CH4_O2_Stanford_Skeletal.yaml"),
        )
    ),
)


class TestOrientation:
    @classmethod
    def setup_class(self):
        pass

    @classmethod
    def teardown_class(self):
        pass

    ##############################################
    # Test for all positive i aligned orientations
    ##############################################
    def test_123(self, my_setup, adv, spdata):

        tb = twoblock123(adv, spdata)
        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        # Reorient and update communication info
        tb.mb.setBlockCommunication()
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        passfail = []
        for var, shape, off, nLayers in zip(
            tb.varList,
            tb.varShapes,
            tb.offsets,
            tb.nLayers,
        ):
            check0 = True
            check1 = True
            for k in range(shape[2]):
                for j in range(shape[1]):
                    for i in range(nLayers):
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

    def test_135(self, my_setup, adv, spdata):

        tb = twoblock123(adv, spdata)
        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(
                np.flip(blk1.array[var], axis=2), (0, 1, 2), (0, 2, 1)
            )

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

        blk1.nj = tb.xshape[2] - 2 * ng
        blk1.nk = tb.xshape[1] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "135"
        blk1.getFace(1).orientation = "162"

        tb.mb.setBlockCommunication()
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, shape, off, nLayers in zip(
            tb.varList,
            tb.varShapes,
            tb.offsets,
            tb.nLayers,
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

    def test_162(self, my_setup, adv, spdata):
        tb = twoblock123(adv, spdata)
        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(
                np.flip(blk1.array[var], axis=1), (0, 1, 2), (0, 2, 1)
            )

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

        blk1.nj = tb.xshape[2] - 2 * ng
        blk1.nk = tb.xshape[1] - 2 * ng

        # Reorient second block and update communication info
        blk0.getFace(2).orientation = "162"
        blk1.getFace(1).orientation = "135"

        tb.mb.setBlockCommunication()
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            tb.varList,
            [0, 0, 0, 1, 1, 1, 1],
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
    def test_231(self, my_setup, adv, spdata):
        tb = twoblock123(adv, spdata)

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (1, 2, 0))

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

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
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            tb.varList,
            [0, 0, 0, 1, 1, 1, 1],
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
    def test_312(self, my_setup, adv, spdata):
        tb = twoblock123(adv, spdata)

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (2, 0, 1))

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

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
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            tb.varList,
            [0, 0, 0, 1, 1, 1, 1],
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
    def test_432(self, my_setup, adv, spdata):
        tb = twoblock123(adv, spdata)

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (0, 2, 1))

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

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
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            tb.varList,
            [0, 0, 0, 1, 1, 1, 1],
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
    def test_513(self, my_setup, adv, spdata):
        tb = twoblock123(adv, spdata)

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (1, 0, 2))

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

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
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            tb.varList,
            [0, 0, 0, 1, 1, 1, 1],
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
    def test_621(self, my_setup, adv, spdata):
        tb = twoblock123(adv, spdata)

        blk0 = tb.mb[0]
        blk1 = tb.mb[1]
        ng = blk0.ng

        for var in tb.varList:
            blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (2, 1, 0))

        # HACK: It seems like previous test data still exists
        # We need to clear out residual arrays from previous tests
        for v in blk0.array.keys():
            if v not in tb.varList:
                blk0.array[v] = None
                blk0.mirror[v] = None
                blk1.array[v] = None
                blk1.mirror[v] = None

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
        tb.mb.initSolverArrays(tb.config)
        # Execute communication
        for blk in tb.mb:
            blk.updateDeviceView(tb.varList)
        pg.mpiComm.communicate(tb.mb, tb.varList)
        for blk in tb.mb:
            blk.updateHostView(tb.varList)

        b02b1 = []
        b12b0 = []
        for var, off in zip(
            tb.varList,
            [0, 0, 0, 1, 1, 1, 1],
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
