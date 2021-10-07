import peregrinepy as pg
import numpy as np


class twoblock123:
    def __init__(self):
        self.config = pg.files.configFile()
        self.mb = pg.multiBlock.solver(2, ["Air"])

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
        for face in blk0.faces:
            face.setBcFunc()
        for face in blk1.faces:
            face.setBcFunc()

        self.xshape = self.mb[0].array["x"].shape
        self.qshape = self.mb[0].array["q"].shape

        for blk in self.mb:
            blk.array["x"][:] = np.random.random((self.xshape))
            blk.array["y"][:] = np.random.random((self.xshape))
            blk.array["z"][:] = np.random.random((self.xshape))
            blk.array["q"][:] = np.random.random((self.qshape))


##############################################
# Test for all positive i aligned orientations
##############################################
def test_123():

    tb = twoblock123()
    blk0 = tb.mb[0]
    blk1 = tb.mb[1]

    # Reorient and update communication info
    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)
    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][0, j, k]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][2 - off, j, k]
                )
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert False not in passfail


def test_135():
    tb = twoblock123()
    blk0 = tb.mb[0]
    blk1 = tb.mb[1]

    for var in ["x", "y", "z", "q"]:
        blk1.array[var] = np.moveaxis(
            np.flip(blk1.array[var], axis=2), (0, 1, 2), (0, 2, 1)
        )
    blk1.nj = tb.xshape[2] - 2
    blk1.nk = tb.xshape[1] - 2

    # Reorient second block and update communication info
    blk0.getFaceConn(2)["orientation"] = "135"
    blk1.getFaceConn(1)["orientation"] = "162"

    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)

    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][0, -(k + 1), j]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][2 - off, -(k + 1), j]
                )
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert False not in passfail


##############################################
# Test for all positive j aligned orientations
##############################################
def test_231():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ["x", "y", "z", "q"]:
        blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (1, 2, 0))
    blk1.ni = tb.xshape[2] - 2
    blk1.nj = tb.xshape[0] - 2
    blk1.nk = tb.xshape[1] - 2

    # Reorient second block and update communication info
    blk0.getFaceConn(2)["orientation"] = "231"

    blk1.getFaceConn(1)["neighbor"] = None
    blk1.getFaceConn(1)["bcType"] = "adiabaticSlipWall"
    blk1.getFaceConn(1)["orientation"] = None
    blk1.getFace(1).commRank = None

    blk1.getFaceConn(3)["neighbor"] = 0
    blk1.getFaceConn(3)["bcType"] = "b0"
    blk1.getFaceConn(3)["orientation"] = "312"
    blk1.getFace(3).commRank = 0
    for face in blk0.faces:
        face.setBcFunc()
    for face in blk1.faces:
        face.setBcFunc()

    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)

    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][k, 0, j]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][k, 2 - off, j]
                )
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert False not in passfail


##############################################
# Test for all positive k aligned orientations
##############################################
def test_321():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ["x", "y", "z", "q"]:
        blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (2, 0, 1))
    blk1.ni = tb.xshape[1] - 2
    blk1.nj = tb.xshape[2] - 2
    blk1.nk = tb.xshape[0] - 2

    # Reorient second block and update communication info
    blk0.getFaceConn(2)["orientation"] = "312"

    blk1.getFaceConn(1)["neighbor"] = None
    blk1.getFaceConn(1)["bcType"] = "adiabaticSlipWall"
    blk1.getFaceConn(1)["orientation"] = None
    blk1.getFace(1).commRank = None

    blk1.getFaceConn(5)["neighbor"] = 0
    blk1.getFaceConn(5)["bcType"] = "b0"
    blk1.getFaceConn(5)["orientation"] = "231"
    blk1.getFace(5).commRank = 0
    for face in blk0.faces:
        face.setBcFunc()
    for face in blk1.faces:
        face.setBcFunc()

    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)

    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][j, k, 0]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][j, k, 2 - off]
                )
                if not check0 or not check1:
                    print(blk0.array[var][-3, j, k], blk1.array[var][j, k, 0])
                    break
            if not check0 or not check1:
                print(blk0.array[var][-1, j, k], blk1.array[var][j, k, 2])
                break
        passfail.append(check0)
        passfail.append(check1)
    passfail = []

    assert False not in passfail


##############################################
# Test for all negative i aligned orientations
##############################################
def test_432():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ["x", "y", "z", "q"]:
        blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (0, 2, 1))
    blk1.ni = tb.xshape[0] - 2
    blk1.nj = tb.xshape[2] - 2
    blk1.nk = tb.xshape[1] - 2

    # Reorient second block and update communication info
    blk0.getFaceConn(2)["orientation"] = "432"

    blk1.getFaceConn(1)["neighbor"] = None
    blk1.getFaceConn(1)["bcType"] = "adiabaticSlipWall"
    blk1.getFaceConn(1)["orientation"] = None
    blk1.getFace(1).commRank = None

    blk1.getFaceConn(2)["neighbor"] = 0
    blk1.getFaceConn(2)["bcType"] = "b0"
    blk1.getFaceConn(2)["orientation"] = "432"
    blk1.getFace(2).commRank = 0
    for face in blk0.faces:
        face.setBcFunc()
    for face in blk1.faces:
        face.setBcFunc()

    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)

    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][-1, k, j]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][-3 + off, k, j]
                )
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert False not in passfail


##############################################
# Test for all negative j aligned orientations
##############################################
def test_513():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ["x", "y", "z", "q"]:
        blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (1, 0, 2))
    blk1.ni = tb.xshape[1] - 2
    blk1.nj = tb.xshape[0] - 2
    blk1.nk = tb.xshape[2] - 2

    # Reorient second block and update communication info
    blk0.getFaceConn(2)["orientation"] = "513"

    blk1.getFaceConn(1)["neighbor"] = None
    blk1.getFaceConn(1)["bcType"] = "adiabaticSlipWall"
    blk1.getFaceConn(1)["orientation"] = None
    blk1.getFace(1).commRank = None

    blk1.getFaceConn(4)["neighbor"] = 0
    blk1.getFaceConn(4)["bcType"] = "b0"
    blk1.getFaceConn(4)["orientation"] = "243"
    blk1.getFace(4).commRank = 0

    for face in blk0.faces:
        face.setBcFunc()
    for face in blk1.faces:
        face.setBcFunc()

    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)

    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][j, -1, k]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][j, -3 + off, k]
                )
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert False not in passfail


##############################################
# Test for all negative k aligned orientations
##############################################
def test_621():
    tb = twoblock123()

    blk0 = tb.mb[0]
    blk1 = tb.mb[1]
    for var in ["x", "y", "z", "q"]:
        blk1.array[var] = np.moveaxis(blk1.array[var], (0, 1, 2), (2, 1, 0))
    blk1.ni = tb.xshape[2] - 2
    blk1.nj = tb.xshape[1] - 2
    blk1.nk = tb.xshape[0] - 2

    # Reorient second block and update communication info
    blk0.getFaceConn(2)["orientation"] = "621"

    blk1.getFaceConn(1)["neighbor"] = None
    blk1.getFaceConn(1)["bcType"] = "adiabaticSlipWall"
    blk1.getFaceConn(1)["orientation"] = None
    blk1.getFace(1).commRank = None

    blk1.getFaceConn(6)["neighbor"] = 0
    blk1.getFaceConn(6)["bcType"] = "b0"
    blk1.getFaceConn(6)["orientation"] = "324"
    blk1.getFace(6).commRank = 0

    for face in blk0.faces:
        face.setBcFunc()
    for face in blk1.faces:
        face.setBcFunc()

    pg.mpiComm.blockComm.setBlockCommunication(tb.mb)

    # Execute communication
    pg.mpiComm.blockComm.communicate(tb.mb, ["x", "y", "z", "q"])

    passfail = []
    for var, shape, off in zip(
        ["x", "y", "z", "q"], [tb.xshape, tb.xshape, tb.xshape, tb.qshape], [0, 0, 0, 1]
    ):
        check0 = True
        check1 = True
        for k in range(shape[2]):
            for j in range(shape[1]):
                check0 = np.all(
                    blk0.array[var][-3 + off, j, k] == blk1.array[var][k, j, -1]
                )
                check1 = np.all(
                    blk0.array[var][-1, j, k] == blk1.array[var][k, j, -3 + off]
                )
                if not check0 or not check1:
                    break
            if not check0 or not check1:
                break
        passfail.append(check0)
        passfail.append(check1)

    assert False not in passfail
