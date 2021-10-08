import numpy as np
from .topologyFace import topologyFace
from ..misc import frozenDict, null
from ..bcs import inlets, exits, walls

s_ = np.s_


class solverFace(topologyFace):

    __slots__ = (
        "s0_",
        "s1_",
        "s2_",
        "bcVals",
        "bcFunc",
        "commRank",
        "orient",
        "sliceS3",
        "sliceS4",
        "sliceR3",
        "sliceR4",
        "tagS",
        "sendBuffer3",
        "sendBuffer4",
        "tagR",
        "recvBuffer3",
        "recvBuffer4",
    )

    def __init__(self, nface):
        super().__init__(nface)
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        # Face slices
        if nface == 1:
            self.s0_ = s_[0, :, :]
            self.s1_ = s_[1, :, :]
            self.s2_ = s_[2, :, :]
        if nface == 2:
            self.s0_ = s_[-1, :, :]
            self.s1_ = s_[-2, :, :]
            self.s2_ = s_[-3, :, :]
        if nface == 3:
            self.s0_ = s_[:, 0, :]
            self.s1_ = s_[:, 1, :]
            self.s2_ = s_[:, 2, :]
        if nface == 4:
            self.s0_ = s_[:, -1, :]
            self.s1_ = s_[:, -2, :]
            self.s2_ = s_[:, -3, :]
        if nface == 5:
            self.s0_ = s_[:, :, 0]
            self.s1_ = s_[:, :, 1]
            self.s2_ = s_[:, :, 2]
        if nface == 6:
            self.s0_ = s_[:, :, -1]
            self.s1_ = s_[:, :, -2]
            self.s2_ = s_[:, :, -3]

        # Boundary condition values
        self.bcVals = frozenDict({})
        # Boundary function
        self.bcFunc = walls.adiabaticSlipWall

        # MPI variables - only set for solver blocks, but we will store them
        # all the time for now
        self.commRank = None

        self.orient = None
        self.sliceS3 = None
        self.sliceR3 = None
        self.sliceS4 = None
        self.sliceR4 = None

        self.tagS = None
        self.sendBuffer3 = None
        self.sendBuffer4 = None
        self.tagR = None
        self.recvBuffer3 = None
        self.recvBuffer4 = None

    def setBcFunc(self):

        bc = self.bcType
        if bc in ["b0", "b1"]:
            self.bcFunc = null
        elif bc == "constantVelocitySubsonicInlet":
            self.bcFunc = inlets.constantVelocitySubsonicInlet
        elif bc == "constantPressureSubsonicExit":
            self.bcFunc = exits.constantPressureSubsonicExit
        elif bc == "adiabaticNoSlipWall":
            self.bcFunc = walls.adiabaticNoSlipWall
        elif bc == "adiabaticSlipWall":
            self.bcFunc = walls.adiabaticSlipWall
        elif bc == "adiabaticMovingWall":
            self.bcFunc = walls.adiabaticMovingWall
        elif bc == "isoTMovingWall":
            self.bcFunc = walls.isoTMovingWall
        else:
            raise KeyError(f"{bc} is not a valid bcType")

    def setCommBuffers(self, ni, nj, nk, ne, nblki):
        assert (
            self.orient is not None
        ), "Must set orientFun before commBuffers, or perhaps you are trying to set buffers for a non communication face."

        # Send and Recv slices
        # The order of the face slice list always goes from smallest index to
        #  largest index, where each slice is a ghost layer.
        #  o----------o----------o|x----------x----------x
        #  |          |           |           |          |
        #  | recv[0]  |  recv[1]  |  send[0]  |  send[1] |
        #  |          |           |           |          |
        #  o----------o----------o|x----------x----------x

        if self.nface == 1:
            self.sliceS3 = s_[2, :, :]
            self.sliceS4 = s_[1, :, :, :]
            commfpshape = (nj + 2, nk + 2)
            commcshape = (nj + 1, nk + 1, ne)
            self.sliceR3 = s_[0, :, :]
            self.sliceR4 = s_[0, :, :, :]
        elif self.nface == 2:

            self.sliceS3 = s_[-3, :, :]
            self.sliceS4 = s_[-2, :, :, :]
            commfpshape = (nj + 2, nk + 2)
            commcshape = (nj + 1, nk + 1, ne)
            self.sliceR3 = s_[-1, :, :]
            self.sliceR4 = s_[-1, :, :, :]

        elif self.nface == 3:
            self.sliceS3 = s_[:, 2, :]
            self.sliceS4 = s_[:, 1, :, :]
            commfpshape = (ni + 2, nk + 2)
            commcshape = (ni + 1, nk + 1, ne)
            self.sliceR3 = s_[:, 0, :]
            self.sliceR4 = s_[:, 0, :, :]

        elif self.nface == 4:
            self.sliceS3 = s_[:, -3, :]
            self.sliceS4 = s_[:, -2, :, :]
            commfpshape = (ni + 2, nk + 2)
            commcshape = (ni + 1, nk + 1, ne)
            self.sliceR3 = s_[:, -1, :]
            self.sliceR4 = s_[:, -1, :, :]

        elif self.nface == 5:
            self.sliceS3 = s_[:, :, 2]
            self.sliceS4 = s_[:, :, 1]
            commfpshape = (ni + 2, nj + 2)
            commcshape = (ni + 1, nj + 1, ne)
            self.sliceR3 = s_[:, :, 0]
            self.sliceR4 = s_[:, :, 0, :]

        elif self.nface == 6:
            self.sliceS3 = s_[:, :, -3]
            self.sliceS4 = s_[:, :, -2, :]
            commfpshape = (ni + 2, nj + 2)
            commcshape = (ni + 1, nj + 1, ne)
            self.sliceR3 = s_[:, :, -1]
            self.sliceR4 = s_[:, :, -1, :]

        # We send the data in the correct shape already
        # Face and point shape
        self.sendBuffer3 = np.ascontiguousarray(self.orient(np.empty(commfpshape)))
        # We revieve the data in the correct shape already
        self.recvBuffer3 = np.ascontiguousarray(np.empty(commfpshape))

        # Cell
        self.sendBuffer4 = np.ascontiguousarray(self.orient(np.empty(commcshape)))
        # We revieve the data in the correct shape already
        self.recvBuffer4 = np.ascontiguousarray(np.empty(commcshape))

        self.tagR = int(f"1{self.neighbor}2{nblki}1{self.nface}")
        self.tagS = int(f"1{nblki}2{self.neighbor}1{self.neighborFace}")

    def setOrientFunc(self, ni, nj, nk, ne):
        assert 0 not in [
            ni,
            nj,
            nk,
        ], "Must get grid before setting block communicaitons."

        ##########################################################
        # Define the mapping based on orientation
        ##########################################################
        faceToOrientIndexMapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        largeIndexMapping = {0: "k", 1: "k", 2: "j"}
        needToTranspose = {
            "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
            "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
        }

        neighbor = self.neighbor
        if neighbor is None:
            return

        neighborFace = self.neighborFace
        orientation = self.orientation

        # What are the orientations of our face plane? i.e. if we are
        #  face #1 with orientation "123" then our face in quetions has
        #  the orientation [2,3]
        faceOrientations = [
            int(i)
            for j, i in enumerate(orientation)
            if j != faceToOrientIndexMapping[self.nface]
        ]
        # What is my face normal index
        normalIndex = [
            j for j in range(3) if j == faceToOrientIndexMapping[self.nface]
        ][0]
        # What is my neighbor's face normal index
        normalIndex2 = [
            j for j in range(3) if j == faceToOrientIndexMapping[neighborFace]
        ][0]

        # What is the larger index for our face orientation? i < j < k
        bigIndex = largeIndexMapping[normalIndex]
        # What is the larger index for our neighbor face orientation? i < j < k
        bigIndex2 = largeIndexMapping[normalIndex2]

        # Set the orienting function for this face
        function = "orient"
        # Do we need to transpose?
        transpose = faceOrientations[1] in needToTranspose[bigIndex][bigIndex2]
        function += "T" if transpose else ""

        # Do we need to flip along 0 axis?
        flip0 = faceOrientations[1] in [4, 5, 6]
        function += "f0" if flip0 else ""

        # Do we need to flip along 1 axis?
        flip1 = faceOrientations[0] in [4, 5, 6]
        function += "f1" if flip1 else ""

        # Do we need to do anything?
        function += "Null" if function == "orient" else ""

        self.orient = getattr(self, function)

    ##########################################################
    # Define the possible reorientation routines
    ##########################################################
    @staticmethod
    def orientT(temp):
        axT = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.transpose(temp, axT)

    @staticmethod
    def orientTf0(temp):
        axT = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.flip(np.transpose(temp, axT), 0)

    @staticmethod
    def orientTf1(temp):
        axT = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.flip(np.transpose(temp, axT), 1)

    @staticmethod
    def orientTf0f1(temp):
        axT = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.flip(np.transpose(temp, axT), (0, 1))

    @staticmethod
    def orientf0(temp):
        return np.flip(temp, 0)

    @staticmethod
    def orientf0f1(temp):
        return np.flip(temp, (0, 1))

    @staticmethod
    def orientf1(temp):
        return np.flip(temp, 1)

    @staticmethod
    def orientNull(temp):
        return temp
