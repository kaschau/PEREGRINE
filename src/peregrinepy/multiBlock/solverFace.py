from kokkos import deep_copy
import numpy as np
from .topologyFace import topologyFace
from ..misc import null, frozenDict, createViewMirrorArray
from ..compute import face_, bcs

s_ = np.s_


class solverFace(topologyFace, face_):

    faceType = "solver"

    def __init__(self, nface, ng):
        face_.__init__(self)
        topologyFace.__init__(self, nface)
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self._ng = ng

        # Face slices
        smallS0 = range(ng - 1, -1, -1)
        smallS2 = range(ng + 1, 2 * ng + 1)
        largeS0 = range(-ng, 0)
        largeS2 = range(-(ng + 2), -(2 * ng + 2), -1)
        if nface == 1:
            self.s0_ = [s_[i, :, :] for i in smallS0]
            self.s1_ = s_[ng, :, :]
            self.s2_ = [s_[i, :, :] for i in smallS2]
        elif nface == 2:
            self.s0_ = [s_[i, :, :] for i in largeS0]
            self.s1_ = s_[-(ng + 1), :, :]
            self.s2_ = [s_[i, :, :] for i in largeS2]
        elif nface == 3:
            self.s0_ = [s_[:, i, :] for i in smallS0]
            self.s1_ = s_[:, ng, :]
            self.s2_ = [s_[:, i, :] for i in smallS2]
        elif nface == 4:
            self.s0_ = [s_[:, i, :] for i in largeS0]
            self.s1_ = s_[:, -(ng + 1), :]
            self.s2_ = [s_[:, i, :] for i in largeS2]
        elif nface == 5:
            self.s0_ = [s_[:, :, i] for i in smallS0]
            self.s1_ = s_[:, :, ng]
            self.s2_ = [s_[:, :, i] for i in smallS2]
        elif nface == 6:
            self.s0_ = [s_[:, :, i] for i in largeS0]
            self.s1_ = s_[:, :, -(ng + 1)]
            self.s2_ = [s_[:, :, i] for i in largeS2]

        # Boundary condition values
        self.array = frozenDict(
            {
                "sendBuffer3": None,
                "sendBuffer4": None,
                "recvBuffer3": None,
                "recvBuffer4": None,
                "tempRecvBuffer3": None,
                "tempRecvBuffer4": None,
                "qBcVals": None,
                "QBcVals": None,
                "periodicRotMatrixUp": None,
                "periodicRotMatrixDown": None,
            }
        )
        self.mirror = frozenDict(
            {
                "sendBuffer3": None,
                "sendBuffer4": None,
                "recvBuffer3": None,
                "recvBuffer4": None,
                "tempRecvBuffer3": None,
                "tempRecvBuffer4": None,
                "qBcVals": None,
                "QBcVals": None,
                "periodicRotMatrixUp": None,
                "periodicRotMatrixDown": None,
            }
        )
        # Boundary function
        self.bcFunc = bcs.walls.adiabaticSlipWall

        # MPI variables
        self.commRank = None

        self.orient = None
        self.sliceS3 = None
        self.sliceR3 = None
        self.sliceS4 = None
        self.sliceR4 = None

        self.tagS = None
        self.tagR = None

    @topologyFace.bcType.setter
    def bcType(self, value):
        super(solverFace, type(self)).bcType.fset(self, value)
        self._setBcFunc()

    def _setBcFunc(self):

        bcType = self.bcType
        if bcType == "b0" or bcType.startswith("periodicTrans"):
            self.bcFunc = null
        else:
            for bcmodule in [bcs.inlets, bcs.exits, bcs.walls, bcs.periodics]:
                try:
                    self.bcFunc = getattr(bcmodule, bcType)
                    break
                except AttributeError:
                    pass
            else:
                raise KeyError(f"{bcType} is not a valid bcType")

    def _setCommBuffers(self, ni, nj, nk, ne, nblki):
        assert (
            self.orient is not None
        ), "Must set orientFunc before commBuffers, or perhaps you are trying to set buffers for a non communication face."
        assert self.ng is not None, "ng must be specified to setCommBuffers"

        # Send and Recv slices
        # The order of the recieve slice list always goes from smallest index to
        #  largest index. Send lists begin in the same way, however if the
        #  orientation requires this order to be flipped such that
        #  the recv buffer arrives in the appropriate order we will do that later
        #  in the funciton. Also note that the send/recv buffers can be seen as list
        #  of slices. I.e. recv[0] is the first slice recieved, recv[1] is the next
        #  slice recieved. This way, we can still treat each slice as a 2d object for
        #  reorientation purposes.
        #
        #        index -------------------------->
        #  o----------o----------o|x----------x----------x
        #  |          |           |           |          |
        #  | recv[0]  |  recv[1]  |  send[0]  |  send[1] |
        #  |          |           |           |          |
        #  o----------o----------o|x----------x----------x
        ng = self.ng

        smallSfp = list(range(ng + 1, 2 * ng + 1))
        smallRfp = list(range(0, ng))
        largeSfp = list(range(-(2 * ng + 1), -(ng + 1)))
        largeRfp = list(range(-ng, 0))

        smallSc = list(range(ng, 2 * ng))
        smallRc = list(range(0, ng))
        largeSc = list(range(-2 * ng, -ng))
        largeRc = list(range(-ng, 0))

        if self.nface == 1:
            self.sliceS3 = [s_[i, :, :] for i in smallSfp]
            self.sliceS4 = [s_[i, :, :, :] for i in smallSc]
            commfpshape = (ng, nj + 2 * ng, nk + 2 * ng)
            commcshape = (ng, nj + 2 * ng - 1, nk + 2 * ng - 1, ne)
            self.sliceR3 = [s_[i, :, :] for i in smallRfp]
            self.sliceR4 = [s_[i, :, :, :] for i in smallRc]
        elif self.nface == 2:
            self.sliceS3 = [s_[ni + 2 * ng + i, :, :] for i in largeSfp]
            self.sliceS4 = [s_[ni + 2 * ng + i - 1, :, :, :] for i in largeSc]
            commfpshape = (ng, nj + 2 * ng, nk + 2 * ng)
            commcshape = (ng, nj + 2 * ng - 1, nk + 2 * ng - 1, ne)
            self.sliceR3 = [s_[ni + 2 * ng + i, :, :] for i in largeRfp]
            self.sliceR4 = [s_[ni + 2 * ng + i - 1, :, :, :] for i in largeRc]
        elif self.nface == 3:
            self.sliceS3 = [s_[:, i, :] for i in smallSfp]
            self.sliceS4 = [s_[:, i, :, :] for i in smallSc]
            commfpshape = (ng, ni + 2 * ng, nk + 2 * ng)
            commcshape = (ng, ni + 2 * ng - 1, nk + 2 * ng - 1, ne)
            self.sliceR3 = [s_[:, i, :] for i in smallRfp]
            self.sliceR4 = [s_[:, i, :, :] for i in smallRc]
        elif self.nface == 4:
            self.sliceS3 = [s_[:, nj + 2 * ng + i, :] for i in largeSfp]
            self.sliceS4 = [s_[:, nj + 2 * ng + i - 1, :, :] for i in largeSc]
            commfpshape = (ng, ni + 2 * ng, nk + 2 * ng)
            commcshape = (ng, ni + 2 * ng - 1, nk + 2 * ng - 1, ne)
            self.sliceR3 = [s_[:, nj + 2 * ng + i, :] for i in largeRfp]
            self.sliceR4 = [s_[:, nj + 2 * ng + i - 1, :, :] for i in largeRc]
        elif self.nface == 5:
            self.sliceS3 = [s_[:, :, i] for i in smallSfp]
            self.sliceS4 = [s_[:, :, i, :] for i in smallSc]
            commfpshape = (ng, ni + 2 * ng, nj + 2 * ng)
            commcshape = (ng, ni + 2 * ng - 1, nj + 2 * ng - 1, ne)
            self.sliceR3 = [s_[:, :, i] for i in smallRfp]
            self.sliceR4 = [s_[:, :, i, :] for i in smallRc]
        elif self.nface == 6:
            self.sliceS3 = [s_[:, :, nk + 2 * ng + i] for i in largeSfp]
            self.sliceS4 = [s_[:, :, nk + 2 * ng + i - 1, :] for i in largeSc]
            commfpshape = (ng, ni + 2 * ng, nj + 2 * ng)
            commcshape = (ng, ni + 2 * ng - 1, nj + 2 * ng - 1, ne)
            self.sliceR3 = [s_[:, :, nk + 2 * ng + i] for i in largeRfp]
            self.sliceR4 = [s_[:, :, nk + 2 * ng + i - 1, :] for i in largeRc]

        # We reverse the order of the send slices if the face's neighbor
        # and this face axis is counter aligned. That way the recv buffer
        # arrives good to go.
        if self.nface in [1, 2]:
            indx = 0
        elif self.nface in [3, 4]:
            indx = 1
        elif self.nface in [5, 6]:
            indx = 2
        if self.orientation[indx] in ["4", "5", "6"]:
            self.sliceS3.reverse()
            self.sliceS4.reverse()

        # We send the data in the correct shape already
        # Face and point shape
        temp = self.orient(np.empty(commfpshape[1::]))
        self.array["sendBuffer3"] = np.ascontiguousarray(
            np.empty(tuple([ng]) + temp.shape)
        )
        shape = self.array["sendBuffer3"].shape
        createViewMirrorArray(self, "sendBuffer3", shape)

        # We recieve the data in the correct shape already
        self.array["recvBuffer3"] = np.ascontiguousarray(np.empty(commfpshape))
        shape = self.array["recvBuffer3"].shape
        createViewMirrorArray(self, "recvBuffer3", shape)
        # We use temporary buffers for some storage
        self.array["tempRecvBuffer3"] = np.ascontiguousarray(np.empty(commfpshape))
        shape = self.array["tempRecvBuffer3"].shape
        createViewMirrorArray(self, "tempRecvBuffer3", shape)

        # Cell
        temp = self.orient(np.empty(commcshape[1::]))
        self.array["sendBuffer4"] = np.ascontiguousarray(
            np.empty(tuple([ng]) + temp.shape)
        )
        shape = self.array["sendBuffer4"].shape
        createViewMirrorArray(self, "sendBuffer4", shape)
        # We revieve the data in the correct shape already
        self.array["recvBuffer4"] = np.ascontiguousarray(np.empty(commcshape))
        shape = self.array["recvBuffer4"].shape
        createViewMirrorArray(self, "recvBuffer4", shape)
        # We use temporary buffers for some storage
        self.array["tempRecvBuffer4"] = np.ascontiguousarray(np.empty(commcshape))
        shape = self.array["tempRecvBuffer4"].shape
        createViewMirrorArray(self, "tempRecvBuffer4", shape)

        # Unique tags.
        self.tagR = int(nblki * 6 + self.nface)
        self.tagS = int(self.neighbor * 6 + self.neighborNface)

    def _setOrientFunc(self, ni, nj, nk, ne):
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

        neighborNface = self.neighborNface
        neighborOrientation = self.neighborOrientation

        # What are the orientations of our neighbot face plane? i.e.
        #  face #1 with orientation "123" then the face in quetions has
        #  the orientation [2,3]
        faceOrientations = [
            int(i)
            for j, i in enumerate(neighborOrientation)
            if j != faceToOrientIndexMapping[neighborNface]
        ]
        # What is my neighbor's face normal index
        normalIndex = [
            j for j in range(3) if j == faceToOrientIndexMapping[neighborNface]
        ][0]
        # What is my face normal index
        normalIndex2 = [
            j for j in range(3) if j == faceToOrientIndexMapping[self.nface]
        ][0]

        # What is the larger index for our neighbor face orientation? i < j < k
        bigIndex = largeIndexMapping[normalIndex]
        # What is the larger index for our face orientation? i < j < k
        bigIndex2 = largeIndexMapping[normalIndex2]

        # Set the orienting function for this face
        function = "orient"
        # Do we need to transpose?
        transpose = faceOrientations[1] in needToTranspose[bigIndex][bigIndex2]
        function += "T" if transpose else ""

        # Do we need to flip along 0 axis?
        flip0 = faceOrientations[0] in [4, 5, 6]
        function += "f0" if flip0 else ""

        # Do we need to flip along 1 axis?
        flip1 = faceOrientations[1] in [4, 5, 6]
        function += "f1" if flip1 else ""

        # Do we need to do anything?
        function += "Null" if function == "orient" else ""

        self.orient = getattr(self, function)

    @property
    def ng(self):
        return self._ng

    def updateDeviceView(self, vars):
        if type(vars) == str:
            vars = [vars]
        for var in vars:
            deep_copy(getattr(self, var), self.mirror[var])

    def updateHostView(self, vars):
        if type(vars) == str:
            vars = [vars]
        for var in vars:
            deep_copy(self.mirror[var], getattr(self, var))

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
