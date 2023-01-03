import numpy as np
from kokkos import deep_copy

from ..compute import bcs, face_
from ..misc import createViewMirrorArray, frozenDict, null
from .gridFace import gridFace
from .topologyFace import topologyFace

s_ = np.s_


class solverFace(gridFace, face_):

    faceType = "solver"

    def __init__(self, nface, ng):
        face_.__init__(self)
        gridFace.__init__(self, nface)
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

        # arrays that faces save
        #
        # List of all possible communicate vars
        commVars = ["x", "y", "z", "q", "Q", "dqdx", "dqdy", "dqdz", "phi"]
        self.mirror = frozenDict()
        for d in commVars:
            self.array["sendBuffer_" + d] = None
            self.array["recvBuffer_" + d] = None
            self.array["tempRecvBuffer_" + d] = None

            self.mirror["sendBuffer_" + d] = None
            self.mirror["recvBuffer_" + d] = None
            self.mirror["tempRecvBuffer_" + d] = None

        self.array["qBcVals"] = None
        self.array["QBcVals"] = None
        self.mirror["qBcVals"] = None
        self.mirror["QBcVals"] = None
        self.mirror["periodicRotMatrixUp"] = None
        self.mirror["periodicRotMatrixDown"] = None

        self.array._freeze()
        self.mirror._freeze()

        # Boundary function
        self.bcFunc = bcs.walls.adiabaticSlipWall

        # MPI variables
        self.commRank = None
        self.orient = None
        self.tagS = None
        self.tagR = None

    @topologyFace.bcType.setter
    def bcType(self, value):
        gridFace.bcType.fset(self, value)
        self._setBcFunc()

    @gridFace.periodicAxis.setter
    def periodicAxis(self, axis):
        gridFace.periodicAxis.fset(self, axis)
        createViewMirrorArray(self, "periodicRotMatrixUp", (3, 3))
        createViewMirrorArray(self, "periodicRotMatrixDown", (3, 3))

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

        # List of node halo indicies
        smallFaceSendNodesSlices = list(range(ng + 1, 2 * ng + 1))
        smallFaceRecvNodesSlices = list(range(0, ng))
        largeFaceSendNodesSlices = list(range(-(2 * ng + 1), -(ng + 1)))
        largeFaceRecvNodesSlices = list(range(-ng, 0))

        # List of cc halo indices
        smallFaceSendCcAll = list(range(ng, 2 * ng))
        smallFaceRecvCcAll = list(range(0, ng))
        largeFaceSendCcAll = list(range(-2 * ng, -ng))
        largeFaceRecvCcAll = list(range(-ng, 0))

        if self.nface == 1:
            self.nodeSendSlices = [s_[i, :, :] for i in smallFaceSendNodesSlices]
            self.ccSendAllSlices = [s_[i, :, :, :] for i in smallFaceSendCcAll]
            nodeShape = (ng, nj + 2 * ng, nk + 2 * ng)
            ccShape = (ng, nj + 2 * ng - 1, nk + 2 * ng - 1)
            self.nodeRecvSlices = [s_[i, :, :] for i in smallFaceRecvNodesSlices]
            self.ccRecvAllSlices = [s_[i, :, :, :] for i in smallFaceRecvCcAll]
        elif self.nface == 2:
            self.nodeSendSlices = [
                s_[ni + 2 * ng + i, :, :] for i in largeFaceSendNodesSlices
            ]
            self.ccSendAllSlices = [
                s_[ni + 2 * ng + i - 1, :, :, :] for i in largeFaceSendCcAll
            ]
            nodeShape = (ng, nj + 2 * ng, nk + 2 * ng)
            ccShape = (ng, nj + 2 * ng - 1, nk + 2 * ng - 1)
            self.nodeRecvSlices = [
                s_[ni + 2 * ng + i, :, :] for i in largeFaceRecvNodesSlices
            ]
            self.ccRecvAllSlices = [
                s_[ni + 2 * ng + i - 1, :, :, :] for i in largeFaceRecvCcAll
            ]
        elif self.nface == 3:
            self.nodeSendSlices = [s_[:, i, :] for i in smallFaceSendNodesSlices]
            self.ccSendAllSlices = [s_[:, i, :, :] for i in smallFaceSendCcAll]
            nodeShape = (ng, ni + 2 * ng, nk + 2 * ng)
            ccShape = (ng, ni + 2 * ng - 1, nk + 2 * ng - 1)
            self.nodeRecvSlices = [s_[:, i, :] for i in smallFaceRecvNodesSlices]
            self.ccRecvAllSlices = [s_[:, i, :, :] for i in smallFaceRecvCcAll]
        elif self.nface == 4:
            self.nodeSendSlices = [
                s_[:, nj + 2 * ng + i, :] for i in largeFaceSendNodesSlices
            ]
            self.ccSendAllSlices = [
                s_[:, nj + 2 * ng + i - 1, :, :] for i in largeFaceSendCcAll
            ]
            nodeShape = (ng, ni + 2 * ng, nk + 2 * ng)
            ccShape = (ng, ni + 2 * ng - 1, nk + 2 * ng - 1)
            self.nodeRecvSlices = [
                s_[:, nj + 2 * ng + i, :] for i in largeFaceRecvNodesSlices
            ]
            self.ccRecvAllSlices = [
                s_[:, nj + 2 * ng + i - 1, :, :] for i in largeFaceRecvCcAll
            ]
        elif self.nface == 5:
            self.nodeSendSlices = [s_[:, :, i] for i in smallFaceSendNodesSlices]
            self.ccSendAllSlices = [s_[:, :, i, :] for i in smallFaceSendCcAll]
            nodeShape = (ng, ni + 2 * ng, nj + 2 * ng)
            ccShape = (ng, ni + 2 * ng - 1, nj + 2 * ng - 1)
            self.nodeRecvSlices = [s_[:, :, i] for i in smallFaceRecvNodesSlices]
            self.ccRecvAllSlices = [s_[:, :, i, :] for i in smallFaceRecvCcAll]
        elif self.nface == 6:
            self.nodeSendSlices = [
                s_[:, :, nk + 2 * ng + i] for i in largeFaceSendNodesSlices
            ]
            self.ccSendAllSlices = [
                s_[:, :, nk + 2 * ng + i - 1, :] for i in largeFaceSendCcAll
            ]
            nodeShape = (ng, ni + 2 * ng, nj + 2 * ng)
            ccShape = (ng, ni + 2 * ng - 1, nj + 2 * ng - 1)
            self.nodeRecvSlices = [
                s_[:, :, nk + 2 * ng + i] for i in largeFaceRecvNodesSlices
            ]
            self.ccRecvAllSlices = [
                s_[:, :, nk + 2 * ng + i - 1, :] for i in largeFaceRecvCcAll
            ]

        # We only need first halo slice for some variables (dqdxyz,phi)
        # make sure we pick up the correct send face, this must be done
        # before the send faces are reversed!
        if self.nface in [1, 3, 5]:
            self.ccSendFirstHaloSlice = [self.ccSendAllSlices[0]]
            self.ccRecvFirstHaloSlice = [self.ccRecvAllSlices[-1]]
        elif self.nface in [2, 4, 6]:
            self.ccSendFirstHaloSlice = [self.ccSendAllSlices[-1]]
            self.ccRecvFirstHaloSlice = [self.ccRecvAllSlices[0]]

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
            self.nodeSendSlices.reverse()
            self.ccSendAllSlices.reverse()

        # We send the data in the correct shape already
        # Node shape
        temp = self.orient(np.empty(nodeShape[1::]))
        for var in ["x", "y", "z"]:
            sendName = "sendBuffer_" + var
            self.array[sendName] = np.ascontiguousarray(
                np.empty(tuple([ng]) + temp.shape)
            )
            shape = self.array[sendName].shape
            createViewMirrorArray(self, sendName, shape)

            # We recieve the data in the correct shape already
            recvName = "recvBuffer_" + var
            self.array[recvName] = np.ascontiguousarray(np.empty(nodeShape))
            shape = self.array[recvName].shape
            createViewMirrorArray(self, recvName, shape)

            # We use temporary buffers for some storage
            tempRecvName = "tempRecvBuffer_" + var
            self.array[tempRecvName] = np.ascontiguousarray(np.empty(nodeShape))
            shape = self.array[tempRecvName].shape
            createViewMirrorArray(self, tempRecvName, shape)

        # Cell Center, ne vars
        for var in ["q", "Q", "dqdx", "dqdy", "dqdz", "phi"]:

            # Only "q" and "Q" need ALL ghost layers
            if var in ["q", "Q"]:
                nLayers = ng
            else:
                nLayers = 1
            # phi only uses 3 last exis elements not ne
            if var in ["phi"]:
                nE = 3
            else:
                nE = ne

            temp = self.orient(np.empty(ccShape[1::] + tuple([nE])))
            sendName = "sendBuffer_" + var
            self.array[sendName] = np.ascontiguousarray(
                np.empty(tuple([nLayers]) + temp.shape)
            )
            shape = self.array[sendName].shape
            createViewMirrorArray(self, sendName, shape)

            # We revieve the data in the correct shape already
            recvName = "recvBuffer_" + var
            self.array[recvName] = np.ascontiguousarray(np.empty(ccShape + tuple([nE])))
            shape = self.array[recvName].shape
            createViewMirrorArray(self, recvName, shape)

            # We use temporary buffers for some storage
            tempRecvName = "tempRecvBuffer_" + var
            self.array[tempRecvName] = np.ascontiguousarray(
                np.empty(ccShape + tuple([nE]))
            )
            shape = self.array[tempRecvName].shape
            createViewMirrorArray(self, tempRecvName, shape)

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

        # What are the orientations of our neighbor face plane? i.e.
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
        return np.moveaxis(temp, (0, 1), (1, 0))

    @staticmethod
    def orientTf0(temp):
        return np.flip(np.moveaxis(temp, (0, 1), (1, 0)), 0)

    @staticmethod
    def orientTf1(temp):
        return np.flip(np.moveaxis(temp, (0, 1), (1, 0)), 1)

    @staticmethod
    def orientTf0f1(temp):
        return np.flip(np.moveaxis(temp, (0, 1), (1, 0)), (0, 1))

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
