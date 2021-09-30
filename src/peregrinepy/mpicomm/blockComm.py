from .mpiutils import getCommRankSize
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request
import numpy as np


def communicate(mb, varis):
    if not isinstance(varis, list):
        varis = [varis]

    comm, rank, size = getCommRankSize()

    for var in varis:
        reqs = []
        # Post non-blocking recieves
        for blk in mb:
            ndim = blk.array[var].ndim
            for face in blk.faces:
                neighbor = face.connectivity["neighbor"]
                if neighbor is None:
                    continue
                commRank = face.commRank
                tag = int(f"1{neighbor}2{blk.nblki}1{face.nface}")

                recv = face.recvBuffer4 if ndim == 4 else face.recvBuffer3
                ssize = recv.size
                reqs.append(
                    comm.Irecv([recv, ssize, MPIDOUBLE], source=commRank, tag=tag)
                )

        # Post non-blocking sends
        for blk in mb:
            ndim = blk.array[var].ndim
            for face in blk.faces:
                neighbor = face.connectivity["neighbor"]
                if neighbor is None:
                    continue
                commRank = face.commRank
                tag = int(f"1{blk.nblki}2{neighbor}1{face.neighborFace}")

                send, sliceS = (
                    (face.sendBuffer4, face.sliceS4)
                    if ndim == 4
                    else (face.sendBuffer3, face.sliceS3)
                )
                send[:] = face.orient(blk.array[var][sliceS])
                ssize = send.size
                comm.Send([send, ssize, MPIDOUBLE], dest=commRank, tag=tag)

        # wait and assign
        count = 0
        for blk in mb:
            ndim = blk.array[var].ndim
            for face in blk.faces:
                neighbor = face.connectivity["neighbor"]
                if neighbor is None:
                    continue
                Request.Wait(reqs[count])
                recv, sliceR = (
                    (face.recvBuffer4, face.sliceR4)
                    if ndim == 4
                    else (face.recvBuffer3, face.sliceR3)
                )
                blk.array[var][sliceR] = recv
                count += 1

        comm.Barrier()


def setBlockCommunication(mb):
    assert 0 not in [
        mb[0].ni,
        mb[0].nj,
        mb[0].nk,
    ], "Must get grid before setting block communicaitons."

    from numpy import s_

    ##########################################################
    # Define the possible reorientation routines
    ##########################################################
    def orientT(temp):
        axes = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.transpose(temp, axes)

    def orientTf0(temp):
        axT = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.flip(np.transpose(temp, axT), 0)

    def orientTf1(temp):
        axT = (1, 0, 2) if temp.ndim == 3 else (1, 0)
        return np.flip(np.transpose(temp, axT), 1)

    def orientTf0f1(temp):
        return np.flip(np.flip(temp, 0), 1)

    def orientf0(temp):
        return np.flip(temp, 0)

    def orientf0f1(temp):
        return np.flip(np.flip(temp, 0), 1)

    def orientf1(temp):
        return np.flip(temp, 1)

    def orientNull(temp):
        return temp

    ##########################################################
    # Define the mapping based on orientation
    ##########################################################
    faceToOrientPlaceMapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    orientToSmallFaceMapping = {1: 2, 2: 4, 3: 6, 4: 1, 5: 3, 6: 5}
    orientToLargeFaceMapping = {1: 1, 2: 3, 3: 5, 4: 2, 5: 4, 6: 6}

    largeIndexMapping = {0: "k", 1: "k", 2: "j"}
    needToTranspose = {
        "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
        "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
    }

    def getNeighborFace(nface, orientation):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        direction = int(orientation[faceToOrientPlaceMapping[nface]])

        if nface in [2, 4, 6]:
            nface2 = orientToLargeFaceMapping[direction]
        elif nface in [1, 3, 5]:
            nface2 = orientToSmallFaceMapping[direction]

        return nface2

    ##########################################################
    # Get the neighbor orientation opposite of each face
    ##########################################################
    comm, rank, size = getCommRankSize()
    for blk in mb:
        for face in blk.faces:
            neighbor = face.connectivity["neighbor"]
            if neighbor is None:
                continue
            orientation = face.connectivity["orientation"]
            commRank = face.commRank
            face.neighborFace = getNeighborFace(face.nface, orientation)
            tag = int(f"1{blk.nblki}2{neighbor}1{face.neighborFace}")
            comm.isend(orientation, dest=commRank, tag=tag)
    neighborOrientations = []
    for blk in mb:
        neighborOrientations.append({})
        for face in blk.faces:
            neighbor = face.connectivity["neighbor"]
            if neighbor is None:
                continue
            orientation = face.connectivity["orientation"]
            commRank = face.commRank
            tag = int(f"1{neighbor}2{blk.nblki}1{face.nface}")
            neighborOrientations[-1][face.nface] = comm.recv(source=commRank, tag=tag)

    comm.Barrier()

    for blk, no in zip(mb, neighborOrientations):
        sliceSfp = {}
        sliceSc = {}
        commfpshape = {}
        commcshape = {}
        sliceRfp = {}
        sliceRc = {}

        sliceSfp[1] = s_[2, :, :]
        sliceSc[1] = s_[1, :, :, :]
        commfpshape[1] = (blk.nj + 2, blk.nk + 2)
        commcshape[1] = (blk.nj + 1, blk.nk + 1, blk.ne)
        sliceRfp[1] = s_[0, :, :]
        sliceRc[1] = s_[0, :, :, :]

        sliceSfp[2] = s_[-3, :, :]
        sliceSc[2] = s_[-2, :, :, :]
        commfpshape[2] = commfpshape[1]
        commcshape[2] = commcshape[1]
        sliceRfp[2] = s_[-1, :, :]
        sliceRc[2] = s_[-1, :, :, :]

        sliceSfp[3] = s_[:, 2, :]
        sliceSc[3] = s_[:, 1, :, :]
        commfpshape[3] = (blk.ni + 2, blk.nk + 2)
        commcshape[3] = (blk.ni + 1, blk.nk + 1, blk.ne)
        sliceRfp[3] = s_[:, 0, :]
        sliceRc[3] = s_[:, 0, :, :]

        sliceSfp[4] = s_[:, -3, :]
        sliceSc[4] = s_[:, -2, :, :]
        commfpshape[4] = commfpshape[3]
        commcshape[4] = commcshape[3]
        sliceRfp[4] = s_[:, -1, :]
        sliceRc[4] = s_[:, -1, :, :]

        sliceSfp[5] = s_[:, :, 2]
        sliceSc[5] = s_[:, :, 1]
        commfpshape[5] = (blk.ni + 2, blk.nj + 2)
        commcshape[5] = (blk.ni + 1, blk.nj + 1, blk.ne)
        sliceRfp[5] = s_[:, :, 0]
        sliceRc[5] = s_[:, :, 0, :]

        sliceSfp[6] = s_[:, :, -3]
        sliceSc[6] = s_[:, :, -2, :]
        commfpshape[6] = commfpshape[5]
        commcshape[6] = commcshape[5]
        sliceRfp[6] = s_[:, :, -1]
        sliceRc[6] = s_[:, :, -1, :]

        for face in blk.faces:
            neighbor = face.connectivity["neighbor"]
            if neighbor is None:
                pass
            else:
                face.sliceS3 = sliceSfp[face.nface]
                face.sliceR3 = sliceRfp[face.nface]

                face.sliceS4 = sliceSc[face.nface]
                face.sliceR4 = sliceRc[face.nface]

                orientation = face.connectivity["orientation"]
                neighbor = face.connectivity["neighbor"]

                face2 = face.neighborFace

                faceOrientations = [
                    int(i)
                    for j, i in enumerate(orientation)
                    if j != faceToOrientPlaceMapping[face.nface]
                ]
                normalIndex = [
                    j for j in range(3) if j == faceToOrientPlaceMapping[face.nface]
                ][0]
                normalIndex2 = [
                    j for j in range(3) if j == faceToOrientPlaceMapping[face2]
                ][0]

                bigIndex = largeIndexMapping[normalIndex]
                bigIndex2 = largeIndexMapping[normalIndex2]

                # Do we need to transpoze?
                if faceOrientations[1] in needToTranspose[bigIndex][bigIndex2]:
                    # Do we need to flip along 0 axis?
                    if faceOrientations[1] in [4, 5, 6]:
                        # Do we need to flip along 1 axis?
                        if faceOrientations[0] in [4, 5, 6]:
                            # Then do all three
                            face.orient = orientTf0f1
                        else:
                            # Then do just T and flip0
                            face.orient = orientTf0
                    else:
                        # Do we need to flip along 1 axis?
                        if faceOrientations[0] in [4, 5, 6]:
                            # Then do just T and flip1
                            face.orient = orientTf1
                        else:
                            # Then do just T
                            face.orient = orientT
                else:
                    # Do we need to flip along 0 axis?
                    if faceOrientations[1] in [4, 5, 6]:
                        # Do we need to flip along 1 axis?
                        if faceOrientations[0] in [4, 5, 6]:
                            # Then do just flip0 and flip1
                            face.orient = orientf0f1
                        else:
                            # Then do just flip0
                            face.orient = orientf0
                    elif faceOrientations[0] in [4, 5, 6]:
                        # Then do just flip1
                        face.orient = orientf1
                    else:
                        # Then do nothing
                        face.orient = orientNull

                # We send the data in the correct shape already
                # Face and point shape
                temp = face.orient(np.empty(commfpshape[face.nface]))
                face.sendBuffer3 = np.ascontiguousarray(temp)
                # We revieve the data in the correct shape already
                temp = np.empty(commfpshape[face.nface])
                face.recvBuffer3 = np.ascontiguousarray(
                    np.empty(commfpshape[face.nface])
                )

                # Cell
                temp = face.orient(np.empty(commcshape[face.nface]))
                face.sendBuffer4 = np.ascontiguousarray(temp)
                # We revieve the data in the correct shape already
                temp = np.empty(commcshape[face.nface])
                face.recvBuffer4 = np.ascontiguousarray(
                    np.empty(commcshape[face.nface])
                )
