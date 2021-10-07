from .mpiUtils import getCommRankSize
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request


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
                blk.array[var][sliceR] = recv[:]
                count += 1

        comm.Barrier()


def setBlockCommunication(mb):

    for blk in mb:
        for face in blk.faces:
            neighbor = face.connectivity["neighbor"]
            if neighbor is None:
                continue
            face.setOrientFunc(blk.ni, blk.nj, blk.nk, blk.ne)
            face.setCommBuffers(blk.ni, blk.nj, blk.nk, blk.ne)
