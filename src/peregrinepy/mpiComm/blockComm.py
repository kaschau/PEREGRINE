from .mpiUtils import getCommRankSize
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request


def communicate(mb, varis):

    varis = list(varis)
    comm, rank, size = getCommRankSize()

    for var in varis:
        reqs = []
        # Post non-blocking recieves
        for blk in mb:
            ndim = blk.array[var].ndim
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                recv = face.recvBuffer4 if ndim == 4 else face.recvBuffer3
                ssize = recv.size
                reqs.append(
                    comm.Irecv(
                        [recv, ssize, MPIDOUBLE], source=face.commRank, tag=face.tagR
                    )
                )

        # Post non-blocking sends
        for blk in mb:
            ndim = blk.array[var].ndim
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                send, sliceS = (
                    (face.sendBuffer4, face.sliceS4)
                    if ndim == 4
                    else (face.sendBuffer3, face.sliceS3)
                )
                for i, sS in enumerate(sliceS):
                    send[i] = face.orient(blk.array[var][sS])
                ssize = send.size
                comm.Send([send, ssize, MPIDOUBLE], dest=face.commRank, tag=face.tagS)

        # wait and assign
        reqs = iter(reqs)
        for blk in mb:
            ndim = blk.array[var].ndim
            for face in blk.faces:
                if face.neighbor is None:
                    continue
                Request.Wait(reqs.__next__())
                recv, sliceR = (
                    (face.recvBuffer4, face.sliceR4)
                    if ndim == 4
                    else (face.recvBuffer3, face.sliceR3)
                )
                for i, sR in enumerate(sliceR):
                    blk.array[var][sR] = recv[i]

        comm.Barrier()


def setBlockCommunication(mb):

    for blk in mb:
        for face in blk.faces:
            if face.neighbor is None:
                continue
            face.setOrientFunc(blk.ni, blk.nj, blk.nk, blk.ne)
            face.setCommBuffers(blk.ni, blk.nj, blk.nk, blk.ne, blk.nblki)
