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

                recv = face.array["recvBuffer_" + var]
                ssize = recv.size
                reqs.append(
                    comm.Irecv(
                        [recv, ssize, MPIDOUBLE], source=face.commRank, tag=face.tagR
                    )
                )

        # Post non-blocking sends
        for blk in mb:
            blk.updateHostView(var)
            ndim = blk.array[var].ndim
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                send = face.array["sendBuffer_" + var]
                if var in ["Q", "q"]:
                    sliceS = face.ccSendAllSlices
                elif var in ["dqdx", "dqdy", "dqdz", "phi"]:
                    sliceS = face.ccSendFirstHaloSlice
                elif var in ["x", "y", "z"]:
                    sliceS = face.nodeSendSlices
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
                recv = face.array["recvBuffer_" + var]
                if var in ["Q", "q"]:
                    sliceR = face.ccRecvFirstHaloSlice
                elif var in ["dqdx", "dqdy", "dqdz", "phi"]:
                    sliceR = face.ccRecvAllSlices
                elif var in ["x", "y", "z"]:
                    sliceR = face.nodeRecvSlices
                for i, sR in enumerate(sliceR):
                    blk.array[var][sR] = recv[i]
            # Push back up the device
            blk.updateDeviceView(var)

        comm.Barrier()
