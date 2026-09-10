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
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                send = face.array["sendBuffer_" + var]
                for i, sS in enumerate(face.sendSlices(var)):
                    send[i] = face.orient(blk.array[var][sS])
                ssize = send.size
                comm.Send([send, ssize, MPIDOUBLE], dest=face.commRank, tag=face.tagS)

        # wait and assign
        reqs = iter(reqs)
        for blk in mb:
            for face in blk.faces:
                if face.neighbor is None:
                    continue
                Request.Wait(reqs.__next__())
                recv = face.array["recvBuffer_" + var]
                for i, sR in enumerate(face.recvSlices(var)):
                    blk.array[var][sR] = recv[i]
            # Push back up the device
            blk.updateDeviceView(var)

        comm.Barrier()
