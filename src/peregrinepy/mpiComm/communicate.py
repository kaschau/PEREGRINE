from .mpiUtils import getCommRankSize
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request
from ..compute.utils import extractSendBuffer, placeRecvBuffer


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
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                send = face.array["sendBuffer_" + var]
                recvName = "tempRecvBuffer_" + var
                if var in ["Q", "q"]:
                    sliceS = face.ccSendAllSlices
                elif var in ["dqdx", "dqdy", "dqdz", "phi"]:
                    sliceS = face.ccSendFirstHaloSlice
                elif var in ["x", "y", "z"]:
                    sliceS = face.nodeSendSlices

                # Get the indices of the send slices from the numpy slice object
                sliceIndxs = [s for f in sliceS for s in f if type(s) is int]
                # populate the temp recv array with the unoriented send data, since its
                # the correct size and shape
                extractSendBuffer(
                    getattr(blk, var), getattr(face, recvName), face, sliceIndxs
                )
                # update the device temp recv buffer
                face.updateHostView(recvName)
                # Now, orient each send slice and place in send buffer
                for i in range(len(sliceIndxs)):
                    send[i] = face.orient(face.array[recvName][i])
                ssize = send.size
                comm.Send([send, ssize, MPIDOUBLE], dest=face.commRank, tag=face.tagS)

        # wait and update
        reqs = iter(reqs)
        for blk in mb:
            for face in blk.faces:
                if face.neighbor is None:
                    continue
                Request.Wait(reqs.__next__())

                recvName = "recvBuffer_" + var
                if var in ["Q", "q"]:
                    sliceR = face.ccRecvAllSlices
                elif var in ["dqdx", "dqdy", "dqdz", "phi"]:
                    sliceR = face.ccRecvFirstHaloSlice
                elif var in ["x", "y", "z"]:
                    sliceR = face.nodeRecvSlices

                # Push back up the device
                face.updateDeviceView(recvName)
                # Get the indices of the send slices from the numpy slice object
                sliceIndxs = [s for f in sliceR for s in f if type(s) is int]
                # Place the recv in the view
                placeRecvBuffer(
                    getattr(blk, var), getattr(face, recvName), face, sliceIndxs
                )

        comm.Barrier()
