from .mpiUtils import getCommRankSize
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request
from ..compute.utils import (
    extract_sendBuffer3,
    extract_sendBuffer4,
    place_recvBuffer3,
    place_recvBuffer4,
)


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
                recvName = "tempRecBuffer_" + var
                if var in ["Q", "q"]:
                    sliceS = face.ccSendAllSlices
                    extract = extract_sendBuffer4
                elif var in ["dqdx", "dqdy", "dqdz", "phi"]:
                    sliceS = face.ccSendFirstHaloSlice
                    extract = extract_sendBuffer4
                elif var in ["x", "y", "z"]:
                    sliceS = face.nodeSendSlices
                    extract = extract_sendBuffer3

                # Get the indices of the send slices from the numpy slice object
                sliceIndxs = [s for f in sliceS for s in f if type(s) is int]
                # populate the temp recv array with the unoriented send data, since its
                # the correct size and shape
                extract(getattr(blk, var), face, sliceIndxs)
                # update the device temp recv buffer
                face.updateHostView(recvName)
                # Now, orient each send slice and place in send buffer
                for i in range(face.ng):
                    send[i] = face.orient(face.array[recvName][i])
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

                recvName = "recvBuffer_" + var
                if var in ["Q", "q"]:
                    sliceR = face.ccRecvFirstHaloSlice
                    place = place_recvBuffer4
                elif var in ["dqdx", "dqdy", "dqdz", "phi"]:
                    sliceR = face.ccRecvAllSlices
                    place = place_recvBuffer4
                elif var in ["x", "y", "z"]:
                    sliceR = face.nodeRecvSlices
                    place = place_recvBuffer3

                # Push back up the device
                face.updateDeviceView(recvName)
                # Get the indices of the send slices from the numpy slice object
                sliceIndxs = [s for f in sliceR for s in f if type(s) is int]
                # Place the recv in the view
                place(getattr(blk, var), face, sliceIndxs)

        comm.Barrier()
