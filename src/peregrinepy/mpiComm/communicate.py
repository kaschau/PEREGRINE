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
            ndim = blk.array[var].ndim
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                recv = (
                    face.array["recvBuffer4"]
                    if ndim == 4
                    else face.array["recvBuffer3"]
                )
                ssize = recv.size
                reqs.append(
                    comm.Irecv(
                        [recv, ssize, MPIDOUBLE], source=face.commRank, tag=face.tagR
                    )
                )

        # Post non-blocking sends
        for blk in mb:
            # Need to update host data
            ndim = blk.array[var].ndim
            for face in blk.faces:
                if face.neighbor is None:
                    continue

                if ndim == 4:
                    send = face.array["sendBuffer4"]
                    recv = "tempRecvBuffer4"
                    sliceS = face.sliceS4
                    extract = extract_sendBuffer4
                else:
                    send = face.array["sendBuffer3"]
                    recv = "tempRecvBuffer3"
                    sliceS = face.sliceS3
                    extract = extract_sendBuffer3
                # Get the indices of the send slices from the numpy slice object
                sliceIndxs = [s for f in sliceS for s in f if type(s) is int]
                # populate the recv array with the unoriented send data, since its
                # the correct size and not need right now
                extract(getattr(blk, var), face, sliceIndxs)
                # update the device recv buffer
                face.updateHostView(recv)
                # Now, orient each send slice and place in send buffer
                for i in range(face.ng):
                    send[i] = face.orient(face.array[recv][i])
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
                if ndim == 4:
                    recv = "recvBuffer4"
                    sliceR = face.sliceR4
                    place = place_recvBuffer4
                else:
                    recv = "recvBuffer3"
                    sliceR = face.sliceR3
                    place = place_recvBuffer3
                # Push back up the device
                face.updateDeviceView(recv)
                # Get the indices of the send slices from the numpy slice object
                sliceIndxs = [s for f in sliceR for s in f if type(s) is int]
                # Place the recv in the view
                place(getattr(blk, var), face, sliceIndxs)

        comm.Barrier()
