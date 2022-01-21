from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request
from .. import mpiComm


def unifySolverGrid(mb):

    for blk in mb:
        assert blk.blockType == "solver", "Only solverBlocks can be unified"
        blk.generateHalo()

    # Lets just be clean and create the edges and corners
    for _ in range(3):
        mpiComm.communicate(mb, ["x", "y", "z"])

    comm, rank, size = mpiComm.mpiUtils.getCommRankSize()

    for var in ["x", "y", "z"]:
        for blk in mb:
            # Need to update host data
            blk.updateHostView(var)
        for _ in range(3):
            reqs = []
            # Post non-blocking recieves
            for blk in mb:
                for face in blk.faces:
                    bc = face.bcType
                    if bc != "b1":
                        continue
                    neighbor = face.neighbor
                    commRank = face.commRank
                    tag = int(f"1{neighbor}2{blk.nblki}1{face.nface}")

                    ssize = face.array["recvBuffer3"].size
                    reqs.append(
                        comm.Irecv(
                            [face.array["recvBuffer3"][:], ssize, MPIDOUBLE],
                            source=commRank,
                            tag=tag,
                        )
                    )

            # Post non-blocking sends
            for blk in mb:
                for face in blk.faces:
                    bc = face.bcType
                    if bc != "b1":
                        continue
                    neighbor = face.neighbor
                    commRank = face.commRank
                    tag = int(f"1{blk.nblki}2{neighbor}1{face.neighborNface}")
                    for i, sS in enumerate(face.sliceS3):
                        face.array["sendBuffer3"][i] = face.orient(
                            blk.array[var][sS] - blk.array[var][face.s1_]
                        )
                    ssize = face.array["sendBuffer3"].size
                    comm.Send(
                        [face.array["sendBuffer3"], ssize, MPIDOUBLE],
                        dest=commRank,
                        tag=tag,
                    )

            # wait and assign
            reqs = iter(reqs)
            for blk in mb:
                for face in blk.faces:
                    bc = face.bcType
                    if bc != "b1":
                        continue
                    Request.Wait(reqs.__next__())
                    for i, sR in enumerate(face.sliceR3):
                        blk.array[var][sR] = (
                            blk.array[var][face.s1_] + face.array["recvBuffer3"][i]
                        )

            comm.Barrier()

        for blk in mb:
            # Push back up the device
            blk.updateDeviceView(var)
