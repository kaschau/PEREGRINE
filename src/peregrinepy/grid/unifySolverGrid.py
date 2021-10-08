from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request
from .. import mpiComm


def unifySolverGrid(mb):

    for blk in mb:
        blk.generateHalo()

    # Lets just be clean and create the edges and corners
    for _ in range(3):
        mpiComm.blockComm.communicate(mb, ["x", "y", "z"])

    comm, rank, size = mpiComm.mpiUtils.getCommRankSize()

    for var in ["x", "y", "z"]:
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

                    ssize = face.recvBuffer3.size
                    reqs.append(
                        comm.Irecv(
                            [face.recvBuffer3[:], ssize, MPIDOUBLE],
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
                    tag = int(f"1{blk.nblki}2{neighbor}1{face.neighborFace}")
                    face.sendBuffer3[:] = face.orient(
                        blk.array[var][face.s2_] - blk.array[var][face.s1_]
                    )
                    ssize = face.sendBuffer3.size
                    comm.Send(
                        [face.sendBuffer3, ssize, MPIDOUBLE], dest=commRank, tag=tag
                    )

            # wait and assign
            count = 0
            for blk in mb:
                for face in blk.faces:
                    bc = face.bcType
                    if bc != "b1":
                        continue
                    Request.Wait(reqs[count])
                    blk.array[var][face.s0_] = (
                        blk.array[var][face.s1_] + face.recvBuffer3[:]
                    )
                    count += 1

            comm.Barrier()
