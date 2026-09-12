from mpi4py import MPI  # noqa: F401
import numpy as np


def getCommRankSize():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    return comm, rank, size


def getRanksPerNode():
    """How many ranks share a node. Taken as the max so every rank agrees even
    when the last node is only partly filled."""
    comm, rank, size = getCommRankSize()
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    ranksPerNode = comm.allreduce(node.size, op=MPI.MAX)
    node.Free()
    return ranksPerNode


def getNumCells(mb):
    comm, rank, size = getCommRankSize()

    nCells = np.array([0], dtype=np.int32)
    for blk in mb:
        nCells[0] += (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)

    comm.Allreduce(MPI.IN_PLACE, nCells, op=MPI.SUM)

    return nCells[0]


def getLoadEfficiency(mb):
    comm, rank, size = getCommRankSize()

    myCells = np.array([0], dtype=np.int32)
    for blk in mb:
        myCells[0] += (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)

    recv = None
    if rank == 0:
        recv = np.empty(size, dtype=np.int32)
    comm.Gather(myCells, recv, root=0)

    if rank == 0:
        perfect = np.mean(recv)
        slowest = perfect / np.max(recv) * 100.0
        slowestProc = np.argmax(recv)
    else:
        slowest = None
        slowestProc = None

    return slowest, slowestProc


def getDtMaxCFL(mb):
    comm, rank, size = getCommRankSize()

    # the max over this rank's blocks, then over ranks; the convective floor
    # keeps the time step finite in a quiescent field
    cfl = np.zeros(3)
    mb.CFLmax(cfl=cfl)
    cfl[1] = max(cfl[1], 1e-16)
    comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)

    if mb.config["timeIntegration"]["variableTimeStep"]:
        cflMAX = mb.config["timeIntegration"]["maxCFL"]
        dt = min(cflMAX / cfl[2], mb.config["timeIntegration"]["maxDt"])
    else:
        dt = mb.config["timeIntegration"]["dt"]

    return dt, cfl[0], cfl[1], cfl[2]


def checkForNan(mb):
    comm, rank, size = getCommRankSize()

    abort = np.array([0], np.int32)
    abort[0] = not mb.allFinite()
    if abort[0] > 0:
        for blk in mb:
            Q = blk.Q.get()
            ng = blk.ng
            nans = np.where(np.sum(np.isnan(Q[ng:-ng, ng:-ng, ng:-ng, :]), axis=-1) > 0)
            if len(nans[0]) == 0:
                continue
            with open(f"nans_{blk.nblki}.log", "w") as f:
                f.write(f"Nan Detection Log: Block {blk.nblki}\n")
                xs = blk.cells[..., 0][ng:-ng, ng:-ng, ng:-ng][nans]
                ys = blk.cells[..., 1][ng:-ng, ng:-ng, ng:-ng][nans]
                zs = blk.cells[..., 2][ng:-ng, ng:-ng, ng:-ng][nans]
                for x, y, z in zip(xs, ys, zs):
                    f.write(f"x = {x} y = {y} z = {z}\n")

    comm.Allreduce(MPI.IN_PLACE, abort, op=MPI.SUM)

    return abort[0]
