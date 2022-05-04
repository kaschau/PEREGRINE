from mpi4py import MPI
from ..compute.utils import CFLmax, checkNan
import numpy as np


def getCommRankSize():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    return comm, rank, size


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

    cfl = np.array(CFLmax(mb), dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)

    if mb.config["simulation"]["variableTimeStep"]:
        cflMAX = mb.config["simulation"]["maxCFL"]
        dt = min(cflMAX / cfl[2], mb.config["simulation"]["maxDt"])

        mb.config["simulation"]["dt"] = dt
    else:
        dt = mb.config["simulation"]["dt"]

    return dt, cfl[0], cfl[1], cfl[2]


def checkForNan(mb):

    comm, rank, size = getCommRankSize()

    abort = np.array([0], np.int32)
    abort[0] = checkNan(mb)
    if abort[0] > 0:
        for blk in mb:
            blk.updateHostView(["Q"])
            with open(f"nans_{blk.nblki}.log", "w") as f:
                f.write(f"Nan Detection Log: Block {blk.nblki}\n")
            for i in range(blk.ne):
                ng = blk.ng
                nans = np.where(np.isnan(blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, i]))
                xs = blk.array["xc"][ng:-ng, ng:-ng, ng:-ng][nans]
                ys = blk.array["yc"][ng:-ng, ng:-ng, ng:-ng][nans]
                zs = blk.array["zc"][ng:-ng, ng:-ng, ng:-ng][nans]
                with open(f"nans_{blk.nblki}.log", "a") as f:
                    for x, y, z in zip(xs, ys, zs):
                        f.write(f"x = {x}, y = {y}, z = {z}, Qindex = {i}\n")

    comm.Allreduce(MPI.IN_PLACE, abort, op=MPI.SUM)

    return abort[0]
