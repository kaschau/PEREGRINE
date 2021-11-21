from mpi4py import MPI
from ..compute.utils import CFLmax
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
        dt = min(cflMAX / cfl[0], cflMAX / cfl[1])
    else:
        dt = mb.config["simulation"]["dt"]

    return dt, cfl[0], cfl[1]


def checkNan(mb):

    comm, rank, size = getCommRankSize()

    abort = np.array([0], np.int32)
    for blk in mb:
        blk.updateHostView("Q")
        if np.any(np.isnan(blk.array["Q"])):
            print(f"nan detected in block {blk.nblki}")
            print(np.argwhere(np.isnan(blk.array["Q"])))
            abort[0] += 1
        if np.any(np.isinf(blk.array["Q"])):
            print(f"inf detected in block {blk.nblki}")
            print(np.argwhere(np.isinf(blk.array["Q"])))
            abort[0] += 1

    comm.Allreduce(MPI.IN_PLACE, abort, op=MPI.SUM)

    return abort[0]
