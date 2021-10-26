from mpi4py import MPI
import numpy as np


def getCommRankSize():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    return comm, rank, size


def getNumCells(mb):

    comm, rank, size = getCommRankSize()

    myCells = 0
    for blk in mb:
        myCells += (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)

    nCells = comm.allreduce(myCells, op=MPI.SUM)

    return nCells


def getLoadEfficiency(mb):

    comm, rank, size = getCommRankSize()

    myCells = np.array([0.0])
    for blk in mb:
        myCells[0] += (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)

    recv = None
    if rank == 0:
        recv = np.empty(size)
    comm.Gather(myCells, recv, root=0)

    if rank == 0:
        perfect = np.mean(recv)
        slowest = perfect / np.max(recv) * 100.0
        slowestProc = np.argmax(recv)
    else:
        slowest = None
        slowestProc = None

    return slowest, slowestProc
