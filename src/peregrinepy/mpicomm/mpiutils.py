from mpi4py import MPI


def getCommRankSize():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    return comm, rank, size
