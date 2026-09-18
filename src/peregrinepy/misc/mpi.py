from mpi4py import MPI


def getCommRankSize():
    comm = MPI.COMM_WORLD
    return comm, comm.rank, comm.size
