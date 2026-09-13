from mpi4py import MPI  # noqa: F401


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
