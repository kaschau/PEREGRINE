from mpi4py import MPI


def getCommRankSize():
    comm = MPI.COMM_WORLD
    return comm, comm.rank, comm.size


def getRanksPerNode():
    """Gives how many ranks share a node, taken as the max so every rank
    agrees even when the last node is only partly filled."""
    comm, rank, size = getCommRankSize()
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    ranksPerNode = comm.allreduce(node.size, op=MPI.MAX)
    node.Free()
    return ranksPerNode
