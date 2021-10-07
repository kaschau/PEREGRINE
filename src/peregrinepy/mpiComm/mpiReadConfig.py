from ..readers import readConfigFile
from .mpiUtils import getCommRankSize


def mpiReadConfig(file_path):

    comm, rank, size = getCommRankSize()
    if rank == 0:
        config = readConfigFile(file_path)
    else:
        config = None

    config = comm.bcast(config, root=0)
    comm.Barrier()

    return config
