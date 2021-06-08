from mpi4py import MPI
from ..files import config_file
from ..readers import read_config_file

def init_config(file_path):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        config = read_config_file(file_path)

    else:
        config = None

    config = comm.bcast(config,root=0)

    return config
