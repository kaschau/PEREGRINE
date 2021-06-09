from ..files import config_file
from ..readers import read_config_file
from ..mpicomm import mpiutils

def init_config(file_path):

    comm,rank,size = mpiutils.get_comm_rank_size()
    if rank == 0:
        config = read_config_file(file_path)

    else:
        config = None

    config = comm.bcast(config,root=0)
    comm.Barrier()

    return config
