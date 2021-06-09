from ..block import block
from ..readers import read_grid
from ..mpicomm import mpiutils

def init_grid(mb,config):

    comm,rank,size = mpiutils.get_comm_rank_size()
    read_grid(mb,config)

    comm.Barrier()

    #return mb (dont need?)
