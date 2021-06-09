from ..block import block
from ..readers import read_grid
from ..mpicomm import mpiutils, halo_grid

def init_grid(mb,config):

    comm,rank,size = mpiutils.get_comm_rank_size()

    for blk in mb:
        blk.ngls = config['RunTime']['ngls']

    read_grid(mb,config)

    comm.Barrier()

    halo_grid(mb,config)

    #return mb (dont need?)
