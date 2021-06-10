from ..block import block
from ..readers import read_grid
from ..mpicomm import mpiutils,blockcomm
from ..ghost import ghost_grid

def init_grid(mb,config):

    comm,rank,size = mpiutils.get_comm_rank_size()

    read_grid(mb,config)
    comm.Barrier()

    #Generate the grid halos (extrapolate BC)
    ghost_grid(mb,config)

    #Communicate halos
    blockcomm.communicate(mb,['x','y','z'])


