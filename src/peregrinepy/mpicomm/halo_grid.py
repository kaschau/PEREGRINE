from .mpiutils import get_comm_rank_size

import numpy as np


def halo_grid(mb):

    comm,rank,size = get_comm_rank_size()
    block_list = mb.block_list
    for face in [str(i+1) for i in range(6)]:
        for blk in mb:
            neighbor = blk.connectivity[face]['connection']
            #if neighbor
