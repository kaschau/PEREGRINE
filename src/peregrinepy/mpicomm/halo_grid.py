from .mpiutils import get_comm_rank_size

import numpy as np


def halo_grid(mb,config):

    comm,rank,size = get_comm_rank_size()
    block_list = mb.block_list
    for blk in mb:
        for face in [str(i+1) for i in range(6)]:
            neighbor    = blk.connectivity[face]['neighbor']
            orientation = blk.connectivity[face]['orientation']
            comm_rank   = blk.connectivity[face]['comm_rank']
            #if neighbor
