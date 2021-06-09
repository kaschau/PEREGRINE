from .mpiutils import get_comm_rank_size


def halo_grid(mb,config):

    comm,rank,size = get_comm_rank_size()
    block_list = mb.block_list
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor    = blk.connectivity[face]['neighbor']
            if neighbor is None:
                continue
            orientation = blk.connectivity[face]['orientation']
            comm_rank   = blk.connectivity[face]['comm_rank']
            #if neighbor
