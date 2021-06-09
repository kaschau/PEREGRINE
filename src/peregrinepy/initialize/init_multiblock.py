
from ..multiblock import multiblock
from ..readers import read_blocks4procs,read_connectivity
from ..mpicomm import mpiutils

def init_multiblock(config):

    comm,rank,size = mpiutils.get_comm_rank_size()
    ################################################################
    ##### First we determine what bocks we are responsible for #####
    ################################################################
    if rank == 0:
        blocks4procs = read_blocks4procs(config)
    else:
        blocks4procs = None
    blocks4procs = comm.bcast(blocks4procs,root=0)

    myblocks = blocks4procs[rank]
    mb = multiblock(len(myblocks))

    for i,nblki in enumerate(myblocks):
        mb[i].nblki = nblki


    ################################################################
    ##### Read in the connectivity
    ################################################################
    if rank == 0:
        conn = read_connectivity(config)
    else:
        conn = None
    conn = comm.bcast(conn,root=0)

    ################################################################
    ##### Now we figure out which processor each blocks neighbor is on
    ################################################################

    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor = int(blk.connectivity[face]['neighbor'])
            for otherrank,proc in enumerate(blocks4procs):
                if neighbor in proc:
                    blk.connectivity[face]['comm_proc'] = otherrank

    return mb
