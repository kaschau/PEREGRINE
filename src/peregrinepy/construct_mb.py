from . import multiblock
from .readers import read_blocks4procs,read_connectivity,read_grid
from .mpicomm import mpiutils,blockcomm

def construct_mb(config):

    comm,rank,size = mpiutils.get_comm_rank_size()
    ################################################################
    ##### First we determine what bocks we are responsible for #####
    ################################################################
    if rank == 0:
        blocks4procs = read_blocks4procs(config['io']['inputdir'])
    else:
        blocks4procs = None
    blocks4procs = comm.bcast(blocks4procs,root=0)
    comm.Barrier()

    if len(blocks4procs) != size:
        if rank == 0:
            print('ERROR!! Number of requested processors does not equal number of processors!')
        comm.Abort()

    myblocks = blocks4procs[rank]
    mb = multiblock.generate_multiblock_solver(len(myblocks),config)

    #We have to overwrite the default value of nblki in parallel
    for i,nblki in enumerate(myblocks):
        mb[i].nblki = nblki

    ################################################################
    ##### Read in the grid
    ################################################################
    read_grid(mb,config['io']['griddir'])

    ################################################################
    ##### Read in the connectivity
    ################################################################
    if rank == 0:
        conn = read_connectivity(None,config['io']['inputdir'])
    else:
        conn = None
    conn = comm.bcast(conn,root=0)
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            for k1 in conn[blk.nblki][face].keys():
                blk.connectivity[face][k1] = conn[blk.nblki][face][k1]

    ################################################################
    ##### Now we figure out which processor each block's neighbor
    ##### is on
    ################################################################

    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            neighbor = blk.connectivity[face]['neighbor']
            if neighbor == None:
                blk.connectivity[face]['comm_rank'] = None
                continue
            for otherrank,proc in enumerate(blocks4procs):
                if neighbor in proc:
                    blk.connectivity[face]['comm_rank'] = otherrank

    ################################################################
    ##### Now set the MPI communication info for each block
    ################################################################
    blockcomm.set_block_communication(mb,config)

    return mb
