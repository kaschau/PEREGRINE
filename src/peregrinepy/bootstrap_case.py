import peregrinepy as pg

def bootstrap_case(config):

    comm,rank,size = pg.mpicomm.mpiutils.get_comm_rank_size()
    ################################################################
    ##### First we determine what bocks we are responsible for #####
    ################################################################
    if rank == 0:
        blocks4procs = pg.readers.read_blocks4procs(config['io']['inputdir'])
    else:
        blocks4procs = None
    blocks4procs = comm.bcast(blocks4procs,root=0)
    comm.Barrier()

    if len(blocks4procs) != size:
        if rank == 0:
            print('ERROR!! Number of requested processors in blocks4procs does not equal number of processors!')
        comm.Abort()

    myblocks = blocks4procs[rank]
    mb = pg.multiblock.generate_multiblock_solver(len(myblocks),config, myblocks)

    ################################################################
    ##### Read in the connectivity
    ################################################################
    pg.readers.read_connectivity(mb, config['io']['inputdir'])

    ################################################################
    ##### Now we figure out which processor each block's neighbor
    ##### is on
    ################################################################

    for blk in mb:
        for face in blk.faces:
            neighbor = face.connectivity['neighbor']
            if neighbor == None:
                face.comm_rank = None
                continue
            for otherrank,proc in enumerate(blocks4procs):
                if neighbor in proc:
                    face.comm_rank = otherrank

    ################################################################
    ##### Read in the grid
    ################################################################
    pg.readers.read_grid(mb,config['io']['griddir'])

    ################################################################
    ##### Now set the MPI communication info for each block
    ################################################################
    pg.mpicomm.blockcomm.set_block_communication(mb)

    ################################################################
    ##### Initialize the solver arrays
    ################################################################
    mb.init_solver_arrays(config)

    ################################################################
    ##### Unify the grid via halo construction, compute metrics
    ################################################################
    pg.grid.unify_solver_grid(mb)
    pg.compute.metrics(mb)

    ################################################################
    ##### Read in boundary conditions
    ################################################################
    pg.readers.read_bcs(mb,
                        config['io']['inputdir'])

    ################################################################
    ##### Read in restart
    ################################################################
    pg.readers.read_restart(mb,
                            config['io']['outputdir'],
                            config['simulation']['restart_from'],
                            config['simulation']['animate'])

    #Generate conserved variables
    for blk in mb:
        mb.eos(blk, mb.thermdat, 0, 'prims')

    #Consistify total flow field
    pg.consistify(mb)

    return mb
