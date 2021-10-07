import peregrinepy as pg


def bootstrapCase(config):

    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    ################################################################
    # First we determine what bocks we are responsible for
    ################################################################
    if rank == 0:
        blocksForProcs = pg.readers.readBlocksForProcs(config["io"]["inputdir"])
    else:
        blocksForProcs = None
    blocksForProcs = comm.bcast(blocksForProcs, root=0)
    comm.Barrier()

    if len(blocksForProcs) != size:
        if rank == 0:
            print(
                "ERROR!! Number of requested processors in blocksForProcs does not equal number of processors!"
            )
        comm.Abort()

    myblocks = blocksForProcs[rank]
    mb = pg.multiBlock.generateMultiBlockSolver(len(myblocks), config, myblocks)

    ################################################################
    # Read in the connectivity
    ################################################################
    pg.readers.readConnectivity(mb, config["io"]["inputdir"])

    ################################################################
    # Now we figure out which processor each block's neighbor
    # is on
    ################################################################

    for blk in mb:
        for face in blk.faces:
            neighbor = face.connectivity["neighbor"]
            if neighbor is None:
                face.commRank = None
                continue
            for otherrank, proc in enumerate(blocksForProcs):
                if neighbor in proc:
                    face.commRank = otherrank

    ################################################################
    # Read in the grid
    ################################################################
    pg.readers.readGrid(mb, config["io"]["griddir"])

    ################################################################
    # Now set the MPI communication info for each block
    ################################################################
    pg.mpiComm.blockComm.setBlockCommunication(mb)

    ################################################################
    # Initialize the solver arrays
    ################################################################
    mb.initSolverArrays(config)

    ################################################################
    # Unify the grid via halo construction, compute metrics
    ################################################################
    mb.unifyGrid()
    mb.computeMetrics()

    ################################################################
    # Read in boundary conditions
    ################################################################
    pg.readers.readBcs(mb, config["io"]["inputdir"])
    for blk in mb:
        for face in blk.faces:
            face.setBcFunc()

    ################################################################
    # Read in restart
    ################################################################
    pg.readers.readRestart(
        mb,
        config["io"]["outputdir"],
        config["simulation"]["restartFrom"],
        config["simulation"]["animate"],
    )

    ################################################################
    # Register parallel writer
    ################################################################
    pg.writers.parallelWriter.registerParallelXdmf(
        mb, config["io"]["outputdir"], gridPath=f"../{config['io']['griddir']}"
    )

    # Generate conserved variables
    for blk in mb:
        mb.eos(blk, mb.thtrdat, 0, "prims")

    # Consistify total flow field
    pg.consistify(mb)

    return mb
