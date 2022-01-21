import peregrinepy as pg


def bootstrapCase(config):

    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    ################################################################
    # First we determine what bocks we are responsible for
    ################################################################
    blocksForProcs = pg.readers.readBlocksForProcs(config["io"]["inputdir"])
    if rank == 0:
        print("Read blocsForProcs.")

    # Check that we have correct number of processors
    if len(blocksForProcs) != size:
        if rank == 0:
            print(
                "ERROR!! Number of requested processors in blocksForProcs does not equal number of processors!"
            )
        comm.Abort()
    # Check we have correct number of blks
    summ = 0
    maxx = 0
    for group in blocksForProcs:
        summ += len(group)
        maxx = max(maxx, max(group))
    if summ - 1 != maxx:
        if rank == 0:
            print(
                "ERROR!! Number of blocks in blockForProcs.inp does not equal total number of blocks."
            )
        comm.Abort()

    myblocks = blocksForProcs[rank]
    mb = pg.multiBlock.generateMultiBlockSolver(len(myblocks), config, myblocks)
    if rank == 0:
        print("Generated multiblock.")

    ################################################################
    # Read in the connectivity
    ################################################################
    pg.readers.readConnectivity(mb, config["io"]["inputdir"])
    if rank == 0:
        print("Read connectivity.")

    ################################################################
    # Now we figure out which processor each block's neighbor
    # is on
    ################################################################
    for blk in mb:
        for face in blk.faces:
            neighbor = face.neighbor
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
    if rank == 0:
        print("Read grid.")

    ################################################################
    # Read in restart
    ################################################################
    pg.readers.readRestart(
        mb,
        config["io"]["outputdir"],
        config["simulation"]["restartFrom"],
        config["simulation"]["animate"],
    )
    if rank == 0:
        print("Read restart.")

    ################################################################
    # Now set the MPI communication info for each block
    ################################################################
    mb.setBlockCommunication()

    ################################################################
    # Initialize the solver arrays
    ################################################################
    mb.initSolverArrays(config)

    ################################################################
    # Unify the grid via halo construction, compute metrics
    ################################################################
    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])
    if rank == 0:
        print("Unified grid.")

    ################################################################
    # Read in boundary conditions
    ################################################################
    pg.readers.readBcs(mb, config["io"]["inputdir"])

    ################################################################
    # Register parallel writer
    ################################################################
    pg.writers.parallelWriter.registerParallelXdmf(
        mb,
        path=config["io"]["outputdir"],
        gridPath=f"../{config['io']['griddir']}",
        animate=config["simulation"]["animate"],
    )

    ################################################################
    # Prepare interior fields
    ################################################################
    # Generate conserved variables
    for blk in mb:
        mb.eos(blk, mb.thtrdat, 0, "prims")

    # Consistify total flow field
    pg.consistify(mb)

    ################################################################
    # Initialize coprocessor
    ################################################################
    if config["Catalyst"]["coprocess"]:
        from .coproc import coprocessor

        mb.coproc = coprocessor(mb)
    else:
        mb.coproc = pg.misc.null

    if rank == 0:
        print("Ready to solve.")

    return mb
