import peregrinepy as pg


def bootstrapCase(config):

    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    ################################################################
    # First we determine what bocks we are responsible for
    ################################################################
    blocksForProcs = pg.readers.readBlocksForProcs(
        config["io"]["inputDir"], parallel=True
    )
    comm.Barrier()
    if rank == 0:
        print("Read blocksForProcs.")
    # If blocksForProcs.inp is not found, blocksForProcs will be
    # None, so we assume one block per proc
    if blocksForProcs is None:
        blocksForProcs = [[i] for i in range(size)]
        if rank == 0:
            print(
                "No blocksForProcs.inp found.\n",
                f"assuming {size} blocks with one block per proc.",
            )

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
    comm.Barrier()
    if rank == 0:
        print("Generated multiblock.")

    ################################################################
    # Read in the connectivity
    ################################################################
    pg.readers.readConnectivity(mb, config["io"]["inputDir"])
    comm.Barrier()
    if rank == 0:
        print("Read connectivity.")
        # Check we have the correct number of blocks.
        totalCheck = 0
        for proc in blocksForProcs:
            totalCheck += len(proc)
        if mb.totalBlocks != totalCheck:
            print(
                "ERROR!! Number of blocks in conn.yaml does not equal number of total blocks"
            )
            comm.Abort()

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
    pg.readers.readGrid(mb, path=config["io"]["gridDir"], lump=config["io"]["lumpIO"])
    comm.Barrier()
    if rank == 0:
        print("Read grid.")

    ################################################################
    # Read in restart
    ################################################################
    pg.readers.readRestart(
        mb,
        path=config["io"]["restartDir"],
        nrt=config["simulation"]["restartFrom"],
        animate=config["io"]["animateRestart"],
        lump=config["io"]["lumpIO"],
    )
    comm.Barrier()
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
    # Read in periodic boundary condition info
    ################################################################
    pg.readers.readBcs(mb, config["io"]["inputDir"], justPeriodic=True)

    ################################################################
    # Unify the grid via halo construction, compute metrics
    ################################################################
    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])
    comm.Barrier()
    if rank == 0:
        print("Unified grid.")

    ################################################################
    # Read in boundary conditions
    ################################################################
    pg.readers.readBcs(mb, config["io"]["inputDir"], justPeriodic=False)
    comm.Barrier()
    if rank == 0:
        print("Set boundary conditions.")

    ################################################################
    # Register parallel restart/archive writers
    ################################################################
    mb.parallelRestartXdmf = mb.pg.writers.parallelWriter.registerParallelMetaData(
        mb,
        blocksForProcs,
        gridPath=f"../{config['io']['gridDir']}",
        precision="double",
        animate=config["io"]["animateRestart"],
        lump=config["io"]["lumpIO"],
    )
    mb.parallelArchiveXdmf = mb.pg.writers.parallelWriter.registerParallelMetaData(
        mb,
        blocksForProcs,
        gridPath=f"../{config['io']['gridDir']}",
        precision="single",
        animate=config["io"]["animateArchive"],
        lump=config["io"]["lumpIO"],
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
    # Dual time initialization
    ################################################################
    if mb.step.stepType == "dualTime":
        mb.initializeDualTime()

    ################################################################
    # Initialize coprocessor
    ################################################################
    mb.coproc = pg.coproc.coprocessor(mb)

    comm.Barrier()
    if rank == 0:
        print("Ready to solve.")

    return mb
