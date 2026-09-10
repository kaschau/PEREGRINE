import peregrinepy as pg

"""
This function is used for directory based peregrine cases (non-script cases).

The function will start with a config object
(see PEREGRINE/src/peregrinepy/files/config.py),
which is basically a dictionary, and build a multiBlock solver case based on
what is in the input config, as well as reading files from the directory
structure of the peregrine case.
"""


def bootstrapCase(config):
    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    ################################################################
    # First we determine what bocks we are responsible for
    ################################################################
    gridReader = pg.readers.GridReader(config["io"]["gridDir"])
    try:
        blocksForProcs = gridReader.partition(
            size, pg.mpiComm.mpiUtils.getRanksPerNode()
        )
    except ValueError as e:
        if rank == 0:
            print(f"ERROR!! {e}")
        comm.Abort()
    comm.Barrier()
    if rank == 0:
        print("Read partition.")

    myblocks = blocksForProcs[rank]
    # Generate the multiBlock solver object for each MPI process, given the number of
    # blocks each process is responsible for
    mb = pg.multiBlock.generateMultiBlockSolver(len(myblocks), config, myblocks)
    comm.Barrier()
    if rank == 0:
        print("Generated multiblock.")

    ################################################################
    # Read in the connectivity
    ################################################################
    gridReader.readConnectivity(mb)
    comm.Barrier()
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
    gridReader.readGrid(mb)
    gridReader.close()
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
    # Read in any periodic boundary condition info
    ################################################################
    pg.readers.readBcs(mb, config["io"]["inputDir"], justPeriodic=True)

    ################################################################
    # Unify the grid via halo construction, compute metrics
    ################################################################
    mb.unifyGrid()
    mb.computeMetrics()
    comm.Barrier()
    if rank == 0:
        print("Unified grid.")

    ################################################################
    # Read in all non-periodic boundary conditions
    ################################################################
    pg.readers.readBcs(mb, config["io"]["inputDir"], justPeriodic=False)
    comm.Barrier()
    if rank == 0:
        print("Set boundary conditions.")

    ################################################################
    # Register parallel restart/archive writers
    ################################################################
    mb.restartMetaData = pg.writers.parallelWriter.registerParallelMetaData(
        mb,
        blocksForProcs,
        gridPath=f"../{config['io']['gridDir']}",
        precision="double",
        animate=config["io"]["animateRestart"],
    )
    mb.archiveMetaData = pg.writers.parallelWriter.registerParallelMetaData(
        mb,
        blocksForProcs,
        gridPath=f"../{config['io']['gridDir']}",
        precision="single",
        animate=config["io"]["animateArchive"],
    )
    for extraVar in config["io"]["saveExtraVars"]:
        meta = pg.writers.parallelWriter.registerParallelMetaData(
            mb,
            blocksForProcs,
            gridPath=f"../{config['io']['gridDir']}",
            precision="single",
            animate=config["io"]["animateArchive"],
            arrayName=extraVar,
        )
        mb.extraMetaData.append(meta)

    ################################################################
    # Prepare interior fields
    ################################################################
    # Generate conserved variables
    for blk in mb:
        mb.eos(blk, mb.thtrdat, -1, "prims")

    # Consistify total flow field
    pg.consistify(mb)

    ################################################################
    # Dual time initialization
    ################################################################
    if mb.stepType == "dualTime":
        mb.initializeDualTime()

    ################################################################
    # Initialize coprocessor
    ################################################################
    mb.coproc = pg.coproc.coprocessor(mb)

    comm.Barrier()
    if rank == 0:
        print("Ready to solve.")

    return mb
