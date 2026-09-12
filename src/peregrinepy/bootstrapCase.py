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
    myblocks = gridReader.partition(size, pg.mpiComm.mpiUtils.getRanksPerNode())
    comm.Barrier()
    if rank == 0:
        print("Read partition.")

    # Generate the multiBlock solver object for each MPI process, given the number of
    # blocks each process is responsible for
    mb = pg.multiBlock.buildSolver(config, myblocks=myblocks)
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
        path=config["io"]["resultsDir"],
        nrt=config["simulation"]["restartFrom"],
    )
    comm.Barrier()
    if rank == 0:
        print("Read restart.")

    ################################################################
    # Now set the MPI communication info for each block
    ################################################################
    mb.setBlockCommunication()

    ################################################################
    # Unify the grid via halo construction, compute metrics
    ################################################################
    mb.unifyGrid()
    mb.computeMetrics()
    comm.Barrier()
    if rank == 0:
        print("Unified grid.")

    ################################################################
    # Put the case's boundary values on the faces that take them
    ################################################################
    pg.bcs.applyBcValues(mb)
    comm.Barrier()
    if rank == 0:
        print("Set boundary conditions.")

    ################################################################
    # Build the writer this case reports its results through
    ################################################################
    mb.resultsWriter = pg.writers.RestartWriter(
        mb,
        path=config["io"]["resultsDir"],
        gridPath=f"../{config['io']['gridDir']}",
        precision="single",
    )

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
