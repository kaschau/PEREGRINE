import h5py


def readRestart(mb, path="./", nrt=0):
    """This function reads in all the HDF5 grid files in :path:
    and adds the coordinate data to a supplied peregrinepy.multiBlock.restart
    object (or one of its descendants)

    Parameters
    ----------

    mb : peregrinepy.multiBlock.restart (or a descendant)

    path : str
        Path to find all the HDF5 grid files to be read in

    Returns
    -------
    None

    """

    qf = h5py.File(f"{path}/q.{nrt:08d}.h5", "r")

    for blk in mb:
        # Create the "q" array
        blk.initRestartArrays()

        variables = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]

        blk.nrt = int(list(qf["iter"]["nrt"])[0])
        blk.tme = float(list(qf["iter"]["tme"])[0])

        # read from base slab
        resS = qf[f"results_{blk.baseNblki:06d}"]
        dest = blk.array["q"]
        for i, var in enumerate(variables):
            # a case may carry species the result it restarts from did not,
            # and those keep whatever initRestartArrays gave them
            if var not in resS:
                if blk.nblki == 0:
                    print(f"Warning, {var} not found in restart. Leaving as is.")
                continue
            if dest.flags["F_CONTIGUOUS"]:
                resS[var].read_direct(
                    dest.T, source_sel=blk.baseCellSlab, dest_sel=(i,) + blk.interior
                )
            else:
                dest[blk.interior + tuple([i])] = resS[var][blk.baseCellSlab].T

        mb.progress(blk.nblki + 1, f"Reading in restartBlock {blk.nblki}")
        blk.fillHaloWithNearest("q")

    qf.close()

    # Set the mb values as well
    mb.nrt = mb[0].nrt
    mb.tme = mb[0].tme
