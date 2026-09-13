"""
Reading a PEREGRINE restart.

One q.<nrt>.h5 holds the primitives of every block at one time.
"""

import h5py

from ..misc import Progress


class RestartReader:
    """The result numbered :nrt: in :path:.

    Making one reads the step and time it was written at. The file is opened
    again for the block reads of fill().
    """

    def __init__(self, path="./", nrt=0, quiet=False):
        self.fileName = f"{path}/q.{nrt:08d}.h5"
        self.quiet = quiet
        with h5py.File(self.fileName, "r") as f:
            self.nrt = int(f["iter"]["nrt"][0])
            self.tme = float(f["iter"]["tme"][0])

    def fill(self, mb):
        """Fill in the primitives of every block of mb, and the step and time
        they are at."""
        with h5py.File(self.fileName, "r") as f, Progress(
            len(mb.blocks), self.quiet
        ) as bar:
            for blk in mb.blocks:
                variables = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]

                # read from base slab
                resS = f[f"results_{blk.baseNblki:06d}"]
                dest = blk.hostCopy("q")
                for i, var in enumerate(variables):
                    # a case may carry species the result it restarts from did
                    # not, and those keep the zeros they were allocated with
                    if var not in resS:
                        if blk.nblki == 0:
                            print(
                                f"Warning, {var} not found in restart. Leaving as is."
                            )
                        continue
                    # the device layout takes the file straight in; a host
                    # array pays a copy through a temporary
                    if dest.flags["F_CONTIGUOUS"]:
                        resS[var].read_direct(
                            dest.T,
                            source_sel=blk.baseCellSlab,
                            dest_sel=(i,) + blk.interior,
                        )
                    else:
                        dest[blk.interior + tuple([i])] = resS[var][blk.baseCellSlab].T

                blk.store("q", dest)
                bar.step(f"Reading in block {blk.nblki}")

        mb.nrt = self.nrt
        mb.tme = self.tme
