"""Reading a PEREGRINE result, q.<nrt>.h5; what it holds is docs/files.md."""

from pathlib import Path

import h5py
import yaml

from ..files import configFile
from ..misc import Progress


class RestartReader:
    """The result file :fileName:.

    Making one reads the step and time it was written at. The file is opened
    again for the block reads of fill().
    """

    def __init__(self, fileName, quiet=True):
        self.fileName = fileName
        self.quiet = quiet
        with h5py.File(self.fileName, "r") as f:
            self.nrt, self.tme = int(f.attrs["nrt"]), float(f.attrs["tme"])
            names = lambda key: [s.decode() for s in f.attrs[key]]
            # the least a case starts from, and everything the file holds
            self.primVars, self.variables = names("primVars"), names("variables")
            # what the result stores per block beyond the state
            self.extras = names("extras")
            # the grid it sits on, and the case that wrote it, when it was one
            self.grid = str(Path(fileName).parent / f.attrs["grid"])
            text = f.attrs["config"]
            self.config = configFile.fromDict(yaml.safe_load(text)) if text else None
        # what fill() read beyond the state
        self.found = set()

    def fill(self, mb):
        """Fill in the primitives of every block of mb, and the step and time
        they are at; and any array a block declares that the result stores
        beyond the state, an integrator's."""
        with (
            h5py.File(self.fileName, "r") as f,
            Progress(len(mb.blocks), self.quiet) as bar,
        ):
            for blk in mb.blocks:
                variables = blk.primVars

                # read from base slab
                resS = f[f"results_{blk.baseNblki:06d}"]
                dest = blk.prims.get()
                for i, var in enumerate(variables):
                    # a case may carry species the result it restarts from did
                    # not, and those keep the zeros they were allocated with
                    if var not in resS:
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

                blk.prims.set(dest)

                for name in self.extras:
                    if name not in getattr(blk, "declared", ()):
                        continue
                    self.found.add(name)
                    dest = getattr(blk, name).get()
                    comps = (slice(None),) * (dest.ndim - 3)
                    if dest.flags["F_CONTIGUOUS"]:
                        resS[name].read_direct(
                            dest.T,
                            source_sel=comps + blk.baseCellSlab,
                            dest_sel=comps + blk.interior,
                        )
                    else:
                        dest[blk.interior] = resS[name][comps + blk.baseCellSlab].T
                    getattr(blk, name).set(dest)
                bar.step(f"Reading in block {blk.nblki}")

        mb.nrt = self.nrt
        mb.tme = self.tme
