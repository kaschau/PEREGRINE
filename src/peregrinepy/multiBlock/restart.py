from ..readers import GridReader, RestartReader
from .arrays import CellCenterArray
from .grid import grid
from .restartBlock import restartBlock


class restart(grid):
    """A grid with a state on it: the primitives of every block at one
    time, and the species they are of."""

    def _newBlock(self, nblki):
        return restartBlock(nblki, self)

    @classmethod
    def fromResult(cls, fileName, quiet=True):
        """A restart from a result file, which says its own species and the
        grid it sits on."""
        reader = RestartReader(fileName, quiet=quiet)
        mb = cls(reader.species)
        GridReader(reader.grid, quiet=quiet).fill(mb)
        reader.fill(mb)
        return mb

    def __init__(self, spNames):
        super().__init__()
        self.speciesNames = spNames
        self.ns = len(spNames)
        # the step count and the time the state is at
        self.nrt = 0
        self.tme = 0.0
        # the primitive vector p, u, v, w, T, Y(0 .. ns - 2), what a result
        # file holds of the state
        self.declareArray("prims", CellCenterArray, components=5 + self.ns - 1)

    def checkSpeciesSum(self, normalize=False):
        """Loop through each block to check that the sum of all
        species does not exceed 1.0 anywhere in the domain."""

        anyBad = False
        for blk in self.blocks:
            goodSum = blk.verifySpeciesSum(normalize)
            if not goodSum:
                anyBad = True

        if anyBad:
            if not normalize:
                print(
                    "\nRe-run checkSpeciesSum sum with arg normalize=True to normalize species mass fraction.\n"
                )
            else:
                print(
                    "Normalizing species mass fraction such that sum is <= one everywhere."
                )
