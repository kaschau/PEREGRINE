from .grid import grid
from .restartBlock import restartBlock


class restart(grid):
    """A grid with a state on it: the primitives of every block at one
    time, and the species they are of."""

    # a restart holds the primatives it was written with, not conserved variables
    hasConservatives = False

    def _newBlock(self, nblki):
        return restartBlock(nblki, self.speciesNames)

    def __init__(self, spNames):
        super().__init__()
        self.speciesNames = spNames
        self.ns = len(spNames)

        self.__nrt = 0
        self.__tme = 0.0

    # We will make the nrt and tme attribues of the restart containter
    # properties with a setter so that setting the container value will
    # also set the block object values as well.
    @property
    def nrt(self):
        return self.__nrt

    @nrt.setter
    def nrt(self, val):
        self.__nrt = val
        for blk in self.blocks:
            blk.nrt = val

    @property
    def tme(self):
        return self.__tme

    @tme.setter
    def tme(self, val):
        self.__tme = val
        for blk in self.blocks:
            blk.tme = val

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
