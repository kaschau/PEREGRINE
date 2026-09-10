import numpy as np
from .gridBlock import gridBlock


class restartBlock(gridBlock):
    """
    restartBlock object holds all the information that a PEREGRINE restart
    would need to know about a block.
    """

    def __init__(self, nblki, speciesNames, ng=0):
        super().__init__(nblki, ng)

        self.nrt = 0
        self.tme = 0.0

        self.speciesNames = speciesNames
        self.ns = len(speciesNames)
        if self.ns < 1:
            raise ValueError("Number of species must be >=1")

        # Primative variables
        self.declare("q", kind="cell", components=5 + self.ns - 1)

    def fillHaloWithNearest(self, name):
        """No halo to fill."""

    def verifySpeciesSum(self, normalize=False):
        """Function to verify that the sum of species in any cell is not greater than unity"""

        assert (
            self.ns > 1
        ), "You are trying to check species sum on a case where ns = 1."
        summation = np.sum(self.array["q"][:, :, :, 5::], axis=-1)
        if np.max(summation) > 1.0:
            print(
                "Warning! Species sum of",
                np.max(summation),
                "found at",
                np.unravel_index(np.argmax(summation, axis=None), summation.shape),
                "in block",
                self.nblki,
            )
            if normalize:
                self.array["q"][:, :, :, 5::] = np.where(
                    summation[:, :, :, np.newaxis] > 1.0,
                    self.array["q"][:, :, :, 5::] / summation[:, :, :, np.newaxis],
                    self.array["q"][:, :, :, 5::],
                )
            return False
