import numpy as np

from .gridBlock import gridBlock


class restartBlock(gridBlock):
    """
    restartBlock object holds all the information that a PEREGRINE restart
    would need to know about a block.
    """

    def __init__(self, nblki, mb):
        super().__init__(nblki, mb)
        self.speciesNames = mb.speciesNames
        self.ns = len(self.speciesNames)
        if self.ns < 1:
            raise ValueError("Number of species must be >=1")

    def primitives(self):
        """The primitive vector of every cell, p, u, v, w, T, Y(0 .. ns - 2),
        as a host array."""
        return self.prims.get()

    def verifySpeciesSum(self, normalize=False):
        """Function to verify that the sum of species in any cell is not greater than unity"""

        assert (
            self.ns > 1
        ), "You are trying to check species sum on a case where ns = 1."
        q = self.prims.get()
        summation = np.sum(q[:, :, :, 5::], axis=-1)
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
                q[:, :, :, 5::] = np.where(
                    summation[:, :, :, np.newaxis] > 1.0,
                    q[:, :, :, 5::] / summation[:, :, :, np.newaxis],
                    q[:, :, :, 5::],
                )
                self.prims.set(q)
            return False
