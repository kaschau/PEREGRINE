# -*- coding: utf-8 -*-

""" multiBlock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiBlock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .grid import grid
from .restartBlock import restartBlock


class restart(grid):
    """A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiBlock.grid"""

    mbType = "restart"

    def __init__(self, nblks, spNames, ls=None):
        if ls is None:
            temp = [restartBlock(i, spNames) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

        self.__nrt = 0
        self.__tme = 0.0

    def initRestartArrays(self):
        for blk in self:
            blk.initRestartArrays()

    # We will make the nrt and tme attribues of the restart containter
    # properties with a setter so that setting the container value will
    # also set the block object values as well.
    @property
    def nrt(self):
        return self.__nrt

    @nrt.setter
    def nrt(self, val):
        self.__nrt = val
        for blk in self:
            blk.nrt = val

    @property
    def tme(self):
        return self.__tme

    @tme.setter
    def tme(self, val):
        self.__tme = val
        for blk in self:
            blk.tme = val

    def checkSpeciesSum(self, normalize=False):
        """Loop through each block to check that the sum of all species does not exceed 1.0 anywhere in the domain

        Parameters
        ----------

        normalize: bool
            Whether to normalize the species mass fraction such that the summ is less than one.

        Returns
        -------
        None

        """

        anyBad = False
        for blk in self:
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
