# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .grid import grid
from .restart_block import restart_block


class restart(grid):
    """A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid"""

    mb_type = "restart"

    def __init__(self, nblks, sp_names, ls=[]):

        if ls == []:
            temp = [restart_block(i, sp_names) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

        self.nrt = 0
        self.tme = 0

    def init_restart_arrays(self):
        for blk in self:
            blk.init_restart_arrays()

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

    def check_species_sum(self, normalize=False):
        """Loop through each block to check that the sum of all species does not exceed 1.0 anywhere in the domain

        Parameters
        ----------

        normalize: bool
            Whether to normalize the species mass fraction such that the summ is less than one.

        Returns
        -------
        None

        """

        any_bad = False
        for blk in self:
            good_sum = blk.verify_species_sum(normalize)
            if not good_sum:
                any_bad = True

        if any_bad:
            if not normalize:
                print(
                    "\nRe-run check_species_sum sum with arg normalize=True to normalize species mass fraction.\n"
                )
            else:
                print(
                    "Normalizing species mass fraction such that sum is <= one everywhere."
                )
