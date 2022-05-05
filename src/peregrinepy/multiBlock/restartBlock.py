# -*- coding: utf-8 -*-

""" restartBlock.py

Authors:

Kyle Schau

This module defines a restart_block object that is used to compose a peregrinepy.multiBlock.restart (see multiBlock.py).

"""

import numpy as np
from .gridBlock import gridBlock


class restartBlock(gridBlock):
    """
    restartBlock object holds all the information that a PEREGRINE restart
    would need to know about a block.
    """

    blockType = "restart"

    def __init__(self, nblki, speciesNames):
        super().__init__(nblki)

        self.nrt = 0
        self.tme = 0.0

        self.speciesNames = speciesNames
        # If we are a solver block, and we have hard coded ns at compile
        # time, it is already set at this point. So we will check if it is,
        # and make sure it is the same as the number of species we want.
        if hasattr(self, "ns"):
            # Then it is alreay set and hard coded
            if self.ns != len(speciesNames):
                raise ValueError(
                    f"ERROR!! You are trying to use {len(speciesNames)} species, but pg.compute\n"
                    f"    was precompiled for {self.ns} species."
                )
        else:
            self.ns = len(speciesNames)
        if self.ns < 1:
            raise ValueError("Number of species must be >=1")

        #########################################################
        # Primative Variables
        #########################################################
        # Conserved variables
        for d in ["q"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None

        if self.blockType == "restart":
            self.array._freeze()

    def initRestartArrays(self):
        """
        Create zeroed numpy arrays of correct size.
        """
        if self.blockType == "solver":
            ng = self.ng
        else:
            ng = 0

        cQshape = (
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            5 + self.ns - 1,
        )
        self.array["q"] = np.zeros((cQshape))

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
                    summation > 1.0,
                    self.array["q"][:, :, :, 5::] / summation[:, :, :, np.newaxis],
                    self.array["q"][:, :, :, 5::],
                )
            return False
