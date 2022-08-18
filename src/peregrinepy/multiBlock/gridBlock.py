# -*- coding: utf-8 -*-

""" gridBlock.py

Authors:

Kyle Schau

This module defines a gridBlock object that is used to
compose a peregrinepy.multiBlock.grid (or one of its descendants)

"""

import numpy as np
from .topologyBlock import topologyBlock
from ..misc import frozenDict
from ..grid import metrics
from ..grid import generateHalo


class gridBlock(topologyBlock):
    """
    gridBlock object holds all the information that a grid
    would need to know about a block.
    """

    blockType = "grid"

    def __init__(self, nblki):
        super().__init__(nblki)

        self.ni = 0
        self.nj = 0
        self.nk = 0

        #########################################################
        # Data arrays
        #########################################################
        # Python side data
        self.array = frozenDict()
        # Kokkos mirrors (only used for solverBlocks)
        self.mirror = frozenDict()
        # Coordinate arrays
        for d in ["x", "y", "z"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Grid metrics
        # Cell centers
        for d in ["xc", "yc", "zc", "J"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Cell center metrics
        for d in [
            "dEdx",
            "dEdy",
            "dEdz",
            "dNdx",
            "dNdy",
            "dNdz",
            "dCdx",
            "dCdy",
            "dCdz",
        ]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # i face area vectors
        for d in ["ixc", "iyc", "izc", "isx", "isy", "isz", "iS", "inx", "iny", "inz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # j face area vectors
        for d in ["jxc", "jyc", "jzc", "jsx", "jsy", "jsz", "jS", "jnx", "jny", "jnz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # k face area vectors
        for d in ["kxc", "kyc", "kzc", "ksx", "ksy", "ksz", "kS", "knx", "kny", "knz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None

        if self.blockType == "grid":
            self.array._freeze()

    def initGridArrays(self):
        """
        Create zeroed numpy arrays of correct size.
        """
        if self.blockType == "solver":
            ng = self.ng
        else:
            ng = 0

        # Primary grid coordinates
        shape = [self.ni + 2 * ng, self.nj + 2 * ng, self.nk + 2 * ng]
        for name in ["x", "y", "z"]:
            self.array[name] = np.zeros((shape))

        # Cell center locations, volumes, diffusive metrics
        shape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        for name in [
            "xc",
            "yc",
            "zc",
            "J",
            "dEdx",
            "dEdy",
            "dEdz",
            "dNdx",
            "dNdy",
            "dNdz",
            "dCdx",
            "dCdy",
            "dCdz",
        ]:
            self.array[name] = np.zeros((shape))

        # i face normal, area vectors
        shape = [self.ni + 2 * ng, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        for name in [
            "ixc",
            "iyc",
            "izc",
            "isx",
            "isy",
            "isz",
            "iS",
            "inx",
            "iny",
            "inz",
        ]:
            self.array[name] = np.zeros((shape))

        # j face normal, area vectors
        shape = [self.ni + 2 * ng - 1, self.nj + 2 * ng, self.nk + 2 * ng - 1]
        for name in [
            "jxc",
            "jyc",
            "jzc",
            "jsx",
            "jsy",
            "jsz",
            "jS",
            "jnx",
            "jny",
            "jnz",
        ]:
            self.array[name] = np.zeros((shape))

        # k face normal, area vectors
        shape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng]
        for name in [
            "kxc",
            "kyc",
            "kzc",
            "ksx",
            "ksy",
            "ksz",
            "kS",
            "knx",
            "kny",
            "knz",
        ]:
            self.array[name] = np.zeros((shape))

    def computeMetrics(self, fdOrder, xcOnly=False):
        metrics(self, fdOrder, xcOnly)

    def generateHalo(self):
        generateHalo(self)
