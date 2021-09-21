# -*- coding: utf-8 -*-

""" grid_block.py

Authors:

Kyle Schau

This module defines a grid_block object that is used to compose a raptorpy.multiblock.grid (or one of its descendants) (see multiblock.py)

"""

import numpy as np
from .topology_block import topology_block
from ..misc import FrozenDict
from ..grid import metrics
from ..grid import generate_halo


class grid_block(topology_block):
    """grid_block object holds all the information that a grid would need to know about a block."""

    block_type = "grid"

    def __init__(self, nblki):
        super().__init__(nblki)

        self.ni = 0
        self.nj = 0
        self.nk = 0

        ################################################################################################################
        ############## Data arrays
        ################################################################################################################
        # Python side data
        self.array = FrozenDict()
        # Coordinate arrays
        for d in ["x", "y", "z"]:
            self.array[f"{d}"] = None
        # Grid metrics
        # Cell centers
        for d in ["xc", "yc", "zc", "J"]:
            self.array[f"{d}"] = None
        # Cell center metrics
        for d in [
            "dEdx",
            "dEdy",
            "dEdz",
            "dNdx",
            "dNdy",
            "dNdz",
            "dXdx",
            "dXdy",
            "dXdz",
        ]:
            self.array[f"{d}"] = None
        # i face area vectors
        for d in ["isx", "isy", "isz", "iS", "inx", "iny", "inz"]:
            self.array[f"{d}"] = None
        # j face area vectors
        for d in ["jsx", "jsy", "jsz", "jS", "jnx", "jny", "jnz"]:
            self.array[f"{d}"] = None
        # k face area vectors
        for d in ["ksx", "ksy", "ksz", "kS", "knx", "kny", "knz"]:
            self.array[f"{d}"] = None

        if self.block_type == "grid":
            self.array._freeze()

    def init_grid_arrays(self):
        """
        Create zeroed numpy arrays of correct size.
        """

        # Primary grid coordinates
        shape = (
            [self.ni + 2, self.nj + 2, self.nk + 2]
            if self.block_type == "solver"
            else [self.ni, self.nj, self.nk]
        )
        for name in ["x", "y", "z"]:
            self.array[name] = np.zeros((shape))

        # Cell center locations, volumes, diffusive metrics
        shape = (
            [self.ni + 1, self.nj + 1, self.nk + 1]
            if self.block_type == "solver"
            else [self.ni - 1, self.nj - 1, self.nk - 1]
        )
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
            "dXdx",
            "dXdy",
            "dXdz",
        ]:
            self.array[name] = np.zeros((shape))

        # i face normal, area vectors
        shape = (
            [self.ni + 2, self.nj + 1, self.nk + 1]
            if self.block_type == "solver"
            else [self.ni, self.nj - 1, self.nk - 1]
        )
        for name in ["isx", "isy", "isz", "iS", "inx", "iny", "inz"]:
            self.array[name] = np.zeros((shape))

        # j face normal, area vectors
        shape = (
            [self.ni + 1, self.nj + 2, self.nk + 1]
            if self.block_type == "solver"
            else [self.ni - 1, self.nj, self.nk - 1]
        )
        for name in ["jsx", "jsy", "jsz", "jS", "jnx", "jny", "jnz"]:
            self.array[name] = np.zeros((shape))

        # k face normal, area vectors
        shape = (
            [self.ni + 1, self.nj + 1, self.nk + 2]
            if self.block_type == "solver"
            else [self.ni - 1, self.nj - 1, self.nk]
        )
        for name in ["ksx", "ksy", "ksz", "kS", "knx", "kny", "knz"]:
            self.array[name] = np.zeros((shape))

    def compute_metrics(self):
        metrics(self)

    def generate_halo(self):
        generate_halo(self)
