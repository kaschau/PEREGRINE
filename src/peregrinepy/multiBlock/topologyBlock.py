# -*- coding: utf-8 -*-
from .topologyFace import topologyFace

""" topologyBlock.py

Authors:

Kyle Schau

This module defines the topology block class.
This object is the most basic object that a multiBlock
composition can be comprised of.

"""


class topologyBlock:
    """topologyBlock object is the most basic object a peregrinepy.multiBlock
    (or one of its descendants) can be.

    Attributes
    ---------
    nblki : int
        Block number (first block number is 0)

    """

    blockType = "topology"

    def __init__(self, nblki):
        self.nblki = nblki

        self.faces = []
        if self.blockType in ["topology", "grid", "restart"]:
            for fn in [1, 2, 3, 4, 5, 6]:
                self.faces.append(topologyFace(fn))

    def getFace(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.faces[int(nface) - 1]
