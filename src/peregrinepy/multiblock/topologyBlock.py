# -*- coding: utf-8 -*-
from .face import face

""" topologyBlock.py

Authors:

Kyle Schau

This module defines the topology block class.
This object is the most basic object that a multiblock
composition can be comprised of.

"""


class topologyBlock:
    """topologyBlock object is the most basic object a peregrinepy.multiblock
    (or one of its descendants) can be.

    Attributes
    ---------
    nblki : int
        Block number (first block number is 0)

    """

    def __init__(self, nblki):

        self.nblki = nblki

        self.faces = []
        for fn in [1, 2, 3, 4, 5, 6]:
            self.faces.append(face(fn))

    def getFace(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.faces[int(nface) - 1]

    def getFaceConn(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.faces[int(nface) - 1].connectivity

    def connectivity(self):
        conn = {}
        for fc in self.faces:
            conn[f"Face{fc.nface}"] = {}
            f = conn[f"Face{fc.nface}"]
            for key in fc.connectivity.keys():
                f[key] = fc.connectivity[key]
        return conn
