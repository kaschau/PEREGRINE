# -*- coding: utf-8 -*-
from ..misc import FrozenDict
from .face import face

""" block.py

Authors:

Kyle Schau

This module defines the top block class. This object is the most basic object that a multiblock datasets (see multiblock.py) can be composed of.

"""


class topology_block:
    """block object is the most basic object a raptorpy.multiblock.dataset (or one of its descendants) can be.

    Attributes
    ---------
    nblki : int
        Block number (not the index, so the first block has nblki = 1)

    """

    def __init__(self, nblki):

        self.nblki = nblki

        self.faces = []
        for fn in [1, 2, 3, 4, 5, 6]:
            self.faces.append(face(fn))

    def get_face(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.faces[int(nface) - 1]

    def get_face_conn(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.faces[int(nface) - 1].connectivity

    def connectivity(self):
        conn = {}
        for face in self.faces:
            conn[f"Face{face.nface}"] = {}
            f = conn[f"Face{face.nface}"]
            for key in face.connectivity.keys():
                f[key] = face.connectivity[key]
        return conn
