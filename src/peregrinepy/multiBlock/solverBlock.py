# -*- coding: utf-8 -*-
import kokkos
from ..compute import block_
from .restartBlock import restartBlock
from .solverFace import solverFace
from ..misc import createViewMirrorArray

""" block.py

Authors:

Kyle Schau

This module defines the top block class. This object is the most basic object
that a multiBlock datasets (see multiBlock.py) can be composed of.

"""


class solverBlock(restartBlock, block_):
    """

    Attributes
    ---------

    """

    blockType = "solver"

    def __init__(self, nblki, sp_names, ng):
        # The c++ stuff must be instantiated first,
        # so that inhereted python side
        # attributes are assigned values, not defined
        # in the upstream __init__s
        block_.__init__(self)

        # Flag to determine if a block's solver arrays are
        # initialized or not
        self._isInitialized = False

        self.ng = ng

        restartBlock.__init__(self, nblki, sp_names)

        for fn in [1, 2, 3, 4, 5, 6]:
            self.faces.append(solverFace(fn, self.ng))

        self.ne = 5 + self.ns - 1

        #######################################################################
        # Solution Variables
        #######################################################################
        # Conserved variables
        for d in ["Q", "dQ"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Spatial derivative of primative array
        for d in ["dqdx", "dqdy", "dqdz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # thermo,trans arrays
        for d in ["qh", "qt"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # chemistry
        for d in ["omega"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # RK stages
        for d in ["rhs0", "rhs1", "rhs2", "rhs3"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Face fluxes
        for d in ["iF", "jF", "kF"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Face flux switches
        for d in ["phi"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None

        self.array._freeze()
        self.mirror._freeze()

    def initSolverArrays(self, config):
        """
        Create the Kokkos work arrays and python side numpy wrappers
        """
        self._isInitialized = True

        ng = self.ng
        ccshape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        ifshape = [self.ni + 2 * ng, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        jfshape = [self.ni + 2 * ng - 1, self.nj + 2 * ng, self.nk + 2 * ng - 1]
        kfshape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng]

        cQshape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            5 + self.ns - 1,
        ]
        ifQshape = [
            self.ni + 2 * ng,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            5 + self.ns - 1,
        ]
        jfQshape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng,
            self.nk + 2 * ng - 1,
            5 + self.ns - 1,
        ]
        kfQshape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng,
            5 + self.ns - 1,
        ]

        #######################################################################
        # Grid Arrays
        #######################################################################
        # ------------------------------------------------------------------- #
        #       Primary grid coordinates
        # ------------------------------------------------------------------- #
        shape = [self.ni + 2 * ng, self.nj + 2 * ng, self.nk + 2 * ng]
        createViewMirrorArray(self, ["x", "y", "z"], shape)

        # ------------------------------------------------------------------- #
        #       Cell center
        # ------------------------------------------------------------------- #
        shape = ccshape
        createViewMirrorArray(self, ["xc", "yc", "zc", "J"], shape)
        # Cell center metrics
        createViewMirrorArray(
            self,
            ["dEdx", "dEdy", "dEdz", "dNdx", "dNdy", "dNdz", "dXdx", "dXdy", "dXdz"],
            shape,
        )

        # ------------------------------------------------------------------- #
        #       i face vector components and areas
        # ------------------------------------------------------------------- #
        createViewMirrorArray(self, ["ixc", "iyc", "izc"], ifshape)
        createViewMirrorArray(
            self, ["isx", "isy", "isz", "iS", "inx", "iny", "inz"], ifshape
        )

        # ------------------------------------------------------------------- #
        #       j face vector components and areas
        # ------------------------------------------------------------------- #
        createViewMirrorArray(self, ["jxc", "jyc", "jzc"], jfshape)
        createViewMirrorArray(
            self, ["jsx", "jsy", "jsz", "jS", "jnx", "jny", "jnz"], jfshape
        )

        # ------------------------------------------------------------------- #
        #       k face vector components and areas
        # ------------------------------------------------------------------- #
        createViewMirrorArray(self, ["kxc", "kyc", "kzc"], kfshape)
        createViewMirrorArray(
            self, ["ksx", "ksy", "ksz", "kS", "knx", "kny", "knz"], kfshape
        )

        #######################################################################
        # Flow Arrays
        #######################################################################
        # ------------------------------------------------------------------- #
        #       Conservative, Primative, dQ
        # ------------------------------------------------------------------- #
        createViewMirrorArray(self, ["Q", "q", "dQ"], cQshape)

        # ------------------------------------------------------------------- #
        #       Spatial derivative of primative array
        # ------------------------------------------------------------------- #
        createViewMirrorArray(self, ["dqdx", "dqdy", "dqdz"], cQshape)

        # ------------------------------------------------------------------- #
        #       Thermo
        # ------------------------------------------------------------------- #
        shape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            5 + self.ns,
        ]
        createViewMirrorArray(self, ["qh"], shape)

        # ------------------------------------------------------------------- #
        #       Transport
        # ------------------------------------------------------------------- #
        shape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            2 + self.ns - 1,
        ]
        createViewMirrorArray(self, ["qt"], shape)

        # ------------------------------------------------------------------- #
        #       Chemistry
        # ------------------------------------------------------------------- #
        if config["thermochem"]["chemistry"]:
            shape = [
                self.ni + 2 * ng - 1,
                self.nj + 2 * ng - 1,
                self.nk + 2 * ng - 1,
                1 + self.ns,
            ]
            createViewMirrorArray(self, ["omega"], shape)

        # ------------------------------------------------------------------- #
        #       RK Stages
        # ------------------------------------------------------------------- #
        nstorage = {"rk1": 0, "rk3": 2, "rk4": 4, "strang": 2}
        names = [
            f"rhs{i}" for i in range(nstorage[config["solver"]["timeIntegration"]])
        ]
        createViewMirrorArray(self, names, cQshape)

        # ------------------------------------------------------------------- #
        #       Fluxes
        # ------------------------------------------------------------------- #
        for shape, names in (
            (ifQshape, ["iF"]),
            (jfQshape, ["jF"]),
            (kfQshape, ["kF"]),
        ):
            createViewMirrorArray(self, names, shape)

        # ------------------------------------------------------------------- #
        #       Switches
        # ------------------------------------------------------------------- #
        createViewMirrorArray(self, ["phi"], cQshape)

    def setBlockCommunication(self):

        for face in self.faces:
            if face.neighbor is None:
                continue
            face.setOrientFunc(self.ni, self.nj, self.nk, self.ne)
            face.setCommBuffers(self.ni, self.nj, self.nk, self.ne, self.nblki)

    def updateDeviceView(self, var):
        kokkos.deep_copy(getattr(self, var), self.mirror[var])

    def updateHostView(self, var):
        kokkos.deep_copy(self.mirror[var], getattr(self, var))
