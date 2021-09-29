# -*- coding: utf-8 -*-
import kokkos
import numpy as np
from ..compute import block_
from .restartBlock import restartBlock

""" block.py

Authors:

Kyle Schau

This module defines the top block class. This object is the most basic object
that a multiblock datasets (see multiblock.py) can be composed of.

"""


class solverBlock(restartBlock, block_):
    """
    block object is the most basic object a raptorpy.multiblock.dataset
    (or one of its descendants) can be.

    Attributes
    ---------

    """

    blockType = "solver"

    def __init__(self, nblki, sp_names):
        # The c++ stuff must be instantiated first,
        # so that inhereted python side
        # attributes are assigned values, not defined
        # in the upstream __init__s
        block_.__init__(self)
        restartBlock.__init__(self, nblki, sp_names)

        self.ne = 5 + self.ns - 1

        #######################################################################
        # Solution Variables
        #######################################################################
        # Conserved variables
        for d in ["Q", "dQ"]:
            self.array[f"{d}"] = None
        # Spatial derivative of primative array
        for d in ["dqdx", "dqdy", "dqdz"]:
            self.array[f"{d}"] = None
        # thermo,trans arrays
        for d in ["qh", "qt"]:
            self.array[f"{d}"] = None
        # chemistry
        for d in ["omega"]:
            self.array[f"{d}"] = None
        # RK stages
        for d in ["rhs0", "rhs1", "rhs2", "rhs3"]:
            self.array[f"{d}"] = None
        # Face fluxes
        for d in ["iF", "jF", "kF"]:
            self.array[f"{d}"] = None
        # Face flux switches
        for d in ["phi"]:
            self.array[f"{d}"] = None

        if self.blockType == "solver":
            self.array._freeze()

    def initSolverArrays(self, config):
        """
        Create the Kokkos work arrays and python side numpy wrappers
        """

        if config["Kokkos"]["Space"] in ["OpenMP", "Serial", "Default"]:
            space = kokkos.HostSpace
        else:
            raise ValueError("Are we ready for that?")

        ccshape = [self.ni + 1, self.nj + 1, self.nk + 1]
        ifshape = [self.ni + 2, self.nj + 1, self.nk + 1]
        jfshape = [self.ni + 1, self.nj + 2, self.nk + 1]
        kfshape = [self.ni + 1, self.nj + 1, self.nk + 2]

        cQshape = [self.ni + 1, self.nj + 1, self.nk + 1, 5 + self.ns - 1]
        ifQshape = [self.ni + 2, self.nj + 1, self.nk + 1, 5 + self.ns - 1]
        jfQshape = [self.ni + 1, self.nj + 2, self.nk + 1, 5 + self.ns - 1]
        kfQshape = [self.ni + 1, self.nj + 1, self.nk + 2, 5 + self.ns - 1]

        def npOrKokkos(names, shape):
            for name in names:
                if self.array[name] is None:
                    setattr(
                        self,
                        name,
                        kokkos.array(
                            name,
                            shape=shape,
                            dtype=kokkos.double,
                            space=space,
                            dynamic=False,
                        ),
                    )
                    self.array[name] = np.array(getattr(self, name), copy=False)
                else:
                    setattr(
                        self,
                        name,
                        kokkos.array(
                            self.array[name],
                            dtype=kokkos.double,
                            space=space,
                            dynamic=False,
                        ),
                    )

        #######################################################################
        # Grid Arrays
        #######################################################################
        # ------------------------------------------------------------------- #
        #       Primary grid coordinates
        # ------------------------------------------------------------------- #
        shape = [self.ni + 2, self.nj + 2, self.nk + 2]
        npOrKokkos(["x", "y", "z"], shape)

        # ------------------------------------------------------------------- #
        #       Cell center
        # ------------------------------------------------------------------- #
        shape = ccshape
        npOrKokkos(["xc", "yc", "zc", "J"], shape)
        # Cell center metrics
        npOrKokkos(
            ["dEdx", "dEdy", "dEdz", "dNdx", "dNdy", "dNdz", "dXdx", "dXdy", "dXdz"],
            shape,
        )

        # ------------------------------------------------------------------- #
        #       i face vector components and areas
        # ------------------------------------------------------------------- #
        shape = ifshape
        npOrKokkos(["isx", "isy", "isz", "iS", "inx", "iny", "inz"], ifshape)

        # ------------------------------------------------------------------- #
        #       j face vector components and areas
        # ------------------------------------------------------------------- #
        shape = jfshape
        npOrKokkos(["jsx", "jsy", "jsz", "jS", "jnx", "jny", "jnz"], shape)

        # ------------------------------------------------------------------- #
        #       k face vector components and areas
        # ------------------------------------------------------------------- #
        shape = kfshape
        npOrKokkos(["ksx", "ksy", "ksz", "kS", "knx", "kny", "knz"], shape)

        #######################################################################
        # Flow Arrays
        #######################################################################
        # ------------------------------------------------------------------- #
        #       Conservative, Primative, dQ
        # ------------------------------------------------------------------- #
        shape = cQshape
        npOrKokkos(["Q", "q", "dQ"], shape)

        # ------------------------------------------------------------------- #
        #       Spatial derivative of primative array
        # ------------------------------------------------------------------- #
        shape = cQshape
        npOrKokkos(["dqdx", "dqdy", "dqdz"], shape)

        # ------------------------------------------------------------------- #
        #       Thermo
        # ------------------------------------------------------------------- #
        shape = [self.ni + 1, self.nj + 1, self.nk + 1, 5 + self.ns]
        npOrKokkos(["qh"], shape)

        # ------------------------------------------------------------------- #
        #       Transport
        # ------------------------------------------------------------------- #
        shape = [self.ni + 1, self.nj + 1, self.nk + 1, 2 + self.ns - 1]
        npOrKokkos(["qt"], shape)

        # ------------------------------------------------------------------- #
        #       Chemistry
        # ------------------------------------------------------------------- #
        if config["thermochem"]["chemistry"]:
            shape = [self.ni + 1, self.nj + 1, self.nk + 1, 1 + self.ns]
            npOrKokkos(["omega"], shape)

        # ------------------------------------------------------------------- #
        #       RK Stages
        # ------------------------------------------------------------------- #
        shape = cQshape
        nstorage = {"rk1": 0, "rk3": 2, "rk4": 4, "strang": 2}
        names = [
            f"rhs{i}" for i in range(nstorage[config["solver"]["timeIntegration"]])
        ]
        npOrKokkos(names, shape)

        # ------------------------------------------------------------------- #
        #       Fluxes
        # ------------------------------------------------------------------- #
        for shape, names in (
            (ifQshape, ["iF"]),
            (jfQshape, ["jF"]),
            (kfQshape, ["kF"]),
        ):
            npOrKokkos(names, shape)

        # ------------------------------------------------------------------- #
        #       Switches
        # ------------------------------------------------------------------- #
        shape = [self.ni + 1, self.nj + 1, self.nk + 1, 3]
        npOrKokkos(["phi"], shape)