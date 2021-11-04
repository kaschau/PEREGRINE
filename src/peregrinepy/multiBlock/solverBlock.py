# -*- coding: utf-8 -*-
import kokkos
import numpy as np
from ..compute import block_
from .restartBlock import restartBlock
from .solverFace import solverFace

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
        elif config["Kokkos"]["Space"] in ["Cuda"]:
            space = kokkos.CudaSpace
        else:
            raise ValueError("What space?")

        ng = self.ng

        ccshape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        ifshape = [self.ni + 2 * ng, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        jfshape = [self.ni + 1, self.nj + 2 * ng, self.nk + 2 * ng - 1]
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
        shape = [self.ni + 2 * ng, self.nj + 2 * ng, self.nk + 2 * ng]
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
        shape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            5 + self.ns,
        ]
        npOrKokkos(["qh"], shape)

        # ------------------------------------------------------------------- #
        #       Transport
        # ------------------------------------------------------------------- #
        shape = [
            self.ni + 2 * ng - 1,
            self.nj + 2 * ng - 1,
            self.nk + 2 * ng - 1,
            2 + self.ns,
        ]
        npOrKokkos(["qt"], shape)

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
        shape = cQshape
        npOrKokkos(["phi"], shape)
