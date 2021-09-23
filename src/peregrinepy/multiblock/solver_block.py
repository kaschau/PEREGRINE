# -*- coding: utf-8 -*-
import kokkos
import numpy as np
from ..compute import block_
from .restart_block import restart_block

""" block.py

Authors:

Kyle Schau

This module defines the top block class. This object is the most basic object
that a multiblock datasets (see multiblock.py) can be composed of.

"""


class solver_block(restart_block, block_):
    """
    block object is the most basic object a raptorpy.multiblock.dataset
    (or one of its descendants) can be.

    Attributes
    ---------

    """

    block_type = "solver"

    def __init__(self, nblki, sp_names):
        # The c++ stuff must be instantiated first,
        # so that inhereted python side
        # attributes are assigned values, not defined
        # in the upstream __init__s
        block_.__init__(self)
        restart_block.__init__(self, nblki, sp_names)

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

        if self.block_type == "solver":
            self.array._freeze()

    def init_solver_arrays(self, config):
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

        cQshape  = [self.ni + 1, self.nj + 1, self.nk + 1, 5 + self.ns - 1]
        ifQshape = [self.ni + 2, self.nj + 1, self.nk + 1, 5 + self.ns - 1]
        jfQshape = [self.ni + 1, self.nj + 2, self.nk + 1, 5 + self.ns - 1]
        kfQshape = [self.ni + 1, self.nj + 1, self.nk + 2, 5 + self.ns - 1]

        def np_or_kokkos(names, shape):
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
                    self.array[name] = np.array(getattr(self, name),
                                                copy=False)
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
        np_or_kokkos(["x", "y", "z"], shape)

        # ------------------------------------------------------------------- #
        #       Cell center
        # ------------------------------------------------------------------- #
        shape = ccshape
        np_or_kokkos(["xc", "yc", "zc", "J"], shape)
        # Cell center metrics
        np_or_kokkos(["dEdx", "dEdy", "dEdz",
                      "dNdx", "dNdy", "dNdz",
                      "dXdx", "dXdy", "dXdz"], shape)

        # ------------------------------------------------------------------- #
        #       i face vector components and areas
        # ------------------------------------------------------------------- #
        shape = ifshape
        np_or_kokkos(["isx", "isy", "isz", "iS",
                      "inx", "iny", "inz"], ifshape)

        # ------------------------------------------------------------------- #
        #       j face vector components and areas
        # ------------------------------------------------------------------- #
        shape = jfshape
        np_or_kokkos(["jsx", "jsy", "jsz", "jS",
                      "jnx", "jny", "jnz"], shape)

        # ------------------------------------------------------------------- #
        #       k face vector components and areas
        # ------------------------------------------------------------------- #
        shape = kfshape
        np_or_kokkos(["ksx", "ksy", "ksz", "kS",
                      "knx", "kny", "knz"], shape)

        #######################################################################
        # Flow Arrays
        #######################################################################
        # ------------------------------------------------------------------- #
        #       Conservative, Primative, dQ
        # ------------------------------------------------------------------- #
        shape = cQshape
        np_or_kokkos(["Q", "q", "dQ"], shape)

        # ------------------------------------------------------------------- #
        #       Spatial derivative of primative array
        # ------------------------------------------------------------------- #
        shape = cQshape
        np_or_kokkos(["dqdx", "dqdy", "dqdz"], shape)

        # ------------------------------------------------------------------- #
        #       Thermo
        # ------------------------------------------------------------------- #
        shape = [self.ni + 1, self.nj + 1, self.nk + 1, 5 + self.ns]
        np_or_kokkos(["qh"], shape)

        # ------------------------------------------------------------------- #
        #       Transport
        # ------------------------------------------------------------------- #
        shape = [self.ni + 1, self.nj + 1, self.nk + 1, 2 + self.ns - 1]
        np_or_kokkos(["qt"], shape)

        # ------------------------------------------------------------------- #
        #       Chemistry
        # ------------------------------------------------------------------- #
        if config["thermochem"]["chemistry"]:
            shape = [self.ni + 1, self.nj + 1, self.nk + 1, 1 + self.ns]
            np_or_kokkos(["omega"], shape)

        # ------------------------------------------------------------------- #
        #       RK Stages
        # ------------------------------------------------------------------- #
        shape = cQshape
        nstorage = {"rk1": 0, "rk3": 2, "rk4": 4, "simpler": 2}
        names = [f"rhs{i}" for i in range(nstorage[config["solver"]
                                                   ["time_integration"]])]
        np_or_kokkos(names, shape)

        # ------------------------------------------------------------------- #
        #       Fluxes
        # ------------------------------------------------------------------- #
        for shape, names in ((ifQshape, ["iF"]),
                             (jfQshape, ["jF"]),
                             (kfQshape, ["kF"])):
            np_or_kokkos(names, shape)
